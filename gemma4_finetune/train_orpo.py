#!/usr/bin/env python3
"""ORPO loop-penalty LoRA trainer for Gemma 4 E2B Text (16 GB-safe).

mlx_lm 0.31.3 has no ORPO/DPO trainer, so this drives a custom loop with the
odds-ratio loss in orpo_loss.py over the existing mlx_lm LoRA + grad-checkpoint
machinery. It loads the int4 base via gemma4_text_patch, the same way train_v2.py
does for SFT.

CRASH SAFETY (the 2026-06-02 smoke OOM-froze the laptop): ORPO forwards BOTH the
chosen and rejected sequence through the model in one autograd graph, so it needs
~2x the activation memory of SFT. Mitigations, all on by default:
  - DEFAULT ACTION IS A PROBE, not training. A real run needs --train.
  - hard MLX memory cap (--mem-gb, default 12) so an overrun raises instead of
    swapping macOS to death;
  - batch size forced to 1;
  - gradient checkpointing (mlx_lm.tuner.trainer.grad_checkpoint);
  - cache cleared every --clear-every steps;
  - the probe measures peak memory on the LONGEST pairs (worst case) and exits,
    so we learn the footprint before committing to a long run.

Usage:
  # 1) measure peak memory on the worst-case pairs at a given seq length, then exit
  python train_orpo.py --probe-steps 2 --max-seq 1024
  python train_orpo.py --probe-steps 2 --max-seq 2048

  # 2) only after the probe looks safe: the real run
  python train_orpo.py --train --iters 8   --max-seq 2048 --adapter-path adapters_orpo_smoke
  python train_orpo.py --train --iters 600 --max-seq 2048 --adapter-path adapters_orpo_v6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import gemma4_text_patch  # noqa: F401  -- patches mlx_lm.models.gemma4_text on import

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.tuner.trainer import grad_checkpoint as apply_grad_checkpoint

from orpo_loss import orpo_terms

GB = 1024 ** 3
DEFAULT_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


# --------------------------------------------------------------------------- #
# Data: pre-tokenise {prompt, chosen, rejected} into (ids, prompt_offset) pairs.
# Tokenisation mirrors mlx_lm CompletionsDataset.process exactly, so the training
# format matches what the student sees at inference.
# --------------------------------------------------------------------------- #
def encode(tokenizer, prompt, completion):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full = tokenizer.apply_chat_template(messages, return_dict=False)
    offset = len(
        tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False
        )
    )
    return full, offset


def load_pairs(path, tokenizer, max_seq):
    raw = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    pairs, dropped = [], 0
    for r in raw:
        c_ids, c_off = encode(tokenizer, r["prompt"], r["chosen"])
        j_ids, j_off = encode(tokenizer, r["prompt"], r["rejected"])
        if len(c_ids) > max_seq or len(j_ids) > max_seq:
            dropped += 1
            continue
        pairs.append((c_ids, c_off, j_ids, j_off))
    return pairs, dropped


def make_batch(pair):
    """B=1 batch -> (chosen_ids, c_off, c_total, rejected_ids, j_off, j_total).

    Offsets/totals are python ints so the loss can statically slice the response
    span before the output head. total = last token index in target space (L-1).
    """
    c_ids, c_off, j_ids, j_off = pair
    cid = mx.array([c_ids])
    jid = mx.array([j_ids])
    return cid, c_off, len(c_ids) - 1, jid, j_off, len(j_ids) - 1


def report_mem(tag):
    print(
        f"  [{tag}] active={mx.get_active_memory()/GB:5.2f}GB  "
        f"peak={mx.get_peak_memory()/GB:5.2f}GB  cache={mx.get_cache_memory()/GB:5.2f}GB",
        flush=True,
    )


def evaluate(model, pairs, n, beta):
    """Mean ORPO loss + preference accuracy over n val pairs (no grad)."""
    tot_loss = tot_or = tot_acc = 0.0
    k = min(n, len(pairs))
    for i in range(k):
        t = orpo_terms(model, *make_batch(pairs[i]), beta=beta)
        mx.eval(t["loss"], t["l_or"], t["acc"])
        tot_loss += float(t["loss"]); tot_or += float(t["l_or"]); tot_acc += float(t["acc"])
        mx.clear_cache()
    return tot_loss / k, tot_or / k, tot_acc / k


def save_adapter(model, num_layers, lora_cfg, adapter_path, fname="adapters.safetensors"):
    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_path / fname), weights)
    # mlx_lm-compatible config so the bench/scorers can load this adapter later.
    (adapter_path / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_cfg,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Gemma4-E2B-IT-Text-int4")
    ap.add_argument("--data", default=str(THIS_DIR / "dataset_orpo_pilot"))
    ap.add_argument("--adapter-path", default=str(THIS_DIR / "adapters_orpo_smoke"))
    ap.add_argument("--train", action="store_true",
                    help="run real training; without it, only a memory probe runs")
    ap.add_argument("--probe-steps", type=int, default=2,
                    help="probe: full steps on the longest pairs, then exit")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--steps-per-report", type=int, default=2)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--clear-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=12.0,
                    help="hard-ish MLX memory cap; overrun raises instead of OS crash")
    args = ap.parse_args()

    # ---- memory cap FIRST, before any allocation ----
    prev = mx.set_memory_limit(int(args.mem_gb * GB))
    mx.reset_peak_memory()
    print(f"MLX memory limit set to {args.mem_gb:.1f}GB (was {prev/GB:.1f}GB)", flush=True)

    mx.random.seed(args.seed)
    random.seed(args.seed)

    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    report_mem("model loaded")

    lora_cfg = {"rank": args.rank, "scale": args.scale, "dropout": args.dropout,
                "keys": DEFAULT_KEYS}
    model.freeze()
    linear_to_lora_layers(model, args.num_layers, lora_cfg)
    apply_grad_checkpoint(model.layers[0])   # checkpoints the shared layer class
    print_trainable_parameters(model)

    tr_pairs, tr_drop = load_pairs(Path(args.data) / "train.jsonl", tokenizer, args.max_seq)
    va_pairs, va_drop = load_pairs(Path(args.data) / "valid.jsonl", tokenizer, args.max_seq)
    print(f"train pairs: {len(tr_pairs)} (dropped {tr_drop} > {args.max_seq} tok)  "
          f"valid pairs: {len(va_pairs)} (dropped {va_drop})", flush=True)
    if not tr_pairs:
        sys.exit("no training pairs fit the seq cap; raise --max-seq or check data")

    # One forward+backward per step. value_and_grad differentiates the loss and
    # passes the reporting metrics (L_OR, preference accuracy) through as aux, so we
    # never recompute the forward just to log -- which matters under the memory cap.
    def loss_aux(model, cid, c_off, c_total, jid, j_off, j_total):
        t = orpo_terms(model, cid, c_off, c_total, jid, j_off, j_total, beta=args.beta)
        return t["loss"], (t["ntoks"], t["l_or"], t["acc"])

    lvg = nn.value_and_grad(model, loss_aux)
    opt = optim.AdamW(learning_rate=args.lr)

    def step_once(pair):
        (lval, (ntoks, l_or, acc)), grad = lvg(model, *make_batch(pair))
        opt.update(model, grad)
        mx.eval(model.trainable_parameters(), opt.state, lval)
        return float(lval), float(l_or), float(acc)

    # ----------------------------------------------------------------------- #
    # PROBE: worst-case memory on the longest pairs, then exit. No long run.
    # ----------------------------------------------------------------------- #
    if not args.train:
        worst = sorted(tr_pairs, key=lambda p: len(p[0]) + len(p[2]), reverse=True)
        print(f"\nPROBE: {args.probe_steps} full step(s) on the longest pairs "
              f"at max_seq={args.max_seq}", flush=True)
        for i in range(args.probe_steps):
            pair = worst[i % len(worst)]
            ctoks, jtoks = len(pair[0]), len(pair[2])
            mx.reset_peak_memory()
            t0 = time.time()
            lval, l_or, acc = step_once(pair)
            dt = time.time() - t0
            print(f"step {i}: chosen={ctoks}tok rejected={jtoks}tok  "
                  f"loss={lval:.4f}  {dt:.1f}s", flush=True)
            report_mem(f"probe {i}")
            mx.clear_cache()
        print("\nPROBE DONE. If peak stayed well under the cap, rerun with --train.",
              flush=True)
        return

    # ----------------------------------------------------------------------- #
    # TRAIN
    # ----------------------------------------------------------------------- #
    print(f"\nTRAIN: iters={args.iters} beta={args.beta} lr={args.lr} "
          f"max_seq={args.max_seq} grad_checkpoint=on batch=1", flush=True)
    order = list(range(len(tr_pairs)))
    random.shuffle(order)
    ptr = 0
    run_loss = run_or = run_acc = 0.0
    t_start = time.time()
    for step in range(1, args.iters + 1):
        if ptr >= len(order):
            random.shuffle(order); ptr = 0
        pair = tr_pairs[order[ptr]]; ptr += 1

        lval, l_or, acc = step_once(pair)
        run_loss += lval; run_or += l_or; run_acc += acc
        if step % args.clear_every == 0:
            mx.clear_cache()

        if step % args.steps_per_report == 0:
            n = args.steps_per_report
            its = step / (time.time() - t_start)
            print(f"step {step:4d}/{args.iters}  loss={run_loss/n:.4f}  "
                  f"L_OR={run_or/n:.4f}  pref_acc={run_acc/n:.2f}  "
                  f"peak={mx.get_peak_memory()/GB:.2f}GB  {its:.2f} it/s", flush=True)
            run_loss = run_or = run_acc = 0.0

        if va_pairs and step % args.steps_per_eval == 0:
            vl, vor, vacc = evaluate(model, va_pairs, args.val_batches, args.beta)
            print(f"  [val] loss={vl:.4f}  L_OR={vor:.4f}  pref_acc={vacc:.2f}", flush=True)

        if step % args.save_every == 0:
            save_adapter(model, args.num_layers, lora_cfg, args.adapter_path,
                         fname=f"{step:07d}_adapters.safetensors")
            print(f"  saved checkpoint at step {step}", flush=True)

    save_adapter(model, args.num_layers, lora_cfg, args.adapter_path)
    if va_pairs:
        vl, vor, vacc = evaluate(model, va_pairs, args.val_batches, args.beta)
        print(f"FINAL [val] loss={vl:.4f}  L_OR={vor:.4f}  pref_acc={vacc:.2f}", flush=True)
    print(f"saved adapter -> {args.adapter_path}  (peak {mx.get_peak_memory()/GB:.2f}GB)",
          flush=True)


if __name__ == "__main__":
    main()
