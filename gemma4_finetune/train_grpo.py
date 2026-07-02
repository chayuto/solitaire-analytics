#!/usr/bin/env python3
"""GRPO / RLOO policy-gradient LoRA trainer for Gemma 4 E2B Text (16 GB-safe).

Iteration-0 on-policy RFT: CONTINUE-TRAIN the volume adapter (the shipped lead
policy, and the policy that generated the best-of-N rollouts) on its own rollouts
weighted by group-relative advantage. This is the first lever that can push the
policy AWAY from its losing trajectories, which SFT-on-self-wins never could
(that line closed at parity 2026-07-01).

The loss (grpo_loss.grpo_terms) is advantage-weighted REINFORCE with the ORPO
slice-trick for 16 GB safety. Because we do ONE pass (a couple of epochs) over
freshly-sampled on-policy rollouts, plain REINFORCE with a group baseline is the
exact GRPO objective minus the PPO clip/KL (which only matter across reused
rollout epochs). NO explicit KL in this first cut: the regularisers are (1) we
resume the volume LoRA rather than restart, (2) a small LR, (3) low-rank LoRA,
(4) few epochs. If the eval shows JSON-format collapse (the known KL-free risk on
top of volume's existing JSON regression), that motivates adding a reference-KL
term next.

Gradient is ACCUMULATED over a minibatch of turns before each optimizer step, so
advantage MAGNITUDE is preserved within the batch (Adam would otherwise normalise
away the difference between a +2.0 and a +0.05 turn if stepped per-example).
Activation peak stays at one sequence (turns are summed into a grad accumulator
one at a time), so the minibatch does not raise memory.

CRASH SAFETY mirrors train_orpo.py: hard MLX memory cap (--mem-gb), batch forced
to B=1 forwards, gradient checkpointing, cache cleared each step, and a PROBE
mode (default, no --train) that measures peak on the longest turns then exits.

Usage:
  # 1) memory probe on the longest turns at seq 2048, then exit
  .venv/bin/python gemma4_finetune/train_grpo.py --probe-steps 2

  # 2) the real run: resume volume, 2 epochs, gentle LR
  .venv/bin/python gemma4_finetune/train_grpo.py --train \
      --data gemma4_finetune/grpo_corpus_iter1.jsonl \
      --resume-adapter gemma4_finetune/adapters_volume/adapters.safetensors \
      --adapter-path gemma4_finetune/adapters_grpo_iter1 \
      --epochs 2 --lr 2e-5 --minibatch 8
"""
from __future__ import annotations

import argparse
import json
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
from mlx.utils import tree_flatten, tree_map

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.tuner.trainer import grad_checkpoint as apply_grad_checkpoint

from grpo_loss import grpo_terms, mean_logp
from train_orpo import encode, save_adapter  # reuse the exact encode + adapter writer

GB = 1024 ** 3
DEFAULT_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_rows(path, tokenizer, max_seq):
    raw = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows, dropped = [], 0
    for r in raw:
        ids, off = encode(tokenizer, r["prompt"], r["completion"])
        if len(ids) > max_seq:
            dropped += 1
            continue
        # 4-tuple: ref_logp (4th) is filled by the pre-pass when a trust region
        # is on, else stays None (KL-free path = run 1).
        rows.append((ids, off, float(r["advantage"]), None))
    return rows, dropped


def resume_volume(model, path: Path):
    """Load the volume LoRA weights into the freshly-created LoRA layers and
    VERIFY they landed (fresh linear_to_lora_layers zero-inits every lora_b, so a
    non-zero lora_b after load proves the resume worked and we are not silently
    training from the identity adapter)."""
    tparams = dict(tree_flatten(model.trainable_parameters()))
    vw = mx.load(str(path))
    matched = [k for k in vw if k in tparams]
    if not matched:
        raise SystemExit(f"resume: NONE of {len(vw)} volume keys matched the "
                         f"{len(tparams)} trainable keys -- structure mismatch")
    model.load_weights(str(path), strict=False)
    mx.eval(model.parameters())
    nonzero_b = 0
    for k, v in tree_flatten(model.trainable_parameters()):
        if k.endswith("lora_b"):
            if float(mx.abs(v).sum()) > 0:
                nonzero_b += 1
    if nonzero_b == 0:
        raise SystemExit("resume: every lora_b is still zero after load -- "
                         "volume weights did not apply")
    print(f"resumed volume: {len(matched)}/{len(vw)} keys matched, "
          f"{nonzero_b} non-zero lora_b tensors", flush=True)


def report_mem(tag):
    print(f"  [{tag}] active={mx.get_active_memory()/GB:5.2f}GB  "
          f"peak={mx.get_peak_memory()/GB:5.2f}GB  cache={mx.get_cache_memory()/GB:5.2f}GB",
          flush=True)


def diag_logp(model, rows):
    """Mean length-normalised logp over a fixed diagnostic sample (no grad)."""
    if not rows:
        return float("nan")
    tot = 0.0
    for (ids, off, _adv, _ref) in rows:
        v = mean_logp(model, mx.array([ids]), off, len(ids) - 1)
        mx.eval(v)
        tot += float(v)
        mx.clear_cache()
    return tot / len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Gemma4-E2B-IT-Text-int4")
    ap.add_argument("--data", default=str(THIS_DIR / "grpo_corpus_iter1.jsonl"))
    ap.add_argument("--resume-adapter",
                    default=str(THIS_DIR / "adapters_volume" / "adapters.safetensors"),
                    help="volume LoRA weights to continue-train (the RFT start policy)")
    ap.add_argument("--adapter-path", default=str(THIS_DIR / "adapters_grpo_iter1"))
    ap.add_argument("--train", action="store_true",
                    help="run real training; without it, only a memory probe runs")
    ap.add_argument("--probe-steps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="subsample the corpus to this many turns (0 = all), "
                         "stratified to preserve the positive/negative advantage "
                         "ratio. The per-turn body forward on ~2000-tok v1.6 "
                         "prompts is the compute bottleneck, so a full 12k-row "
                         "corpus is ~20h; ~1000 rows keeps it to an overnight run. "
                         "The full corpus stays on disk for later iterations.")
    ap.add_argument("--minibatch", type=int, default=8,
                    help="turns accumulated per optimizer step (preserves advantage "
                         "magnitude within the batch); activation peak stays B=1")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip-grad-norm", type=float, default=0.0,
                    help="clip the global gradient norm to this before each step "
                         "(0 = off). NOTE: run-1's collapse was slow DRIFT (grad "
                         "norm stayed ~0.3), not spikes, so clipping does not bind; "
                         "use --trust-region-beta instead.")
    ap.add_argument("--trust-region-beta", type=float, default=0.0,
                    help="one-sided downward trust region (0 = off = KL-free run-1). "
                         "When >0, a pre-pass caches each turn's reference logp under "
                         "volume, and the loss adds beta*(ref_logp - logp)^2 whenever "
                         "logp drops below reference -- opposing the collapse "
                         "direction while leaving positive learning free.")
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="0.0 = deterministic logp = the deployed (inference) policy, "
                         "so the RFT gradient is on the exact policy that sampled "
                         "the rollouts")
    ap.add_argument("--steps-per-report", type=int, default=20)
    ap.add_argument("--steps-per-eval", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=13.0)
    args = ap.parse_args()

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
    resume_volume(model, Path(args.resume_adapter))
    apply_grad_checkpoint(model.layers[0])
    print_trainable_parameters(model)

    rows, dropped = load_rows(Path(args.data), tokenizer, args.max_seq)
    print(f"train turns: {len(rows)} (dropped {dropped} > {args.max_seq} tok)", flush=True)
    if not rows:
        sys.exit("no training rows fit the seq cap")

    if args.max_rows and len(rows) > args.max_rows:
        pos = [r for r in rows if r[2] > 0]
        neg = [r for r in rows if r[2] <= 0]
        frac = args.max_rows / len(rows)
        rng = random.Random(args.seed)
        rng.shuffle(pos); rng.shuffle(neg)
        keep = pos[:max(1, round(len(pos) * frac))] + neg[:max(1, round(len(neg) * frac))]
        rng.shuffle(keep)
        print(f"subsampled {len(rows)} -> {len(keep)} turns "
              f"({sum(1 for r in keep if r[2] > 0)} pos / "
              f"{sum(1 for r in keep if r[2] <= 0)} neg), stratified by advantage sign",
              flush=True)
        rows = keep

    # ---- Trust-region pre-pass: cache each turn's reference (volume) logp ----
    # The policy currently == volume (resumed, not yet updated), so a no-grad
    # forward now yields the reference logp the trust region anchors to. Fills the
    # 4th tuple slot; skipped (ref stays None) when the trust region is off.
    if args.trust_region_beta > 0:
        print(f"trust-region pre-pass: caching reference logp for {len(rows)} turns "
              f"(beta={args.trust_region_beta}) ...", flush=True)
        t_pp = time.time()
        newrows = []
        for i, (ids, off, adv, _ref) in enumerate(rows):
            rlp = float(mean_logp(model, mx.array([ids]), off, len(ids) - 1))
            newrows.append((ids, off, adv, rlp))
            mx.clear_cache()
            if (i + 1) % 100 == 0:
                print(f"  ref-logp {i+1}/{len(rows)}  ({time.time()-t_pp:.0f}s)", flush=True)
        rows = newrows
        refs = [r[3] for r in rows]
        print(f"  done in {time.time()-t_pp:.0f}s; reference logp mean "
              f"{sum(refs)/len(refs):+.3f}, range [{min(refs):+.3f}, {max(refs):+.3f}]",
              flush=True)

    # Fixed diagnostic samples: the most positive and most negative advantage
    # turns. Expect pos logp to hold/rise and neg logp to fall as RFT proceeds
    # (the training-set analog of the loss self-test).
    by_adv = sorted(rows, key=lambda r: r[2])
    diag_neg = by_adv[:24]
    diag_pos = by_adv[-24:]

    def loss_aux(model, cid, c_off, c_total, adv, ref):
        t = grpo_terms(model, cid, c_off, c_total, adv,
                       ref_logp=ref, tr_beta=args.trust_region_beta)
        return t["loss"], (t["ntoks"], t["logp"])

    lvg = nn.value_and_grad(model, loss_aux)
    opt = optim.AdamW(learning_rate=args.lr)

    def accum_step(batch):
        """Sum the advantage-weighted gradient over the minibatch, optionally
        clip its global norm (the trust-region proxy that stops the KL-free
        negative branch from running away), step once. Returns
        (mean_loss, pre_clip_grad_norm)."""
        grads = None
        tot_loss = 0.0
        for (ids, off, adv, ref) in batch:
            cid = mx.array([ids])
            (lval, _aux), g = lvg(model, cid, off, len(ids) - 1, adv, ref)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            tot_loss += float(lval)
        grads = tree_map(lambda a: a / len(batch), grads)
        if args.clip_grad_norm > 0:
            grads, gnorm = optim.clip_grad_norm(grads, args.clip_grad_norm)
        else:
            _, gnorm = optim.clip_grad_norm(grads, 1e9)  # measure only, no clip
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        return tot_loss / len(batch), float(gnorm)

    # ---- PROBE: worst-case memory on the longest turns, then exit ----
    if not args.train:
        worst = sorted(rows, key=lambda r: len(r[0]), reverse=True)
        print(f"\nPROBE: {args.probe_steps} step(s) of minibatch={args.minibatch} on "
              f"the longest turns at max_seq={args.max_seq}", flush=True)
        for i in range(args.probe_steps):
            batch = worst[i * args.minibatch:(i + 1) * args.minibatch] or worst[:args.minibatch]
            mx.reset_peak_memory()
            t0 = time.time()
            lv, gn = accum_step(batch)
            print(f"step {i}: longest={len(batch[0][0])}tok  loss={lv:.4f}  "
                  f"gnorm={gn:.2f}  {time.time()-t0:.1f}s", flush=True)
            report_mem(f"probe {i}")
            mx.clear_cache()
        print("\nPROBE DONE. If peak stayed under the cap, rerun with --train.", flush=True)
        return

    # ---- TRAIN ----
    steps_per_epoch = (len(rows) + args.minibatch - 1) // args.minibatch
    total_steps = args.epochs * steps_per_epoch
    print(f"\nTRAIN: epochs={args.epochs} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} minibatch={args.minibatch} lr={args.lr} "
          f"grad_checkpoint=on", flush=True)
    print(f"  diag logp @ start:  pos={diag_logp(model, diag_pos):+.4f}  "
          f"neg={diag_logp(model, diag_neg):+.4f}", flush=True)

    order = list(range(len(rows)))
    step = 0
    run_loss = 0.0
    run_gn = 0.0
    t_start = time.time()
    for epoch in range(args.epochs):
        random.shuffle(order)
        for bstart in range(0, len(rows), args.minibatch):
            batch = [rows[order[i]] for i in range(bstart, min(bstart + args.minibatch, len(rows)))]
            l, gn = accum_step(batch)
            run_loss += l
            run_gn += gn
            mx.clear_cache()
            step += 1

            if step % args.steps_per_report == 0:
                its = step / (time.time() - t_start)
                print(f"step {step:4d}/{total_steps}  loss={run_loss/args.steps_per_report:+.4f}  "
                      f"gnorm={run_gn/args.steps_per_report:6.1f}  "
                      f"peak={mx.get_peak_memory()/GB:.2f}GB  {its:.2f} it/s", flush=True)
                run_loss = 0.0
                run_gn = 0.0

            if step % args.steps_per_eval == 0:
                dp, dn = diag_logp(model, diag_pos), diag_logp(model, diag_neg)
                print(f"  [diag] pos-adv logp={dp:+.4f}  neg-adv logp={dn:+.4f}  "
                      f"(want pos up / neg down)", flush=True)

            if step % args.save_every == 0:
                save_adapter(model, args.num_layers, lora_cfg, args.adapter_path,
                             fname=f"{step:07d}_adapters.safetensors")
                print(f"  saved checkpoint at step {step}", flush=True)

    save_adapter(model, args.num_layers, lora_cfg, args.adapter_path)
    dp, dn = diag_logp(model, diag_pos), diag_logp(model, diag_neg)
    print(f"\nFINAL diag logp:  pos={dp:+.4f}  neg={dn:+.4f}", flush=True)
    print(f"saved adapter -> {args.adapter_path}  (peak {mx.get_peak_memory()/GB:.2f}GB)",
          flush=True)


if __name__ == "__main__":
    main()
