#!/usr/bin/env python3
"""Bench the PRIOR REASONING truncation prompt-hygiene edit (edit 4 from
the 20260526 harvester ask) against the 20-state Phase 1.5 bench.

For each of the 20 bench prompts, renders TWO variants:
  control   : the prompt unchanged
  truncated : the PRIOR REASONING block kept as headers only (each entry
              is reduced to its 'move:' line; the multi-paragraph 'why:'
              text is dropped)

Runs the model (Gemma 4 E2B text-only, optional LoRA adapter) on both
variants and produces a side-by-side comparison: JSON validity rate,
move-index parse rate, match-to-teacher rate, confidence median, response
length, and call seconds.

The hygiene edit ships only if the hard guardrails hold (JSON validity
stays at 100%) and the soft guardrails hold (match-to-teacher does not
drop by more than 5 percentage points; confidence median shift under
0.05).

Usage:
  .venv/bin/python gemma4_finetune/bench_prior_reasoning_truncation.py \\
      --adapter-path gemma4_finetune/adapters/v1_iter750 \\
      --out-name v1_iter750.json

  # Untuned baseline (no adapter):
  .venv/bin/python gemma4_finetune/bench_prior_reasoning_truncation.py \\
      --out-name untuned.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median

THIS_DIR = Path(__file__).resolve().parent

MODEL_ID = "mlx-community/Gemma4-E2B-IT-Text-int4"
REPO = THIS_DIR.parent
PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
OUT_DIR = THIS_DIR / "bench_prior_reasoning_truncation"
TEACHER_LOOKUP = THIS_DIR / "teacher_picks_n20.json"


def truncate_prior_reasoning(prompt: str) -> str:
    """Drop the multi-line `why:` explanations inside PRIOR REASONING,
    keeping only the `move:` header line for each entry."""
    m = re.search(
        r"(PRIOR REASONING[^\n]*\n)(.*?)(\nNow choose|\Z)",
        prompt,
        re.DOTALL,
    )
    if not m:
        return prompt
    header, body, tail = m.group(1), m.group(2), m.group(3)
    truncated_body = re.sub(
        r"\n     why:.*?(?=(\n  \d+\. move:)|\Z)",
        "",
        body,
        flags=re.DOTALL,
    )
    return prompt.replace(header + body + tail, header + truncated_body + tail)


def extract_move_and_confidence(text: str):
    """Pull the move_index and confidence out of the model's JSON
    response. Returns (move_index_or_None, confidence_or_None, json_ok)."""
    candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fd = obj.get("final_decision")
        if isinstance(fd, dict):
            mi = fd.get("move_index")
            cf = fd.get("confidence")
            if isinstance(mi, int):
                return mi, cf if isinstance(cf, (int, float)) else None, True
        if isinstance(obj.get("move_index"), int):
            cf = obj.get("confidence")
            return obj["move_index"], cf if isinstance(cf, (int, float)) else None, True
    return None, None, False


def run_variant(model, tokenizer, prompts, variant_name, max_tokens, resp_dir, teacher):
    """Run the model on a list of (state_id, prompt) under one variant.
    Returns the per-state result rows."""
    results = []
    for state_id, prompt in prompts:
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        mx.reset_peak_memory()
        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=wrapped, max_tokens=max_tokens, verbose=False,
        )
        t = time.time() - t0
        peak = mx.get_peak_memory() / 1e9

        (resp_dir / f"{state_id}.txt").write_text(response)
        idx, conf, json_ok = extract_move_and_confidence(response)
        teacher_idx = teacher.get(state_id)
        row = {
            "state_id": state_id,
            "category": state_id.split("-", 1)[0],
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "call_seconds": round(t, 2),
            "call_peak_gb": round(peak, 2),
            "json_valid": json_ok,
            "move_index": idx,
            "confidence": conf,
            "teacher_move_index": teacher_idx,
            "agreement": json_ok and idx is not None and idx == teacher_idx,
        }
        results.append(row)
        print(
            f"  [{variant_name:<9}|{state_id[:24]:<24}] "
            f"{t:>5.1f}s peak={peak:.2f}GB "
            f"move={idx} teacher={teacher_idx} conf={conf} "
            f"json={json_ok} agree={row['agreement']}",
            flush=True,
        )
    return results


def summarize(results, variant_name):
    n = len(results)
    json_ok = sum(1 for r in results if r["json_valid"])
    parsed = sum(1 for r in results if r["move_index"] is not None)
    agree = sum(1 for r in results if r["agreement"])
    confs = [r["confidence"] for r in results if isinstance(r["confidence"], (int, float))]
    prompt_chars = [r["prompt_chars"] for r in results]
    resp_chars = [r["response_chars"] for r in results]
    return {
        "variant": variant_name,
        "n": n,
        "json_valid_rate": round(json_ok / n, 3),
        "move_parsed_rate": round(parsed / n, 3),
        "teacher_match_rate": round(agree / n, 3),
        "confidence_mean": round(mean(confs), 3) if confs else None,
        "confidence_median": round(median(confs), 3) if confs else None,
        "prompt_chars_mean": round(mean(prompt_chars), 1),
        "response_chars_mean": round(mean(resp_chars), 1),
        "call_seconds_mean": round(mean(r["call_seconds"] for r in results), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", default=None,
                    help="Optional LoRA adapter; omit for untuned base.")
    ap.add_argument("--out-name", default="bench.json",
                    help="Filename under bench_prior_reasoning_truncation/")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    # Defer mlx imports so the truncation helper can be unit-tested
    # without the full ML stack installed.
    sys.path.insert(0, str(THIS_DIR))
    import gemma4_text_patch  # noqa: F401
    import mlx.core as mx
    from mlx_lm import generate, load
    globals()["mx"] = mx
    globals()["generate"] = generate
    globals()["load"] = load

    OUT_DIR.mkdir(exist_ok=True)
    ctrl_dir = OUT_DIR / "control_responses"
    trunc_dir = OUT_DIR / "truncated_responses"
    ctrl_dir.mkdir(exist_ok=True)
    trunc_dir.mkdir(exist_ok=True)

    teacher = json.loads(TEACHER_LOOKUP.read_text())
    state_dirs = sorted(PROMPTS_DIR.iterdir())
    assert len(state_dirs) == 20, f"expected 20 bench states, got {len(state_dirs)}"

    control_prompts = []
    truncated_prompts = []
    for sd in state_dirs:
        prompt = (sd / "prompt.txt").read_text()
        control_prompts.append((sd.name, prompt))
        truncated_prompts.append((sd.name, truncate_prior_reasoning(prompt)))

    avg_ctrl = mean(len(p) for _, p in control_prompts)
    avg_trunc = mean(len(p) for _, p in truncated_prompts)
    print(
        f"Prompt size: control mean={avg_ctrl:.0f} chars, "
        f"truncated mean={avg_trunc:.0f} chars "
        f"(reduction {(avg_ctrl - avg_trunc) / avg_ctrl * 100:.1f}%)",
        flush=True,
    )

    print(
        f"Loading {MODEL_ID}"
        + (f" + adapter={args.adapter_path}" if args.adapter_path else " (no adapter)")
        + " ...",
        flush=True,
    )
    mx.reset_peak_memory()
    t0 = time.time()
    if args.adapter_path:
        model, tokenizer = load(MODEL_ID, adapter_path=args.adapter_path)
    else:
        model, tokenizer = load(MODEL_ID)
    print(
        f"  load: {time.time() - t0:.1f}s, peak after load = "
        f"{mx.get_peak_memory() / 1e9:.2f} GB",
        flush=True,
    )

    print("\n=== CONTROL (full PRIOR REASONING) ===", flush=True)
    ctrl = run_variant(
        model, tokenizer, control_prompts, "control", args.max_tokens, ctrl_dir, teacher,
    )
    print("\n=== TRUNCATED (move-only PRIOR REASONING) ===", flush=True)
    trunc = run_variant(
        model, tokenizer, truncated_prompts, "truncated", args.max_tokens, trunc_dir, teacher,
    )

    ctrl_summary = summarize(ctrl, "control")
    trunc_summary = summarize(trunc, "truncated")
    delta = {
        "json_valid_delta_pp": round(
            (trunc_summary["json_valid_rate"] - ctrl_summary["json_valid_rate"]) * 100, 1
        ),
        "move_parsed_delta_pp": round(
            (trunc_summary["move_parsed_rate"] - ctrl_summary["move_parsed_rate"]) * 100, 1
        ),
        "teacher_match_delta_pp": round(
            (trunc_summary["teacher_match_rate"] - ctrl_summary["teacher_match_rate"]) * 100, 1
        ),
        "confidence_median_delta": (
            round(trunc_summary["confidence_median"] - ctrl_summary["confidence_median"], 3)
            if ctrl_summary["confidence_median"] is not None
               and trunc_summary["confidence_median"] is not None
            else None
        ),
        "prompt_chars_saved": round(
            ctrl_summary["prompt_chars_mean"] - trunc_summary["prompt_chars_mean"], 1
        ),
        "response_chars_delta": round(
            trunc_summary["response_chars_mean"] - ctrl_summary["response_chars_mean"], 1
        ),
        "call_seconds_delta": round(
            trunc_summary["call_seconds_mean"] - ctrl_summary["call_seconds_mean"], 2
        ),
    }

    # Guardrail verdict
    hard_pass = (
        trunc_summary["json_valid_rate"] >= 1.0
        and trunc_summary["move_parsed_rate"] >= 1.0
    )
    soft_pass = abs(delta["teacher_match_delta_pp"]) <= 5.0
    conf_pass = (
        delta["confidence_median_delta"] is None
        or abs(delta["confidence_median_delta"]) <= 0.05
    )
    verdict = (
        "SHIP" if (hard_pass and soft_pass and conf_pass) else "REVISE"
    )

    out = {
        "model": MODEL_ID,
        "adapter_path": args.adapter_path,
        "max_tokens": args.max_tokens,
        "control": ctrl_summary,
        "truncated": trunc_summary,
        "delta": delta,
        "guardrails": {
            "hard_pass_json_and_parse_at_100": hard_pass,
            "soft_pass_match_within_5pp": soft_pass,
            "conf_pass_median_shift_within_0.05": conf_pass,
            "verdict": verdict,
        },
        "per_state": {
            "control": ctrl,
            "truncated": trunc,
        },
    }

    out_path = OUT_DIR / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  variant      json%   parse%  match%  conf_med  pmt_chars  resp_chars  call_s")
    for s in (ctrl_summary, trunc_summary):
        cm = s["confidence_median"]
        print(
            f"  {s['variant']:<12} "
            f"{s['json_valid_rate'] * 100:>5.1f}   "
            f"{s['move_parsed_rate'] * 100:>5.1f}   "
            f"{s['teacher_match_rate'] * 100:>5.1f}   "
            f"{cm if cm is not None else '-':>7}   "
            f"{s['prompt_chars_mean']:>8.0f}   "
            f"{s['response_chars_mean']:>9.0f}   "
            f"{s['call_seconds_mean']:>5.1f}"
        )
    print(f"\n  delta: json {delta['json_valid_delta_pp']:+.1f}pp  "
          f"match {delta['teacher_match_delta_pp']:+.1f}pp  "
          f"conf_med {delta['confidence_median_delta']:+}  "
          f"prompt -{delta['prompt_chars_saved']:.0f} chars  "
          f"resp {delta['response_chars_delta']:+.0f} chars  "
          f"call {delta['call_seconds_delta']:+.2f}s")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
