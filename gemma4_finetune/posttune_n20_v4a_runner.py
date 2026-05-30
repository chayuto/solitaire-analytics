#!/usr/bin/env python3
"""Re-run the N=20 Phase 1.5 bench with a v4-A LoRA adapter loaded on top of
Gemma 3n E2B. Same structure as posttune_n20_runner.py (the v1 gemma-3n path) --
v4-A shares that base, so there is NO gemma4_text_patch here. Only the output
directory and labels differ, so v4-A checkpoint evals stack into their own
baseline_n20_v4a/ tree without colliding with the v1 runs.

max_tokens defaults to 512 to match the v1.1 bench protocol (the locked
pre-registration baseline of tier 3.15 / agreement 11/20 was measured at 512);
keep it 512 for the v4-A-vs-v1.1 comparison to stay apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

MODEL_ID = "mlx-community/gemma-3n-E2B-it-text-4bit-dwq"
REPO = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
OUT_DIR = Path(__file__).parent / "baseline_n20_v4a"
TEACHER_LOOKUP = Path(__file__).parent / "teacher_picks_n20.json"


def extract_move_index(text: str):
    candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fd = obj.get("final_decision")
        if isinstance(fd, dict) and isinstance(fd.get("move_index"), int):
            return fd["move_index"], True
        if isinstance(obj.get("move_index"), int):
            return obj["move_index"], True
    return None, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True,
                    help="path to the LoRA adapter directory")
    ap.add_argument("--out-name", default="posttune.json",
                    help="filename under baseline_n20_v4a/ to write")
    ap.add_argument("--responses-subdir", default="posttune_responses",
                    help="subdir under baseline_n20_v4a/ for raw responses")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="generation cap; 512 matches the v1.1 bench protocol")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    resp_dir = OUT_DIR / args.responses_subdir
    resp_dir.mkdir(exist_ok=True)
    teacher = json.loads(TEACHER_LOOKUP.read_text())

    state_dirs = sorted(PROMPTS_DIR.iterdir())
    assert len(state_dirs) == 20

    print(f"Loading {MODEL_ID} + adapter={args.adapter_path} ...", flush=True)
    mx.reset_peak_memory()
    t0 = time.time()
    model, tokenizer = load(MODEL_ID, adapter_path=args.adapter_path)
    print(f"  load: {time.time()-t0:.1f}s, peak after load = "
          f"{mx.get_peak_memory()/1e9:.2f} GB", flush=True)

    results = []
    overall_peak = mx.get_peak_memory() / 1e9
    for sd in state_dirs:
        state_id = sd.name
        prompt = (sd / "prompt.txt").read_text()
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        mx.reset_peak_memory()
        t_call_start = time.time()
        response = generate(
            model, tokenizer, prompt=wrapped, max_tokens=args.max_tokens,
            verbose=False,
        )
        t_call = time.time() - t_call_start
        peak = mx.get_peak_memory() / 1e9
        overall_peak = max(overall_peak, peak)

        (resp_dir / f"{state_id}.txt").write_text(response)
        idx, json_ok = extract_move_index(response)
        teacher_idx = teacher.get(state_id)
        category = state_id.split("-", 1)[0]
        row = {
            "state_id": state_id,
            "category": category,
            "call_seconds": round(t_call, 2),
            "call_peak_gb": round(peak, 2),
            "response_chars": len(response),
            "json_valid": json_ok,
            "e2b_move_index": idx,
            "teacher_move_index": teacher_idx,
            "agreement": json_ok and idx is not None and idx == teacher_idx,
        }
        results.append(row)
        print(
            f"  [{state_id[:24]:<24}] {t_call:>5.1f}s peak={peak:.2f}GB "
            f"v4a={idx} teacher={teacher_idx} json={json_ok} "
            f"agree={row['agreement']}",
            flush=True,
        )

    summary = {
        "model": MODEL_ID,
        "adapter_path": args.adapter_path,
        "max_tokens": args.max_tokens,
        "n": len(results),
        "overall_peak_gb": round(overall_peak, 2),
        "mean_call_seconds": round(
            sum(r["call_seconds"] for r in results) / len(results), 2
        ),
        "json_valid_count": sum(r["json_valid"] for r in results),
        "agreement_count": sum(r["agreement"] for r in results),
        "results": results,
    }
    (OUT_DIR / args.out_name).write_text(json.dumps(summary, indent=2))
    print(
        f"\nDone. json={summary['json_valid_count']}/{summary['n']} "
        f"agree={summary['agreement_count']}/{summary['n']} "
        f"peak={overall_peak:.2f}GB -> {OUT_DIR / args.out_name}"
    )


if __name__ == "__main__":
    main()
