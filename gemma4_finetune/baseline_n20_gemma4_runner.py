#!/usr/bin/env python3
"""Rung 2 of the Gemma 4 E2B v2 exploration: untuned Gemma 4 E2B on the same
Phase 1.5 20-state bench used for the Gemma 3n baseline.

Mirrors baseline_n20_runner.py exactly but routes load + generate through
mlx-vlm (where the gemma4 architecture lives) instead of mlx-lm. Writes to
baseline_n20_gemma4/ to keep the gemma-3n results at baseline_n20/ untouched.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_ID = "mlx-community/gemma-4-E2B-it-4bit"
REPO = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
OUT_DIR = Path(__file__).parent / "baseline_n20_gemma4"
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
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "responses").mkdir(exist_ok=True)
    teacher = json.loads(TEACHER_LOOKUP.read_text())

    state_dirs = sorted(PROMPTS_DIR.iterdir())
    assert len(state_dirs) == 20, f"expected 20 states, got {len(state_dirs)}"

    print(f"Loading {MODEL_ID} via mlx-vlm ...", flush=True)
    mx.reset_peak_memory()
    t_load = time.time()
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    load_seconds = time.time() - t_load
    load_peak_gb = mx.get_peak_memory() / 1e9
    print(f"  load: {load_seconds:.1f}s, peak after load = {load_peak_gb:.2f} GB", flush=True)

    results = []
    overall_peak = load_peak_gb
    for sd in state_dirs:
        state_id = sd.name
        prompt = (sd / "prompt.txt").read_text()
        wrapped = apply_chat_template(processor, config, prompt, num_images=0)

        mx.reset_peak_memory()
        t0 = time.time()
        out = generate(
            model,
            processor,
            wrapped,
            max_tokens=512,
            temperature=0.0,
            verbose=False,
        )
        t_call = time.time() - t0
        peak = mx.get_peak_memory() / 1e9
        overall_peak = max(overall_peak, peak)

        response = out.text if hasattr(out, "text") else str(out)
        (OUT_DIR / "responses" / f"{state_id}.txt").write_text(response)
        idx, json_ok = extract_move_index(response)
        teacher_idx = teacher.get(state_id)
        category = state_id.split("-", 1)[0]

        row = {
            "state_id": state_id,
            "category": category,
            "prompt_chars": len(prompt),
            "wrapped_chars": len(wrapped),
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
            f"g4={idx} teacher={teacher_idx} json={json_ok} "
            f"agree={row['agreement']}",
            flush=True,
        )

    summary = {
        "model": MODEL_ID,
        "n": len(results),
        "load_seconds": round(load_seconds, 1),
        "overall_peak_gb": round(overall_peak, 2),
        "mean_call_seconds": round(
            sum(r["call_seconds"] for r in results) / len(results), 2
        ),
        "json_valid_count": sum(r["json_valid"] for r in results),
        "agreement_count": sum(r["agreement"] for r in results),
        "results": results,
    }
    (OUT_DIR / "baseline_n20_gemma4.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nDone. json={summary['json_valid_count']}/{summary['n']} "
        f"agree={summary['agreement_count']}/{summary['n']} peak={overall_peak:.2f}GB "
        f"-> {OUT_DIR / 'baseline_n20_gemma4.json'}"
    )


if __name__ == "__main__":
    main()
