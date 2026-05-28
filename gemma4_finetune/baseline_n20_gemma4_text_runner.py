#!/usr/bin/env python3
"""Rung 2b: same Phase 1.5 N=20 bench, but against the patched text-only
Gemma 4 E2B (audio Conformer + vision encoder physically absent, loaded via
mlx-lm with the gemma4_text loader bug worked around).

Result is the apples-to-apples v2 starting baseline: same toolchain as v1
(mlx_lm.load / generate, same tokenizer chat-template wrapping), same
prompts, same scoring path. Writes to baseline_n20_gemma4_text/ to keep both
prior runs (gemma-3n and gemma-4 multimodal) intact for comparison.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import gemma4_text_patch  # noqa: F401 -- patches mlx_lm.models.gemma4_text on import

import mlx.core as mx
from mlx_lm import generate, load

MODEL_ID = "mlx-community/Gemma4-E2B-IT-Text-int4"
REPO = THIS_DIR.parent
PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
OUT_DIR = THIS_DIR / "baseline_n20_gemma4_text"
TEACHER_LOOKUP = THIS_DIR / "teacher_picks_n20.json"


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

    print(f"Loading {MODEL_ID} (patched gemma4_text) ...", flush=True)
    mx.reset_peak_memory()
    t_load = time.time()
    model, tokenizer = load(MODEL_ID)
    load_seconds = time.time() - t_load
    load_peak_gb = mx.get_peak_memory() / 1e9
    print(f"  load: {load_seconds:.1f}s, peak after load = {load_peak_gb:.2f} GB", flush=True)

    results = []
    overall_peak = load_peak_gb
    for sd in state_dirs:
        state_id = sd.name
        prompt = (sd / "prompt.txt").read_text()
        messages = [{"role": "user", "content": prompt}]
        wrapped = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        mx.reset_peak_memory()
        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=wrapped, max_tokens=2048, verbose=False
        )
        t_call = time.time() - t0
        peak = mx.get_peak_memory() / 1e9
        overall_peak = max(overall_peak, peak)

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
            f"g4t={idx} teacher={teacher_idx} json={json_ok} "
            f"agree={row['agreement']}",
            flush=True,
        )

    summary = {
        "model": MODEL_ID,
        "patch": "gemma4_text_patch (strips redundant KV-shared layer weights)",
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
    (OUT_DIR / "baseline_n20_gemma4_text.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nDone. json={summary['json_valid_count']}/{summary['n']} "
        f"agree={summary['agreement_count']}/{summary['n']} peak={overall_peak:.2f}GB "
        f"-> {OUT_DIR / 'baseline_n20_gemma4_text.json'}"
    )


if __name__ == "__main__":
    main()
