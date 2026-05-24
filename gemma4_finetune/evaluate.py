#!/usr/bin/env python3
"""Evaluate a fine-tuned E2B advisor against the 31B teacher.

For each held-out test example: generate with the LoRA-adapted model, parse the
output, and score it. Success criterion is "matches the teacher", so the primary
metric is top-1 `move_index` agreement.

Metrics reported:
  - validity   : output parses as JSON with the 3 required keys
  - legal      : chosen move_index exists in the prompt's legalMoves
  - agreement  : E2B move_index == teacher move_index   (PRIMARY)
  - baseline   : share of the single most common teacher move (trivial-guess floor)

Usage:
  python evaluate.py --test dataset/test.jsonl --adapter-path adapters
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = ("board_analysis", "strategic_plan", "final_decision")


def extract_legal_move_count(prompt: str) -> int | None:
    """Count entries in the prompt's legalMoves array via bracket matching."""
    m = re.search(r'"legalMoves"\s*:', prompt)
    if not m:
        return None
    s = prompt.index("[", m.end())
    depth = 0
    for j in range(s, len(prompt)):
        if prompt[j] == "[":
            depth += 1
        elif prompt[j] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return len(json.loads(prompt[s : j + 1]))
                except json.JSONDecodeError:
                    return None
    return None


def parse_decision(text: str):
    """Return (move_index, ok_3keys) from a model output string."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # tolerate prose around the object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, False
    ok = all(k in obj for k in REQUIRED_KEYS)
    mi = obj.get("final_decision", {}).get("move_index") if isinstance(obj, dict) else None
    return mi, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default="dataset/test.jsonl")
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--adapter-path", default="adapters")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    from mlx_lm import generate, load  # imported here so --help works without mlx

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    rows = [json.loads(line) for line in Path(args.test).read_text().splitlines() if line.strip()]
    teacher_moves = [parse_decision(r["completion"])[0] for r in rows]
    baseline = max(Counter(teacher_moves).values()) / len(rows) if rows else 0.0

    n = len(rows)
    valid = legal = agree = 0
    for r in rows:
        out = generate(
            model, tokenizer, prompt=r["prompt"], max_tokens=args.max_tokens, verbose=False
        )
        mi, ok = parse_decision(out)
        if ok:
            valid += 1
        nlegal = extract_legal_move_count(r["prompt"])
        if mi is not None and nlegal is not None and 0 <= mi < nlegal:
            legal += 1
        if mi is not None and mi == parse_decision(r["completion"])[0]:
            agree += 1

    pct = lambda x: f"{100 * x / n:.1f}%" if n else "n/a"
    print(f"test examples     : {n}")
    print(f"JSON validity     : {pct(valid)}  ({valid}/{n})")
    print(f"legal move        : {pct(legal)}  ({legal}/{n})")
    print(f"teacher agreement : {pct(agree)}  ({agree}/{n})   <- PRIMARY")
    print(f"trivial baseline  : {100 * baseline:.1f}%  (most-common teacher move)")
    if n:
        verdict = "PASS" if (valid / n >= 0.98 and legal / n >= 0.98 and agree / n > baseline) else "REVIEW"
        print(f"verdict           : {verdict}")


if __name__ == "__main__":
    main()
