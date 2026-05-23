#!/usr/bin/env python3
"""Generate a synthetic dataset for the memory smoke test.

The smoke test must stress the trainer with examples the same *shape* as the real
Solitaire advisor data (~1764-token prompt + ~289-token completion ≈ 2050 tokens),
WITHOUT needing the real (pilot-blocked) dataset. Content is filler — only token
length matters here, since the goal is to measure peak memory at seq-len 2048.

Output: smoke_data/{train,valid}.jsonl in mlx-lm completions format.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Rough heuristic: ~4 characters per token for English/JSON text.
CHARS_PER_TOKEN = 4
PROMPT_TOKENS = 1764
COMPLETION_TOKENS = 289

_WORDS = (
    "tableau foundation stock waste column hidden card reveal sequence king "
    "alternating descending suit hearts spades clubs diamonds draw move legal "
    "strategic analysis board decision confidence priority heuristic".split()
)


def _filler(n_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    target_chars = n_tokens * CHARS_PER_TOKEN
    out: list[str] = []
    total = 0
    while total < target_chars:
        w = rng.choice(_WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)


def make_example(i: int) -> dict:
    prompt = (
        "You are an expert Klondike Solitaire strategist acting as an advisor.\n"
        + _filler(PROMPT_TOKENS, seed=i)
    )
    completion = json.dumps(
        {
            "board_analysis": _filler(COMPLETION_TOKENS // 2, seed=i + 10_000),
            "strategic_plan": _filler(COMPLETION_TOKENS // 2, seed=i + 20_000),
            "final_decision": {
                "move_index": 0,
                "confidence": 0.9,
                "alternative_move_index": 1,
            },
        }
    )
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="smoke_data", help="output directory")
    ap.add_argument("--train", type=int, default=64, help="num train examples")
    ap.add_argument("--valid", type=int, default=16, help="num valid examples")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, count, base in (("train", args.train, 0), ("valid", args.valid, 100_000)):
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for i in range(count):
                f.write(json.dumps(make_example(base + i)) + "\n")
        print(f"wrote {count:>4} examples -> {path}")
    print(
        f"\nExample shape: ~{PROMPT_TOKENS}+{COMPLETION_TOKENS} tokens "
        f"(~{PROMPT_TOKENS + COMPLETION_TOKENS} total) — matches real data."
    )


if __name__ == "__main__":
    main()
