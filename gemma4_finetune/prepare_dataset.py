#!/usr/bin/env python3
"""Convert a Solitaire-advisor collection log into mlx-lm training files.

Input : either a raw collection export (JSON object with an `interactions`
        array, or a JSON array of interactions), or a JSONL file with one
        interaction per line. The ingest pipeline's `data/dataset/training.jsonl`
        is JSONL and is the intended input.
Output: dataset/{train,valid,test}.jsonl in mlx-lm completions format,
        {"prompt": <full prompt>, "completion": <rawResponse JSON string>}.
        mlx-lm masks the prompt from the loss automatically.

Splitting is done at the GAME level (all turns of one game land entirely in one
split) to avoid leakage between near-identical consecutive turns. The game key
is `gameId`, `gameSeed`, or `sessionId`, whichever is present. Without any of
them the script falls back to a row-level split and prints a loud warning.

Usage:
  python prepare_dataset.py --log ../data/dataset/training.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REQUIRED_KEYS = ("board_analysis", "strategic_plan", "final_decision")


def load_interactions(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSONL: one interaction per line.
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        return data.get("interactions", [])
    return data


def clean_examples(interactions: list[dict]) -> tuple[list[dict], dict]:
    """Filter to usable rows and emit {prompt, completion, _game} dicts."""
    stats = {"total": len(interactions), "not_success": 0, "bad_json": 0, "kept": 0}
    out: list[dict] = []
    for it in interactions:
        if it.get("outcome") != "success":
            stats["not_success"] += 1
            continue
        raw = it.get("rawResponse")
        prompt = it.get("prompt")
        if not raw or not prompt:
            stats["bad_json"] += 1
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            stats["bad_json"] += 1
            continue
        if not all(k in parsed for k in REQUIRED_KEYS):
            stats["bad_json"] += 1
            continue
        # game id for leakage-free splitting; fall back to a per-row unique id.
        game = it.get("gameId") or it.get("gameSeed") or it.get("sessionId")
        out.append(
            {
                "prompt": prompt,
                "completion": raw,
                "_game": str(game) if game is not None else f"__row_{it.get('id')}",
            }
        )
        stats["kept"] += 1
    return out, stats


def split_by_game(examples: list[dict], ratios: tuple[float, float, float], seed: int):
    games = sorted({e["_game"] for e in examples})
    have_game_ids = any(not g.startswith("__row_") for g in games)
    if not have_game_ids:
        print(
            "WARNING: no `gameId`/`gameSeed` in the log — falling back to a ROW-LEVEL "
            "split. Consecutive turns of one game may leak across train/test. "
            "This is acceptable only for the current pre-P0 sample.",
            file=sys.stderr,
        )
    rng = random.Random(seed)
    rng.shuffle(games)
    n = len(games)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    buckets = {
        "train": set(games[:n_train]),
        "valid": set(games[n_train : n_train + n_valid]),
        "test": set(games[n_train + n_valid :]),
    }
    split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for e in examples:
        for name, ids in buckets.items():
            if e["_game"] in ids:
                split[name].append(e)
                break
    return split, have_game_ids, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="collection log JSON path")
    ap.add_argument("--out", default="dataset", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ratios", default="0.8,0.1,0.1", help="train,valid,test fractions"
    )
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    interactions = load_interactions(Path(args.log))
    examples, stats = clean_examples(interactions)
    print(
        f"interactions: {stats['total']}  "
        f"dropped(non-success): {stats['not_success']}  "
        f"dropped(bad JSON): {stats['bad_json']}  "
        f"kept: {stats['kept']}"
    )
    if not examples:
        sys.exit("ERROR: no usable examples — nothing to write.")

    split, have_game_ids, n_games = split_by_game(examples, ratios, args.seed)
    print(f"games: {n_games}  (game-level split: {have_game_ids})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
        print(f"  {name:<5} {len(rows):>5} examples -> {path}")

    if stats["kept"] < 200:
        print(
            "\nNOTE: small dataset — keep `iters` low (~2-3 epochs) and watch the "
            "validation loss for overfitting (see implementation plan §8)."
        )


if __name__ == "__main__":
    main()
