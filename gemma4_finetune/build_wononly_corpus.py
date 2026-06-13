#!/usr/bin/env python3
"""Build the v5 WON-GAMES-ONLY training log for Gemma 4 E2B distillation.

Game-level win/loss filter (vs the move-level reversal filter that produced v3/
v4-A). Keeps every turn of a game the teacher WON, drops every turn of a game it
did not win. This targets the v2 finding (lost-game doom-loop turns poison
training) at the game level instead of guessing which individual moves are bad.

Method:
  - Win records live at data/raw/solitaire-win-<gid>-<ts>.json; each carries a
    `gameSessionId` and `gameWon: true`.
  - Training turns live at data/dataset/training.jsonl, each tagged `sessionId`.
  - A training turn is WON iff its sessionId is the gameSessionId of some win
    record. (Verified 2026-05-30: 7 won sessions, 1923 turns, 51.2% of corpus;
    the won/lost split is clean -- no lost-attempt sessions share a won seed.)

Output: data/dataset/training_wononly.jsonl (same row schema as training.jsonl,
filtered). Feed it to prepare_dataset.py exactly like the raw log.

Run:
  .venv/bin/python gemma4_finetune/build_wononly_corpus.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_LOG = REPO / "data" / "dataset" / "training.jsonl"
WIN_GLOB = str(REPO / "data" / "raw" / "solitaire-win-*.json")
OUT_LOG = REPO / "data" / "dataset" / "training_wononly.jsonl"


def won_session_ids() -> set[str]:
    sids: set[str] = set()
    for f in sorted(glob.glob(WIN_GLOB)):
        try:
            wr = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(wr, dict):
            continue
        if wr.get("gameWon") is not True:
            continue
        gsid = wr.get("gameSessionId")
        if gsid:
            sids.add(str(gsid))
    return sids


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-seeds", default=None,
                    help="Comma-separated deck seeds to HOLD OUT of the corpus "
                         "(eval decks must not appear in training; the 24-deck "
                         "benchmark is built FROM win records, so without this "
                         "every benchmark eval is an in-training memorization "
                         "probe).")
    ap.add_argument("--out", default=str(OUT_LOG),
                    help="output jsonl path")
    args = ap.parse_args()
    excluded = ({int(s) for s in args.exclude_seeds.split(",") if s.strip()}
                if args.exclude_seeds else set())
    out_log = Path(args.out)

    won = won_session_ids()
    rows = [
        json.loads(l)
        for l in TRAIN_LOG.read_text().splitlines()
        if l.strip()
    ]
    kept_all = [r for r in rows if str(r.get("sessionId")) in won]
    kept = [r for r in kept_all
            if not (r.get("seed") and int(r["seed"]) in excluded)]

    out_log.write_text("".join(json.dumps(r) + "\n" for r in kept))

    per_seed = Counter(r.get("seed") for r in kept)
    print(f"won sessions (gameWon=true): {len(won)}")
    print(f"training rows total:         {len(rows)}")
    print(f"won-only rows (pre-holdout): {len(kept_all)}")
    if excluded:
        print(f"held-out seeds:              {len(excluded)} "
              f"({len(kept_all) - len(kept)} rows dropped)")
    print(f"won-only rows kept:          {len(kept)}  "
          f"({100*len(kept)/max(len(rows),1):.1f}%)")
    print(f"distinct seeds in won-only:  {len(per_seed)}")
    print("per-seed turn counts:")
    for seed, n in per_seed.most_common():
        print(f"    {seed}: {n}  ({100*n/max(len(kept),1):.0f}%)")
    print(f"-> {out_log}")


if __name__ == "__main__":
    main()
