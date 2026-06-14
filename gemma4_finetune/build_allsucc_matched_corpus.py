#!/usr/bin/env python3
"""Build the FILTER-VS-VOLUME ablation corpus for the won-only gate.

The won-only gate changed two things vs the prior all-success SFTs at once:
the won/lost FILTER (keep only turns from games the teacher won) AND scale
(36 won sessions). This corpus isolates the filter: same row budget as the
gate (2492 rows after the 13 eval decks are held out), but drawn from the
NATURAL won+lost success mix instead of won-only. Same eval-deck holdout,
same downstream recipe.

Decision rule (plan doc 10.3 / 11.6):
  gate >> all-succ-matched  -> the won-only FILTER is the lever; scale wins.
  gate ~= all-succ-matched  -> it was volume/recency, not the filter.

Whole games are sampled (never split a game across the boundary) via a fixed
shuffle seed so the build is reproducible. Reports the won/lost split of the
sample so the contrast against the 100%-won gate is explicit.

Run:
  .venv/bin/python gemma4_finetune/build_allsucc_matched_corpus.py
  # -> data/dataset/training_allsucc_matched.jsonl
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "data" / "dataset" / "training.jsonl"
OUT = REPO / "data" / "dataset" / "training_allsucc_matched.jsonl"
EVAL_SEEDS = {495097115, 1388178981, 3123337720, 4250754298, 239901548,
              350743738, 3263196305, 2703165610, 405489085, 4221577640,
              4161700176, 3841057237, 4197389931}
TARGET_ROWS = 2492   # match the won-only gate corpus after eval holdout
SHUFFLE_SEED = 13     # fixed for reproducibility (no Date/random at runtime)


def won_session_ids() -> set[str]:
    sids: set[str] = set()
    for f in sorted(glob.glob(str(REPO / "data" / "raw" / "solitaire-win-*.json"))):
        try:
            wr = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(wr, dict) and wr.get("gameWon") is True and wr.get("gameSessionId"):
            sids.add(str(wr["gameSessionId"]))
    return sids


def main() -> None:
    import argparse
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=TARGET_ROWS,
                    help="row budget; whole games sampled until reached. "
                         "0 = use the ENTIRE non-eval success pool (the "
                         "volume-scaling arm: max data, natural mix).")
    ap.add_argument("--out", default=str(OUT), help="output jsonl path")
    args = ap.parse_args()
    target = args.max_rows
    out = Path(args.out)

    won = won_session_ids()
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    pool = [r for r in rows if not (r.get("seed") and int(r["seed"]) in EVAL_SEEDS)]

    by_game: dict[str, list] = defaultdict(list)
    for r in pool:
        by_game[str(r.get("sessionId"))].append(r)

    game_ids = sorted(by_game)          # deterministic order before the shuffle
    random.Random(SHUFFLE_SEED).shuffle(game_ids)

    kept_rows: list = []
    kept_games: list[str] = []
    for sid in game_ids:
        if target and len(kept_rows) >= target:
            break
        kept_rows.extend(by_game[sid])
        kept_games.append(sid)

    out.write_text("".join(json.dumps(r) + "\n" for r in kept_rows))

    won_g = sum(1 for sid in kept_games if sid in won)
    won_r = sum(len(by_game[sid]) for sid in kept_games if sid in won)
    print(f"non-eval pool: {len(pool)} rows / {len(by_game)} games "
          f"({sum(len(by_game[s]) for s in by_game if s in won)} won-rows)")
    print(f"matched all-success sample (seed {SHUFFLE_SEED}):")
    print(f"  rows:  {len(kept_rows)}  (target {TARGET_ROWS})")
    print(f"  games: {len(kept_games)}  ({won_g} won / {len(kept_games)-won_g} lost)")
    print(f"  won-row fraction: {100*won_r/max(len(kept_rows),1):.0f}%  "
          f"(gate arm = 100%); this is the ablation contrast")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
