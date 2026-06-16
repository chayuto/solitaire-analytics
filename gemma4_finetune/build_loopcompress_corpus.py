#!/usr/bin/env python3
"""Build the LOOP-COMPRESSION spike corpus (2026-06-16).

Hypothesis (plan doc 20260614_generalization_run_plan.md section 5.1): the
teacher's doom-loops poison imitation, so compress each game down to its
progressing spine, keeping the ESCAPE move with its real loop-context.

Mechanism: EXACT board-state cycle collapse. Replay each game's logged
decisions in turn order; hash the OBSERVABLE physical board the model faced
(foundations + tableau faceUp/faceDownCount + waste top + stock count +
recycle flag), EXCLUDING history (recentMoves / drawTimeline / seenDrawPile)
since those differ even at an identical board. When a board hash recurs, the
decisions between the two visits formed a net no-op cycle: keep only the
decision made at the LAST occurrence of that board (the escape that finally
left it) and drop the loop body. The kept rows are the ORIGINAL rows,
unmodified, so the escape still shows the loop in its RECENT MOVES (the
loop-context is what makes the escape teachable).

Why EXACT state (not a looser key): the gap between exact (~5.5% cut) and a
tableau-only relaxed key (~51% cut) is almost entirely DRAW decisions ("tableau
unchanged while the stock cycles"). Cutting those would starve the corpus of
draws and reproduce the v7 ORPO draw-starvation failure (a corpus that under-
represents an action makes the student stop taking it). Exact-state cycles are
draw-safe: they only collapse genuine net-no-op repeats. The cut is small
because pure draw-free tableau loops are rare; that is a finding, not a bug.

Same 13 eval decks held out and same downstream recipe as the volume arm, so
loopcompress-vs-volume is a clean paired comparison (loopcompress is a strict
subset of the volume pool with the exact-cycle bodies removed).

Run:
  .venv/bin/python gemma4_finetune/build_loopcompress_corpus.py
  # -> data/dataset/training_loopcompress.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))
from load_export import parse_board  # noqa: E402

TRAIN = REPO / "data" / "dataset" / "training.jsonl"
OUT = REPO / "data" / "dataset" / "training_loopcompress.jsonl"
EVAL_SEEDS = {495097115, 1388178981, 3123337720, 4250754298, 239901548,
              350743738, 3263196305, 2703165610, 405489085, 4221577640,
              4161700176, 3841057237, 4197389931}


def board_key(prompt: str) -> str | None:
    """Canonical hash of the OBSERVABLE physical board (history excluded)."""
    b = parse_board(prompt)
    if not b:
        return None
    tab = tuple((tuple(c.get("faceUp") or []), c.get("faceDownCount"))
                for c in (b.get("tableau") or []))
    return json.dumps([b.get("foundations"), tab, b.get("discardTop"),
                       b.get("drawPileCount"), b.get("canRecycleStock")],
                      sort_keys=True, default=str)


def collapse(keys: list[str | None]) -> list[int]:
    """Keep the decision at the LAST occurrence of each board (the escape);
    drop the no-op cycle bodies in between. Returns kept indices, in order."""
    last = {k: i for i, k in enumerate(keys) if k is not None}
    kept: list[int] = []
    k = 0
    while k < len(keys):
        j = last[keys[k]] if keys[k] is not None else k
        kept.append(j)
        k = j + 1
    return kept


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output jsonl path")
    args = ap.parse_args()
    out = Path(args.out)

    won = won_session_ids()
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    pool = [r for r in rows if not (r.get("seed") and int(r["seed"]) in EVAL_SEEDS)]

    by_game: dict[str, list] = defaultdict(list)
    for r in pool:
        by_game[str(r.get("sessionId"))].append(r)

    kept_rows: list = []
    games_with_cycles = 0
    cut_by_outcome = {"won": 0, "lost": 0}
    for sid in sorted(by_game):
        rs = sorted(by_game[sid], key=lambda r: r.get("turnIndex", 0))
        keys = [board_key(r["prompt"]) for r in rs]
        kept_idx = collapse(keys)
        if len(kept_idx) < len(rs):
            games_with_cycles += 1
            cut_by_outcome["won" if sid in won else "lost"] += len(rs) - len(kept_idx)
        kept_rows.extend(rs[i] for i in kept_idx)

    out.write_text("".join(json.dumps(r) + "\n" for r in kept_rows))

    n_in, n_out = len(pool), len(kept_rows)
    won_r = sum(1 for r in kept_rows if str(r.get("sessionId")) in won)
    print(f"non-eval pool:      {n_in} rows / {len(by_game)} games")
    print(f"loop-compressed:    {n_out} rows  (cut {n_in - n_out} = "
          f"{100 * (n_in - n_out) / n_in:.1f}%; exact-state cycle bodies)")
    print(f"games with >=1 collapsed cycle: {games_with_cycles}/{len(by_game)}")
    print(f"rows cut by outcome: won {cut_by_outcome['won']} / lost {cut_by_outcome['lost']}")
    print(f"won-row fraction kept: {100 * won_r / max(n_out, 1):.0f}% "
          f"(volume arm = 36%); should be ~unchanged since cycles span both")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
