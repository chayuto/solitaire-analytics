#!/usr/bin/env python3
"""Build the CLOSE-OUT augmentation corpus (2026-06-17).

Diagnosis (this session, replay-verified on the 4 loopcompress false-resign
decks 2703165610/3123337720/4221577640/3263196305): the student reaches
winnable, fully-revealed (fd=0) endgames and then RESIGNS with direct legal
foundation plays sitting at legalMoves[0]. Its board_analysis evaluates moves
only through the reveal-priority frame ("does this reveal a hidden card?"); at
fd=0 nothing reveals, so it concludes "no move advances the foundations" -- FALSE
-- and emits move_index -1. The failure is an EMERGENT close-out gap, not teacher
imitation: the loopcompress corpus contains ~0 resign rows, so there is nothing
to strip. The lever is to AUGMENT the won-game close-out behaviour (take the
foundation play, grind a revealed endgame to 52), per plan doc
20260617_loopcompress_spike_and_next_steps.md section 5.1.

Mechanism: pure REWEIGHTING on top of the pinned loopcompress split. We start
from the EXISTING dataset_loopcompress/{train,valid,test}.jsonl (so the split is
byte-identical to loopcompress -- closeout-vs-loopcompress isolates exactly the
augmentation, and val/test loss stays comparable). In TRAIN ONLY, each row drawn
from a WON session AND at a revealed endgame (faceDownTotal <= FD_MAX) is emitted
COPIES+1 times; every other train row and all of valid/test pass through once.
No new rows, no corpus growth, no new eval-seed leakage.

Why train-only + won-only + fd<=2:
  - train-only: never inflate valid/test with duplicates (keeps loss comparable).
  - won-only: lost-game fd<=2 rows are the dead-board flailing we do NOT want to
    teach; only WON endgames show the correct close-out.
  - fd<=2: the false-resigns all occurred at fd=0; fd<=2 captures the close-out
    and its immediate lead-in. These rows are foundation-rich (~69% foundation
    plays) -- exactly the action the resigners failed to take.

Draw-safety (the v7 ORPO lesson): the base loopcompress corpus is 45.9% draws, so
oversampling foundation-rich close-out rows cannot starve draws. The script PRINTS
the post-augmentation train move-kind mix; draw fraction must stay well clear of
the v7 failure (>~30%).

Run:
  .venv/bin/python gemma4_finetune/build_closeout_corpus.py
  # -> gemma4_finetune/dataset_closeout/{train,valid,test}.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))
from load_export import parse_board  # noqa: E402

SRC_SPLIT = THIS / "dataset_loopcompress"          # pinned loopcompress split
RAW = REPO / "data" / "dataset" / "training_loopcompress.jsonl"  # carries sessionId/seed
OUT = THIS / "dataset_closeout"

RANK = {"A": 1, "J": 11, "Q": 12, "K": 13, "T": 10, "10": 10,
        **{str(i): i for i in range(2, 11)}}


def fd_of(board) -> int:
    return sum((c.get("faceDownCount") or 0) for c in (board.get("tableau") or []))


def won_session_ids() -> set[str]:
    sids: set[str] = set()
    for f in glob.glob(str(REPO / "data" / "raw" / "solitaire-win-*.json")):
        try:
            wr = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(wr, dict) and wr.get("gameWon") is True and wr.get("gameSessionId"):
            sids.add(str(wr["gameSessionId"]))
    return sids


def move_kind(prompt: str, completion: str) -> str:
    """Classify the taught move (for the draw-safety guard)."""
    try:
        mi = json.loads(completion).get("final_decision", {}).get("move_index")
    except (json.JSONDecodeError, AttributeError):
        return "?"
    if mi is None:
        return "?"
    if mi == -1:
        return "resign"
    b = parse_board(prompt) or {}
    lm = b.get("legalMoves") or []
    if not (0 <= mi < len(lm)):
        return "?"
    t = (lm[mi].get("describe") or lm[mi].get("description") or str(lm[mi])).lower()
    if "foundation" in t:
        return "foundation"
    if "draw" in t:
        return "draw"
    if "recycle" in t:
        return "recycle"
    return "tableau"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-max", type=int, default=2, help="close-out faceDown ceiling")
    ap.add_argument("--copies", type=int, default=2, help="EXTRA copies of each close-out train row")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    won = won_session_ids()
    # prompt -> (sessionId, fd) from the raw loopcompress rows
    meta: dict[str, tuple[str, int]] = {}
    for line in RAW.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        b = parse_board(r["prompt"])
        if b is None:
            continue
        meta[r["prompt"]] = (str(r.get("sessionId")), fd_of(b))

    # valid/test: verbatim passthrough (must equal loopcompress)
    for name in ("valid", "test"):
        (out / f"{name}.jsonl").write_text((SRC_SPLIT / f"{name}.jsonl").read_text())

    train_in = [json.loads(l) for l in (SRC_SPLIT / "train.jsonl").read_text().splitlines() if l.strip()]
    train_out: list[dict] = []
    n_closeout_rows = 0
    n_dupes = 0
    unmatched = 0
    for row in train_in:
        train_out.append(row)
        sid_fd = meta.get(row["prompt"])
        if sid_fd is None:
            unmatched += 1
            continue
        sid, fd = sid_fd
        if sid in won and fd <= args.fd_max:
            n_closeout_rows += 1
            for _ in range(args.copies):
                train_out.append(row)
                n_dupes += 1

    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_out))

    # ---- report + draw-safety guard ----
    base_train = len(train_in)
    kinds = Counter(move_kind(r["prompt"], r["completion"]) for r in train_out)
    tot = sum(kinds.values())
    draw_frac = 100 * kinds.get("draw", 0) / max(tot, 1)
    found_frac = 100 * kinds.get("foundation", 0) / max(tot, 1)
    print(f"base loopcompress train rows:   {base_train}")
    print(f"close-out rows (won & fd<={args.fd_max}):  {n_closeout_rows}  "
          f"({100*n_closeout_rows/base_train:.0f}% of train)")
    print(f"extra copies appended (x{args.copies}):    {n_dupes}")
    print(f"unmatched train prompts:        {unmatched}  (expect 0)")
    print(f"-> dataset_closeout train rows: {len(train_out)}  "
          f"(valid {sum(1 for _ in open(out/'valid.jsonl'))} / "
          f"test {sum(1 for _ in open(out/'test.jsonl'))} -- unchanged vs loopcompress)")
    print(f"\npost-augmentation TRAIN move-kind mix: {dict(kinds)}")
    print(f"  draw fraction      = {draw_frac:.1f}%   (base 45.9%; v7-starvation was 0.5%)")
    print(f"  foundation fraction = {found_frac:.1f}%   (base ~22%)")
    if draw_frac < 25:
        print("  !! WARNING: draw fraction under 25% -- risk of v7-style draw starvation")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
