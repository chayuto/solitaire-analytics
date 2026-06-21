#!/usr/bin/env python3
"""Build the GENTLE close-out augmentation on the VOLUME corpus (2026-06-19).

Follow-on to docs/reports/20260619_closeout_augmentation_eval.md section 7.2.

What last night taught us: the `closeout` arm (loopcompress split + 2x oversample
of won & fd<=2 close-out rows) eliminated the false-resign and won 2 of the 4
diagnosed decks, but it did NOT beat volume and it REGRESSED mid-game reach -- 5
of its 9 stalls were on winnable boards still deep in face-down (fd 9-20), where
volume/loopcompress only ever stall at fd<=1. The 2x oversample of the fd<=2
endgame over-specialized the endgame at the cost of the mid-game excavation that
gets you there (the v7 draw-starvation lesson one level up: we guarded the draw
fraction but skewed the game-PHASE distribution).

This arm applies the lesson:
  - BASE = VOLUME, not loopcompress. Volume is the lead student: 5 wins, 0
    resigns, the best mid-game reach (it reaches fd<=1 endgames). Its 4 non-win
    winnable boards are NEAR-WIN cap-stalls (350743738 fc31, 405489085 fc27,
    3123337720, 4197389931, all fd0/1) -- it gets to the endgame and does not
    close it within 300 turns. Endgame-closing is exactly what the close-out rows
    teach, and volume's mid-game is robust enough to absorb a gentle nudge.
  - GENTLE: 1 extra copy (2x presence), not closeout's 2 (3x). One knob, turned
    down, on a stronger base.
  - BAND kept at fd<=2 (the actual close-out target). We deliberately do NOT
    widen the band: widening to fd<=N oversamples the mid-game too and dilutes the
    endgame-closing signal, which is the only thing we want to add here. The
    multiplier, not the band, is the gentleness lever.

Mechanism = pure TRAIN-ONLY reweighting on the pinned dataset_volume split
(valid/test byte-identical to dataset_volume, so volcloseout-vs-volume isolates
exactly the augmentation and val loss stays comparable). No new rows, no growth,
no eval-seed leakage.

Draw-safety + the gate that exposed closeout's failure: the script prints the
BASE vs post-aug move-kind mix. Draw fraction must stay well clear of the v7
failure (>~30%); and the eval must GATE on meanFC not dropping vs volume (the
metric that caught the mid-game regression).

Run:
  .venv/bin/python gemma4_finetune/build_volcloseout_corpus.py
  # -> gemma4_finetune/dataset_volcloseout/{train,valid,test}.jsonl
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

SRC_SPLIT = THIS / "dataset_volume"                                   # pinned volume split
RAW = REPO / "data" / "dataset" / "training_volume_full.jsonl"        # carries sessionId/seed
OUT = THIS / "dataset_volcloseout"


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


def _mix(rows):
    kinds = Counter(move_kind(r["prompt"], r["completion"]) for r in rows)
    tot = max(sum(kinds.values()), 1)
    return kinds, 100 * kinds.get("draw", 0) / tot, 100 * kinds.get("foundation", 0) / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-max", type=int, default=2, help="close-out faceDown ceiling")
    ap.add_argument("--copies", type=int, default=1,
                    help="EXTRA copies of each close-out train row (gentle: 1; closeout used 2)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--src", default=str(SRC_SPLIT),
                    help="source split dir to reweight (default: dataset_volume)")
    ap.add_argument("--raw", default=str(RAW),
                    help="raw pool jsonl with sessionId, matched to the split by prompt")
    args = ap.parse_args()
    out = Path(args.out)
    src_split = Path(args.src)
    raw = Path(args.raw)
    out.mkdir(parents=True, exist_ok=True)

    won = won_session_ids()
    # prompt -> (sessionId, fd) from the raw volume rows
    meta: dict[str, tuple[str, int]] = {}
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        b = parse_board(r["prompt"])
        if b is None:
            continue
        meta[r["prompt"]] = (str(r.get("sessionId")), fd_of(b))

    # valid/test: verbatim passthrough (must equal volume)
    for name in ("valid", "test"):
        (out / f"{name}.jsonl").write_text((src_split / f"{name}.jsonl").read_text())

    train_in = [json.loads(l) for l in (src_split / "train.jsonl").read_text().splitlines() if l.strip()]
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

    # ---- report + draw-safety guard (base vs post-aug) ----
    base_train = len(train_in)
    _, base_draw, base_found = _mix(train_in)
    kinds, draw_frac, found_frac = _mix(train_out)
    print(f"base volume train rows:         {base_train}")
    print(f"close-out rows (won & fd<={args.fd_max}):  {n_closeout_rows}  "
          f"({100*n_closeout_rows/base_train:.0f}% of train)")
    print(f"extra copies appended (x{args.copies}):    {n_dupes}")
    print(f"unmatched train prompts:        {unmatched}  (expect 0)")
    print(f"-> dataset_volcloseout train rows: {len(train_out)}  "
          f"(valid {sum(1 for _ in open(out/'valid.jsonl'))} / "
          f"test {sum(1 for _ in open(out/'test.jsonl'))} -- unchanged vs volume)")
    print(f"\nmove-kind mix   BASE -> POST-AUG:")
    print(f"  draw fraction       = {base_draw:.1f}% -> {draw_frac:.1f}%   (v7-starvation was 0.5%)")
    print(f"  foundation fraction = {base_found:.1f}% -> {found_frac:.1f}%")
    print(f"  full post-aug kinds : {dict(kinds)}")
    if draw_frac < 25:
        print("  !! WARNING: draw fraction under 25% -- risk of v7-style draw starvation")
    if found_frac - base_found > 12:
        print("  !! NOTE: foundation share jumped >12pts -- this is the closeout over-specialization;")
        print("           consider --copies 0 is impossible, so this is already the gentlest 2x.")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
