#!/usr/bin/env python3
"""Mint MOVE-CONTRAST ORPO pairs: chosen and rejected differ ONLY in the move.

Why this exists: the first ORPO pilot (dataset_orpo_pilot) used pairs whose
chosen and rejected were whole, different JSON responses. The model already
separated them by overall fluency, so preference accuracy was 1.0 from step one
and the odds-ratio term never engaged; the run was effectively SFT-on-chosen.
See docs/reports/20260608_orpo_pilot_and_fullgame_eval.md.

This minter fixes that. For each decision state that offers BOTH a progress move
(reveal a hidden card, or send a card to a foundation) AND a no-progress tableau
shuffle, it emits a pair with:
  - identical prompt,
  - identical board_analysis and strategic_plan (the teacher's real text for
    that state),
  - final_decision.move_index = the progress move (chosen) vs the shuffle
    (rejected).
Only the move index differs, so the length-normalised log-prob difference is
dominated by that one token and the odds-ratio gradient lands on the move
choice, not on prose style.

Choice rules:
  chosen   = a foundation move if available, else a reveal move (strongest
             progress first).
  rejected = a shuffle that REVERSES a recent move if one exists (the actual
             loop move), else any no-progress tableau shuffle.

This is the loop-penalty preference label, not a heuristic injected into the
prompt: it supervises "prefer the progress move over the shuffle at this state".

Usage:
  .venv/bin/python gemma4_finetune/mint_move_swap_pairs.py \
      --log data/dataset/training.jsonl --out gemma4_finetune/dataset_orpo_moveswap
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from filter_shuffles import parse_state, detect_reversal  # noqa: E402


def classify_moves(legal, recent):
    """Return (chosen_progress_idx, rejected_shuffle_idx) or (None, None)."""
    found, reveal, shuf = [], [], []
    for i, lm in enumerate(legal):
        d = (lm.get("describe") or "").lower()
        t = (lm.get("type") or "").lower()
        if "foundation" in d or "foundation" in t:
            found.append(i)
        elif "reveals a hidden card" in d:
            reveal.append(i)
        elif t == "tableau_to_tableau":
            shuf.append(i)
    chosen = found[0] if found else (reveal[0] if reveal else None)
    if chosen is None or not shuf:
        return None, None
    # prefer the shuffle that reverses a recent move (the loop move) as rejected
    state = {"legalMoves": legal, "recentMoves": recent}
    rejected = next((i for i in shuf if _safe_reversal(state, i)), shuf[0])
    return chosen, rejected


def _safe_reversal(state, idx):
    try:
        return detect_reversal(state, idx)
    except Exception:
        return False


def response_text(move_index, board_analysis, strategic_plan):
    return json.dumps({
        "board_analysis": board_analysis,
        "strategic_plan": strategic_plan,
        "final_decision": {"move_index": move_index},
    })


def parse_response_prose(row):
    """Pull (board_analysis, strategic_plan) from the row, with a fallback."""
    raw = row.get("rawResponse")
    if raw:
        try:
            r = json.loads(raw)
            ba = r.get("board_analysis"); sp = r.get("strategic_plan")
            if ba and sp:
                return ba, sp
        except (json.JSONDecodeError, TypeError):
            pass
    dec = row.get("decision") or {}
    ba = dec.get("boardAnalysis") or dec.get("board_analysis")
    sp = dec.get("reasoning") or dec.get("strategic_plan") or dec.get("strategicPlan")
    if ba and sp:
        return ba, sp
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="data/dataset/training.jsonl")
    ap.add_argument("--out", default=str(THIS_DIR / "dataset_orpo_moveswap"))
    ap.add_argument("--per-session-cap", type=int, default=30,
                    help="max pairs per session, for diversity (v5 discipline)")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.log).read_text().splitlines() if l.strip()]
    by_session = {}
    stats = {"rows": len(rows), "no_state": 0, "no_pair": 0, "no_prose": 0,
             "found_chosen": 0, "reveal_chosen": 0, "reversal_rejected": 0, "kept": 0}

    for r in rows:
        st = parse_state(r.get("prompt") or "")
        if not st:
            stats["no_state"] += 1
            continue
        legal = st["legalMoves"]; recent = st.get("recentMoves") or []
        c, j = classify_moves(legal, recent)
        if c is None:
            stats["no_pair"] += 1
            continue
        ba, sp = parse_response_prose(r)
        if not ba or not sp:
            stats["no_prose"] += 1
            continue
        cdesc = (legal[c].get("describe") or "").lower()
        stats["found_chosen" if "foundation" in cdesc else "reveal_chosen"] += 1
        if _safe_reversal({"legalMoves": legal, "recentMoves": recent}, j):
            stats["reversal_rejected"] += 1
        sid = str(r.get("sessionId") or r.get("gameSeed") or "?")
        by_session.setdefault(sid, []).append({
            "prompt": r["prompt"],
            "chosen": response_text(c, ba, sp),
            "rejected": response_text(j, ba, sp),
            "_meta": {"session": sid, "chosen_mi": c, "rejected_mi": j,
                      "chosen_desc": legal[c].get("describe"),
                      "rejected_desc": legal[j].get("describe")},
        })

    # per-session cap for diversity, then game-level split (no session in both)
    rng = random.Random(args.seed)
    sessions = sorted(by_session)
    rng.shuffle(sessions)
    capped = {s: (by_session[s] if len(by_session[s]) <= args.per_session_cap
                  else rng.sample(by_session[s], args.per_session_cap))
              for s in sessions}
    n_valid_sessions = max(1, int(len(sessions) * args.valid_frac))
    valid_sessions = set(sessions[:n_valid_sessions])

    train, valid = [], []
    for s in sessions:
        (valid if s in valid_sessions else train).extend(capped[s])
    for k in ("kept",):
        stats[k] = len(train) + len(valid)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(p) + "\n" for p in train))
    (out / "valid.jsonl").write_text("".join(json.dumps(p) + "\n" for p in valid))

    print(f"rows={stats['rows']}  no_state={stats['no_state']}  "
          f"no_both_moves={stats['no_pair']}  no_prose={stats['no_prose']}")
    print(f"chosen: foundation={stats['found_chosen']} reveal={stats['reveal_chosen']}  "
          f"rejected: reversal-loop-move={stats['reversal_rejected']}")
    print(f"sessions={len(sessions)}  per-session-cap={args.per_session_cap}")
    print(f"-> train {len(train)}  valid {len(valid)}  ({out})")


if __name__ == "__main__":
    main()
