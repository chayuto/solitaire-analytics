#!/usr/bin/env python3
"""Analyze one play_deck_with_student.py run directory.

Loads summary.json + turns.jsonl, applies the pre-registered decision
rules from docs/reports/20260526_full_game_play_sequence_plan.md, and
outputs a structured diagnostic: outcome classification, foundation
trajectory, plateau windows, oscillation signature if any, key reasoning
excerpts at decision points.

Usage:
  .venv/bin/python gemma4_finetune/analyze_play_run.py \\
      gemma4_finetune/play_runs/v1_seed3263196305_run2

Or analyze all runs in play_runs/:
  for d in gemma4_finetune/play_runs/*/; do
      .venv/bin/python gemma4_finetune/analyze_play_run.py "$d"
  done
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def classify_outcome(summary: dict, turns: list[dict]) -> tuple[str, str]:
    """Apply the pre-registered decision rules. Returns (class, rationale)."""
    outcome = summary.get("outcome")
    fc = summary.get("final_foundation_cards", 0)
    fd = summary.get("final_face_down", 0)
    plateau = summary.get("plateau_at_end_turns", 0)
    turns_played = summary.get("turns_played", 0)

    if outcome == "won":
        if turns_played <= 250:
            return "WIN", f"won in {turns_played} turns"
        return "WIN_LONG", f"won but took {turns_played} turns (>250)"

    if outcome == "parse_failure":
        return "PARSE_FAILURE", "JSON parse failures clustered at end"

    if outcome == "engine_violation":
        return "ENGINE_VIOLATION", "apply_move returned None for a legal move"

    # Revised rule: classify on plateau + oscillation, not foundation count alone
    osc_sig = detect_oscillation(turns)
    if osc_sig and plateau >= 15:
        return "MIDGAME_STALL_DOOMLOOP", (
            f"{plateau}-turn plateau at fc={fc} with {osc_sig['pattern']} "
            f"oscillation ({osc_sig['count']}x in last 10 moves)"
        )

    if outcome == "illegal_move":
        # Illegal at end is usually the symptom of upstream doom-loop;
        # the runner aborted after 3 illegal picks in a row
        if osc_sig:
            return "MIDGAME_STALL_DOOMLOOP", (
                f"illegal_move trigger after {osc_sig['pattern']} oscillation "
                f"({osc_sig['count']}x); plateau={plateau} turns at fc={fc}"
            )
        return "MOVE_INDEX_FIXATION", (
            f"3 illegal picks at end without clear oscillation; "
            f"fc={fc}, plateau={plateau}"
        )

    if outcome == "max_turns":
        if fc >= 45:
            return "HIT_MAX_TURNS_NEAR_WIN", f"fc={fc}, just ran out of budget"
        if fc >= 30:
            return "CLOSEOUT_FAIL", f"reached fc={fc}, stalled at endgame"
        if fc >= 10:
            return "MIDGAME_STALL", f"fc={fc} with plateau={plateau} turns"
        return "EARLY_FAIL", f"never broke fc=10 in {turns_played} turns"

    if outcome == "stalled":
        return "STALLED_NO_LEGAL", f"engine returned zero legal moves at fc={fc}"

    return "UNCLASSIFIED", f"outcome={outcome!r} fc={fc} plateau={plateau}"


def detect_oscillation(turns: list[dict]) -> dict | None:
    """Scan the last 10-15 played moves for a 2-card or 3-card oscillation
    signature. Returns {pattern, count} or None."""
    successful = [t for t in turns if t.get("json_ok") and t.get("move_text")]
    if len(successful) < 6:
        return None

    # Build "canonical" move signatures (card + from-col + to-col)
    sigs = []
    for t in successful[-15:]:
        mtxt = t.get("move_text", "")
        m = re.match(r"Move (\S+)(?: plus \d+ more)? from column (\d+) to column (\d+)", mtxt)
        if m:
            sigs.append(f"{m.group(1)} {m.group(2)}->{m.group(3)}")
        else:
            sigs.append(None)

    # Check for the canonical AB-BA pair pattern: a 2-move signature
    # repeating multiple times in the window
    pairs = Counter()
    for i in range(len(sigs) - 1):
        if sigs[i] and sigs[i + 1]:
            # Canonical pair: sort the two sides so AB and BA both fold into the same key
            a, b = sigs[i], sigs[i + 1]
            key = tuple(sorted([a, b]))
            pairs[key] += 1

    if not pairs:
        return None

    most, n = pairs.most_common(1)[0]
    if n < 4:  # at least 4 occurrences in the 15-move window
        return None

    # Render as a human-readable pattern
    a, b = most
    return {"pattern": f"{a} / {b}", "count": n}


def foundation_trajectory(turns: list[dict], every_n: int = 10) -> list[tuple[int, int]]:
    """Sample (turn, fc) every N turns for a compact trajectory view."""
    out = []
    last_t = -every_n
    for t in turns:
        if not t.get("json_ok"):
            continue
        ti = t.get("turn", -1)
        fc = t.get("fc", 0)
        if ti - last_t >= every_n:
            out.append((ti, fc))
            last_t = ti
    # Always include the final point
    final = next((t for t in reversed(turns) if t.get("json_ok")), None)
    if final and (not out or out[-1][0] != final.get("turn")):
        out.append((final.get("turn"), final.get("fc", 0)))
    return out


def plateau_windows(turns: list[dict]) -> list[tuple[int, int, int, int]]:
    """Return list of (start_turn, end_turn, length, fc_during) for plateaus
    >=10 turns where fc stayed constant. Useful for spotting doom-loop windows."""
    out = []
    if not turns:
        return out
    cur_fc = None
    cur_start = 0
    for t in turns:
        if not t.get("json_ok"):
            continue
        ti = t.get("turn", -1)
        fc = t.get("fc", -1)
        if fc != cur_fc:
            if cur_fc is not None and ti - cur_start >= 10:
                out.append((cur_start, ti - 1, ti - cur_start, cur_fc))
            cur_fc = fc
            cur_start = ti
    # Tail
    last_ti = turns[-1].get("turn", 0)
    if cur_fc is not None and last_ti - cur_start >= 10:
        out.append((cur_start, last_ti, last_ti - cur_start + 1, cur_fc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="play_runs/<run-name>/ directory to analyze")
    ap.add_argument("--json", action="store_true",
                    help="Emit structured JSON instead of plain-text report")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    turns_path = run_dir / "turns.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    if not turns_path.exists():
        raise SystemExit(f"missing {turns_path}")

    summary = json.loads(summary_path.read_text())
    turns = [json.loads(l) for l in turns_path.open()]

    cls, rationale = classify_outcome(summary, turns)
    osc = detect_oscillation(turns)
    plateaus = plateau_windows(turns)
    trajectory = foundation_trajectory(turns, every_n=10)

    if args.json:
        out = {
            "run_dir": str(run_dir),
            "summary": summary,
            "classification": cls,
            "rationale": rationale,
            "oscillation": osc,
            "plateau_windows": [
                {"start": s, "end": e, "length": l, "fc": fc}
                for s, e, l, fc in plateaus
            ],
            "foundation_trajectory": [
                {"turn": ti, "fc": fc} for ti, fc in trajectory
            ],
        }
        print(json.dumps(out, indent=2))
        return

    print(f"=== Play run analysis: {run_dir} ===\n")
    print(f"Model    : {summary.get('model_id')}")
    print(f"Adapter  : {summary.get('adapter_path') or '(no adapter)'}")
    print(f"Deck seed: {summary.get('deck_seed')}")
    print(f"Outcome  : {summary.get('outcome')} after {summary.get('turns_played')} turns "
          f"(wallclock {summary.get('wallclock_seconds', 0)/60:.1f} min)")
    print(f"Final    : fc={summary.get('final_foundation_cards')}/52  "
          f"fd={summary.get('final_face_down')}  "
          f"plateau_at_end={summary.get('plateau_at_end_turns')} turns")
    print()
    print(f"CLASS    : {cls}")
    print(f"WHY      : {rationale}")
    print()
    if osc:
        print(f"Oscillation signature: {osc['pattern']}  ({osc['count']}x in tail window)")
        print()
    if plateaus:
        print(f"Plateau windows (>=10 turns at same fc):")
        for s, e, l, fc in plateaus:
            print(f"  turn {s:>3}-{e:<3}  ({l} turns)  fc={fc}")
        print()
    print("Foundation trajectory (sampled every 10 turns):")
    for ti, fc in trajectory:
        bar = "*" * fc if fc <= 52 else "*" * 52
        print(f"  turn {ti:>3}  fc={fc:>2}  {bar}")


if __name__ == "__main__":
    main()
