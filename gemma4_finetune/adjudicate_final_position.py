#!/usr/bin/env python3
"""Exact post-hoc adjudication of cap-truncated play-harness games.

For each game dir whose summary says outcome=max_turns, deterministically
replay the recorded decision stream through the engine (mirroring the
harness's auto-forced loop), assert zero fc/fd drift at every decision, and
hand the exact final position to the sound best-first solver
(winnability_solver.solve_winnable).

Verdicts per game:
  SOLVED n=<k>      the final position is winnable (k explored states; tiny k
                    means a near-forced cascade was cut off by the cap)
  UNSOLVABLE n=<k>  the final position is structurally dead: the cap did not
                    truncate anything, the game was already lost
  UNKNOWN n=<k>     node cap hit first; inconclusive

First use (2026-06-12): tourA_v16_rescue game 1 (seed 495097115) ended
max_turns at fc=40/fd=0 mid-cascade; replay matched all 200 decisions with
zero drift and the final position is SOLVED in 15 states, i.e. the first
faithful base-policy win was truncated by the 200-turn cap, not by play.

Usage:
  .venv/bin/python gemma4_finetune/adjudicate_final_position.py \
      gemma4_finetune/play_runs/tourA_v16_rescue/base/seed*/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))

import play_deck_with_student as H  # noqa: E402
from solitaire_analytics.engine import apply_move  # noqa: E402
from winnability_solver import solve_winnable  # noqa: E402


def apply_one(state, mv):
    if mv.move_type.value == "recycle_stock":
        return H.apply_recycle(state), []
    ns = apply_move(state, mv)
    if ns is None:
        raise RuntimeError("engine rejected a replayed move")
    return H.auto_flip(ns)


def _replay_records(state, turns_path: Path, until_turn=None, check_drift=True):
    """Replay decision records from a turns.jsonl onto state (mirroring the
    harness loop: auto-forced phase before each decision). Returns
    (state, n_decisions)."""
    n = 0
    for line in turns_path.open():
        rec = json.loads(line)
        if rec.get("move_index") is None or rec.get("illegal") or not rec.get("json_ok"):
            continue
        if rec.get("resigned"):
            break
        if until_turn is not None and rec["turn"] >= until_turn:
            break
        while True:
            if sum(len(f) for f in state.foundations) == 52:
                break
            forced = H.visible_legal_moves(state)
            if len(forced) != 1:
                break
            state, _ = apply_one(state, forced[0])
        state, _ = apply_one(state, H.visible_legal_moves(state)[rec["move_index"]])
        n += 1
        fc = sum(len(f) for f in state.foundations)
        fd = sum(1 for col in state.tableau for c in col if not c.face_up)
        if check_drift and (fc, fd) != (rec["fc"], rec["fd"]):
            raise RuntimeError(f"replay drift at decision {n} (turn {rec.get('turn')}): "
                               f"replayed fc/fd {fc}/{fd} vs recorded {rec['fc']}/{rec['fd']}")
    return state, n


def replay_final_state(game_dir: Path):
    """Replay the recorded decisions; return (state, n_decisions). Raises on
    any fc/fd drift (the replay must reproduce the run exactly). Runs that
    were launched with --warm-start-from replay their source first."""
    summary = json.loads((game_dir / "summary.json").read_text())
    seed = summary["deck_seed"]
    decks = json.loads(H.DECK_PATH.read_text())["decks"]
    deck = next(d for d in decks if d.get("seed") == seed)
    state = H.deck_to_state(deck)

    n = 0
    ws = summary.get("warm_start_from")
    if ws:
        ws_path = Path(ws)
        if not ws_path.is_absolute():
            ws_path = REPO / ws_path
        state, n_ws = _replay_records(state, ws_path,
                                      until_turn=summary.get("warm_start_resume_turn"))
        n += n_ws
    state, n_live = _replay_records(state, game_dir / "turns.jsonl")
    n += n_live
    # Trailing auto-forced moves (after the LAST decision) are part of the run
    # too: the harness plays them at the top of the next turn before its won
    # check. Without this, a game whose final card was auto-forced replays to
    # 51/52 (seen on the seed 495097115 banking win).
    while True:
        if sum(len(f) for f in state.foundations) == 52:
            break
        forced = H.visible_legal_moves(state)
        if len(forced) != 1:
            break
        state, _ = apply_one(state, forced[0])
    return state, n


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    node_cap = 300_000
    rows = []
    for arg in argv:
        game_dir = Path(arg)
        summ_path = game_dir / "summary.json"
        if not summ_path.exists():
            continue
        s = json.loads(summ_path.read_text())
        if s.get("outcome") != "max_turns":
            rows.append((game_dir.name, s.get("outcome"), s.get("final_foundation_cards"),
                         s.get("final_face_down"), "-", "-"))
            continue
        state, n = replay_final_state(game_dir)
        verdict, nodes = solve_winnable(state, node_cap=node_cap)
        rows.append((game_dir.name, s.get("outcome"), s.get("final_foundation_cards"),
                     s.get("final_face_down"), verdict, nodes))
        print(f"{game_dir.name}: replayed {n} decisions zero-drift, "
              f"fc={s.get('final_foundation_cards')} fd={s.get('final_face_down')} "
              f"-> {verdict} (n={nodes})", flush=True)
    print("\nseed                outcome      fc  fd  final-position  nodes")
    for name, out, fc, fd, v, nds in rows:
        print(f"{name:<19} {out:<12} {fc!s:>2}  {fd!s:>2}  {v:<14} {nds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
