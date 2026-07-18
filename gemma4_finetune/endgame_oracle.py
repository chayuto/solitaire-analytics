#!/usr/bin/env python3
"""Endgame oracle: exact winning-LINE solver over the harness action space.

The deployed advisor's cheapest certain win (pre-registered in
docs/reports/20260704_next_move_deep_research.md, strand C): when the game
reaches a position the exact solver can settle, stop asking the model and
either play a proven winning line or fold on a proven-dead board. This
deletes two documented loss classes outright:
  - false-resign at fd=0 (volume resigned #9000237 at 50/52 on a SOLVED board);
  - endgame loops/stalls on winnable boards (the 26B 124-move endgame loop
    class; volume's fc=40 cap-outs).

Design notes:
  - Search space = the HARNESS action space: H.visible_legal_moves (flips
    auto-fired, ace-to-foundation deduped to the suit pile, recycle
    synthesized when stock is empty and waste is not). The engine never
    generates foundation-to-tableau pullbacks (undo-only MoveType), so this
    space equals the game's real move set and every returned line is playable
    verbatim by the harness AND the production harvester UI.
  - Verdicts mirror winnability_solver.solve_winnable exactly (same
    heuristic, same safe-autoplay forcing, same transposition keying):
    SOLVED (constructive line), UNSOLVABLE (full space exhausted under the
    cap: sound disproof), UNKNOWN (cap hit: caller must fall back to the
    model, never fold).
  - Line reconstruction via parent pointers keyed on the position key, so
    memory stays O(seen states).

This module deliberately does NOT touch play_deck_with_student.py (the eval
chain runs it right now); the --endgame-oracle harness flag is wired in a
separate change after the chain finishes. Importing H here is read-only.

Validation CLI (offline, engine-side only):
  .venv/bin/python gemma4_finetune/endgame_oracle.py \
      --game-dir gemma4_finetune/play_runs/bestofN_winrate_losses/seed9000237/s1 \
      --drop-last 12
replays the recorded game to (decisions - drop_last), runs the oracle there,
verifies a SOLVED line actually replays to 52/52, and cross-checks the
verdict against solve_winnable.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))

import play_deck_with_student as H  # noqa: E402  (read-only import)
from solitaire_analytics.engine import apply_move  # noqa: E402
from winnability_solver import (  # noqa: E402
    _key, _heuristic, _find_safe_autoplay_move, solve_winnable,
)


def _step(state, action):
    """Apply one oracle action (engine Move or H.RECYCLE_MOVE) + auto-flips."""
    if action.move_type.value == "recycle_stock":
        ns = H.apply_recycle(state)
    else:
        ns = apply_move(state, action)
        if ns is None:
            raise RuntimeError("oracle line contained an illegal move")
    ns, _flipped = H.auto_flip(ns)
    return ns


def solve_winning_line(state, node_cap: int = 200_000):
    """Best-first search for a full winning line in the harness action space.

    Returns (verdict, line, nodes):
      verdict "SOLVED"     -> line is a list of actions (engine Move objects /
                              H.RECYCLE_MOVE) that provably reaches 52/52 from
                              `state` (constructive; replay-verified in tests).
      verdict "UNSOLVABLE" -> line is None; full reachable space exhausted.
      verdict "UNKNOWN"    -> line is None; node cap hit (caller falls back).
    """
    import heapq

    root_key = _key(state)
    parent = {root_key: None}          # key -> (parent_key, action)
    seen = {root_key}
    nodes = 0
    capped = False
    counter = 0

    heap = [(_heuristic(state), 0, state)]
    while heap:
        _, _, s = heapq.heappop(heap)
        nodes += 1
        if s.is_won():
            # Reconstruct root -> win.
            line = []
            k = _key(s)
            while parent[k] is not None:
                pk, action = parent[k]
                line.append(action)
                k = pk
            line.reverse()
            return "SOLVED", line, nodes
        if nodes >= node_cap:
            capped = True
            break

        moves = H.visible_legal_moves(s)  # includes synthesized recycle
        forced = _find_safe_autoplay_move(s, moves)
        actions = [forced] if forced is not None else moves

        sk = _key(s)
        for action in actions:
            try:
                ns = _step(s, action)
            except RuntimeError:
                continue
            k = _key(ns)
            if k in seen:
                continue
            seen.add(k)
            parent[k] = (sk, action)
            counter += 1
            heapq.heappush(heap, (_heuristic(ns), counter, ns))

    return ("UNKNOWN" if capped else "UNSOLVABLE"), None, nodes


def replay_line(state, line):
    """Apply a line to a copy of state; return final foundation count."""
    s = state
    for action in line:
        s = _step(s, action)
    return sum(len(f) for f in s.foundations)


# --------------------------------------------------------------------------- #
# Validation CLI: replay a recorded game to a late position, oracle it, and
# verify constructively.
# --------------------------------------------------------------------------- #
def _load_deck(seed: int):
    for p in sorted(glob.glob(str(REPO / "data" / "benchmarks" / "*.json"))):
        try:
            decks = json.loads(Path(p).read_text()).get("decks", [])
        except Exception:
            continue
        d = next((d for d in decks if d.get("seed") == seed), None)
        if d is not None:
            return d
    raise SystemExit(f"seed {seed} not in any data/benchmarks/*.json")


def _replay_to(game_dir: Path, drop_last: int):
    """Replay the recorded decision stream, stopping drop_last decisions
    before the end. Mirrors the harness loop (auto-forced drain before each
    decision). Returns (state, decisions_replayed, fc, fd)."""
    summary = json.loads((game_dir / "summary.json").read_text())
    deck = _load_deck(summary["deck_seed"])
    state = H.deck_to_state(deck)
    records = [
        json.loads(l) for l in (game_dir / "turns.jsonl").open() if l.strip()
    ]
    decisions = [
        r for r in records
        if not r.get("auto_forced") and r.get("move_index") is not None
        and not r.get("illegal") and r.get("json_ok") and not r.get("resigned")
    ]
    stop_at = max(0, len(decisions) - drop_last)
    n = 0
    for rec in decisions[:stop_at]:
        # auto-forced drain, exactly as the harness plays it
        guard = 0
        while guard < 400:
            if sum(len(f) for f in state.foundations) == 52:
                break
            forced = H.visible_legal_moves(state)
            if len(forced) != 1:
                break
            state = _step(state, forced[0])
            guard += 1
        legal = H.visible_legal_moves(state)
        mi = rec["move_index"]
        if not (0 <= mi < len(legal)):
            raise SystemExit(f"replay: recorded move_index {mi} out of range")
        state = _step(state, legal[mi])
        n += 1
    fc = sum(len(f) for f in state.foundations)
    fd = sum(1 for col in state.tableau for c in col if not c.face_up)
    return state, n, fc, fd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", required=True)
    ap.add_argument("--drop-last", type=int, default=12,
                    help="stop this many decisions before the recorded end")
    ap.add_argument("--node-cap", type=int, default=200_000)
    args = ap.parse_args()

    game_dir = Path(args.game_dir)
    state, n, fc, fd = _replay_to(game_dir, args.drop_last)
    print(f"replayed {n} decisions -> fc={fc} fd={fd}")

    t0 = time.time()
    verdict, line, nodes = solve_winning_line(state, node_cap=args.node_cap)
    dt = time.time() - t0
    print(f"oracle: {verdict} in {nodes} nodes, {dt:.2f}s"
          + (f", line length {len(line)}" if line else ""))

    t0 = time.time()
    ref_verdict, ref_nodes = solve_winnable(state, node_cap=args.node_cap)
    print(f"solve_winnable agreement check: {ref_verdict} "
          f"({ref_nodes} nodes, {time.time()-t0:.2f}s) -> "
          f"{'AGREE' if ref_verdict == verdict else 'DISAGREE'}")

    if verdict == "SOLVED":
        final_fc = replay_line(state, line)
        ok = final_fc == 52
        print(f"constructive check: line replays to fc={final_fc} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return 0 if ok and ref_verdict == verdict else 1
    return 0 if ref_verdict == verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
