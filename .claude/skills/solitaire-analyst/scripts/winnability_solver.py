#!/usr/bin/env python3
"""Sound winnability solver over the repo engine (handles mid-game positions).

Unlike pyksolve (turn-0 deals only, and `load_pysol` is broken in the installed
0.0.15 build), this solves an arbitrary `GameState` -- non-empty foundations,
arbitrary column sizes, partially-cycled stock -- by searching the real engine
(`generate_moves`/`apply_move`). Because every transition is a real legal move,
the verdict is sound by construction:

  * SOLVED      -- a concrete legal sequence reaches all-foundations (a win
                   exists; this is a constructive proof).
  * UNSOLVABLE  -- the FULL reachable space was enumerated under the node cap
                   and contains no win (a disproof, relative to the engine's
                   move set -- which is the move set the game itself allows;
                   the harvester never offers foundation-to-tableau pullbacks
                   either, so this matches real winnability).
  * UNKNOWN     -- the node cap was hit before either was established. Never
                   reported as winnable or dead; the caller must treat it as
                   inconclusive.

Speed comes from three win-/loss-preserving reductions:
  * safe autoplay -- if an exposed card can never be needed in the tableau
    (rank <= 2, or both opposite-colour foundations are already at >= rank-1),
    it is forced to the foundation as the only move. Standard Klondike result.
  * a transposition table keyed on the full position.
  * move ordering -- reveals first, then foundation plays, then waste/tableau,
    then draw -- so winnable lines are found early.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from solitaire_analytics.models import GameState, Card
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves, apply_move


def _recycle_state(state: GameState):
    """Stock recycle, matching GameSession._recycle: when the stock is empty and
    the waste is non-empty, the waste returns to the stock reversed (face-down)
    so the next pass re-draws it in the original order. generate_moves() does
    NOT emit this (recycle lives in the session layer), so the search must add
    it or it cannot cycle the stock past the first pass."""
    if state.stock or not state.waste:
        return None
    ns = state.copy()
    ns.stock = [Card(rank=c.rank, suit=c.suit, face_up=False) for c in reversed(ns.waste)]
    ns.waste = []
    return ns


def _foundation_rank_by_suit(state: GameState) -> dict:
    """Map suit-value -> top rank of that suit's foundation (0 if empty)."""
    out = {}
    for pile in state.foundations:
        if pile:
            out[pile[-1].suit.value] = pile[-1].rank
    return out


_RED = {"hearts", "diamonds"}


def _opposite_color_suits(suit_value: str):
    return ("clubs", "spades") if suit_value in _RED else ("hearts", "diamonds")


def _is_safe_autoplay(card, fr: dict) -> bool:
    """True if `card` can be sent to foundation without ever being needed in
    the tableau. Safe rule: rank <= 2, or both opposite-colour foundations are
    already at >= rank-1."""
    if card.rank <= 2:
        return True
    o1, o2 = _opposite_color_suits(card.suit.value)
    return fr.get(o1, 0) >= card.rank - 1 and fr.get(o2, 0) >= card.rank - 1


def _find_safe_autoplay_move(state: GameState, moves):
    """Return a single safe-autoplay foundation move if one exists, else None."""
    fr = _foundation_rank_by_suit(state)
    for m in moves:
        if m.move_type == MoveType.TABLEAU_TO_FOUNDATION:
            card = state.tableau[m.source_pile][-1]
        elif m.move_type == MoveType.WASTE_TO_FOUNDATION:
            card = state.waste[-1]
        else:
            continue
        if _is_safe_autoplay(card, fr):
            return m
    return None


def _key(state: GameState):
    fk = tuple((p[-1].rank, p[-1].suit.value) if p else (0, "_") for p in state.foundations)
    tk = tuple(tuple((c.rank, c.suit.value, c.face_up) for c in col) for col in state.tableau)
    wk = tuple((c.rank, c.suit.value) for c in state.waste)
    sk = tuple((c.rank, c.suit.value) for c in state.stock)
    return (fk, tk, wk, sk)


def _heuristic(state: GameState):
    """Best-first priority (lower = explored first): most foundation cards,
    then fewest face-down tableau cards. Guides toward winning lines while the
    search still exhausts a small dead space for a sound UNSOLVABLE."""
    found = sum(len(p) for p in state.foundations)
    facedown = sum(1 for col in state.tableau for c in col if not c.face_up)
    return (-found, facedown)


def solve_winnable(state: GameState, node_cap: int = 300_000):
    """Return (verdict, nodes) where verdict in {SOLVED, UNSOLVABLE, UNKNOWN}.

    Best-first over the engine: SOLVED is a constructive win, UNSOLVABLE means
    the full reachable space was exhausted under the cap with no win, UNKNOWN
    means the cap was hit first (inconclusive -- never read as winnable/dead)."""
    import heapq

    seen = set()
    nodes = 0
    capped = False
    counter = 0  # tie-breaker so GameStates are never compared

    heap = [(_heuristic(state), counter, state)]
    seen.add(_key(state))

    while heap:
        _, _, s = heapq.heappop(heap)
        nodes += 1
        if s.is_won():
            return "SOLVED", nodes
        if nodes >= node_cap:
            capped = True
            break

        moves = generate_moves(s)

        forced = _find_safe_autoplay_move(s, moves)
        if forced is not None:
            children = [apply_move(s, forced)]
        else:
            children = [apply_move(s, m) for m in moves]
            rec = _recycle_state(s)
            if rec is not None:
                children.append(rec)

        for ns in children:
            if ns is None:
                continue
            k = _key(ns)
            if k in seen:
                continue
            seen.add(k)
            counter += 1
            heapq.heappush(heap, (_heuristic(ns), counter, ns))

    return ("UNKNOWN" if capped else "UNSOLVABLE"), nodes
