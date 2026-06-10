#!/usr/bin/env python3
"""Exact (perfect-information) winnability for a deck-logging export.

Builds the TRUE GameState at the export's latest board and runs the sound
engine solver on it once. No Monte Carlo: builds 2af3ae5/c39046e/262581c and
later write `initialBoardSetup` (the full 52-card deal) into the ai-log, and
in Klondike a face-down tableau card never moves, so the current face-down
cards of column i are exactly the first `faceDownCount` cards dealt to
column i. Visible cards come from the latest parsed board; the remaining
cards are the stock/waste cycle.

Stock/waste order caveat: the harvester's recycle re-draws the previous pass
in REVERSE order (verified on #a29c9a: deal-relative order 6C 7H TD 9C JC,
observed post-recycle draws JC 9C TD 7H 6C), so this script orders the
cycle by reversed deal order, rotated so `discardTop` is the waste top. With
unlimited recycles modelled in the solver, winnability is insensitive to
that rotation; the order only affects move counts. The reconstruction is
printed so it can be eyeballed against RECENT MOVES.

Verdicts (same semantics as check_winnability's engine backend, but now about
THE deal rather than sampled worlds):
  * SOLVED      -> the real board is winnable (constructive line exists).
  * UNSOLVABLE  -> the real board is dead. A resign/stall here is CORRECT.
  * UNKNOWN     -> node cap hit; raise --node-cap, never read as a verdict.

Run:
    .venv/bin/python .claude/skills/solitaire-analyst/scripts/true_world_winnability.py \
        data/raw/solitaire-ai-log-a29c9a-1781085771631.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from load_export import load_export, latest_board, foundation_cards, face_down_total
from check_winnability import (
    _RANK_FROM_LETTER, _SUIT_FROM_LETTER, full_deck, is_empty_marker, parse_card,
    card_to_str,
)

from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from winnability_solver import solve_winnable

_DEAL_RANK = {"A": 1, "J": 11, "Q": 12, "K": 13, "10": 10,
              **{str(n): n for n in range(2, 10)}}
_DEAL_SUIT = {"hearts": Suit.HEARTS, "diamonds": Suit.DIAMONDS,
              "clubs": Suit.CLUBS, "spades": Suit.SPADES}


def deal_card(entry: dict, face_up: bool = False) -> Card:
    return Card(rank=_DEAL_RANK[entry["rank"]], suit=_DEAL_SUIT[entry["suit"]],
                face_up=face_up)


def build_true_state(board: dict, setup: dict) -> GameState:
    """Assemble the exact GameState for the parsed board using the logged deal."""
    state = GameState()
    placed: list[Card] = []

    f = board.get("foundations") or {}
    for idx, suit in enumerate([Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]):
        top = f.get(suit.value)
        if not top:
            continue
        for r in range(1, _RANK_FROM_LETTER[top[0]] + 1):
            c = Card(rank=r, suit=suit, face_up=True)
            state.foundations[idx].append(c)
            placed.append(c)

    dealt_cols = [[deal_card(e) for e in col] for col in setup["tableau"]]
    for col_idx, col in enumerate(board.get("tableau") or []):
        fd = int(col.get("faceDownCount", 0))
        dealt_fd = max(len(dealt_cols[col_idx]) - 1, 0)  # last dealt card was face-up
        if fd > dealt_fd:
            raise ValueError(
                f"column {col_idx + 1}: board shows {fd} face-down but only "
                f"{dealt_fd} were dealt face-down; deal/board mismatch")
        for c in dealt_cols[col_idx][:fd]:
            state.tableau[col_idx].append(Card(rank=c.rank, suit=c.suit, face_up=False))
            placed.append(c)
        for s in col.get("faceUp") or []:
            if is_empty_marker(s):
                continue
            c = parse_card(s, face_up=True)
            state.tableau[col_idx].append(c)
            placed.append(c)

    placed_keys = {(c.rank, c.suit) for c in placed}
    if len(placed_keys) != len(placed):
        raise ValueError("duplicate card placed from board + deal")

    # Whatever is not on the board is still in the draw cycle, and must be a
    # subset of the dealt draw pile (cards never re-enter the cycle).
    cycle = [deal_card(e, face_up=False) for e in setup["drawPile"]
             if (_DEAL_RANK[e["rank"]], _DEAL_SUIT[e["suit"]]) not in placed_keys]
    expect_cycle = [c for c in full_deck() if (c.rank, c.suit) not in placed_keys]
    if {(c.rank, c.suit) for c in cycle} != {(c.rank, c.suit) for c in expect_cycle}:
        raise ValueError("draw-cycle remainder does not match deck complement")

    stock_n = int(board.get("drawPileCount") or 0)
    waste_n = len(cycle) - stock_n
    if waste_n < 0:
        raise ValueError(f"drawPileCount {stock_n} exceeds remaining cycle {len(cycle)}")

    # Reversed deal order, rotated so discardTop sits at waste-top position.
    order = list(reversed(cycle))
    discard = board.get("discardTop")
    if waste_n and discard and not is_empty_marker(discard):
        dc = parse_card(discard)
        pos = next((i for i, c in enumerate(order)
                    if (c.rank, c.suit) == (dc.rank, dc.suit)), None)
        if pos is None:
            raise ValueError(f"discardTop {discard} not found in remaining cycle")
        k = pos + 1 - waste_n  # cyclic rotation puts discardTop at waste-top slot
        order = order[k:] + order[:k]
    elif waste_n == 0 and discard and not is_empty_marker(discard):
        raise ValueError("board shows a discardTop but reconstruction has empty waste")

    waste, stock_draw_order = order[:waste_n], order[waste_n:]
    for c in waste:
        state.waste.append(Card(rank=c.rank, suit=c.suit, face_up=True))
    # Engine draws stock.pop(): first-to-draw goes LAST in the list.
    for c in reversed(stock_draw_order):
        state.stock.append(Card(rank=c.rank, suit=c.suit, face_up=False))

    total = (sum(len(p) for p in state.foundations) + sum(len(p) for p in state.tableau)
             + len(state.stock) + len(state.waste))
    if total != 52:
        raise ValueError(f"reconstructed state has {total} cards, not 52")
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="path to a deck-logging solitaire-ai-log export JSON")
    ap.add_argument("--node-cap", type=int, default=500_000,
                    help="max states before UNKNOWN (default 500k)")
    args = ap.parse_args(argv)

    raw = json.loads(Path(args.path).read_text())
    setup = raw.get("initialBoardSetup")
    if not setup or not setup.get("drawPile"):
        print("error: no initialBoardSetup in this export (needs a deck-logging "
              "build: 2af3ae5 / c39046e / 262581c or later)")
        return 1

    doc = load_export(args.path)
    board = latest_board(doc)
    if board is None:
        print("error: no parseable board in any interaction")
        return 1

    state = build_true_state(board, setup)

    fc = foundation_cards(board) or 0
    fd = face_down_total(board) or 0
    print(f"session {doc.short_session}: foundationCards={fc} faceDownTotal={fd}")
    for i, col in enumerate(state.tableau):
        hidden = " ".join(card_to_str(c) for c in col if not c.face_up) or "-"
        if hidden != "-":
            print(f"  col {i + 1} true hidden (bottom->top): {hidden}")
    print(f"  waste (bottom->top): {' '.join(card_to_str(c) for c in state.waste) or '-'}")
    print(f"  stock (next-drawn first): "
          f"{' '.join(card_to_str(c) for c in reversed(state.stock)) or '-'}")

    print(f"solving the TRUE board (node_cap={args.node_cap:,}, recycle modelled)...")
    verdict, nodes = solve_winnable(state, node_cap=args.node_cap)
    print(f"  states explored: {nodes:,}")
    if verdict == "SOLVED":
        print("  verdict: WINNABLE -- the real deal has a constructive winning line. "
              "A resign/stall here is a WRONG fold, behavioural not structural.")
    elif verdict == "UNSOLVABLE":
        print("  verdict: STRUCTURALLY DEAD -- the full reachable space is exhausted. "
              "A resign here is CORRECT.")
    else:
        print("  verdict: UNKNOWN -- node cap hit; raise --node-cap before reading "
              "anything into this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
