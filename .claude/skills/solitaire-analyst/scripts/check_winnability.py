#!/usr/bin/env python3
"""Monte Carlo winnability check for a real (imperfect-info) export.

The harvester runs with `seeHiddenCards: false`, so the export's CURRENT GAME
(JSON) block reveals only the face-up cards, the foundation tops, the waste
top, and counts for face-down tableau slots and the remaining stock. The
identities of the hidden cards are unknown to the agent and to us.

Without ground-truth hidden cards we can't ask "is THIS deal winnable?". But
we can sample: build a GameState in which every unknown slot is filled with
a card drawn (without replacement) from the set of cards consistent with what
we *can* see, then ask the solver to win it. Repeating gives a Monte Carlo
estimate of the fraction of consistent worlds in which the agent has a
winning continuation.

Two solver backends are available (controlled by --solver):

  * **pyksolve** (default, recommended): Cython wrapper around ShootMe's
    Klondike-Solver, a DFS with dominance pruning specifically designed for
    Klondike. Solves typical deals in 10-500 ms. Correct on confirmed-winnable
    seeds (10/10 on seed 3263196305 turn-0, vs 0/5 for beam).
  * **beam**: the in-repo ParallelSolver. Beam search, much weaker; misses
    most winnable deals. Kept for back-compat and educational comparison.

Interpret the rate with care regardless of backend:
  * A single solved sample proves the *observed-state class* is sometimes
    winnable, useful evidence against "definitely dead".
  * Zero solves across N samples is suggestive but not proof of unwinnable
    (consistent worlds aren't drawn from the true posterior).

With pyksolve as default, running this is cheap (~0.1s for 10 samples on a
typical deal). The "don't run routinely" warning that applied to the beam
backend no longer applies.

Run:
    .venv/bin/python .claude/skills/solitaire-analyst/scripts/check_winnability.py \
        data/raw/solitaire-ai-log-29a7f5-1779361593611.json \
        --samples 10
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running from anywhere by injecting both the repo and the skill scripts dir.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]  # .claude/skills/solitaire-analyst/scripts → repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from load_export import load_export, latest_board, foundation_cards, face_down_total

from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.solvers import ParallelSolver

try:
    from pyksolve.solver import Solitaire as PyksolveSolitaire
    _PYKSOLVE_OK = True
except ImportError:
    _PYKSOLVE_OK = False


_RANK_TO_LETTER = {1: "A", 10: "T", 11: "J", 12: "Q", 13: "K",
                   **{n: str(n) for n in range(2, 10)}}
_SUIT_TO_LETTER = {Suit.HEARTS: "H", Suit.DIAMONDS: "D",
                   Suit.CLUBS: "C", Suit.SPADES: "S"}


def card_to_str(c: Card) -> str:
    return _RANK_TO_LETTER[c.rank] + _SUIT_TO_LETTER[c.suit]


def gamestate_to_pysol(state: GameState) -> str:
    """Convert a GameState into pyksolve's pysol input format.

    Foundation cards are omitted; pyksolve infers them as the missing low
    cards per suit (foundations build A->K and the K-S/D/C/H ordering is
    standard, so 'rank N missing from S' means S foundation is at >= N).

    Talon = stock then waste in draw order: pyksolve treats the talon as a
    single linear pile, drawing from one end.
    """
    talon_cards = [card_to_str(c) for c in (list(state.stock) + list(state.waste))]
    lines = ["Talon: " + " ".join(talon_cards)]
    for col in state.tableau:
        parts = []
        for c in col:
            cs = card_to_str(c)
            parts.append(cs if c.face_up else f"<{cs}>")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# Card-string parsing — matches the harvester's notation.
# Ranks: A 2-9 T J Q K. Suits: H D C S.
_RANK_FROM_LETTER = {
    "A": 1, "T": 10, "J": 11, "Q": 12, "K": 13,
    **{str(n): n for n in range(2, 10)},
}
_SUIT_FROM_LETTER = {
    "H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS, "S": Suit.SPADES,
}


def is_empty_marker(s: str) -> bool:
    """True for the harvester's empty-column placeholder (e.g. '<empty>')."""
    return s.strip().upper() == "<EMPTY>"


def parse_card(s: str, face_up: bool = True) -> Card:
    """Parse a card string like '4D' or 'TH' into a Card."""
    s = s.strip().upper()
    if len(s) != 2:
        raise ValueError(f"unrecognised card string: {s!r}")
    return Card(rank=_RANK_FROM_LETTER[s[0]], suit=_SUIT_FROM_LETTER[s[1]], face_up=face_up)


def full_deck() -> list[Card]:
    return [Card(rank=r, suit=s, face_up=False) for s in Suit for r in range(1, 14)]


def known_cards_from_board(board: dict) -> list[Card]:
    """Cards whose identity the export reveals.

    Includes foundation contents (every card from Ace up to the foundation top
    for each suit), all face-up tableau cards, and the waste's discardTop.
    """
    known: list[Card] = []
    f = board.get("foundations") or {}
    for suit_name, top in f.items():
        if not top:
            continue
        suit = _SUIT_FROM_LETTER[top[-1]]  # 'H'/'D'/'C'/'S'
        top_rank = _RANK_FROM_LETTER[top[0]]
        for r in range(1, top_rank + 1):
            known.append(Card(rank=r, suit=suit, face_up=True))
    for col in board.get("tableau") or []:
        for card_str in col.get("faceUp") or []:
            if is_empty_marker(card_str):
                continue
            known.append(parse_card(card_str, face_up=True))
    discard = board.get("discardTop")
    if discard:
        known.append(parse_card(discard, face_up=True))
    return known


def sample_state(board: dict, rng: random.Random) -> GameState:
    """Build one fully-determined GameState consistent with the observed board.

    Unknown slots (face-down tableau + remaining stock) are filled by drawing
    without replacement from the set of cards not yet accounted for.
    """
    known = known_cards_from_board(board)
    known_keys = {(c.rank, c.suit) for c in known}
    unknown = [c for c in full_deck() if (c.rank, c.suit) not in known_keys]
    rng.shuffle(unknown)

    state = GameState()

    # Foundations: rebuild from board.foundations using known suit + top rank.
    f = board.get("foundations") or {}
    for foundation_idx, suit in enumerate([Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]):
        # The harvester labels foundations by suit name; map to index 0..3.
        suit_name = suit.value  # 'hearts' / 'diamonds' / 'clubs' / 'spades'
        top = f.get(suit_name)
        if not top:
            continue
        top_rank = _RANK_FROM_LETTER[top[0]]
        for r in range(1, top_rank + 1):
            state.foundations[foundation_idx].append(Card(rank=r, suit=suit, face_up=True))

    # Tableau columns: face-down first (popped from unknown), then face-up.
    for col_idx, col in enumerate(board.get("tableau") or []):
        fd_count = int(col.get("faceDownCount", 0))
        for _ in range(fd_count):
            c = unknown.pop()
            state.tableau[col_idx].append(Card(rank=c.rank, suit=c.suit, face_up=False))
        for card_str in col.get("faceUp") or []:
            if is_empty_marker(card_str):
                continue
            state.tableau[col_idx].append(parse_card(card_str, face_up=True))

    # Waste: just the discardTop.
    discard = board.get("discardTop")
    if discard:
        state.waste.append(parse_card(discard, face_up=True))

    # Stock: remaining unknown cards, up to drawPileCount.
    draw_count = int(board.get("drawPileCount") or 0)
    for _ in range(min(draw_count, len(unknown))):
        c = unknown.pop()
        state.stock.append(Card(rank=c.rank, suit=c.suit, face_up=False))

    # Any leftover (shouldn't happen if board accounts add to 52) — drop on stock.
    while unknown:
        c = unknown.pop()
        state.stock.append(Card(rank=c.rank, suit=c.suit, face_up=False))

    return state


def _solve_one_pyksolve(state: GameState, draw_count: int, max_closed: int) -> tuple[bool, float]:
    """Solve a single GameState via pyksolve. Returns (solved, elapsed_sec)."""
    import time as _time
    sol = PyksolveSolitaire()
    sol.draw_count = draw_count
    sol.load_pysol(gamestate_to_pysol(state))
    sol.reset_game()
    t0 = _time.time()
    r = sol.solve_fast(max_closed_count=max_closed)
    return abs(r.value) == 1, _time.time() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="path to a solitaire-ai-log export JSON")
    ap.add_argument("--samples", type=int, default=10,
                    help="Monte Carlo samples (default 10)")
    ap.add_argument("--solver", choices=["pyksolve", "beam"], default="pyksolve",
                    help="backend: pyksolve (default, fast/correct) or beam (legacy)")
    ap.add_argument("--draw-count", type=int, default=1, choices=[1, 3],
                    help="Klondike draw count for pyksolve (default 1)")
    ap.add_argument("--max-closed", type=int, default=2_000_000,
                    help="pyksolve max_closed_count budget per sample (default 2M)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="beam: solver timeout per sample, seconds (default 30)")
    ap.add_argument("--beam-width", type=int, default=2000,
                    help="beam: solver beam width (default 2000)")
    ap.add_argument("--max-depth", type=int, default=200,
                    help="beam: solver max depth (default 200)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible sampling")
    args = ap.parse_args(argv)

    if args.solver == "pyksolve" and not _PYKSOLVE_OK:
        print("error: pyksolve not installed. Run: pip install pyksolve")
        print("       Or fall back to: --solver beam")
        return 1

    doc = load_export(args.path)
    board = latest_board(doc)
    if board is None:
        print("error: no parseable CURRENT GAME (JSON) block in any interaction")
        return 1

    fc = foundation_cards(board) or 0
    fd = face_down_total(board) or 0
    print(f"session {doc.short_session}: foundationCards={fc} faceDownTotal={fd}")
    if args.solver == "pyksolve":
        print(f"sampling {args.samples} consistent worlds via pyksolve "
              f"(draw={args.draw_count}, max_closed={args.max_closed:,})")
    else:
        print(f"sampling {args.samples} consistent worlds via beam search "
              f"(beam_width={args.beam_width}, timeout={args.timeout}s each)")
    print("(progress: '.'=unsolved, 'W'=won, '?'=error)")

    rng = random.Random(args.seed)
    if args.solver == "beam":
        solver = ParallelSolver(
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            timeout=args.timeout,
            n_jobs=1,
        )
    solved = 0
    elapsed_per_sample: list[float] = []
    explored_per_sample: list[int] = []
    for i in range(args.samples):
        try:
            state = sample_state(board, rng)
            if args.solver == "pyksolve":
                ok, elapsed = _solve_one_pyksolve(state, args.draw_count, args.max_closed)
                elapsed_per_sample.append(elapsed)
                if ok:
                    solved += 1
                    print("W", end="", flush=True)
                else:
                    print(".", end="", flush=True)
            else:
                result = solver.solve(state)
                if result.success:
                    solved += 1
                    print("W", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                explored_per_sample.append(result.states_explored)
        except Exception as exc:
            print("?", end="", flush=True)
            sys.stderr.write(f"\nsample {i} errored: {exc}\n")
    print()

    rate = solved / args.samples if args.samples else 0
    print()
    print(f"  solved   : {solved}/{args.samples} ({100*rate:.0f}%)")
    if args.solver == "pyksolve" and elapsed_per_sample:
        print(f"  mean solve time: {sum(elapsed_per_sample)/len(elapsed_per_sample)*1000:.0f} ms")
    elif explored_per_sample:
        print(f"  avg states explored per sample: "
              f"{sum(explored_per_sample) / max(1, len(explored_per_sample)):.0f}")

    if solved == 0:
        verdict = "STRONG dead-deal signal: zero solves across sampled worlds"
    elif rate < 0.2:
        verdict = "likely dead: solver finds wins in <20% of consistent worlds"
    elif rate < 0.6:
        verdict = "mixed: board is sometimes winnable; doom-loop diagnosis depends on heuristics"
    else:
        verdict = "winnable in most worlds: failure is behavioural, not structural"
    print(f"  verdict  : {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
