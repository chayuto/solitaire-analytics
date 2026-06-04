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

Solver backends (controlled by --solver):

  * **engine** (default, recommended): `winnability_solver.solve_winnable`, a
    best-first search over the repo engine (`generate_moves`/`apply_move`) with
    safe-autoplay, a transposition table, and stock recycle modelled. Handles
    mid-game positions (non-empty foundations, arbitrary columns). Verdicts are
    sound by construction -- SOLVED is a constructive win, UNSOLVABLE means the
    full reachable space was exhausted under the node cap, UNKNOWN means the cap
    was hit (never read as winnable or dead).
  * **pyksolve** / **beam**: DEPRECATED. pyksolve's `load_pysol` is broken in
    the installed 0.0.15 build (it solves a default deck regardless of input),
    so it returns "winnable" for essentially any board -- do not trust it. beam
    (the in-repo ParallelSolver) is too weak and misses most winnable deals.
    Both kept only for back-compat; see DATASET_NOTES "Operating notes"
    (2026-06-03) for the defect history.

Interpreting the engine verdict per sampled world:
  * SOLVED is definitive: a concrete winning line exists for that world.
  * UNSOLVABLE is definitive: that world is unwinnable (full space exhausted).
  * UNKNOWN is inconclusive: raise --node-cap; never read it as winnable/dead.
Across samples, "0 winnable, >0 dead, 0 unknown" => structurally dead (a stall
or resign there is correct). Low-progress boards (many face-down) may UNKNOWN
under the default cap; raise --node-cap to settle them.

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
from winnability_solver import solve_winnable

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


def known_stock_waste_cards(board: dict) -> list[Card]:
    """All stock+waste cards whose identity the export reveals: every entry of
    `seenDrawPileCards` (the cards that have passed through the waste at least
    once), plus the current `discardTop`. After a full stock cycle this is the
    entire stock+waste pile; in the first cycle the still-unseen stock cards are
    absent (they remain in the unknown pool)."""
    out: list[Card] = []
    seen_keys = set()
    for cs in (board.get("seenDrawPileCards") or []):
        if is_empty_marker(cs):
            continue
        c = parse_card(cs, face_up=True)
        out.append(c)
        seen_keys.add((c.rank, c.suit))
    discard = board.get("discardTop")
    if discard and not is_empty_marker(discard):
        c = parse_card(discard, face_up=True)
        if (c.rank, c.suit) not in seen_keys:
            out.append(c)
    return out


def known_cards_from_board(board: dict) -> list[Card]:
    """Cards whose identity the export reveals.

    Includes foundation contents (every card from Ace up to the foundation top
    per suit), all face-up tableau cards, and EVERY known stock+waste card
    (`seenDrawPileCards` + `discardTop`) -- not just the waste top. Counting
    only `discardTop` (the old bug) leaked the other known stock/waste cards
    into the face-down sample pool, letting genuinely-buried cards (e.g. a
    buried 3C) be sampled into a drawable position and inflate winnability.
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
    known.extend(known_stock_waste_cards(board))
    return known


def sample_state(board: dict, rng: random.Random) -> GameState:
    """Build one fully-determined GameState consistent with the observed board.

    The only genuinely-unknown cards are the face-down tableau cards plus any
    still-unseen stock cards (first cycle only). Everything else -- foundation
    contents, face-up tableau, and every `seenDrawPileCards`/`discardTop` card
    -- is placed at its real location. Unknowns are dealt without replacement:
    `faceDownTotal` of them into the face-down tableau slots, the remainder into
    the stock as the unseen cards.

    Stock/waste split: known seen cards go to the waste (face up, `discardTop`
    on top), unseen unknowns go to the stock (face down). With recycle modelled
    in the solver this reproduces the cyclic stock faithfully; the exact split
    point matters little once the pile can be cycled.
    """
    known = known_cards_from_board(board)
    known_keys = {(c.rank, c.suit) for c in known}
    unknown = [c for c in full_deck() if (c.rank, c.suit) not in known_keys]
    rng.shuffle(unknown)

    state = GameState()

    # Foundations: rebuild from board.foundations using known suit + top rank.
    f = board.get("foundations") or {}
    for foundation_idx, suit in enumerate([Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]):
        suit_name = suit.value
        top = f.get(suit_name)
        if not top:
            continue
        top_rank = _RANK_FROM_LETTER[top[0]]
        for r in range(1, top_rank + 1):
            state.foundations[foundation_idx].append(Card(rank=r, suit=suit, face_up=True))

    # Tableau columns: face-down first (sampled from unknown), then face-up.
    for col_idx, col in enumerate(board.get("tableau") or []):
        fd_count = int(col.get("faceDownCount", 0))
        for _ in range(fd_count):
            if not unknown:
                break
            c = unknown.pop()
            state.tableau[col_idx].append(Card(rank=c.rank, suit=c.suit, face_up=False))
        for card_str in col.get("faceUp") or []:
            if is_empty_marker(card_str):
                continue
            state.tableau[col_idx].append(parse_card(card_str, face_up=True))

    # Waste: the known seen stock/waste cards, discardTop placed last (= top).
    seen = known_stock_waste_cards(board)
    discard_str = board.get("discardTop")
    discard_key = None
    if discard_str and not is_empty_marker(discard_str):
        dc = parse_card(discard_str, face_up=True)
        discard_key = (dc.rank, dc.suit)
    waste_cards = [c for c in seen if (c.rank, c.suit) != discard_key]
    for c in waste_cards:
        state.waste.append(Card(rank=c.rank, suit=c.suit, face_up=True))
    if discard_key is not None:
        state.waste.append(Card(rank=discard_key[0], suit=discard_key[1], face_up=True))

    # Stock: the remaining unknown cards = still-unseen stock (face down).
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
    ap.add_argument("--solver", choices=["engine", "pyksolve", "beam"], default="engine",
                    help="backend: engine (default; repo-engine best-first, handles "
                         "mid-game boards) | pyksolve (BROKEN for mid-game in 0.0.15, "
                         "do not use) | beam (legacy, weak)")
    ap.add_argument("--node-cap", type=int, default=500_000,
                    help="engine: max states per sampled world before UNKNOWN (default 500k)")
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
        return 1
    if args.solver == "pyksolve":
        print("WARNING: pyksolve's load_pysol is broken in the installed 0.0.15 build "
              "(it solves a default deck, not your board), so this backend's verdicts "
              "are meaningless for mid-game positions. Use --solver engine.")

    doc = load_export(args.path)
    board = latest_board(doc)
    if board is None:
        print("error: no parseable CURRENT GAME (JSON) block in any interaction")
        return 1

    fc = foundation_cards(board) or 0
    fd = face_down_total(board) or 0
    print(f"session {doc.short_session}: foundationCards={fc} faceDownTotal={fd}")
    rng = random.Random(args.seed)

    # ---- engine backend (default, recommended): sound 3-way verdict ----
    if args.solver == "engine":
        print(f"sampling {args.samples} consistent worlds via repo-engine best-first "
              f"(node_cap={args.node_cap:,}, recycle modelled)")
        print("(progress: 'W'=winnable, 'X'=proved dead, '?'=hit cap)")
        from collections import Counter
        tally: Counter = Counter()
        nodes_list: list[int] = []
        for _ in range(args.samples):
            state = sample_state(board, rng)
            verdict, nodes = solve_winnable(state, node_cap=args.node_cap)
            tally[verdict] += 1
            nodes_list.append(nodes)
            print({"SOLVED": "W", "UNSOLVABLE": "X", "UNKNOWN": "?"}[verdict],
                  end="", flush=True)
        print("\n")
        won = tally["SOLVED"]
        dead = tally["UNSOLVABLE"]
        unk = tally["UNKNOWN"]
        n = args.samples
        print(f"  winnable (constructive win) : {won}/{n}")
        print(f"  proved dead (exhausted)     : {dead}/{n}")
        print(f"  inconclusive (hit cap)      : {unk}/{n}")
        if nodes_list:
            print(f"  mean states / sample        : {sum(nodes_list)/len(nodes_list):.0f}")

        if won == 0 and unk == 0 and dead > 0:
            verdict = ("STRUCTURALLY DEAD: every sampled world is provably unwinnable "
                       "(engine-exhaustive). A stall or resign here is CORRECT, not behavioural.")
        elif won == 0 and dead > 0:
            verdict = (f"LIKELY DEAD: 0 winnable, {dead} proved unwinnable, {unk} inconclusive. "
                       "Lean structural; raise --node-cap to settle the inconclusive worlds.")
        elif won == 0 and dead == 0:
            verdict = (f"INCONCLUSIVE: all {unk} worlds hit the node cap with no win and no "
                       "exhaustion. Raise --node-cap; do not call this winnable or dead.")
        elif won > 0 and dead == 0 and unk == 0:
            verdict = ("WINNABLE: a concrete winning line exists in every sampled world. "
                       "A stall here would be behavioural, not structural.")
        else:
            verdict = (f"WINNABLE in {won}/{n} sampled worlds (each a constructive win); "
                       f"{dead} dead, {unk} inconclusive. At least partly behavioural.")
        print(f"  verdict  : {verdict}")
        return 0

    # ---- legacy backends (deprecated) ----
    if args.solver == "beam":
        print(f"sampling {args.samples} consistent worlds via beam search "
              f"(beam_width={args.beam_width}, timeout={args.timeout}s each)")
        solver = ParallelSolver(max_depth=args.max_depth, beam_width=args.beam_width,
                                timeout=args.timeout, n_jobs=1)
    else:
        print(f"sampling {args.samples} consistent worlds via pyksolve "
              f"(draw={args.draw_count}, max_closed={args.max_closed:,}) [DO NOT TRUST]")
    print("(progress: '.'=unsolved, 'W'=won, '?'=error)")
    solved = 0
    for i in range(args.samples):
        try:
            state = sample_state(board, rng)
            if args.solver == "pyksolve":
                ok, _ = _solve_one_pyksolve(state, args.draw_count, args.max_closed)
            else:
                ok = solver.solve(state).success
            solved += int(ok)
            print("W" if ok else ".", end="", flush=True)
        except Exception as exc:
            print("?", end="", flush=True)
            sys.stderr.write(f"\nsample {i} errored: {exc}\n")
    print()
    print(f"\n  solved   : {solved}/{args.samples} (legacy backend; engine is authoritative)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
