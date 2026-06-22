#!/usr/bin/env python3
"""Form-3 SOLVER-GROUNDED, play-matched SFT rows (2026-06-21).

Step 4 of docs/reports/20260621_data_volume_and_strategy_text_eval.md: escalate
the strategy-text lever from hand-authored declarative Q&A (form 1, volstrategy:
7/13) to solver-grounded DECISIONS in the exact v1.6 play format. Each row is a
byte-faithful v1.6 board prompt (render_prompt_v16) whose target completion picks
a move that lies on a concrete WINNING line found by the sound best-first solver,
with a rationale in natural Klondike-strategy terms. Unlike the teacher rows
(Gemma 31B, ~31% ceiling), the move is on a solver-proven win, so the rows teach
play ABOVE the teacher's behavioural ceiling -- the point of solver-as-teacher.

How a clean winning line is produced (and why a greedy "pick any winning-
preserving move" does NOT work): greedily re-picking a move that merely keeps the
board SOLVED wanders the winnable set forever (reveals/builds stay winnable
without ever completing the foundations -- measured: foundation stuck at 1 for
300 moves). Instead we search ONCE per deal for a full winning PATH and walk that
exact acyclic sequence, which strictly converges to the win.

The search runs in the HARNESS move space (visible_legal_moves + auto_flip +
recycle), NOT raw engine moves. This matters: raw generate_moves includes
foundation-to-tableau pullbacks and flip-as-choice that the live harness never
offers, so a raw-space win can be unreproducible by the student. Searching the
harness space guarantees every path move maps to a real legal-move index.

Generation walks fresh, unseen winnable deals (seed range 8000001+, disjoint from
every eval/benchmark/training seed) and emits a row at each decision point
(>=2 legal moves) the live harness would actually prompt on, with the harness's
own bookkeeping (cycle, ever_seen, the two stall counters, RECENT MOVES)
replicated move-for-move from apply_and_track. So the rows span the real play
distribution (opening, mid-game, endgame). Single-legal positions are auto-fired
without a row, exactly as the harness does.

  build_solver_grounded_corpus.py --max-rows 700 --node-cap 200000 --seed-start 8000001
    -> solver_grounded_rows.jsonl   ({prompt, completion} rows, ready to mix)

The rationale text deliberately NEVER mentions a solver (the model cannot run
one; teaching it to recite "the solver says" would induce hallucinated
solver-talk). The solver is ground truth for SELECTING the move; the prose is
human strategic reasoning, so the model learns to pair good reasoning with the
winning move.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))

import play_deck_with_student as H  # noqa: E402
from solitaire_analytics.game import deal_klondike  # noqa: E402
from solitaire_analytics.engine import apply_move  # noqa: E402
from winnability_solver import _key as state_key, _heuristic  # noqa: E402

OUT_DEFAULT = THIS / "solver_grounded_rows.jsonl"
RANK_INT_TO_STR = {v: k for k, v in H.RANK_STR_TO_INT.items()}


# ---------- state construction (identical to build_generalization_decks) ----------

def _serialize(state) -> dict:
    def card(c):
        return {"rank": RANK_INT_TO_STR[c.rank], "suit": c.suit.value,
                "face_up": c.face_up}
    tableau = [[card(c) for c in col] for col in state.tableau]
    stock = [{"rank": RANK_INT_TO_STR[c.rank], "suit": c.suit.value}
             for c in reversed(state.stock)]
    return {"tableau": tableau, "stock": stock}


def fresh_state(seed: int):
    """A turn-0 deal constructed exactly as the eval harness constructs decks
    (deal -> serialize -> deck_to_state), so stock ordering matches the harness."""
    g = deal_klondike(seed)
    rec = _serialize(g)
    rec["seed"] = seed
    rec["draw_count"] = 1
    return H.deck_to_state(rec)


# ---------- move helpers ----------

def is_recycle(m) -> bool:
    return m is H.RECYCLE_MOVE or getattr(
        getattr(m, "move_type", None), "value", None) == "recycle_stock"


def apply_legal(state, m):
    if is_recycle(m):
        return H.apply_recycle(state)
    return apply_move(state, m)


def is_won(state) -> bool:
    return sum(len(f) for f in state.foundations) == 52


def index_of(move, legal) -> int | None:
    """Index of `move` (a path move) within a visible_legal_moves list. Both come
    from visible_legal_moves on the same state sequence, so match by attributes."""
    if is_recycle(move):
        for j, m in enumerate(legal):
            if is_recycle(m):
                return j
        return None
    for j, m in enumerate(legal):
        if is_recycle(m):
            continue
        if (m.move_type == move.move_type and m.source_pile == move.source_pile
                and m.dest_pile == move.dest_pile and m.num_cards == move.num_cards):
            return j
    return None


# ---------- harness-space winning-path solver ----------

def solve_path_harness(state, node_cap: int):
    """Best-first search over the HARNESS move space (visible_legal_moves +
    auto_flip after each move + recycle). Returns (verdict, path, nodes); on
    SOLVED, path is the list of harness moves from `state` to a won state -- a
    concrete winning line every move of which the live harness can play."""
    seen = set()
    nodes = 0
    capped = False
    counter = 0
    parent = {0: (None, None)}  # counter -> (parent_counter, move_into_this)

    s0, _ = H.auto_flip(state)
    heap = [(_heuristic(s0), 0, s0)]
    seen.add(state_key(s0))

    while heap:
        _, cid, s = heapq.heappop(heap)
        nodes += 1
        if is_won(s):
            path = []
            c = cid
            while parent[c][0] is not None:
                pc, mv = parent[c]
                path.append(mv)
                c = pc
            path.reverse()
            return "SOLVED", path, nodes
        if nodes >= node_cap:
            capped = True
            break
        for L in H.visible_legal_moves(s):
            ns = apply_legal(s, L)
            if ns is None:
                continue
            ns, _ = H.auto_flip(ns)
            k = state_key(ns)
            if k in seen:
                continue
            seen.add(k)
            counter += 1
            parent[counter] = (cid, L)
            heapq.heappush(heap, (_heuristic(ns), counter, ns))
    return ("UNKNOWN" if capped else "UNSOLVABLE"), [], nodes


# ---------- move classification (for the rationale tag) ----------

def classify_move(state, m) -> str:
    if is_recycle(m):
        return "recycle"
    mt = m.move_type.value
    if mt == "stock_to_waste":
        return "draw"
    if mt in ("tableau_to_foundation", "waste_to_foundation"):
        card = (state.tableau[m.source_pile][-1] if mt == "tableau_to_foundation"
                else state.waste[-1])
        return "found_low" if card.rank <= 2 else "found_high"
    if mt == "tableau_to_tableau":
        col = state.tableau[m.source_pile]
        lead_idx = len(col) - m.num_cards
        if lead_idx == 0:
            return "frees_col"
        if not col[lead_idx - 1].face_up:
            return "reveal_t2t"
        return "build"
    if mt == "waste_to_tableau":
        return "to_empty" if not state.tableau[m.dest_pile] else "waste_build"
    return "other"


# ---------- rationale templates (human strategy prose, ASCII only) ----------
# Two variants per tag for variety (anti-homogeneity); chosen by row index.

_TEMPLATES = {
    "reveal_t2t": [
        ("{desc}.",
         "Revealing a face-down card is the priority in Klondike, since each one hides a card and a potential play. This move uncovers one in {src} and keeps the game progressing."),
        ("A face-down card in {src} can be uncovered this turn.",
         "Uncovering hidden cards opens new options and is the right kind of progress, so this reveal is the move."),
    ],
    "frees_col": [
        ("{desc} clears its source column.",
         "An empty column is a flexible slot for a King-headed run, so freeing one here is worthwhile."),
        ("This move empties a tableau column.",
         "Opening a column gives space to relocate a King and its run later, so taking it now is sound."),
    ],
    "found_low": [
        ("An Ace or Two is available to play up.",
         "Aces and Twos never help the tableau, so sending them to the foundation is pure gain and frees flexibility."),
        ("A lowest-rank card can go to its foundation.",
         "Playing an Ace or Two up is always safe, so doing it now is correct."),
    ],
    "found_high": [
        ("This card can advance its suit on the foundation.",
         "Sending it up makes progress while a winning line remains, so it is the move here."),
        ("A foundation play is available for this suit.",
         "Advancing the suit is the right choice in this position."),
    ],
    "to_empty": [
        ("An empty column can receive this card.",
         "Filling the open column keeps the tableau workable, so this is the move."),
        ("This card can move into an open column.",
         "Using the empty space here sets up further play, so take it."),
    ],
    "build": [
        ("{desc} extends a descending alternating-color sequence.",
         "Consolidating the run keeps cards ordered toward a later reveal, so this build is correct."),
        ("A tableau build is available in {src}.",
         "Building the sequence now positions it to expose a face-down card soon, so this is the move."),
    ],
    "waste_build": [
        ("The waste card can join a tableau sequence.",
         "Placing it into the tableau frees the waste and keeps it usable, so take it."),
        ("This places the waste card onto a tableau build.",
         "Moving it off the waste keeps options open, so this is the move."),
    ],
    "draw": [
        ("No tableau move makes progress this turn.",
         "With no reveal or foundation play available, drawing a fresh card from the stock is the right move."),
        ("The tableau offers no progressing move right now.",
         "Drawing from the stock to find a usable card is correct when nothing on the board advances."),
    ],
    "recycle": [
        ("The stock is empty and the waste still holds cards.",
         "Recycling restores access to cards still needed for another pass, so it is the move."),
        ("The stock has run out this cycle.",
         "Recycling re-exposes the waste cards for another pass, which is correct here."),
    ],
    "other": [
        ("This move keeps the game progressing.",
         "It keeps a winning line available, so it is the right choice in this position."),
    ],
}


def make_completion(state, m, idx: int, tag: str, variant_i: int) -> str:
    desc = H.describe_move(m, state)
    src = (f"column {m.source_pile + 1}"
           if getattr(getattr(m, "move_type", None), "value", None) == "tableau_to_tableau"
           else "the tableau")
    variants = _TEMPLATES.get(tag, _TEMPLATES["other"])
    analysis_fmt, plan_fmt = variants[variant_i % len(variants)]
    obj = {
        "board_analysis": analysis_fmt.format(desc=desc, src=src),
        "strategic_plan": plan_fmt.format(desc=desc, src=src),
        "final_decision": {"move_index": idx},
    }
    return json.dumps(obj, indent=2)


# ---------- trajectory walker (replicates the harness bookkeeping) ----------

class Walk:
    def __init__(self, state):
        self.state, _ = H.auto_flip(state)
        self.recent_moves: list = []
        self.cycle = 1
        self.ever_seen: set = set()
        self.since_foundation = 0
        self.since_reveal = 0

    def apply(self, chosen) -> bool:
        """Replicate apply_and_track (v16 path): recycle/draw tracking, history
        entry, auto-flip, and the two stall counters. False on engine violation."""
        st = self.state
        fc0 = sum(len(f) for f in st.foundations)
        mt = "recycle_stock" if is_recycle(chosen) else chosen.move_type.value
        drawn_card = H.card_short(st.stock[-1]) if mt == "stock_to_waste" else None
        state_before = st
        if mt == "recycle_stock":
            self.state = H.apply_recycle(st)
            self.cycle += 1
        else:
            ns = apply_move(st, chosen)
            if ns is None:
                return False
            self.state = ns
        if drawn_card:
            self.ever_seen.add(drawn_card)
        self.recent_moves.append(
            H.build_history_entry(chosen, state_before, drawn_card=drawn_card))
        self.state, flipped = H.auto_flip(self.state)
        new_fc = sum(len(f) for f in self.state.foundations)
        self.since_foundation = 0 if new_fc > fc0 else self.since_foundation + 1
        self.since_reveal = 0 if flipped else self.since_reveal + 1
        return True

    def prompt(self, legal, header: str) -> str:
        return H.render_prompt_v16(self.state, legal, self.recent_moves, self.cycle,
                                   self.ever_seen, self.since_foundation,
                                   self.since_reveal, header)


def walk_seed(seed: int, header: str, node_cap: int, per_seed_cap: int, start_rows: int):
    """Yield play-matched {prompt, completion} rows along a solver winning line.

    Follows the precomputed harness-space winning path, collecting a row at every
    decision point (>=2 legal moves) and silently auto-firing single-legal
    positions, so the walk strictly converges to the win (no greedy wandering).
    A full game is ~110-175 correlated decisions, so for DIVERSITY we keep only an
    EVENLY-SPACED subsample of up to per_seed_cap rows (opening, mid, endgame all
    represented), letting 700 rows come from many distinct deals rather than ~5."""
    state = fresh_state(seed)
    verdict, path, _ = solve_path_harness(state, node_cap)
    if verdict != "SOLVED":
        return  # not harness-winnable under the cap; skip

    w = Walk(state)
    cand: list = []  # (state_at_decision, move, idx, tag, prompt)
    for mv in path:
        if is_won(w.state):
            break
        legal = H.visible_legal_moves(w.state)
        idx = index_of(mv, legal)
        if idx is None:
            return  # path desync (should not happen); abandon seed
        if len(legal) >= 2:
            prompt = w.prompt(legal, header)
            if f"[{idx}] " not in prompt:
                return  # rendered index missing; abandon seed
            cand.append((w.state, legal[idx], idx, classify_move(w.state, legal[idx]), prompt))
        if not w.apply(legal[idx]):
            return

    if per_seed_cap and len(cand) > per_seed_cap:
        step = len(cand) / per_seed_cap
        picks = [cand[int(i * step)] for i in range(per_seed_cap)]
    else:
        picks = cand
    for k, (st, mv, idx, tag, prompt) in enumerate(picks):
        yield {"prompt": prompt,
               "completion": make_completion(st, mv, idx, tag, start_rows + k)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=700)
    ap.add_argument("--node-cap", type=int, default=200_000,
                    help="best-first node cap for the per-deal winning-path search")
    ap.add_argument("--seed-start", type=int, default=8_000_001)
    ap.add_argument("--max-seeds", type=int, default=400,
                    help="cap on seeds attempted (each winnable deal yields a whole game)")
    ap.add_argument("--per-seed-cap", type=int, default=25,
                    help="evenly-spaced rows kept per game (diversity over correlation)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    header = H.PROMPT_HEADER_V16_PATH.read_text()
    out = Path(args.out)

    rows: list = []
    seed = args.seed_start
    seeds_used = winnable = 0
    with out.open("w") as fh:
        while len(rows) < args.max_rows and seeds_used < args.max_seeds:
            seeds_used += 1
            got = 0
            for row in walk_seed(seed, header, args.node_cap, args.per_seed_cap, len(rows)):
                fh.write(json.dumps(row) + "\n")
                rows.append(row)
                got += 1
                if len(rows) >= args.max_rows:
                    break
            if got:
                winnable += 1
                print(f"  seed {seed}: +{got} rows  (total {len(rows)}/{args.max_rows})",
                      flush=True)
            seed += 1

    print(f"\nseeds attempted {seeds_used}, winnable {winnable}; "
          f"wrote {len(rows)} rows -> {out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
