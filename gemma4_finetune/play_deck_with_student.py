#!/usr/bin/env python3
"""Play one full Klondike game with the deployed LoRA student.

Loads a deck from data/benchmarks/winnable_decks.json, hydrates a
GameState, then loops: render hybrid-v1 prompt -> call model ->
parse move_index -> apply move via engine -> repeat. Auto-fires
flip-card moves on the harness side (matches the harvester, which
flips revealed face-down tableau cards automatically). Tracks
RECENT MOVES (last 10) and PRIOR REASONING (last 5) the same way
the harvester does so the student sees prompts on its training
distribution.

Termination:
  - won            : all 52 cards in foundations
  - stalled        : engine returns zero legal moves (excluding flips)
  - max_turns      : safety cap (default 300)
  - parse_failure  : model emitted no parseable JSON (configurable
                     allowance with --max-parse-failures)
  - illegal_move   : model picked an out-of-range index (same)

Writes:
  out_dir/summary.json       : outcome, turn count, foundation trajectory
  out_dir/turns.jsonl        : per-turn record (prompt, response, decision,
                               applied move, post-move state digest)
  out_dir/responses/*.txt    : raw model responses for inspection

Usage:
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \\
      --deck-seed 3263196305 \\
      --adapter-path gemma4_finetune/adapters_t5_at750 \\
      --out-dir gemma4_finetune/play_runs/v1_seed3263196305
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
DECK_PATH = REPO / "data" / "benchmarks" / "winnable_decks.json"

# Defer heavy imports until main() so the helpers can be unit-tested
# without mlx / engine installed.

RANK_STR_TO_INT = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 11, "Q": 12, "K": 13,
}
RANK_INT_TO_STR = {v: k for k, v in RANK_STR_TO_INT.items()}
# pysol-style short rank notation: T for 10, others identical
RANK_INT_TO_SHORT = {**RANK_INT_TO_STR, 10: "T"}
SUIT_NAMES = ["hearts", "diamonds", "clubs", "spades"]
SUIT_INITIAL = {"hearts": "H", "diamonds": "D", "clubs": "C", "spades": "S"}

# Engine move-type to harvester move-type names. flip_tableau_card and
# foundation_to_tableau are auto-fired / suppressed; the harvester does
# not surface them in LEGAL MOVES.
ENGINE_TO_HARVESTER_MOVETYPE = {
    "tableau_to_tableau":   "tableau_to_tableau",
    "tableau_to_foundation": "tableau_to_foundation",
    "waste_to_tableau":     "discard_to_tableau",
    "waste_to_foundation":  "discard_to_foundation",
    "stock_to_waste":       "draw_card",
    "recycle_stock":        "recycle_stock",
}

# The engine has NO recycle move (generate_moves only emits STOCK_TO_WASTE while
# the stock is non-empty), so before 2026-06-10 every full-game run permanently
# lost the waste once the stock emptied -- a rules infidelity vs the harvester,
# whose wins routinely recycle 2-4 times. The harness now synthesizes a recycle
# as a pseudo-move and applies it itself (waste reversed back into the stock,
# preserving draw order), matching harvester behaviour.
class _RecycleMoveType:
    value = "recycle_stock"


class RecycleMove:
    move_type = _RecycleMoveType()
    num_cards = 0
    source_pile = None
    dest_pile = None


RECYCLE_MOVE = RecycleMove()

# v1.6 static prompt header, extracted byte-verbatim from corpus session
# #57947c (templateHash 7d2c6cad..., identical across 3 sessions / 3 builds,
# sha 4d02ed26d7). Loaded from the sidecar file so no transcription can drift;
# em-dashes and all spacing are the harvester's own bytes (byte fidelity to the
# training distribution is the point).
PROMPT_HEADER_V16_PATH = THIS_DIR / "prompt_header_v16.txt"

STATIC_PROMPT_HEADER = """\
You are an expert Klondike Solitaire strategist acting as an advisor.

KLONDIKE SOLITAIRE RULES (this variant):
- There are 7 tableau columns, 4 foundations (one per suit), a stock (draw) pile and a waste (discard) pile.
- Tableau columns are numbered 1 to 7. Always refer to a column by that 1-based number, never 0-based.
- Foundations are built UP by suit, starting from the Ace: A, 2, 3, ... up to King.
- Tableau columns are built DOWN in alternating colors (red on black, black on red). Example: a black 7 can go on a red 8.
- Only a King (or a valid sequence headed by a King) may be moved onto an EMPTY tableau column.
- A face-up run of cards in a tableau column may be moved together as a unit onto another column.
- The top card of a column, or a valid run, may move to another column; the top card may move to a foundation.
- The top (most recent) card of the waste pile may move to a tableau column or a foundation.
- Drawing turns the next stock card face-up onto the waste. When the stock is empty it can be recycled from the waste.
- When a face-down tableau card is exposed by a move, it flips face-up automatically.
- The game is WON when all 52 cards reach the foundations.

THE GOAL: choose the single move that gives the best chance of eventually winning.

STRATEGY GUIDANCE (heuristics, not absolute rules):
- Prioritize moves that turn over (reveal) a face-down tableau card -- hidden cards are the main obstacle.
- Play Aces and 2s to the foundations promptly; they are rarely useful in the tableau.
- Be cautious sending higher cards to the foundations too early -- they are sometimes needed to receive tableau cards.
- Do not empty a column unless you have a King ready to occupy it.
- Prefer exposing new cards and creating useful sequences over shuffling cards between columns with no gain.
- Drawing from the stock is reasonable when no productive tableau/foundation move exists.
- Avoid moves that simply undo a recent move or lead nowhere.

RESPONSE FORMAT:
You will receive the current game as plain-text blocks (NOTATION, FOUNDATIONS, STOCK,
TABLEAU, RECENT MOVES, SEEN IN WASTE, LEGAL MOVES, PROGRESS -- some are optional). The
LEGAL MOVES block is a numbered list; each line begins with [index], the canonical
move identifier you must return.
Reason step by step, then respond with ONLY a single JSON object containing exactly
these three keys, in this order (no prose or markdown fences outside the object):
{
  "board_analysis": <string>,
  "strategic_plan": <string>,
  "final_decision": { "move_index": <number>, "confidence": <number>, "alternative_move_index": <number> }
}
- board_analysis: assess the current board -- hidden cards, blocked columns, foundation
  progress, and the opportunities each legal move opens or closes.
- strategic_plan: explain your plan and why the chosen move is best, given that analysis.
- final_decision.move_index: the [index] of your chosen move from the LEGAL MOVES block.
- final_decision.confidence: a calibrated probability (0 to 1) that this move is
  objectively the best one available -- a genuine estimate, not a feeling. Use the
  full range honestly:
    1.0-0.9  forced, or clearly dominant -- any other move would be a mistake.
    0.9-0.7  strong -- one plausible alternative exists, but this move is better.
    0.7-0.5  a real toss-up between two or three reasonable moves.
    0.5-0.3  a guess -- the board is unclear or several moves look about equal.
    below 0.3  little better than picking at random.
  If you would not bet on the move, do not report high confidence.
- final_decision.alternative_move_index: optional; the index of your second-choice move.
Produce the keys in the order above: analyse the board first, then plan, then decide.

"""


def card_short(card) -> str:
    """Render a card as its short string: rank (T for 10) + suit initial.
    Works on engine Card objects."""
    return RANK_INT_TO_SHORT[card.rank] + SUIT_INITIAL[card.suit.value]


def deck_to_state(deck: dict):
    """Hydrate a winnable_decks.json deck record into an engine GameState."""
    from solitaire_analytics.models import GameState, Card
    from solitaire_analytics.models.card import Suit

    suit_map = {s: getattr(Suit, s.upper()) for s in SUIT_NAMES}

    def to_card(c, default_face_up=False):
        return Card(
            rank=RANK_STR_TO_INT[c["rank"]],
            suit=suit_map[c["suit"]],
            face_up=c.get("face_up", default_face_up),
        )

    tableau = [[to_card(c) for c in col] for col in deck["tableau"]]
    # Hydrate stock cards as face_up=True. The engine's STOCK_TO_WASTE apply
    # does not flip the card, so a face-down stock card would land in the
    # waste face-down and then in the tableau face-down via WASTE_TO_TABLEAU,
    # breaking legality checks downstream. The harvester operationally
    # treats drawn waste cards as visible/playable, and the engine does not
    # enforce face-down stock anywhere in validator or apply_move.
    #
    # Stock orientation: the deck JSON lists drawPile top-of-stock-first
    # (i.e. drawPile[0] is the FIRST card the harvester draws). The engine's
    # STOCK_TO_WASTE pops from the END of state.stock (state.stock[-1] is
    # top-of-stock per engine convention). Reverse the list so the engine
    # draws cards in the same order the harvester did.
    stock = [to_card(c, default_face_up=True) for c in reversed(deck["stock"])]
    return GameState(
        tableau=tableau,
        foundations=[[], [], [], []],
        stock=stock,
        waste=[],
        move_count=0,
        score=0,
    )


def visible_legal_moves(state, allow_recycle: bool = True):
    """Engine moves the agent should see: exclude flips (auto-fired)
    and foundation-to-tableau (harvester doesn't surface this). Appends the
    synthesized recycle pseudo-move when the stock is empty and the waste has
    cards (the engine itself has no recycle; see RecycleMove above)."""
    from solitaire_analytics.engine import generate_moves
    moves = []
    for m in generate_moves(state):
        mt = m.move_type.value
        if mt in ("flip_tableau_card", "foundation_to_tableau"):
            continue
        # The engine emits one ace-to-foundation move per EMPTY foundation pile
        # (4 duplicates with identical describe text); the harvester surfaces
        # exactly one, onto the card's own suit pile. Keep only the canonical
        # one. This also keeps foundations[i] suit-aligned with the FOUNDATIONS
        # render (H,D,C,S) -- previously a model picking the first duplicate
        # could land AC on the hearts-indexed pile and every later FOUNDATIONS
        # line rendered it under the wrong suit label.
        if mt in ("tableau_to_foundation", "waste_to_foundation"):
            card = (state.tableau[m.source_pile][-1] if mt == "tableau_to_foundation"
                    else state.waste[-1])
            if m.dest_pile != SUIT_NAMES.index(card.suit.value):
                continue
        moves.append(m)
    if allow_recycle and not state.stock and state.waste:
        moves.append(RECYCLE_MOVE)
    return moves


def apply_recycle(state):
    """Apply the synthesized recycle: flip the waste back into the stock,
    preserving draw order (waste[0] was drawn first this cycle, so it must be
    drawn first next cycle; the engine pops stock[-1])."""
    new_state = state.copy()
    new_state.stock = list(reversed(new_state.waste))
    new_state.waste = []
    return new_state


def auto_flip(state):
    """Apply any pending flip_tableau_card moves until none remain.
    Mimics the harvester's behaviour where a revealed face-down tableau
    card flips automatically. Returns (new_state, list_of_flipped_cards)."""
    from solitaire_analytics.engine import generate_moves, apply_move
    flipped = []
    while True:
        flips = [m for m in generate_moves(state) if m.move_type.value == "flip_tableau_card"]
        if not flips:
            return state, flipped
        m = flips[0]
        revealed_card = state.tableau[m.source_pile][-1]
        state = apply_move(state, m)
        flipped.append({
            "column": m.source_pile + 1,
            "card": card_short(revealed_card),
        })


def describe_move(move, state) -> str:
    """Render an engine Move in the same prose form the harvester
    emits ('Move X plus N more from column A to column B', etc.).
    Column indexing in the output is 1-based."""
    mt = move.move_type.value
    if mt == "stock_to_waste":
        return "Draw the next card from the stock onto the waste"
    if mt == "recycle_stock":
        return "Recycle the waste pile back into the stock"
    if mt == "tableau_to_tableau":
        col = state.tableau[move.source_pile]
        # The lead card is the deepest of the run being moved
        lead_idx = len(col) - move.num_cards
        lead = card_short(col[lead_idx])
        plus = f" plus {move.num_cards - 1} more" if move.num_cards > 1 else ""
        dst_col = state.tableau[move.dest_pile]
        empty = " (empty)" if not dst_col else ""
        reveals = ""
        # Reveals a hidden card if the slot beneath the lead is face-down
        if lead_idx > 0 and not col[lead_idx - 1].face_up:
            reveals = " (reveals a hidden card)"
        # Corpus shows BOTH tags when both apply: "(empty) (reveals a hidden card)"
        return (f"Move {lead}{plus} from column {move.source_pile + 1} "
                f"to column {move.dest_pile + 1}{empty}{reveals}")
    if mt == "tableau_to_foundation":
        col = state.tableau[move.source_pile]
        card = card_short(col[-1])
        suit = col[-1].suit.value
        return f"Send {card} from column {move.source_pile + 1} to the {suit} foundation"
    if mt == "waste_to_tableau":
        card = card_short(state.waste[-1])
        empty = " (empty)" if not state.tableau[move.dest_pile] else ""
        return f"Move {card} from the waste to column {move.dest_pile + 1}{empty}"
    if mt == "waste_to_foundation":
        card = state.waste[-1]
        return f"Send {card_short(card)} from the waste to the {card.suit.value} foundation"
    return f"({mt})"  # fallback


def render_recent_moves_line(history_entry) -> str:
    """Render one historical move in the RECENT MOVES compact form
    the harvester uses (no 'Move' or 'Send' prefix; column index)."""
    et = history_entry["engine_move_type"]
    if et == "stock_to_waste":
        return f"draw {history_entry.get('drawn_card', '?')}"
    if et == "recycle_stock":
        return "recycle stock"
    if et == "tableau_to_tableau":
        return (f"move {history_entry['lead_card']} col {history_entry['from_col']} "
                f"-> col {history_entry['to_col']}")
    if et == "tableau_to_foundation":
        return (f"{history_entry['card']} col {history_entry['from_col']} "
                f"-> {history_entry['foundation_suit']} foundation")
    if et == "waste_to_tableau":
        return f"{history_entry['card']} waste -> col {history_entry['to_col']}"
    if et == "waste_to_foundation":
        return f"{history_entry['card']} waste -> {history_entry['foundation_suit']} foundation"
    return f"({et})"


def render_prompt(
    state,
    legal_moves,
    recent_moves: list,
    prior_decisions: list,
    seen_in_waste: list,
    recycle_available: bool,
    stall_info: Optional[dict] = None,
) -> str:
    """Render the hybrid-v1 prompt for the current state.

    stall_info: optional {"no_progress_moves", "position_seen_before"} for the
    --stall-field A/B. When None (default), the STALL/REPEAT block is omitted
    and the prompt is byte-identical to the baseline template.
    """
    lines = []

    # CURRENT GAME header
    lines.append("CURRENT GAME:")
    lines.append("NOTATION: rank+suit (A 2-9 T J Q K; H D C S). ?? = face-down. "
                 "In each column the top of the stack is the rightmost card.")
    lines.append("")

    # FOUNDATIONS
    def foundation_top(suit_idx) -> str:
        f = state.foundations[suit_idx]
        if not f:
            return "--"
        c = f[-1]
        return RANK_INT_TO_SHORT[c.rank] + SUIT_INITIAL[c.suit.value]
    f_h = foundation_top(0)
    f_d = foundation_top(1)
    f_c = foundation_top(2)
    f_s = foundation_top(3)
    lines.append(f"FOUNDATIONS:   H: {f_h}   D: {f_d}   C: {f_c}   S: {f_s}")

    # STOCK + WASTE + recycle
    waste_top = card_short(state.waste[-1]) if state.waste else "--"
    recycle = "yes" if recycle_available else "no"
    lines.append(f"STOCK: {len(state.stock)} cards   WASTE top: {waste_top}   recycle stock: {recycle}")
    lines.append("")

    # TABLEAU
    lines.append("TABLEAU:")
    for i, col in enumerate(state.tableau):
        if not col:
            lines.append(f"  col{i + 1}: <empty>")
            continue
        parts = []
        for c in col:
            s = RANK_INT_TO_SHORT[c.rank] + SUIT_INITIAL[c.suit.value]
            parts.append(s if c.face_up else "??")
        lines.append(f"  col{i + 1}: " + " ".join(parts))
    lines.append("")

    # RECENT MOVES (last 10, oldest first)
    if recent_moves:
        lines.append("RECENT MOVES (oldest -> newest; review before picking, do not undo your own work):")
        for i, m in enumerate(recent_moves[-10:], start=1):
            lines.append(f"  {i:>2}. {render_recent_moves_line(m)}")
        lines.append("")

    # SEEN IN WASTE THIS CYCLE
    if seen_in_waste:
        lines.append("SEEN IN WASTE THIS CYCLE: " + " ".join(seen_in_waste))
        lines.append("")

    # LEGAL MOVES
    lines.append("LEGAL MOVES (respond with the index of your chosen move):")
    for i, m in enumerate(legal_moves):
        mt_h = ENGINE_TO_HARVESTER_MOVETYPE.get(m.move_type.value, m.move_type.value)
        lines.append(f"  [{i}] {mt_h:<24}  {describe_move(m, state)}")
    lines.append("")

    # PROGRESS
    foundation_cards = sum(len(f) for f in state.foundations)
    face_down = sum(1 for col in state.tableau for c in col if not c.face_up)
    completion = round(foundation_cards / 52 * 100)
    lines.append(f"PROGRESS: foundation={foundation_cards}/52, face-down remaining={face_down}, completion={completion}%")
    lines.append("")

    # STALL / REPEAT block (only when --stall-field passes stall_info; empty
    # early-game so baseline prompts are unchanged until a stall exists)
    if stall_info is not None:
        from stall_field import stall_lines
        sl = stall_lines(stall_info.get("no_progress_moves", 0),
                         stall_info.get("position_seen_before", 0))
        if sl:
            lines.extend(sl)
            lines.append("")

    # PRIOR REASONING (last 5)
    if prior_decisions:
        lines.append("PRIOR REASONING (may be obsolete; verify against current state):")
        for i, p in enumerate(prior_decisions[-5:], start=1):
            lines.append(f"  {i}. move: {p['move_text']}")
            lines.append(f"     why: {p['why']}")
        lines.append("")

    lines.append("Now choose the best move and reply with only the JSON object.")
    return STATIC_PROMPT_HEADER + "\n".join(lines)


def render_prompt_v16(
    state,
    legal_moves,
    recent_moves: list,
    cycle: int,
    ever_seen: set,
    since_foundation: int,
    since_reveal: int,
    header: str,
) -> str:
    """Render the hybrid-v1.6 prompt, byte-faithful to the harvester's render
    (templateHash 7d2c6cad...). Every format decision below was verified against
    real corpus prompts (#57947c / #0b0f2e / #4c73b8), see the 2026-06-10 session:
      - STOCK line carries CYCLE; no NOTATION line here (it lives in the header);
      - no SEEN IN WASTE and no PRIOR REASONING blocks (v1.0-era only);
      - RECENT MOVES indices right-align to the widest index in the window;
      - DRAW TIMELINE renders ONLY when the waste is non-empty (the {top} token
        is the anchor); stock prints bottom->top left of it (so the token
        directly left of the brace is the next draw), earlier-drawn waste cards
        print to its right, most recent first; ??? = never-yet-drawn (cycle 1);
      - legal-move type names pad to 25 columns;
      - PROGRESS ends with the two v1.5 stall counters.
    """
    lines = []
    lines.append("CURRENT GAME:")

    def foundation_top(suit_idx) -> str:
        f = state.foundations[suit_idx]
        if not f:
            return "--"
        c = f[-1]
        return RANK_INT_TO_SHORT[c.rank] + SUIT_INITIAL[c.suit.value]

    lines.append(f"FOUNDATIONS:   H: {foundation_top(0)}   D: {foundation_top(1)}"
                 f"   C: {foundation_top(2)}   S: {foundation_top(3)}")
    waste_top = card_short(state.waste[-1]) if state.waste else "--"
    recycle = "yes" if (not state.stock and state.waste) else "no"
    lines.append(f"STOCK: {len(state.stock)} cards   CYCLE: {cycle}   "
                 f"WASTE top: {waste_top}   recycle stock: {recycle}")
    lines.append("")

    lines.append("TABLEAU:")
    for i, col in enumerate(state.tableau):
        if not col:
            lines.append(f"  col{i + 1}: <empty>")
            continue
        parts = [
            (RANK_INT_TO_SHORT[c.rank] + SUIT_INITIAL[c.suit.value]) if c.face_up else "??"
            for c in col
        ]
        lines.append(f"  col{i + 1}: " + " ".join(parts))
    lines.append("")

    if recent_moves:
        window = recent_moves[-10:]
        width = len(str(len(window)))
        lines.append("RECENT MOVES (oldest -> newest; review before picking, "
                     "do not undo your own work):")
        for i, m in enumerate(window, start=1):
            lines.append(f"  {i:>{width}}. {render_recent_moves_line(m)}")
        lines.append("")

    if state.waste:
        tokens = []
        for c in state.stock:  # bottom -> top; token left of brace = next draw
            s = card_short(c)
            tokens.append(s if s in ever_seen else "???")
        tokens.append("{" + card_short(state.waste[-1]) + "}")
        for c in reversed(state.waste[:-1]):  # drawn earlier, most recent first
            tokens.append(card_short(c))
        lines.append("DRAW TIMELINE:")
        lines.append("  " + " ".join(tokens))
        lines.append("")

    lines.append("LEGAL MOVES (respond with the index of your chosen move):")
    for i, m in enumerate(legal_moves):
        mt_h = ENGINE_TO_HARVESTER_MOVETYPE.get(m.move_type.value, m.move_type.value)
        lines.append(f"  [{i}] {mt_h:<25}{describe_move(m, state)}")
    lines.append("")

    foundation_cards = sum(len(f) for f in state.foundations)
    face_down = sum(1 for col in state.tableau for c in col if not c.face_up)
    completion = round(foundation_cards / 52 * 100)
    lines.append(f"PROGRESS: foundation={foundation_cards}/52, "
                 f"face-down remaining={face_down}, completion={completion}%, "
                 f"turns since foundation grew: {since_foundation}, "
                 f"turns since a card was revealed: {since_reveal}")
    lines.append("")

    lines.append("Now choose the best move and reply with only the JSON object.")
    return header + "\n".join(lines)


def _decision_from_json_text(text: str):
    """Strict path: find candidate {...} blocks, parse, return the decision
    4-tuple (move_index, confidence, board_analysis, strategic_plan) or None."""
    cands = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    for c in reversed(cands):
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fd = obj.get("final_decision")
        if isinstance(fd, dict) and isinstance(fd.get("move_index"), int):
            return (
                fd["move_index"],
                fd.get("confidence") if isinstance(fd.get("confidence"), (int, float)) else None,
                obj.get("board_analysis"),
                obj.get("strategic_plan") or obj.get("strategicPlan"),
            )
    return None


def _repair_inner_quotes(text: str) -> str:
    """Escape unescaped double quotes inside single-line JSON string values.

    Targets the dominant recorded parse-failure mode (tourA_v16 forensics):
    the model echoes the prompt's own quoted phrases, e.g. (reveals a hidden
    card) wrapped in literal quotes, unescaped inside board_analysis prose.
    Validated offline against the 27 recorded tourA_v16 failures: rescues all
    6 base-arm death events, agreeing 18/18 with independent field extraction.
    """
    out = []
    for ln in text.splitlines():
        m = re.match(r'^(\s*"[A-Za-z_]+"\s*:\s*")(.*)(",?\s*)$', ln)
        if m and '"' in m.group(2):
            body = (m.group(2).replace('\\"', "\x00")
                    .replace('"', '\\"').replace("\x00", '\\"'))
            ln = m.group(1) + body + m.group(3)
        out.append(ln)
    return "\n".join(out)


def extract_decision(text: str):
    """Pull (move_index, confidence, board_analysis, strategic_plan, json_ok,
    via) from the model's response, trying three tiers in order:
      strict -- the response parses as-is (the only tier before 2026-06-11)
      repair -- parses after escaping unescaped inner quotes in string values
      field  -- last "move_index": <int> in the raw text (the move is still
                the model's own stated choice; only the JSON wrapper broke)
    Returns (None, None, None, None, False, None) when no tier can read it."""
    got = _decision_from_json_text(text)
    if got is not None:
        return (*got, True, "strict")
    got = _decision_from_json_text(_repair_inner_quotes(text))
    if got is not None:
        return (*got, True, "repair")
    ms = re.findall(r'"move_index"\s*:\s*(-?\d+)', text)
    if ms:
        return (int(ms[-1]), None, None, None, True, "field")
    return (None, None, None, None, False, None)


def build_history_entry(move, state_before, drawn_card: Optional[str] = None) -> dict:
    """Capture enough of a chosen move to render it in future RECENT MOVES."""
    mt = move.move_type.value
    entry = {"engine_move_type": mt}
    if mt == "recycle_stock":
        return entry
    if mt == "stock_to_waste":
        entry["drawn_card"] = drawn_card or "?"
    elif mt == "tableau_to_tableau":
        col = state_before.tableau[move.source_pile]
        lead_idx = len(col) - move.num_cards
        entry["lead_card"] = card_short(col[lead_idx])
        entry["from_col"] = move.source_pile + 1
        entry["to_col"] = move.dest_pile + 1
    elif mt == "tableau_to_foundation":
        col = state_before.tableau[move.source_pile]
        entry["card"] = card_short(col[-1])
        entry["from_col"] = move.source_pile + 1
        entry["foundation_suit"] = col[-1].suit.value
    elif mt == "waste_to_tableau":
        c = state_before.waste[-1]
        entry["card"] = card_short(c)
        entry["to_col"] = move.dest_pile + 1
    elif mt == "waste_to_foundation":
        c = state_before.waste[-1]
        entry["card"] = card_short(c)
        entry["foundation_suit"] = c.suit.value
    return entry


def compute_recycle_available(state, stock_cycle_drawn: int) -> bool:
    """Recycle is available when stock is empty and the waste has cards."""
    return len(state.stock) == 0 and len(state.waste) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-seed", type=int, required=True,
                    help="Seed of the deck to play (must be in winnable_decks.json)")
    ap.add_argument("--adapter-path", default=None,
                    help="Optional LoRA adapter path; omit for untuned base")
    ap.add_argument("--model-id", default="mlx-community/gemma-3n-E2B-it-text-4bit-dwq",
                    help="HF model id; default is the v1.1 deployed base")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write summary.json + turns.jsonl + responses/")
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-parse-failures", type=int, default=3,
                    help="Abort after this many consecutive JSON parse failures")
    ap.add_argument("--parse-retry-temp", type=float, default=0.3,
                    help="Sampling temperature armed for the retry after a "
                         "parse failure (greedy retries of an unchanged prompt "
                         "are byte-identical, so temp 0 would just repeat the "
                         "same bad response; 0.3 = production teacher temp). "
                         "Reset to greedy after the next successful parse.")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="Policy sampling temperature for EVERY call (0 = "
                         "greedy, the paired-eval default). Use with "
                         "--sample-seed for best-of-N probes.")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="mx.random seed. REQUIRED for meaningful best-of-N: "
                         "without it mlx's default key makes every process "
                         "sample identically.")
    ap.add_argument("--warm-start-from", default=None,
                    help="Path to a recorded turns.jsonl: replay its decisions "
                         "engine-side (no model calls, seconds) and start live "
                         "play where it left off. v1.6 only. Every replayed "
                         "decision's re-rendered prompt must match the recorded "
                         "prompt_chars or the run aborts (drift gate).")
    ap.add_argument("--warm-start-until", type=int, default=None,
                    help="Stop the warm-start replay before this recorded turn "
                         "index and go live there (e.g. resume from a stall "
                         "onset mid-run). Default: replay everything.")
    ap.add_argument("--max-illegal-moves", type=int, default=3,
                    help="Abort after this many illegal move-index picks in a row")
    ap.add_argument("--prompt-version", choices=["v1.6", "v1.0"], default="v1.6",
                    help="v1.6 = byte-faithful current harvester prompt (default); "
                         "v1.0 = the legacy header this harness used before "
                         "2026-06-10 (kept for back-compat with old runs)")
    ap.add_argument("--no-auto-forced", action="store_true",
                    help="send forced single-legal-move positions to the model "
                         "instead of auto-playing them (auto-play matches the "
                         "production harvester and is the v1.6-mode default)")
    ap.add_argument("--stall-field", action="store_true",
                    help="Append the STALL/REPEAT temporal-state block to the prompt "
                         "(harvester-recommendation A/B). Off = baseline template.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir / "responses"
    resp_dir.mkdir(exist_ok=True)
    turns_path = out_dir / "turns.jsonl"
    summary_path = out_dir / "summary.json"

    # Load deck
    decks = json.loads(DECK_PATH.read_text())["decks"]
    deck = next((d for d in decks if d["seed"] == args.deck_seed), None)
    if deck is None:
        sys.exit(f"deck-seed {args.deck_seed} not in {DECK_PATH}")
    state = deck_to_state(deck)
    print(f"Loaded seed {args.deck_seed} from {deck['source_file']}", flush=True)

    # Defer model imports
    sys.path.insert(0, str(THIS_DIR))
    if "Gemma4" in args.model_id:
        import gemma4_text_patch  # noqa: F401
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from solitaire_analytics.engine import apply_move

    print(f"Loading {args.model_id}"
          + (f" + adapter={args.adapter_path}" if args.adapter_path else " (no adapter)"),
          flush=True)
    t0 = time.time()
    mx.reset_peak_memory()
    if args.adapter_path:
        model, tokenizer = load(args.model_id, adapter_path=args.adapter_path)
    else:
        model, tokenizer = load(args.model_id)
    print(f"  load: {time.time() - t0:.1f}s, peak after load = {mx.get_peak_memory() / 1e9:.2f} GB",
          flush=True)

    recent_moves: list = []
    prior_decisions: list = []
    seen_in_waste: list = []
    no_progress = 0           # consecutive moves with no fc/fd change (stall signal)

    v16 = args.prompt_version == "v1.6"
    header_v16 = ""
    if v16:
        if not PROMPT_HEADER_V16_PATH.exists():
            sys.exit(f"missing {PROMPT_HEADER_V16_PATH} -- the v1.6 header is "
                     f"extracted from a corpus ai-log; re-extract it (see the "
                     f"2026-06-10 session notes) or pass --prompt-version v1.0")
        header_v16 = PROMPT_HEADER_V16_PATH.read_text()
    auto_forced = v16 and not args.no_auto_forced
    cycle = 1                 # stock cycle counter (CYCLE field; recycles = cycle-1)
    ever_seen: set = set()    # card shorts ever drawn (??? rendering in the timeline)
    since_foundation = 0      # the two v1.5 stall counters on the PROGRESS line
    since_reveal = 0
    auto_forced_count = 0
    position_counts: dict = {}  # board signature -> times seen (repeat signal)
    consecutive_parse_failures = 0
    consecutive_illegal_moves = 0
    if args.sample_seed is not None:
        mx.random.seed(args.sample_seed)  # distinct best-of-N samples need distinct keys
    base_sampler = make_sampler(temp=args.temp) if args.temp > 0 else None
    base_temp = args.temp if args.temp > 0 else None
    retry_temp = args.temp if args.temp > 0 else args.parse_retry_temp
    retry_sampler = base_sampler or make_sampler(temp=args.parse_retry_temp)
    sampler = base_sampler    # greedy unless --temp; armed hotter after a parse failure
    sampler_temp = base_temp
    rescued_turns = 0         # parses that needed the repair/field tier
    temp_rescued_turns = 0    # successful parses produced by a post-failure retry
    plateau_foundation = -1
    plateau_start_turn = 0

    turns_out = turns_path.open("w")
    overall_t0 = time.time()
    outcome = "max_turns"
    end_reason: Optional[str] = None

    def apply_and_track(chosen, turn):
        """Apply a chosen (or auto-forced) move with full bookkeeping: recycle
        synthesis, draw tracking, auto-flips, plateau and the two v1.5 stall
        counters. Returns (move_text, flipped, new_fc, new_fd), or None on an
        engine contract violation (caller aborts)."""
        nonlocal state, cycle, since_foundation, since_reveal, no_progress
        nonlocal plateau_foundation, plateau_start_turn, seen_in_waste
        move_text = describe_move(chosen, state)
        fc0 = sum(len(f) for f in state.foundations)
        fd0 = sum(1 for col in state.tableau for c in col if not c.face_up)
        drawn_card = None
        if chosen.move_type.value == "stock_to_waste":
            drawn_card = card_short(state.stock[-1])
        state_before = state
        if chosen.move_type.value == "recycle_stock":
            state = apply_recycle(state)
            cycle += 1
            seen_in_waste = []  # the v1.0 SEEN IN WASTE block is per-cycle
        else:
            state = apply_move(state, chosen)
            if state is None:
                state = state_before
                return None
        if drawn_card:
            ever_seen.add(drawn_card)
            if drawn_card not in seen_in_waste:
                seen_in_waste.append(drawn_card)
        recent_moves.append(build_history_entry(chosen, state_before,
                                                drawn_card=drawn_card))
        state, flipped = auto_flip(state)
        # The harvester does NOT log flips into RECENT MOVES (verified against
        # v1.6 corpus prompts); only the legacy v1.0 render kept them.
        if not v16:
            for f in flipped:
                recent_moves.append({"engine_move_type": "flip_card",
                                     "card": f["card"], "from_col": f["column"]})
        new_fc = sum(len(f) for f in state.foundations)
        new_fd = sum(1 for col in state.tableau for c in col if not c.face_up)
        if new_fc != plateau_foundation:
            plateau_foundation = new_fc
            plateau_start_turn = turn
        since_foundation = 0 if new_fc > fc0 else since_foundation + 1
        since_reveal = 0 if flipped else since_reveal + 1
        no_progress = 0 if (new_fc != fc0 or new_fd != fd0) else no_progress + 1
        return move_text, flipped, new_fc, new_fd

    # ---- Warm start: replay a recorded run engine-side (no model calls) ----
    start_turn = 0
    warm_started_decisions = 0
    if args.warm_start_from:
        if not v16:
            sys.exit("--warm-start-from supports --prompt-version v1.6 only")
        ws_t0 = time.time()
        for rec in (json.loads(l) for l in Path(args.warm_start_from).open()):
            if rec.get("move_index") is None or rec.get("illegal") or not rec.get("json_ok"):
                continue
            if rec.get("resigned"):
                break
            if args.warm_start_until is not None and rec["turn"] >= args.warm_start_until:
                break
            wturn = rec["turn"]
            # auto-forced phase, exactly as the live loop below runs it
            while auto_forced and auto_forced_count < 400:
                if sum(len(f) for f in state.foundations) == 52:
                    break
                forced = visible_legal_moves(state)
                if len(forced) != 1:
                    break
                if apply_and_track(forced[0], wturn) is None:
                    sys.exit("warm-start: engine violation in auto-forced replay")
                auto_forced_count += 1
            legal = visible_legal_moves(state)
            # Drift gate: the re-rendered prompt must match the recorded
            # length exactly, or the prompt-side bookkeeping has diverged.
            if rec.get("prompt_chars"):
                p = render_prompt_v16(state, legal, recent_moves, cycle,
                                      ever_seen, since_foundation,
                                      since_reveal, header_v16)
                if len(p) != rec["prompt_chars"]:
                    sys.exit(f"warm-start drift at turn {wturn}: re-rendered "
                             f"prompt {len(p)} chars vs recorded "
                             f"{rec['prompt_chars']}")
            if apply_and_track(legal[rec["move_index"]], wturn) is None:
                sys.exit("warm-start: engine violation replaying recorded move")
            warm_started_decisions += 1
            start_turn = wturn + 1
        if args.warm_start_until is not None:
            start_turn = args.warm_start_until
        if start_turn >= args.max_turns:
            sys.exit(f"warm start consumed all turns (resume at {start_turn}, "
                     f"--max-turns {args.max_turns}); raise --max-turns")
        ws_fc = sum(len(f) for f in state.foundations)
        ws_fd = sum(1 for col in state.tableau for c in col if not c.face_up)
        print(f"warm start: {warm_started_decisions} decisions replayed in "
              f"{time.time() - ws_t0:.1f}s from {args.warm_start_from}; live "
              f"play resumes at turn {start_turn} (fc={ws_fc} fd={ws_fd})",
              flush=True)

    for turn in range(start_turn, args.max_turns):
        # Auto-play forced single-legal-move positions (production harvester
        # behaviour: those positions are never sent to the advisor). Does not
        # consume model-turn budget; capped as a runaway guard.
        if auto_forced:
            while auto_forced_count < 400:
                if sum(len(f) for f in state.foundations) == 52:
                    break
                forced = visible_legal_moves(state)
                if len(forced) != 1:
                    break
                res = apply_and_track(forced[0], turn)
                if res is None:
                    break  # violation surfaces via the normal path below
                auto_forced_count += 1
                a_text, a_flipped, a_fc, a_fd = res
                mt_h = ENGINE_TO_HARVESTER_MOVETYPE.get(
                    forced[0].move_type.value, forced[0].move_type.value)
                print(f"  [{turn:>3}] AUTO  {mt_h:<24} {a_text[:46]:<46}  "
                      f"fc={a_fc:>2} fd={a_fd:>2}", flush=True)
                turns_out.write(json.dumps({
                    "turn": turn, "auto_forced": True,
                    "move_type": forced[0].move_type.value,
                    "move_text": a_text, "fc": a_fc, "fd": a_fd,
                    "flipped": a_flipped, "cycle": cycle,
                }) + "\n")

        legal = visible_legal_moves(state)
        # Won check (no more cards outside foundations)
        if sum(len(f) for f in state.foundations) == 52:
            outcome = "won"; break
        if not legal:
            outcome = "stalled"; end_reason = "no legal moves"; break

        if v16:
            prompt = render_prompt_v16(state, legal, recent_moves, cycle,
                                       ever_seen, since_foundation,
                                       since_reveal, header_v16)
        else:
            recycle = compute_recycle_available(state, len(seen_in_waste))
            stall_info = None
            if args.stall_field:
                from stall_field import board_signature
                sig = board_signature(state)
                seen_before = position_counts.get(sig, 0)
                position_counts[sig] = seen_before + 1
                stall_info = {"no_progress_moves": no_progress,
                              "position_seen_before": seen_before}
            prompt = render_prompt(state, legal, recent_moves, prior_decisions,
                                   seen_in_waste, recycle, stall_info)
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

        t_call_start = time.time()
        mx.reset_peak_memory()
        used_temp = sampler_temp
        response = generate(model, tokenizer, prompt=wrapped,
                            max_tokens=args.max_tokens, verbose=False,
                            sampler=sampler)
        t_call = time.time() - t_call_start
        peak = mx.get_peak_memory() / 1e9

        (resp_dir / f"turn_{turn:03d}.txt").write_text(response)
        mi, conf, ba, sp, json_ok, json_via = extract_decision(response)
        fc = sum(len(f) for f in state.foundations)
        fd = sum(1 for col in state.tableau for c in col if not c.face_up)

        if not json_ok or mi is None:
            consecutive_parse_failures += 1
            # Greedy retries of an unchanged prompt are byte-identical, so at
            # temp 0 a parse failure would repeat forever (tourA_v16: every
            # death was 1 bad response x 3 identical retries). Arm a temp>0
            # sampler for the retry; reset to the base policy sampler on the
            # next good parse.
            sampler = retry_sampler
            sampler_temp = retry_temp
            print(f"  [{turn:>3}] PARSE FAILURE  ({consecutive_parse_failures}/"
                  f"{args.max_parse_failures})  call={t_call:.1f}s  fc={fc} fd={fd}"
                  f"  retry-temp={retry_temp}",
                  flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "json_ok": False, "call_seconds": round(t_call, 2),
                "peak_gb": round(peak, 2), "fc": fc, "fd": fd,
                "prompt_chars": len(prompt), "response_chars": len(response),
                "sampled_at_temp": used_temp,
            }) + "\n")
            if consecutive_parse_failures >= args.max_parse_failures:
                outcome = "parse_failure"; break
            continue
        was_retry = consecutive_parse_failures > 0
        consecutive_parse_failures = 0
        if json_via != "strict":
            rescued_turns += 1
        if was_retry:
            temp_rescued_turns += 1
        sampler = base_sampler  # back to the base policy sampler after a good parse
        sampler_temp = base_temp

        # v1.6 resign action (move_index -1 per the RESPONSE FORMAT)
        if v16 and mi == -1:
            print(f"  [{turn:>3}] RESIGN (move_index=-1)  fc={fc} fd={fd}", flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "json_ok": True, "json_ok_via": json_via,
                "move_index": -1,
                "resigned": True, "fc": fc, "fd": fd,
                "call_seconds": round(t_call, 2), "peak_gb": round(peak, 2),
            }) + "\n")
            outcome = "resigned"; break

        # Validate move-index
        if not (0 <= mi < len(legal)):
            consecutive_illegal_moves += 1
            # Same deterministic-death class as parse failures: a greedy
            # retry of an unchanged prompt repeats the identical out-of-range
            # index (seen on the wononly-gate adapter, 2026-06-13: index 7
            # three times). Production absorbs invalid responses with its
            # stochastic retry budget, so arm the retry sampler here too.
            sampler = retry_sampler
            sampler_temp = retry_temp
            print(f"  [{turn:>3}] ILLEGAL move_index={mi} (legal=[0..{len(legal)-1}])  "
                  f"({consecutive_illegal_moves}/{args.max_illegal_moves})"
                  f"  retry-temp={retry_temp}",
                  flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "json_ok": True, "move_index": mi,
                "illegal": True, "call_seconds": round(t_call, 2),
                "fc": fc, "fd": fd, "confidence": conf,
                "sampled_at_temp": used_temp,
            }) + "\n")
            if consecutive_illegal_moves >= args.max_illegal_moves:
                outcome = "illegal_move"; break
            continue
        consecutive_illegal_moves = 0

        chosen = legal[mi]
        res = apply_and_track(chosen, turn)
        if res is None:
            # generate_moves emitted this move but apply_move rejected it.
            # Treat as a hard engine contract violation; abort cleanly.
            print(f"  [{turn:>3}] ENGINE CONTRACT VIOLATION: apply_move returned None "
                  f"for legal move {chosen.move_type.value} src={chosen.source_pile} "
                  f"dst={chosen.dest_pile} num_cards={chosen.num_cards}", flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "engine_violation": True,
                "move_type": chosen.move_type.value,
                "move_index": mi,
            }) + "\n")
            outcome = "engine_violation"; break
        move_text, flipped, new_fc, new_fd = res

        if not v16:
            prior_decisions.append({"move_text": move_text, "why": sp or ""})
            # Keep prior_decisions short; renderer caps to 5
            prior_decisions = prior_decisions[-5:]

        mt_h = ENGINE_TO_HARVESTER_MOVETYPE.get(chosen.move_type.value, chosen.move_type.value)
        via_tag = "" if json_via == "strict" else f" via={json_via}"
        print(f"  [{turn:>3}] mv=[{mi}] {mt_h:<24} {move_text[:46]:<46}  "
              f"fc={new_fc:>2} fd={new_fd:>2} call={t_call:.1f}s conf={conf} "
              f"flips={len(flipped)}{via_tag}",
              flush=True)

        turns_out.write(json.dumps({
            "turn": turn,
            "json_ok": True,
            "json_ok_via": json_via,
            "sampled_at_temp": used_temp,
            "move_index": mi,
            "move_type": chosen.move_type.value,
            "move_text": move_text,
            "confidence": conf,
            "legal_count": len(legal),
            "fc": new_fc,
            "fd": new_fd,
            "cycle": cycle,
            "call_seconds": round(t_call, 2),
            "peak_gb": round(peak, 2),
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "flipped": flipped,
        }) + "\n")

    turns_out.close()
    overall_t = time.time() - overall_t0
    final_fc = sum(len(f) for f in state.foundations)
    final_fd = sum(1 for col in state.tableau for c in col if not c.face_up)

    summary = {
        "deck_seed": args.deck_seed,
        "deck_source_file": deck["source_file"],
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "prompt_version": args.prompt_version,
        "auto_forced_moves": auto_forced_count,
        "recycles": cycle - 1,
        "rescued_turns": rescued_turns,
        "temp_rescued_turns": temp_rescued_turns,
        "parse_retry_temp": args.parse_retry_temp,
        "policy_temp": args.temp,
        "sample_seed": args.sample_seed,
        "warm_start_from": args.warm_start_from,
        "warm_start_resume_turn": start_turn if args.warm_start_from else None,
        "warm_started_decisions": warm_started_decisions,
        "outcome": outcome,
        "end_reason": end_reason,
        "turns_played": turn + 1 if outcome != "max_turns" else args.max_turns,
        "max_turns": args.max_turns,
        "final_foundation_cards": final_fc,
        "final_face_down": final_fd,
        "plateau_at_end_turns": (turn - plateau_start_turn) if outcome != "won" else 0,
        "plateau_at_end_fc": plateau_foundation,
        "wallclock_seconds": round(overall_t, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("")
    print(f"=== {outcome.upper()} ===")
    print(f"  turns played   : {summary['turns_played']}")
    print(f"  final fc/fd    : {final_fc}/52  /  {final_fd} face-down remaining")
    print(f"  plateau at end : {summary['plateau_at_end_turns']} turns at fc={plateau_foundation}")
    print(f"  wallclock      : {overall_t/60:.1f} min")
    print(f"  saved to       : {out_dir}")


if __name__ == "__main__":
    main()
