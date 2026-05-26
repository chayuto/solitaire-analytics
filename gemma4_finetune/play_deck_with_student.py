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
}

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
    stock = [to_card(c, default_face_up=False) for c in deck["stock"]]
    return GameState(
        tableau=tableau,
        foundations=[[], [], [], []],
        stock=stock,
        waste=[],
        move_count=0,
        score=0,
    )


def visible_legal_moves(state):
    """Engine moves the agent should see: exclude flips (auto-fired)
    and foundation-to-tableau (harvester doesn't surface this)."""
    from solitaire_analytics.engine import generate_moves
    return [
        m for m in generate_moves(state)
        if m.move_type.value not in ("flip_tableau_card", "foundation_to_tableau")
    ]


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
        return (f"Move {lead}{plus} from column {move.source_pile + 1} "
                f"to column {move.dest_pile + 1}{empty or reveals}")
    if mt == "tableau_to_foundation":
        col = state.tableau[move.source_pile]
        card = card_short(col[-1])
        suit = col[-1].suit.value
        return f"Send {card} from column {move.source_pile + 1} to the {suit} foundation"
    if mt == "waste_to_tableau":
        card = card_short(state.waste[-1])
        return f"Move {card} from the waste to column {move.dest_pile + 1}"
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
) -> str:
    """Render the hybrid-v1 prompt for the current state."""
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

    # PRIOR REASONING (last 5)
    if prior_decisions:
        lines.append("PRIOR REASONING (may be obsolete; verify against current state):")
        for i, p in enumerate(prior_decisions[-5:], start=1):
            lines.append(f"  {i}. move: {p['move_text']}")
            lines.append(f"     why: {p['why']}")
        lines.append("")

    lines.append("Now choose the best move and reply with only the JSON object.")
    return STATIC_PROMPT_HEADER + "\n".join(lines)


def extract_decision(text: str):
    """Pull (move_index, confidence, board_analysis, strategic_plan, json_ok)
    from the model's response. Returns Nones on parse failure."""
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
                True,
            )
    return (None, None, None, None, False)


def build_history_entry(move, state_before, drawn_card: Optional[str] = None) -> dict:
    """Capture enough of a chosen move to render it in future RECENT MOVES."""
    mt = move.move_type.value
    entry = {"engine_move_type": mt}
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
    ap.add_argument("--max-illegal-moves", type=int, default=3,
                    help="Abort after this many illegal move-index picks in a row")
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
    consecutive_parse_failures = 0
    consecutive_illegal_moves = 0
    plateau_foundation = -1
    plateau_start_turn = 0

    turns_out = turns_path.open("w")
    overall_t0 = time.time()
    outcome = "max_turns"
    end_reason: Optional[str] = None

    for turn in range(args.max_turns):
        legal = visible_legal_moves(state)
        # Won check (no more cards outside foundations)
        if sum(len(f) for f in state.foundations) == 52:
            outcome = "won"; break
        if not legal:
            outcome = "stalled"; end_reason = "no legal moves"; break

        recycle = compute_recycle_available(state, len(seen_in_waste))
        prompt = render_prompt(state, legal, recent_moves, prior_decisions,
                               seen_in_waste, recycle)
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

        t_call_start = time.time()
        mx.reset_peak_memory()
        response = generate(model, tokenizer, prompt=wrapped,
                            max_tokens=args.max_tokens, verbose=False)
        t_call = time.time() - t_call_start
        peak = mx.get_peak_memory() / 1e9

        (resp_dir / f"turn_{turn:03d}.txt").write_text(response)
        mi, conf, ba, sp, json_ok = extract_decision(response)
        fc = sum(len(f) for f in state.foundations)
        fd = sum(1 for col in state.tableau for c in col if not c.face_up)

        if not json_ok or mi is None:
            consecutive_parse_failures += 1
            print(f"  [{turn:>3}] PARSE FAILURE  ({consecutive_parse_failures}/"
                  f"{args.max_parse_failures})  call={t_call:.1f}s  fc={fc} fd={fd}",
                  flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "json_ok": False, "call_seconds": round(t_call, 2),
                "peak_gb": round(peak, 2), "fc": fc, "fd": fd,
                "prompt_chars": len(prompt), "response_chars": len(response),
            }) + "\n")
            if consecutive_parse_failures >= args.max_parse_failures:
                outcome = "parse_failure"; break
            continue
        consecutive_parse_failures = 0

        # Validate move-index
        if not (0 <= mi < len(legal)):
            consecutive_illegal_moves += 1
            print(f"  [{turn:>3}] ILLEGAL move_index={mi} (legal=[0..{len(legal)-1}])  "
                  f"({consecutive_illegal_moves}/{args.max_illegal_moves})",
                  flush=True)
            turns_out.write(json.dumps({
                "turn": turn, "json_ok": True, "move_index": mi,
                "illegal": True, "call_seconds": round(t_call, 2),
                "fc": fc, "fd": fd, "confidence": conf,
            }) + "\n")
            if consecutive_illegal_moves >= args.max_illegal_moves:
                outcome = "illegal_move"; break
            continue
        consecutive_illegal_moves = 0

        chosen = legal[mi]
        move_text = describe_move(chosen, state)
        # Apply move
        if chosen.move_type.value == "stock_to_waste":
            drawn_card = card_short(state.stock[-1])
        else:
            drawn_card = None
        state_before = state
        state = apply_move(state, chosen)
        # Update seen-in-waste tracking on draw
        if chosen.move_type.value == "stock_to_waste" and drawn_card:
            if drawn_card not in seen_in_waste:
                seen_in_waste.append(drawn_card)
        # Recycle resets the seen list per harvester convention
        if chosen.move_type.value == "stock_to_waste" and len(state_before.stock) == 0:
            seen_in_waste = []
            if drawn_card:
                seen_in_waste.append(drawn_card)

        history_entry = build_history_entry(chosen, state_before, drawn_card=drawn_card)
        recent_moves.append(history_entry)
        # Auto-flip face-downs that are now top of column
        state, flipped = auto_flip(state)
        for f in flipped:
            recent_moves.append({
                "engine_move_type": "flip_card",
                "card": f["card"],
                "from_col": f["column"],
            })

        # Track plateau
        new_fc = sum(len(f) for f in state.foundations)
        if new_fc != plateau_foundation:
            plateau_foundation = new_fc
            plateau_start_turn = turn

        prior_decisions.append({"move_text": move_text, "why": sp or ""})
        # Keep prior_decisions short; renderer caps to 5
        prior_decisions = prior_decisions[-5:]

        new_fd = sum(1 for col in state.tableau for c in col if not c.face_up)
        mt_h = ENGINE_TO_HARVESTER_MOVETYPE.get(chosen.move_type.value, chosen.move_type.value)
        print(f"  [{turn:>3}] mv=[{mi}] {mt_h:<24} {move_text[:46]:<46}  "
              f"fc={new_fc:>2} fd={new_fd:>2} call={t_call:.1f}s conf={conf} flips={len(flipped)}",
              flush=True)

        turns_out.write(json.dumps({
            "turn": turn,
            "json_ok": True,
            "move_index": mi,
            "move_type": chosen.move_type.value,
            "move_text": move_text,
            "confidence": conf,
            "legal_count": len(legal),
            "fc": new_fc,
            "fd": new_fd,
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
