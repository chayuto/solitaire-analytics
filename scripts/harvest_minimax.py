"""Headless harvest of MiniMax responses on known benchmark decks, direct to the API.

Why this exists
---------------
`data/SUMMARY.md` reports MiniMax-M3 at a 15% success rate against 41-45% for the
other teacher models, which reads as "M3 is bad at Klondike". It is not what the
logs say. Across every M3 export in `data/raw/`:

    5715  errorMessage=The model did not respond within 240s.
     126  errorMessage=Could not reach TokenRouter. Check your network connection.
       8  errorMessage=TokenRouter returned an empty answer (finish_reason: error).
       6  errorMessage=TokenRouter returned no message content.
       3  errorMessage=TokenRouter rejected the API key. Check that it is valid.

97.5% of the failures are one client timeout, and the median error duration is
240997 ms -- the wall itself. A *successful* M3 call in the same corpus took
157331 ms and spent 7127 thinking tokens. The 240 s budget sits barely above the
model's typical latency on this prompt, so most calls were cut off mid-thought and
recorded as model failures. The remainder are a "TokenRouter" proxy failing in
front of the model.

So this harvester changes three things and nothing else, to keep the samples
comparable with the existing corpus:

1. It calls the MiniMax endpoint directly. No proxy in the path.
2. It streams, and budgets on a *stall* (no bytes for `--stall` seconds) plus a
   generous hard deadline, rather than a flat wall-clock cap. A model that is
   still emitting tokens is working, not hung.
3. It classifies every failure and retries the retryable ones with exponential
   backoff and jitter, instead of collapsing everything into one "error" outcome
   or, worse, into a silent heuristic fallback the way `strategies/llm.py` does.

The prompt is `hybrid-v1.6`, reproduced from the exports so new rows sit beside
the old ones. Output is the same envelope `scripts/ingest_exports.py` already
reads: `{exportedAt, appCommit, session, initialBoardSetup, count, interactions}`.

Usage
-----
    export MINIMAX_API_KEY=...            # never write the key into the repo
    .venv/bin/python scripts/harvest_minimax.py --decks winnable --games 4 --max-turns 60

Every interaction is appended to a JSONL sidecar as it happens, so a run that is
interrupted keeps everything it had already paid for.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from solitaire_analytics.engine import apply_move, generate_moves
from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.models.card import Suit
from solitaire_analytics.models.move import MoveType

REPO = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO / "data" / "benchmarks"
DEFAULT_OUT = REPO / "data" / "raw"

PROMPT_TEMPLATE_VERSION = "hybrid-v1.6"
APP_COMMIT = "harvest_minimax.py"
DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M3"

RANK_TOKEN = {1: "A", 10: "T", 11: "J", 12: "Q", 13: "K"}
SUIT_TOKEN = {"hearts": "H", "diamonds": "D", "clubs": "C", "spades": "S"}
SUIT_ORDER = ["hearts", "diamonds", "clubs", "spades"]


# --------------------------------------------------------------------------- #
# Board rendering: the hybrid-v1.6 prompt, reproduced from data/raw exports.
# --------------------------------------------------------------------------- #

def card_token(card: Optional[Card]) -> str:
    if card is None:
        return "--"
    if not card.face_up:
        return "??"
    return f"{RANK_TOKEN.get(card.rank, str(card.rank))}{SUIT_TOKEN[card.suit.value]}"


def face_up_token(card: Card) -> str:
    """Identity of a card regardless of whether the player may see it."""
    return f"{RANK_TOKEN.get(card.rank, str(card.rank))}{SUIT_TOKEN[card.suit.value]}"


RULES_BLOCK = """You are an expert Klondike Solitaire strategist acting as an advisor.

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
NOTATION: rank+suit (A 2-9 T J Q K; H D C S). ?? = face-down. In each column the top of the stack is the rightmost card.

THE GOAL: choose the single move that gives the best chance of eventually winning.

STRATEGY GUIDANCE (heuristics, not absolute rules):
- If any legal move is tagged "(reveals a hidden card)", prefer such a move. When several are tagged, prefer the one in the column with the most face-down (??) cards remaining.
- Play Aces and 2s to the foundations promptly; they are rarely useful in the tableau.
- Be cautious sending higher cards to the foundations too early — they are sometimes needed to receive tableau cards.
- Do not empty a column unless you have a King ready to occupy it.
- Prefer a move tagged "(reveals a hidden card)" over a tableau move that is not tagged and only shuffles cards between columns with no gain.
- Do not return a card to a tableau column it occupied in any move shown in RECENT MOVES.

RESPONSE FORMAT:
You will receive the current game as plain-text blocks (FOUNDATIONS, STOCK, TABLEAU,
RECENT MOVES, LEGAL MOVES, PROGRESS — some are optional). The
LEGAL MOVES block is a numbered list; each line begins with [index], the canonical
move identifier you must return.
Reason step by step, then respond with ONLY a single JSON object containing exactly
these three keys, in this order (no prose or markdown fences outside the object):
{
  "board_analysis": <string>,
  "strategic_plan": <string>,
  "final_decision": { "move_index": <number> }
}
- board_analysis: assess the current board, hidden cards, blocked columns, foundation
  progress, and the opportunities each legal move opens or closes.
- strategic_plan: explain your plan and why the chosen move is best, given that analysis.
- final_decision.move_index: the [index] of your chosen move from the LEGAL MOVES
  block, OR -1 to RESIGN. Resign only when no legal move can productively advance
  the game, drawing has been exhausted, and you would not bet on any of the
  available moves to recover. Resign is final and ends the session.
Produce the keys in the order above: analyse the board first, then plan, then decide."""


def describe_move(state: GameState, move: Move) -> tuple[str, str]:
    """A canonical identifier and a human line, matching the exports' phrasing."""
    mt = move.move_type
    if mt is MoveType.STOCK_TO_WASTE:
        return "draw_card", "Draw the next card from the stock onto the waste"
    if mt is MoveType.FLIP_TABLEAU_CARD:
        return "flip_card", f"Flip the face-down card on column {move.source_pile + 1}"
    if mt is MoveType.TABLEAU_TO_FOUNDATION:
        card = state.tableau[move.source_pile][-1]
        return "tableau_to_foundation", f"Move {face_up_token(card)} from column {move.source_pile + 1} to its foundation"
    if mt is MoveType.WASTE_TO_FOUNDATION:
        card = state.waste[-1]
        return "waste_to_foundation", f"Move {face_up_token(card)} from the waste to its foundation"
    if mt is MoveType.WASTE_TO_TABLEAU:
        card = state.waste[-1]
        dest = state.tableau[move.dest_pile]
        where = f"column {move.dest_pile + 1}" + ("" if dest else " (empty)")
        return "waste_to_tableau", f"Move {face_up_token(card)} from the waste to {where}"
    if mt is MoveType.TABLEAU_TO_TABLEAU:
        pile = state.tableau[move.source_pile]
        card = pile[len(pile) - move.num_cards]
        extra = f" plus {move.num_cards - 1} more" if move.num_cards > 1 else ""
        dest = state.tableau[move.dest_pile]
        where = f"column {move.dest_pile + 1}" + ("" if dest else " (empty)")
        return "tableau_to_tableau", f"Move {face_up_token(card)}{extra} from column {move.source_pile + 1} to {where}"
    return mt.name.lower(), str(move)


def offered_moves(state: GameState) -> list[Move]:
    """The choices the model is shown: legal, non-flip, and deduplicated.

    `generate_moves` treats the four foundation piles as interchangeable slots, so an
    Ace on an empty board comes back four times with different `dest_pile` values and
    an identical meaning. Showing all four is noise that makes the model's pick
    arbitrary and inflates the branching factor, so equivalent moves collapse to the
    first one; the Move object behind the index is a real, legal move either way.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Move] = []
    for m in generate_moves(state):
        if m.move_type is MoveType.FLIP_TABLEAU_CARD:
            continue
        key = describe_move(state, m)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def reveals_hidden(state: GameState, move: Move) -> bool:
    """True when applying this move exposes a face-down tableau card."""
    if move.move_type not in (MoveType.TABLEAU_TO_TABLEAU, MoveType.TABLEAU_TO_FOUNDATION):
        return False
    pile = state.tableau[move.source_pile]
    taken = move.num_cards if move.move_type is MoveType.TABLEAU_TO_TABLEAU else 1
    below = len(pile) - taken - 1
    return below >= 0 and not pile[below].face_up


def render_state(state: GameState, moves: list[Move], history: list[str], turn: int) -> str:
    out: list[str] = ["CURRENT GAME:"]

    # This engine's four foundation piles are generic slots, not one-per-suit: an Ace is
    # legal on any empty pile. The prompt still shows the suit-keyed view the corpus used,
    # derived from what each pile actually holds.
    by_suit = {}
    for pile in state.foundations:
        if pile:
            by_suit[pile[-1].suit.value] = pile[-1]
    tops = [f"{SUIT_TOKEN[s]}: {card_token(by_suit.get(s))}" for s in SUIT_ORDER]
    out.append("FOUNDATIONS:   " + "   ".join(tops))

    waste_top = card_token(state.waste[-1]) if state.waste else "--"
    out.append(f"STOCK: {len(state.stock)} cards   WASTE top: {waste_top}   "
               f"recycle stock: {'yes' if not state.stock and state.waste else 'no'}")
    out.append("")

    out.append("TABLEAU:")
    for i, pile in enumerate(state.tableau):
        body = " ".join(card_token(c) for c in pile) if pile else "<empty>"
        out.append(f"  col{i + 1}: {body}")
    out.append("")

    if history:
        out.append("RECENT MOVES (oldest -> newest; review before picking, do not undo your own work):")
        for n, line in enumerate(history[-10:], 1):
            out.append(f"  {n:2d}. {line}")
        out.append("")

    out.append("LEGAL MOVES (respond with the index of your chosen move):")
    for i, mv in enumerate(moves):
        ident, text = describe_move(state, mv)
        tag = "  (reveals a hidden card)" if reveals_hidden(state, mv) else ""
        out.append(f"  [{i}] {ident:24s} {text}{tag}")
    out.append("")

    found = sum(len(f) for f in state.foundations)
    hidden = sum(1 for pile in state.tableau for c in pile if not c.face_up)
    out.append(f"PROGRESS: foundation={found}/52, face-down remaining={hidden}, "
               f"completion={round(found / 52 * 100)}%, turn={turn}")
    out.append("")
    out.append("Now choose the best move and reply with only the JSON object.")
    return "\n".join(out)


def build_prompt(state: GameState, moves: list[Move], history: list[str], turn: int) -> str:
    return RULES_BLOCK + "\n\n" + render_state(state, moves, history, turn)


# --------------------------------------------------------------------------- #
# Deck loading
# --------------------------------------------------------------------------- #

def _card_from(obj: dict) -> Card:
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    raw = obj["rank"]
    rank = rank_map.get(str(raw), None)
    if rank is None:
        rank = int(raw)
    return Card(rank=rank, suit=Suit(obj["suit"]), face_up=bool(obj.get("face_up", True)))


def load_decks(name: str) -> list[dict]:
    path = BENCHMARKS / f"{name}_decks.json"
    if not path.exists():
        raise SystemExit(f"no such deck set: {path}")
    payload = json.loads(path.read_text())
    decks = payload["decks"] if isinstance(payload, dict) else payload
    out = []
    for d in decks:
        out.append({
            "seed": d.get("seed"),
            "draw_count": d.get("draw_count", 1),
            "tableau": [[_card_from(c) for c in col] for col in d["tableau"]],
            "stock": [_card_from(c) for c in d["stock"]],
            "known_winnable": (d.get("pyksolve", {}) or {}).get("draw1", {}).get("verdict", "").startswith("Solved")
                              or bool((d.get("outcome") or {}).get("won")),
            "source_file": d.get("source_file"),
        })
    return out


def state_from_deck(deck: dict) -> GameState:
    return GameState(
        tableau=[list(col) for col in deck["tableau"]],
        foundations=[[], [], [], []],
        stock=list(deck["stock"]),
        waste=[],
    )


# --------------------------------------------------------------------------- #
# The client: streaming, stall-aware, classified errors, retries.
# --------------------------------------------------------------------------- #

RETRYABLE = {"network", "http_5xx", "http_429", "stall", "empty_content", "unparseable", "bad_index"}


@dataclass
class CallResult:
    ok: bool
    kind: str                      # "ok" or an error class
    detail: str = ""
    raw_response: str = ""
    thinking: str = ""
    decision: Optional[dict] = None
    move_index: Optional[int] = None
    duration_ms: int = 0
    http_status: Optional[int] = None
    usage: dict = field(default_factory=dict)
    attempts: int = 1
    first_byte_ms: Optional[int] = None


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Optional[dict]:
    """Pull the decision object out of a reply that may carry reasoning and fences.

    M3 emits `<think>...</think>` inline in `content`; `strategies/llm.py` calls
    json.loads on that string directly, which throws and silently becomes a
    heuristic move. Strip the think block, drop fences, then take the outermost
    balanced object so trailing prose cannot break the parse.
    """
    if not text:
        return None
    body = THINK_RE.sub("", text)
    body = re.sub(r"<think>.*", "", body, flags=re.DOTALL)   # unclosed: truncated mid-thought
    body = FENCE_RE.sub("", body).strip()
    start = body.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(body)):
            ch = body[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = body.find("{", start + 1)
    return None


def read_move_index(decision: dict) -> Optional[int]:
    fd = decision.get("final_decision") or decision.get("finalDecision") or {}
    if isinstance(fd, dict):
        for k in ("move_index", "moveIndex", "index"):
            if isinstance(fd.get(k), (int, float)):
                return int(fd[k])
    for k in ("move_index", "moveIndex"):
        if isinstance(decision.get(k), (int, float)):
            return int(decision[k])
    return None


class MockClient:
    """Answers in M3's real wire shape without touching the network.

    The point is not to simulate play quality; it is that every layer between the
    board and the export gets exercised offline. So the canned reply carries the
    two things that actually break naive parsing in the corpus: an inline
    `<think>` block (what the API returns directly) and a markdown fence (411 of
    1028 logged replies).
    """

    def __init__(self, model="MiniMax-M3-mock", **kw):
        self.model = model
        self.temperature = kw.get("temperature", 0.3)
        self.stall_s = kw.get("stall_s", 0)
        self.deadline_s = kw.get("deadline_s", 0)
        self.max_attempts = kw.get("max_attempts", 1)
        self._n = 0

    def decide(self, prompt: str, n_moves: int) -> CallResult:
        self._n += 1
        idx = 0                                   # first offered move: legal by construction
        body = json.dumps({"board_analysis": "mock", "strategic_plan": "mock",
                           "final_decision": {"move_index": idx}})
        raw = f"<think>\nmock reasoning, turn {self._n}\n</think>\n\n```json\n{body}\n```"
        return CallResult(True, "ok", raw_response=raw, thinking="mock reasoning",
                          decision=extract_json(raw), move_index=idx,
                          duration_ms=1, http_status=200,
                          usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 40,
                                 "total_tokens": len(prompt) // 4 + 40,
                                 "completion_tokens_details": {"reasoning_tokens": 30}})


class MiniMaxClient:
    def __init__(self, api_key: str, base_url: str, model: str, *,
                 temperature: float = 0.3, stall_s: float = 90.0,
                 deadline_s: float = 900.0, max_attempts: int = 4,
                 verbose: bool = False):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.stall_s = stall_s
        self.deadline_s = deadline_s
        self.max_attempts = max_attempts
        self.verbose = verbose

    def _stream_once(self, prompt: str) -> CallResult:
        """One streamed attempt. Budgets on a stall, not on total wall clock."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "Accept": "text/event-stream"},
        )
        started = time.monotonic()
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict = {}
        first_byte: Optional[float] = None
        last_byte = started

        try:
            with urllib.request.urlopen(req, timeout=self.stall_s) as resp:
                status = resp.status
                for raw in resp:
                    now = time.monotonic()
                    if first_byte is None:
                        first_byte = now
                    # A stream that keeps producing is working; one that goes quiet is not.
                    if now - last_byte > self.stall_s:
                        return CallResult(False, "stall",
                                          f"no bytes for {self.stall_s:.0f}s",
                                          duration_ms=int((now - started) * 1000),
                                          http_status=status)
                    if now - started > self.deadline_s:
                        return CallResult(False, "deadline",
                                          f"exceeded {self.deadline_s:.0f}s hard deadline",
                                          "".join(content),
                                          duration_ms=int((now - started) * 1000),
                                          http_status=status)
                    last_byte = now
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning"):
                            reasoning.append(delta["reasoning"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            kind = "http_429" if e.code == 429 else ("http_5xx" if e.code >= 500 else "http_4xx")
            return CallResult(False, kind, f"HTTP {e.code}: {detail}",
                              duration_ms=int((time.monotonic() - started) * 1000),
                              http_status=e.code)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return CallResult(False, "network", f"{type(e).__name__}: {e}",
                              duration_ms=int((time.monotonic() - started) * 1000))

        elapsed = int((time.monotonic() - started) * 1000)
        text = "".join(content)
        think = "".join(reasoning)
        if not text.strip():
            return CallResult(False, "empty_content", "stream closed with no content",
                              duration_ms=elapsed, http_status=200, usage=usage)
        decision = extract_json(text)
        if decision is None:
            return CallResult(False, "unparseable", "no JSON object in reply",
                              raw_response=text, thinking=think,
                              duration_ms=elapsed, http_status=200, usage=usage)
        idx = read_move_index(decision)
        if idx is None:
            return CallResult(False, "unparseable", "no final_decision.move_index",
                              raw_response=text, thinking=think, decision=decision,
                              duration_ms=elapsed, http_status=200, usage=usage)
        return CallResult(True, "ok", raw_response=text, thinking=think, decision=decision,
                          move_index=idx, duration_ms=elapsed, http_status=200, usage=usage,
                          first_byte_ms=int((first_byte - started) * 1000) if first_byte else None)

    def decide(self, prompt: str, n_moves: int) -> CallResult:
        """Retry the retryable, with exponential backoff and jitter."""
        last = CallResult(False, "network", "no attempt made")
        for attempt in range(1, self.max_attempts + 1):
            res = self._stream_once(prompt)
            res.attempts = attempt
            if res.ok and res.move_index is not None:
                if res.move_index == -1 or 0 <= res.move_index < n_moves:
                    return res
                res = CallResult(False, "bad_index",
                                 f"move_index {res.move_index} outside 0..{n_moves - 1}",
                                 res.raw_response, res.thinking, res.decision,
                                 res.move_index, res.duration_ms, res.http_status,
                                 res.usage, attempt, res.first_byte_ms)
            last = res
            if res.kind not in RETRYABLE or attempt == self.max_attempts:
                return res
            wait = min(30.0, 2.0 ** attempt) * (0.6 + 0.8 * random.random())
            if self.verbose:
                print(f"    retry {attempt}/{self.max_attempts - 1} after {res.kind}: "
                      f"{res.detail[:70]} (sleep {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)
        return last


# --------------------------------------------------------------------------- #
# Playing one game
# --------------------------------------------------------------------------- #

def auto_flips(state: GameState) -> tuple[GameState, list[str]]:
    """Apply forced flips the way the harvester does, so they are not model turns."""
    applied: list[str] = []
    while True:
        flips = [m for m in generate_moves(state) if m.move_type is MoveType.FLIP_TABLEAU_CARD]
        if not flips:
            return state, applied
        nxt = apply_move(state, flips[0])
        if nxt is None:
            return state, applied
        state = nxt
        applied.append("flip_card")


def play_game(deck: dict, client: MiniMaxClient, *, max_turns: int, session_id: str,
              sink, lock: threading.Lock, verbose: bool = False) -> dict:
    state = state_from_deck(deck)
    state, _ = auto_flips(state)
    history: list[str] = []
    interactions: list[dict] = []
    stats = {"calls": 0, "ok": 0, "errors": {}, "latency_ms": [], "retries": 0,
             "illegal": 0, "resign": False}
    seed = deck.get("seed")
    initial = state.to_json() if hasattr(state, "to_json") else None

    for turn in range(max_turns):
        moves = offered_moves(state)
        if not moves:
            break
        prompt = build_prompt(state, moves, history, turn)
        res = client.decide(prompt, len(moves))
        stats["calls"] += 1
        stats["retries"] += res.attempts - 1
        stats["latency_ms"].append(res.duration_ms)

        rec = {
            "id": str(uuid.uuid4()),
            "requestId": str(uuid.uuid4()),
            "sessionId": session_id,
            "attempt": res.attempts,
            "timestamp": int(time.time() * 1000),
            "provider": "minimax-direct",
            "model": client.model,
            "appCommit": APP_COMMIT,
            "event": None,          # None marks a real model turn; forced flips are not logged as turns
            "seed": seed,
            "turnIndex": turn,
            "config": {"provider": "minimax-direct", "model": client.model,
                       "preset": "harvest-direct", "seeHiddenCards": False,
                       "includeMoveHistory": True, "moveHistoryLimit": 10,
                       "includeGameMetrics": True, "includeStrategyGuidance": True,
                       "stallSeconds": client.stall_s, "deadlineSeconds": client.deadline_s,
                       "maxAttempts": client.max_attempts},
            "outcome": "success" if res.ok else "error",
            "errorClass": None if res.ok else res.kind,
            "errorMessage": None if res.ok else res.detail,
            "durationMs": res.duration_ms,
            "firstByteMs": res.first_byte_ms,
            "prompt": prompt,
            "promptTemplateVersion": PROMPT_TEMPLATE_VERSION,
            "promptLayoutVersion": PROMPT_TEMPLATE_VERSION,
            "inferenceParams": {"temperature": client.temperature, "stream": True},
            "rawResponse": res.raw_response,
            "thinkingText": res.thinking,
            "decision": json.dumps(res.decision) if res.decision else None,
            "httpStatus": res.http_status,
            "promptTokens": (res.usage or {}).get("prompt_tokens"),
            "thoughtTokens": ((res.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "outputTokens": (res.usage or {}).get("completion_tokens"),
            "totalTokens": (res.usage or {}).get("total_tokens"),
            "movesApplied": [],
        }

        if not res.ok:
            stats["errors"][res.kind] = stats["errors"].get(res.kind, 0) + 1
            interactions.append(rec)
            with lock:
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
            if verbose:
                print(f"  [{session_id[:6]}] turn {turn}: {res.kind} — {res.detail[:60]}", file=sys.stderr)
            break   # a game cannot continue past a move it could not obtain

        if res.move_index == -1:
            stats["resign"] = True
            rec["movesApplied"] = ["resign"]
            interactions.append(rec)
            with lock:
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
            break

        chosen = moves[res.move_index]
        ident, text = describe_move(state, chosen)
        nxt = apply_move(state, chosen)
        if nxt is None:
            # The engine is the authority; a move it refuses is an illegal choice.
            stats["illegal"] += 1
            rec["outcome"] = "error"
            rec["errorClass"] = "illegal_move"
            rec["errorMessage"] = f"engine refused {ident}"
            interactions.append(rec)
            with lock:
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
            break

        state = nxt
        state, flips = auto_flips(state)
        stats["ok"] += 1
        rec["movesApplied"] = [ident] + flips
        history.append(text)
        interactions.append(rec)
        with lock:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
        if verbose:
            found = sum(len(f) for f in state.foundations)
            print(f"  [{session_id[:6]}] turn {turn}: {ident} "
                  f"({res.duration_ms/1000:.0f}s, {res.attempts} att) foundation={found}/52",
                  file=sys.stderr)

        if sum(len(f) for f in state.foundations) == 52:
            break

    found = sum(len(f) for f in state.foundations)
    return {
        "session": {
            "sessionId": session_id,
            "seed": seed,
            "model": client.model,
            "outcome": "won" if found == 52 else ("resigned" if stats["resign"] else "incomplete"),
            "finalProgress": round(found / 52 * 100),
            "moveCount": stats["ok"],
            "knownWinnable": deck.get("known_winnable"),
        },
        "initialBoardSetup": initial,
        "interactions": interactions,
        "stats": stats,
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decks", default="winnable",
                    help="benchmark set: winnable, heldout, winrate, generalization")
    ap.add_argument("--games", type=int, default=2, help="how many decks to play")
    ap.add_argument("--max-turns", type=int, default=40, help="model turns per game")
    ap.add_argument("--concurrency", type=int, default=2, help="games in flight at once")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--stall", type=float, default=90.0,
                    help="seconds of silence before a stream counts as stalled")
    ap.add_argument("--deadline", type=float, default=900.0, help="hard per-call ceiling")
    ap.add_argument("--attempts", type=int, default=4, help="max attempts per move")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--offset", type=int, default=0, help="skip the first N decks")
    ap.add_argument("--key-file", default=os.environ.get("MINIMAX_KEY_FILE"),
                    help="path to a file holding the API key; preferred over an env var, "
                         "because it keeps the secret off the command line and out of the "
                         "environment of every child process")
    ap.add_argument("--mock", action="store_true",
                    help="exercise the whole loop offline with a canned reply, no requests")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.mock:
        client = MockClient(model=args.model + "-mock", temperature=args.temperature)
        return run(args, client)

    key = None
    if args.key_file:
        kf = Path(args.key_file).expanduser()
        if not kf.exists():
            print(f"key file not found: {kf}", file=sys.stderr)
            return 2
        key = kf.read_text().strip()
    key = key or os.environ.get("MINIMAX_API_KEY")
    if not key:
        print("No API key. Pass --key-file, or set MINIMAX_API_KEY. "
              "Never write the key into the repo.", file=sys.stderr)
        return 2

    client = MiniMaxClient(key, args.base_url, args.model, temperature=args.temperature,
                           stall_s=args.stall, deadline_s=args.deadline,
                           max_attempts=args.attempts, verbose=not args.quiet)
    return run(args, client)


def run(args, client) -> int:
    decks = load_decks(args.decks)[args.offset:args.offset + args.games]
    if not decks:
        print("no decks selected", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl = out_dir / f"harvest-minimax-{stamp}.jsonl"
    lock = threading.Lock()

    print(f"harvest: {len(decks)} deck(s) from {args.decks}, up to {args.max_turns} turns each, "
          f"{args.concurrency} in flight\n  model {args.model} at {args.base_url}\n"
          f"  stall {args.stall:.0f}s, deadline {args.deadline:.0f}s, {args.attempts} attempts\n"
          f"  streaming -> {jsonl}", file=sys.stderr)

    results = []
    t0 = time.time()
    with jsonl.open("w") as sink:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {}
            for d in decks:
                sid = str(uuid.uuid4())
                futs[pool.submit(play_game, d, client, max_turns=args.max_turns,
                                 session_id=sid, sink=sink, lock=lock,
                                 verbose=not args.quiet)] = d
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:                      # noqa: BLE001 - report, never lose the run
                    print(f"game crashed: {type(e).__name__}: {e}", file=sys.stderr)

    # One export envelope per session, the shape ingest_exports.py already reads.
    for r in results:
        sid = r["session"]["sessionId"]
        path = out_dir / f"solitaire-ai-log-{sid[-6:]}-{int(time.time()*1000)}.json"
        path.write_text(json.dumps({
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "appCommit": "harvest_minimax.py",
            "appBuildTime": stamp,
            "session": r["session"],
            "initialBoardSetup": r["initialBoardSetup"],
            "count": len(r["interactions"]),
            "interactions": r["interactions"],
        }, indent=1))

    summarise(results, time.time() - t0, jsonl)
    return 0


def summarise(results: list[dict], wall_s: float, jsonl: Path) -> None:
    calls = sum(r["stats"]["calls"] for r in results)
    ok = sum(r["stats"]["ok"] for r in results)
    retries = sum(r["stats"]["retries"] for r in results)
    illegal = sum(r["stats"]["illegal"] for r in results)
    errs: dict[str, int] = {}
    lat: list[int] = []
    for r in results:
        for k, v in r["stats"]["errors"].items():
            errs[k] = errs.get(k, 0) + v
        lat += [x for x in r["stats"]["latency_ms"] if x > 0]
    lat.sort()

    def pct(p: float) -> str:
        return f"{lat[min(len(lat) - 1, int(len(lat) * p))] / 1000:.0f}s" if lat else "-"

    print("\n" + "=" * 68, file=sys.stderr)
    print(f"games {len(results)} | model calls {calls} | usable decisions {ok} "
          f"({ok / calls * 100:.0f}%)" if calls else "no calls", file=sys.stderr)
    print(f"retries spent {retries} | illegal move choices {illegal} | wall {wall_s / 60:.1f} min",
          file=sys.stderr)
    if lat:
        print(f"latency p50 {pct(0.5)}  p90 {pct(0.9)}  max {lat[-1] / 1000:.0f}s", file=sys.stderr)
        for wall in (60, 120, 240, 480):
            over = sum(1 for x in lat if x > wall * 1000)
            print(f"  would time out at {wall:>3}s: {over}/{len(lat)} ({over / len(lat) * 100:.0f}%)",
                  file=sys.stderr)
    if errs:
        print("errors by class:", file=sys.stderr)
        for k, v in sorted(errs.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}", file=sys.stderr)
    for r in results:
        s = r["session"]
        print(f"  seed {s['seed']}: {s['outcome']}, progress {s['finalProgress']}%, "
              f"{s['moveCount']} moves (known winnable: {s['knownWinnable']})", file=sys.stderr)
    print(f"rows -> {jsonl}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
