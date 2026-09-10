"""Tests for the direct MiniMax harvester.

The parser is the part worth pinning. `strategies/llm.py` calls `json.loads` on the
reply body, which fails on 415 of the 1028 real M3 replies in `data/raw` -- 411 of
them for a markdown fence alone -- and every one of those failures is swallowed into
a heuristic fallback and recorded as if the model had chosen it. These tests cover
the shapes that actually occur in the corpus, plus the ones the API returns directly.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from harvest_minimax import (  # noqa: E402
    RETRYABLE,
    auto_flips,
    build_prompt,
    describe_move,
    extract_json,
    load_decks,
    offered_moves,
    read_move_index,
    state_from_deck,
)
from solitaire_analytics.engine import apply_move  # noqa: E402

DECISION = {"board_analysis": "a", "strategic_plan": "b", "final_decision": {"move_index": 3}}
BODY = json.dumps(DECISION)


@pytest.mark.unit
@pytest.mark.parametrize("reply,label", [
    (BODY, "bare object"),
    (f"```json\n{BODY}\n```", "markdown fence (411 of 1028 real replies)"),
    (f"```\n{BODY}\n```", "unlabelled fence"),
    (f"<think>\nreasoning\n</think>\n\n{BODY}", "inline think block, as the API returns"),
    (f"<think>\nreasoning\n</think>\n\n```json\n{BODY}\n```", "think block and fence together"),
    ("Looking at this position, I need to " + BODY, "prose preamble (4 real replies)"),
    (f"{BODY}\n\nHope that helps.", "trailing prose"),
    (f'<think>a "quoted brace {{" inside reasoning</think>{BODY}', "braces inside the think block"),
])
def test_extract_json_handles_every_real_wrapper(reply, label):
    got = extract_json(reply)
    assert got is not None, f"failed to parse: {label}"
    assert read_move_index(got) == 3, label


@pytest.mark.unit
def test_extract_json_survives_a_truncated_think_block():
    """A reply cut off mid-thought has an unclosed <think> and no decision.

    This is exactly what a client timeout produces, and it must read as "no decision"
    rather than throw, so the caller can classify and retry it.
    """
    assert extract_json("<think>\nI was still thinking when the socket closed") is None


@pytest.mark.unit
def test_extract_json_rejects_junk():
    assert extract_json("") is None
    assert extract_json("no object here at all") is None
    assert extract_json("{ not valid json ") is None


@pytest.mark.unit
def test_read_move_index_accepts_the_shapes_the_corpus_uses():
    assert read_move_index({"final_decision": {"move_index": 0}}) == 0
    assert read_move_index({"finalDecision": {"moveIndex": 7}}) == 7   # camelCase, as `decision` is stored
    assert read_move_index({"move_index": 2}) == 2                     # flattened
    assert read_move_index({"final_decision": {"move_index": -1}}) == -1  # resign
    assert read_move_index({"final_decision": {}}) is None
    assert read_move_index({"board_analysis": "no decision"}) is None


@pytest.mark.unit
def test_retryable_covers_transport_and_readback_but_not_auth():
    """A 4xx is the caller's fault and retrying it just burns quota."""
    for kind in ("network", "http_5xx", "http_429", "stall", "empty_content", "unparseable"):
        assert kind in RETRYABLE
    assert "http_4xx" not in RETRYABLE
    assert "deadline" not in RETRYABLE


@pytest.mark.unit
def test_offered_moves_dedupes_the_interchangeable_foundations():
    """`generate_moves` offers an Ace to all four empty foundation slots.

    They are genuinely distinct Moves with the same meaning. Showing four identical
    lines makes the model's choice arbitrary and inflates the branching factor, so
    the prompt collapses them -- while every surviving index is still a legal move.
    """
    deck = load_decks("winnable")[0]
    state, _ = auto_flips(state_from_deck(deck))
    moves = offered_moves(state)
    rendered = [describe_move(state, m) for m in moves]
    assert len(rendered) == len(set(rendered)), "two offered moves read identically"
    for m in moves:
        assert apply_move(state, m) is not None, "an offered move was not legal"


@pytest.mark.unit
def test_prompt_carries_the_blocks_the_template_promises():
    deck = load_decks("winnable")[0]
    state, _ = auto_flips(state_from_deck(deck))
    moves = offered_moves(state)
    prompt = build_prompt(state, moves, ["move 8C col 7 -> col 1"], 4)
    for block in ("FOUNDATIONS:", "TABLEAU:", "LEGAL MOVES", "PROGRESS:", "RECENT MOVES"):
        assert block in prompt, f"missing {block}"
    assert "final_decision" in prompt and "move_index" in prompt
    # Every legal move is offered with the index the model is asked to return.
    for i in range(len(moves)):
        assert f"[{i}]" in prompt


@pytest.mark.unit
def test_known_decks_are_all_solver_confirmed_winnable():
    """The assessment leans on these being winnable; if that stops holding, say so."""
    decks = load_decks("winnable")
    assert decks, "no winnable decks on disk"
    assert all(d["known_winnable"] for d in decks)
    for d in decks:
        assert d["draw_count"] == 1, "the corpus is draw-1 only"
        assert len(d["tableau"]) == 7
        assert sum(len(c) for c in d["tableau"]) + len(d["stock"]) == 52
