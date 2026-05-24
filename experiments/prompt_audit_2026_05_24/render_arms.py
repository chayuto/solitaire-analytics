"""Five prompt-variant renderers for the audit experiment.

Each render_<arm>(full_prompt, current_game) takes:
  full_prompt: the original prompt string (rules + schema + JSON state)
  current_game: parsed JSON dict from the CURRENT GAME (JSON) block
Returns: a new prompt string, ready to send to the model.

Arms:
  C0 — control (passthrough)
  A1 — drop reasoningTrail from JSON
  A2 — drop board_analysis + strategic_plan from response schema
  A3 — A1 + A2
  A4 — hoist `notation` from JSON into rules preamble

Invariants:
  - Game state (foundations, tableau, legalMoves, stock, recentMoves) is identical
    across all arms — we only change what's described/included, not the situation.
  - C0 must produce a byte-equivalent prompt to the input (modulo whitespace).
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

CURRENT_GAME_RE = re.compile(r"(CURRENT GAME \(JSON\):\s*)(\{.*\})", re.DOTALL)

# The exact RESPONSE FORMAT section in the production prompt, so we can swap it.
RESPONSE_FORMAT_RE = re.compile(
    r"RESPONSE FORMAT:\s*\n.*?Produce the keys in the order above:.*?then decide\.",
    re.DOTALL,
)

# A2 replacement: tiny schema, no prose CoT fields.
A2_RESPONSE_FORMAT = """RESPONSE FORMAT:
You will receive the current game as JSON, including a numbered array "legalMoves".
Think carefully, then respond with ONLY a single JSON object containing exactly
this one key (no prose or markdown fences outside the object):
{
  "final_decision": { "move_index": <number>, "confidence": <number>, "alternative_move_index": <number> }
}
- final_decision.move_index: the "index" of your chosen move from the legalMoves array.
- final_decision.confidence: a calibrated probability (0 to 1) that this move is
  objectively the best one available. Use the full range honestly:
    1.0-0.9  forced, or clearly dominant.
    0.9-0.7  strong — one plausible alternative.
    0.7-0.5  a real toss-up between two or three reasonable moves.
    0.5-0.3  a guess — the board is unclear.
    below 0.3  little better than random.
- final_decision.alternative_move_index: optional; the index of your second-choice move."""


def _replace_current_game(full_prompt: str, new_cg: dict) -> str:
    """Replace the CURRENT GAME JSON block with new_cg, preserve everything else."""
    serialized = json.dumps(new_cg)
    return CURRENT_GAME_RE.sub(lambda m: m.group(1) + serialized, full_prompt, count=1)


def render_C0(full_prompt: str, current_game: dict) -> str:
    """Control. Passthrough."""
    return full_prompt


def render_A1(full_prompt: str, current_game: dict) -> str:
    """Drop reasoningTrail from the JSON payload."""
    cg = deepcopy(current_game)
    cg.pop("reasoningTrail", None)
    return _replace_current_game(full_prompt, cg)


def render_A2(full_prompt: str, current_game: dict) -> str:
    """Drop board_analysis + strategic_plan from response schema."""
    if not RESPONSE_FORMAT_RE.search(full_prompt):
        raise RuntimeError("RESPONSE FORMAT section not found — prompt template changed?")
    return RESPONSE_FORMAT_RE.sub(A2_RESPONSE_FORMAT, full_prompt, count=1)


def render_A3(full_prompt: str, current_game: dict) -> str:
    """A1 + A2 combined."""
    step1 = render_A1(full_prompt, current_game)
    return render_A2(step1, current_game)


NOTATION_PREAMBLE_INSERT = (
    "NOTATION (read first): Cards are written as rank then suit "
    "(A 2-9 T J Q K; H D C S). Tableau columns are numbered 1 to 7 by their "
    "\"column\" field; faceUp arrays are bottom-to-top. Always refer to "
    "columns by that 1-based number in your reasoning.\n\n"
)


def render_A4(full_prompt: str, current_game: dict) -> str:
    """Hoist `notation` out of the JSON payload into a rules preamble at the top.

    The notation field is moved from the data payload to immediately AFTER the
    'You are an expert...' opener and BEFORE 'KLONDIKE SOLITAIRE RULES'.
    """
    cg = deepcopy(current_game)
    cg.pop("notation", None)
    intermediate = _replace_current_game(full_prompt, cg)

    rules_marker = "KLONDIKE SOLITAIRE RULES"
    if rules_marker not in intermediate:
        raise RuntimeError(f"Could not find '{rules_marker}' marker for A4 hoist")
    return intermediate.replace(
        rules_marker,
        NOTATION_PREAMBLE_INSERT + rules_marker,
        1,
    )


ARMS = {
    "C0": render_C0,
    "A1": render_A1,
    "A2": render_A2,
    "A3": render_A3,
    "A4": render_A4,
}


def render(arm: str, full_prompt: str, current_game: dict) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return ARMS[arm](full_prompt, current_game)


if __name__ == "__main__":
    # Self-test: load bench, render each arm for state 0, check invariants.
    import sys
    from pathlib import Path

    bench_path = Path(__file__).parent / "bench.json"
    bench = json.loads(bench_path.read_text())
    s0 = bench["states"][0]
    fp = s0["full_prompt"]
    cg = s0["current_game"]

    for arm in ARMS:
        out = render(arm, fp, cg)
        delta = len(out) - len(fp)
        # Check the legalMoves array is intact in every arm
        assert '"legalMoves"' in out, f"{arm}: legalMoves missing!"
        # C0 invariant
        if arm == "C0":
            assert out == fp, "C0 must be byte-identical to input"
        # A1 invariant
        if arm in ("A1", "A3"):
            assert '"reasoningTrail"' not in out, f"{arm}: reasoningTrail must be gone"
        # A2 invariant
        if arm in ("A2", "A3"):
            assert "board_analysis" not in out, f"{arm}: board_analysis must be gone"
            assert "strategic_plan" not in out, f"{arm}: strategic_plan must be gone"
        # A4 invariant: notation gone from JSON, present at top
        if arm == "A4":
            # The original 'notation' field text shouldn't appear inside the JSON anymore,
            # but the preamble should mention it.
            json_block_start = out.find("CURRENT GAME (JSON):")
            assert "NOTATION (read first):" in out[:json_block_start], "A4 preamble missing"

        print(f"{arm}: {len(out):>6} chars  (Δ {delta:+6d} vs C0)")

    print("\nself-test OK")
