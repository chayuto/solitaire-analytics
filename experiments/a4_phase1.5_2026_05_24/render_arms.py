"""Renderers for Phase 1.5: C0 (production passthrough) and A4 (notation hoisted).

C0: the prompt exactly as it appears in production (prompt_template_hash
    0462323c...). Pass-through.

A4: remove the `NOTATION:` line from the CURRENT GAME data section and inject
    it as the last item of the KLONDIKE SOLITAIRE RULES block, framed as a
    parsing rule rather than a header on the data.

The audit-derived rationale: notation rules are reference grammar the model
uses while parsing every turn's data. Keeping them adjacent to the data forces
re-parsing each turn; putting them in the standing rules treats them as a
fixed contract.

Self-test at bottom verifies that A4 removes exactly one NOTATION line, adds
exactly one notation rule to the preamble, leaves all other content intact,
and produces output within 5% of C0 size (H4 sanity).
"""

from __future__ import annotations

import re
from typing import Callable

NOTATION_LINE_PATTERN = re.compile(
    r"\nNOTATION: rank\+suit \(A 2-9 T J Q K; H D C S\)\. \?\? = face-down\."
    r" In each column the top of the stack is the rightmost card\.\n"
)

NOTATION_AS_RULE = (
    "- Notation in the CURRENT GAME data: rank+suit (A 2-9 T J Q K; H D C S);"
    " `??` marks a face-down card; within a column the rightmost card is on top.\n"
)

# Anchor: the last existing bullet of the RULES block, immediately before the
# blank line that separates RULES from "THE GOAL:".
RULES_LAST_BULLET = (
    "- The game is WON when all 52 cards reach the foundations.\n"
)


def render_C0(prompt: str) -> str:
    """Passthrough — return the production prompt unchanged."""
    return prompt


def render_A4(prompt: str) -> str:
    """Hoist the NOTATION line from CURRENT GAME into the RULES preamble."""
    # 1. Remove the inline notation line. Must match exactly once.
    new_prompt, n_removed = NOTATION_LINE_PATTERN.subn("\n", prompt, count=1)
    if n_removed != 1:
        raise ValueError(
            f"A4 renderer expected exactly 1 NOTATION line, found {n_removed}. "
            "Production template may have changed; re-audit before running."
        )
    # 2. Inject the notation rule as the new last bullet of RULES.
    if RULES_LAST_BULLET not in new_prompt:
        raise ValueError(
            "A4 renderer could not find the RULES anchor bullet "
            "('The game is WON when all 52 cards reach the foundations.'). "
            "Production template may have changed."
        )
    new_prompt = new_prompt.replace(
        RULES_LAST_BULLET,
        RULES_LAST_BULLET + NOTATION_AS_RULE,
        1,
    )
    return new_prompt


ARMS: dict[str, Callable[[str], str]] = {
    "C0": render_C0,
    "A4": render_A4,
}


def render(arm: str, prompt: str) -> str:
    return ARMS[arm](prompt)


# === Self-test ===
def _self_test() -> None:
    import json
    from pathlib import Path

    bench = json.loads(
        Path(__file__).parent.joinpath("bench.json").read_text()
    )
    sample = bench["states"][0]["full_prompt"]
    c0 = render_C0(sample)
    a4 = render_A4(sample)

    print("self-test")
    print(f"  C0 chars: {len(c0)}")
    print(f"  A4 chars: {len(a4)}")
    print(f"  delta:    {len(a4) - len(c0):+d}  ({100*(len(a4)-len(c0))/len(c0):+.1f}%)")
    print(f"  H4 (size unchanged within 5%): "
          f"{'PASS' if abs(len(a4)-len(c0))/len(c0) < 0.05 else 'FAIL'}")

    # Structural invariants
    assert "NOTATION:" not in a4.split("CURRENT GAME:")[1], \
        "A4 still has NOTATION line in CURRENT GAME section"
    assert "rank+suit (A 2-9 T J Q K" in a4.split("CURRENT GAME:")[0], \
        "A4 missing notation rule in RULES preamble"
    assert "NOTATION:" in c0, "C0 should still have inline NOTATION"
    assert c0 == sample, "C0 should be pass-through"

    # Check on every bench state — catches templating drift
    n_pass = 0
    n_fail = 0
    for s in bench["states"]:
        try:
            render_A4(s["full_prompt"])
            n_pass += 1
        except ValueError as e:
            n_fail += 1
            print(f"  FAIL on {s['state_id']}: {e}")
    print(f"  rendered all {n_pass}/{n_pass+n_fail} bench states successfully")


if __name__ == "__main__":
    _self_test()
