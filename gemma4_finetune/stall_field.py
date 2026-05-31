#!/usr/bin/env python3
"""Stall / repetition STATE for the play prompt (harvester-recommendation A/B).

Pure helpers, no mlx / engine dependency, so they unit-test in isolation. These
render TEMPORAL state the current prompt omits: how long the game has made no
progress, and how often the exact board position has recurred. They add no rule
or heuristic -- just facts a human player perceives. See
docs/reports/20260531_harvester_recommendation_doomloop_temporal_state.md.

Used by play_deck_with_student.py under --stall-field to A/B the recommendation
on the reproducible doom-loop seed (3263196305) with no new data.
"""
from __future__ import annotations


def board_signature(state) -> str:
    """A hashable digest of the full visible+hidden position.

    Captures tableau (rank/suit/face-up per card), foundations, and stock+waste
    sizes. Two turns with the same signature are the same position, so a repeated
    signature is a literal loop. Cheap: a string join, no model state.
    """
    cols = []
    for col in state.tableau:
        cells = []
        for c in col:
            tag = f"{c.rank}{c.suit.value}" if c.face_up else "?"
            cells.append(tag)
        cols.append(",".join(cells))
    found = ",".join(
        (f"{f[-1].rank}{f[-1].suit.value}" if f else "-") for f in state.foundations
    )
    return f"T[{'|'.join(cols)}]F[{found}]S{len(state.stock)}W{len(state.waste)}"


def stall_lines(no_progress_moves: int, position_seen_before: int) -> list[str]:
    """Render the stall STATE block (zero, one, or two lines).

    no_progress_moves: turns since foundation count OR face-down count last changed.
    position_seen_before: how many earlier turns this exact position already occurred.

    Returns [] when there is nothing notable (fresh progress, novel position) so
    the prompt is byte-identical to baseline early-game; the signal only appears
    once a stall actually exists, matching how a player only notices a loop after
    it has gone on.
    """
    out: list[str] = []
    if no_progress_moves > 0:
        out.append(
            f"STALL: no foundation play and no new card revealed in the last "
            f"{no_progress_moves} moves."
        )
    if position_seen_before > 0:
        out.append(
            f"REPEAT: this exact board position has already occurred "
            f"{position_seen_before} time(s) earlier this game."
        )
    return out
