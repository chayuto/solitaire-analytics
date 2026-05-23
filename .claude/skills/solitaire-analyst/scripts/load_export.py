#!/usr/bin/env python3
"""Load a solitaire-ai-log export and print a one-screen briefing.

The user typically shares a single export path. This script's job is to load
it, surface the fields the analyst cares about, and print a concise briefing
so the analyst can give a verdict in one turn — no second tool round-trip
just to see the basics.

Run:
    python scripts/load_export.py <path-to-export.json>

Import:
    from load_export import load_export, latest_board, doom_loop_signature
    doc = load_export(path)
    board = latest_board(doc)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class Export:
    """A parsed solitaire-ai-log export.

    Attributes:
        path: Where the export was loaded from.
        session_id, seed, model: From the file's `session` block.
        outcome, final_progress, move_count: Session summary fields.
        app_commit, app_build_time, exported_at: Build provenance.
        interactions: All interactions, sorted by (turnIndex, timestamp).
        successes: Just the `outcome == "success"` interactions, same order.
    """
    path: Path
    session_id: str
    seed: int | None
    model: str
    outcome: str
    final_progress: int | None
    move_count: int | None
    app_commit: str | None
    app_build_time: str | None
    exported_at: str | None
    interactions: list[dict] = field(default_factory=list)

    @property
    def successes(self) -> list[dict]:
        return [r for r in self.interactions if r.get("outcome") == "success"]

    @property
    def short_session(self) -> str:
        return f"…{self.session_id[-12:]}" if self.session_id else "(unknown)"


def load_export(path: str | Path) -> Export:
    """Load and normalise an export file."""
    p = Path(path)
    with p.open() as fh:
        doc = json.load(fh)
    sess = doc.get("session") or {}
    raw = doc.get("interactions") or []
    interactions = sorted(
        raw,
        key=lambda r: (
            r.get("turnIndex") if isinstance(r.get("turnIndex"), int) else 10**9,
            r.get("timestamp") or 0,
        ),
    )
    return Export(
        path=p,
        session_id=sess.get("sessionId") or "",
        seed=sess.get("seed"),
        model=sess.get("model") or "",
        outcome=sess.get("outcome") or "",
        final_progress=sess.get("finalProgress"),
        move_count=sess.get("moveCount"),
        app_commit=doc.get("appCommit"),
        app_build_time=doc.get("appBuildTime"),
        exported_at=doc.get("exportedAt"),
        interactions=interactions,
    )


# ----------------------------------------------------------------------
# CURRENT GAME (JSON) extraction
# ----------------------------------------------------------------------

_MARKER = "CURRENT GAME (JSON):"
_END_MARKERS = ("Now choose", "RESPONSE FORMAT")


def parse_board(prompt: str | None) -> dict | None:
    """Pull the embedded CURRENT GAME (JSON) block out of a prompt.

    Mirrors `scripts/ingest_exports.parse_game` exactly so analyst and ingest
    pipeline agree on board state.
    """
    if not prompt:
        return None
    i = prompt.find(_MARKER)
    if i < 0:
        return None
    rest = prompt[i + len(_MARKER):]
    cut = min((rest.find(m) for m in _END_MARKERS if rest.find(m) >= 0), default=-1)
    blob = (rest[:cut] if cut >= 0 else rest).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def latest_board(doc: Export) -> dict | None:
    """The CURRENT GAME (JSON) from the latest successful interaction.

    Falls back to the latest interaction of any kind if no successes exist
    (e.g. fully-failed export).
    """
    for r in reversed(doc.successes):
        board = parse_board(r.get("prompt"))
        if board:
            return board
    for r in reversed(doc.interactions):
        board = parse_board(r.get("prompt"))
        if board:
            return board
    return None


# ----------------------------------------------------------------------
# Progress + stall metrics (mirror ingest_exports.py)
# ----------------------------------------------------------------------

_RANK_FROM_CARD = {
    "A": 1, "T": 10, "J": 11, "Q": 12, "K": 13,
    **{str(n): n for n in range(2, 10)},
}


def foundation_cards(board: dict | None) -> int | None:
    """Sum of foundation ranks (0..52)."""
    if not isinstance(board, dict):
        return None
    f = board.get("foundations") or {}
    if not isinstance(f, dict):
        return None
    try:
        return sum(_RANK_FROM_CARD[v[0]] for v in f.values() if v)
    except (KeyError, TypeError, IndexError):
        return None


def face_down_total(board: dict | None) -> int | None:
    """Sum of faceDownCount across the 7 tableau columns (21 down to 0)."""
    if not isinstance(board, dict):
        return None
    tab = board.get("tableau") or []
    try:
        return sum(int(c.get("faceDownCount", 0)) for c in tab)
    except (TypeError, ValueError):
        return None


def plateau_length(doc: Export) -> tuple[int, tuple[int | None, int | None]]:
    """Length of the trailing plateau on (foundationCards, faceDownTotal).

    Returns (plateau_turns, (fc, fd)). plateau_turns counts how many
    consecutive successful turns at the tail had the same (fc, fd) pair.
    """
    prev: tuple[int, int] | None = None
    streak = 0
    last_pair: tuple[int | None, int | None] = (None, None)
    for r in doc.successes:
        board = parse_board(r.get("prompt"))
        fc, fd = foundation_cards(board), face_down_total(board)
        last_pair = (fc, fd)
        if fc is None or fd is None:
            continue
        if prev is not None and (fc, fd) == prev:
            streak += 1
        else:
            streak = 0
            prev = (fc, fd)
    return streak, last_pair


# ----------------------------------------------------------------------
# Doom-loop detection
# ----------------------------------------------------------------------

# Matches a tableau-shuffle move and captures (card, src_col, dst_col).
_MOVE_RE = re.compile(r"move (\S+) col (\d+) -> col (\d+)")


def doom_loop_signature(recent_moves: list[str] | None) -> dict[str, Any] | None:
    """Detect a 2-card back-and-forth oscillation in a recentMoves window."""
    if not recent_moves:
        return None
    pair_counts: Counter[tuple[str, frozenset[int]]] = Counter()
    for m in recent_moves:
        match = _MOVE_RE.search(m)
        if match:
            card, a, b = match.group(1), int(match.group(2)), int(match.group(3))
            key = (card, frozenset({a, b}))
            pair_counts[key] += 1
    if not pair_counts:
        return None
    (card, cols), n = pair_counts.most_common(1)[0]
    if n < 2:
        return None
    cols_sorted = sorted(cols)
    return {
        "card": card,
        "between_columns": cols_sorted,
        "count_in_window": n,
        "window_size": len(recent_moves),
        "signature": f"{card} oscillating col {cols_sorted[0]} ↔ col {cols_sorted[1]}",
    }


def session_oscillation(doc: Export, top_n: int = 3) -> list[dict[str, Any]]:
    """Aggregate (card, column-pair) move counts across the whole session.

    Each turn's CURRENT GAME (JSON) carries `recentMoves` (last 10). Summed
    across all successful turns, with deduplication on adjacent identical
    windows, the dominant (card, col-pair) is the doom-loop signature even
    when the latest window happens to fall on a non-loop tail (a recycle
    burst, an aborted draw). This catches 29a7f5-style cases where the
    latest recentMoves doesn't show the loop but the session-wide pattern
    obviously does.

    Returns up to `top_n` dicts sorted by count, each with `card`,
    `between_columns`, `count`, and `signature`.
    """
    pair_counts: Counter[tuple[str, frozenset[int]]] = Counter()
    seen_windows: set[tuple[str, ...]] = set()
    for r in doc.successes:
        board = parse_board(r.get("prompt"))
        if not isinstance(board, dict):
            continue
        recent = board.get("recentMoves") or []
        if not recent:
            continue
        key = tuple(recent)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        for m in recent:
            match = _MOVE_RE.search(m)
            if match:
                card, a, b = match.group(1), int(match.group(2)), int(match.group(3))
                pair_counts[(card, frozenset({a, b}))] += 1
    results = []
    for (card, cols), n in pair_counts.most_common(top_n):
        if n < 4:
            break
        cols_sorted = sorted(cols)
        results.append({
            "card": card,
            "between_columns": cols_sorted,
            "count": n,
            "signature": f"{card} col {cols_sorted[0]} ↔ col {cols_sorted[1]}",
        })
    return results


# ----------------------------------------------------------------------
# Briefing
# ----------------------------------------------------------------------

def briefing(doc: Export) -> str:
    """A one-screen text briefing of the export."""
    n_total = len(doc.interactions)
    n_success = len(doc.successes)
    n_error = n_total - n_success

    board = latest_board(doc)
    fc = foundation_cards(board)
    fd = face_down_total(board)
    plateau, _pair = plateau_length(doc)
    recent = (board or {}).get("recentMoves") or []
    loop = doom_loop_signature(recent)
    legal = (board or {}).get("legalMoves") or []
    drawn = (board or {}).get("seenDrawPileCards") or []
    discard = (board or {}).get("discardTop")
    draw_pile = (board or {}).get("drawPileCount")
    can_recycle = (board or {}).get("canRecycleStock")
    session_loops = session_oscillation(doc)

    # Final reasoningTrail entries on the last success
    last_reasoning = ""
    for r in reversed(doc.successes):
        dec = r.get("decision") or {}
        ba = dec.get("boardAnalysis") or ""
        sp = dec.get("strategicPlan") or ""
        if ba or sp:
            last_reasoning = f"\n  boardAnalysis: {ba[:400]}\n  strategicPlan: {sp[:400]}"
            break

    lines = [
        f"## Export briefing — {doc.path.name}",
        f"  session       : {doc.short_session} (full: {doc.session_id})",
        f"  seed          : {doc.seed}",
        f"  model         : {doc.model}",
        f"  build         : {doc.app_commit} @ {doc.app_build_time}",
        f"  exported      : {doc.exported_at}",
        f"  outcome       : {doc.outcome}  finalProgress={doc.final_progress}%  moveCount={doc.move_count}",
        f"  rows in file  : total={n_total}  success={n_success}  error={n_error}",
        "",
        "## Latest board",
        f"  foundationCards={fc}  faceDownTotal={fd}  plateauTurns={plateau}",
        f"  discardTop={discard}  drawPileCount={draw_pile}  canRecycleStock={can_recycle}",
        f"  legalMoves available: {len(legal)}",
        f"  seenDrawPileCards ({len(drawn)}): {', '.join(drawn) if drawn else '(none)'}",
        "",
        "## Doom-loop check",
        (f"  latest-window: {loop['signature']} ({loop['count_in_window']}× in last "
         f"{loop['window_size']} moves)" if loop else "  latest-window: no 2-card oscillation"),
    ]
    if session_loops:
        lines += [f"  session-wide: {x['signature']} ({x['count']}× across plateau)"
                  for x in session_loops]
    else:
        lines += ["  session-wide: no dominant oscillation pattern"]
    lines += [
        "",
        "## recentMoves (last 10, this turn)",
    ]
    lines += [f"  {m}" for m in recent] or ["  (none)"]
    lines += ["", "## Latest reasoning (last successful turn)", last_reasoning or "  (none)"]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="path to a solitaire-ai-log export JSON file")
    args = ap.parse_args(argv)
    doc = load_export(args.path)
    print(briefing(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
