#!/usr/bin/env python3
"""Build a reusable JSON of benchmark winnable decks.

Scans data/raw/solitaire-win-*.json for `gameWon: true` entries, extracts
each one's ground-truth initial deck from `initialBoardSetup`, runs
pyksolve on it (draw-1 and draw-3), and writes a consolidated file at
data/benchmarks/winnable_decks.json.

Consumers:
- pyksolve users: read `decks[*].pysol_format` and feed straight to
  Solitaire.load_pysol().
- Replay / test scripts: read `decks[*].tableau` + `decks[*].stock` as
  structured arrays.
- Cross-version bench arms: anchor the same starting deck across
  experiments without needing the seed-to-deck logic.
- External researchers (HF dataset consumers): a stable, reproducible
  reference for any analysis that needs the exact starting board.

Re-run after harvesting new wins; the file is regenerated from scratch
each time so it always reflects the current corpus.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw"
OUT_PATH = REPO / "data" / "benchmarks" / "winnable_decks.json"
SCHEMA_VERSION = "1"


def card_str(c: dict) -> str:
    """Render a card as the 'RankSuit' string pyksolve / pysol use:
    A 2..9 T J Q K plus H D C S. '10' is normalised to 'T'."""
    rank = c["rank"]
    if rank == "10":
        rank = "T"
    suit = c["suit"][0].upper()  # hearts -> H, etc.
    return f"{rank}{suit}"


def build_pysol(tableau: list[list[dict]], drawpile: list[dict]) -> str:
    """Render the initial board as pysol-format text:

        Talon: <space-separated stock, top first>
        <col1 cards bottom-to-top, face-down wrapped in <>>
        ...
        <col7 cards>
    """
    lines = ["Talon: " + " ".join(card_str(c) for c in drawpile)]
    for col in tableau:
        parts = []
        for c in col:
            s = card_str(c)
            parts.append(s if c.get("faceUp", False) else f"<{s}>")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_pyksolve(pysol_str: str, draw_count: int, max_closed: int = 400_000):
    """Run pyksolve on the given pysol-format board. Returns dict with
    verdict (str), ms (int), value (int)."""
    from pyksolve.solver import Solitaire, SolveResult  # noqa: F401

    sol = Solitaire()
    sol.draw_count = draw_count
    sol.load_pysol(pysol_str)
    sol.reset_game()
    t0 = time.time()
    r = sol.solve_fast(max_closed_count=max_closed)
    return {
        "verdict": r.name,
        "value": int(r.value),
        "ms": int(round((time.time() - t0) * 1000)),
    }


def extract_deck(win_path: Path) -> dict | None:
    """Read a solitaire-win-*.json file and return a deck record, or None
    if the file is not a usable winnable-deck source."""
    doc = json.loads(win_path.read_text())
    if not doc.get("gameWon"):
        return None
    ib = doc.get("initialBoardSetup")
    if not ib:
        return None
    tableau = ib.get("tableau", [])
    drawpile = ib.get("drawPile", [])
    if len(tableau) != 7 or len(drawpile) != 24:
        # Unexpected shape; skip rather than corrupt the output.
        return None
    # Normalise card dicts to a stable schema (drop client-only fields
    # like 'id', keep rank/suit/face_up).
    norm_tableau = [
        [
            {"rank": c["rank"], "suit": c["suit"], "face_up": bool(c.get("faceUp", False))}
            for c in col
        ]
        for col in tableau
    ]
    norm_stock = [{"rank": c["rank"], "suit": c["suit"]} for c in drawpile]
    pysol_str = build_pysol(tableau, drawpile)
    record = {
        "seed": doc.get("seed"),
        "source_session_id": doc.get("gameSessionId"),
        "source_file": str(win_path.relative_to(REPO)),
        "source_app_commit": doc.get("appCommit"),
        "source_app_build_time": doc.get("appBuildTime"),
        # 'difficulty' is the harvester's 1-5 deal-arrangement knob.
        # 3 = true random deal; other values arrange the deck in some
        # documented (by the harvester) way and still seed-randomise
        # within that arrangement. NOT the draw count.
        "harvester_difficulty": doc.get("difficulty"),
        "perceived_difficulty": doc.get("perceivedDifficulty"),
        "outcome": {
            "won": doc.get("gameWon"),
            "moves": len(doc.get("moveHistory", [])),
            "completion_progress": doc.get("completionProgress"),
        },
        "tableau": norm_tableau,
        "stock": norm_stock,
        "foundations": {"hearts": [], "diamonds": [], "clubs": [], "spades": []},
        "pysol_format": pysol_str,
    }
    return record


def main() -> None:
    win_files = sorted(RAW_DIR.glob("solitaire-win-*.json"))
    print(f"Scanning {len(win_files)} solitaire-win-*.json files in {RAW_DIR}")

    decks = []
    skipped = []
    for f in win_files:
        rec = extract_deck(f)
        if rec is None:
            skipped.append(f.name)
            continue
        if rec["seed"] is None:
            # Keep but flag; the deck is reusable for solver work but
            # not for replay against the harvester URL.
            rec["note"] = "no seed in win-record; not replayable via harvester URL"
        # Run pyksolve in both draw modes. The harvester export does not
        # surface a draw-count field, so we evaluate both for completeness
        # (a deck solvable under draw-1 isn't automatically solvable under
        # draw-3 and vice versa).
        rec["pyksolve"] = {
            "draw1": run_pyksolve(rec["pysol_format"], draw_count=1),
            "draw3": run_pyksolve(rec["pysol_format"], draw_count=3),
        }
        decks.append(rec)
        seed = rec["seed"] if rec["seed"] is not None else "(no seed)"
        print(
            f"  + seed={seed!s:<11} "
            f"moves={rec['outcome']['moves']:>3} "
            f"difficulty={rec['harvester_difficulty']} "
            f"draw3={rec['pyksolve']['draw3']['verdict']} "
            f"({rec['pyksolve']['draw3']['ms']} ms)  "
            f"from {f.name}"
        )

    if skipped:
        print(f"Skipped {len(skipped)} files (not winnable or missing initialBoardSetup):")
        for n in skipped:
            print(f"  - {n}")

    # Sort decks by seed for stable diffs (None seeds go last).
    decks.sort(key=lambda d: (d["seed"] is None, d["seed"] or 0))

    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Extracted from initialBoardSetup in data/raw/solitaire-win-*.json",
        "regenerator": "scripts/build_winnable_decks.py",
        "card_notation": "rank in {A,2,3,4,5,6,7,8,9,10,J,Q,K}; suit in {hearts,diamonds,clubs,spades}. "
                         "pysol_format uses T for 10 and the first letter of each suit; face-down wrapped in <>.",
        "harvester_difficulty_note": "harvester_difficulty is the harvester's 1-5 deal-arrangement knob "
                                     "(3 = true random; other values arrange the deck in some documented "
                                     "way and still seed-randomise within that arrangement). It is NOT "
                                     "the draw count. The harvester export does not surface a draw-count "
                                     "field separately, so we run pyksolve in BOTH draw-1 and draw-3 for "
                                     "every deck. perceived_difficulty (47-52 in current decks) is a "
                                     "per-deal metric, likely solver-rated dealability.",
        "n_decks": len(decks),
        "decks": decks,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(REPO)} ({len(decks)} decks, "
          f"{OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
