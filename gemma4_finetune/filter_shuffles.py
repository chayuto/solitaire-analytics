#!/usr/bin/env python3
"""Per-turn shuffle filter for v3 training data.

Drops interactions where the teacher's chosen move is a tableau-to-tableau
move that directly reverses the most recent tableau-to-tableau move in
``recentMoves``. This is the doom-loop / oscillation signature the v2
distillation transferred from the corpus into the student (see
``docs/reports/20260526_v2_gemma4_distillation_lab_log.md`` section 8).

Input:  data/dataset/training.jsonl (one interaction per line; the file
        the v2 pipeline consumed via prepare_dataset.py)
Output: data/dataset/training_shuffle_filtered.jsonl (same shape, fewer rows)

Parses both prompt formats present in the corpus:
- Legacy ``CURRENT GAME (JSON):`` blob: read recentMoves + legalMoves from
  the embedded JSON
- Hybrid-v1 plain-text blocks (appCommit de7dc06+): parse the
  ``RECENT MOVES (oldest -> newest; ...)`` block and the
  ``LEGAL MOVES (respond ...)`` block

A row is dropped iff all of:
1. We can parse a chosen move_index from rawResponse JSON
2. legalMoves[chosen_idx].type starts with "tableau_to_tableau"
3. recentMoves has at least one entry matching ``move XX col N -> col M``
4. The chosen move's describe field encodes the reverse (col M -> col N)

Rows we cannot parse confidently are KEPT (false-negative bias). The
detection is intentionally conservative: only direct one-back reversals,
not multi-step doom loops. False positives on this signature are extremely
rare; false negatives are the dominant error mode and are acceptable for
this first pass.

Usage:
    .venv/bin/python gemma4_finetune/filter_shuffles.py \\
        --in data/dataset/training.jsonl \\
        --out data/dataset/training_shuffle_filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---- Legacy prompt format: CURRENT GAME (JSON): {...} ----------------------

_LEGACY_MARKER = "CURRENT GAME (JSON):"


def _parse_legacy(prompt: str) -> dict | None:
    """Extract {legalMoves, recentMoves} from a legacy CURRENT GAME blob."""
    idx = prompt.find(_LEGACY_MARKER)
    if idx < 0:
        return None
    body_start = idx + len(_LEGACY_MARKER)
    # Find the matching brace-balanced JSON object after the marker.
    depth = 0
    json_start = None
    for i in range(body_start, len(prompt)):
        c = prompt[i]
        if c == "{":
            if json_start is None:
                json_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and json_start is not None:
                json_text = prompt[json_start:i + 1]
                try:
                    obj = json.loads(json_text)
                except json.JSONDecodeError:
                    return None
                return {
                    "legalMoves": obj.get("legalMoves") or [],
                    "recentMoves": obj.get("recentMoves") or [],
                }
    return None


# ---- Hybrid-v1 prompt format (appCommit de7dc06+) --------------------------

_HYBRID_LEGAL_BLOCK_RE = re.compile(
    r"LEGAL MOVES \(respond[^\n]*\):\s*\n((?:\s*\[\d+\][^\n]+\n?)+)",
)
_HYBRID_LEGAL_LINE_RE = re.compile(r"\s*\[(\d+)\]\s+(\S+)\s+(.+)")

_HYBRID_RECENT_BLOCK_RE = re.compile(
    r"RECENT MOVES \(oldest -> newest[^\n]*\):\s*\n((?:\s*\d+\.\s+[^\n]+\n?)+)",
)
_HYBRID_RECENT_LINE_RE = re.compile(r"\s*\d+\.\s+(.+)")


def _parse_hybrid(prompt: str) -> dict | None:
    if "FOUNDATIONS:" not in prompt:
        return None
    legal: list[dict] = []
    lm = _HYBRID_LEGAL_BLOCK_RE.search(prompt)
    if lm:
        for line in lm.group(1).splitlines():
            m = _HYBRID_LEGAL_LINE_RE.match(line)
            if m:
                legal.append({"type": m.group(2), "describe": m.group(3).strip()})
    recent: list[str] = []
    rm = _HYBRID_RECENT_BLOCK_RE.search(prompt)
    if rm:
        for line in rm.group(1).splitlines():
            m = _HYBRID_RECENT_LINE_RE.match(line)
            if m:
                recent.append(m.group(1).strip())
    if not legal:
        return None
    return {"legalMoves": legal, "recentMoves": recent}


def parse_state(prompt: str) -> dict | None:
    if _LEGACY_MARKER in prompt:
        return _parse_legacy(prompt)
    return _parse_hybrid(prompt)


# ---- Reversal detector -----------------------------------------------------

_RECENT_TT_RE = re.compile(
    r"move\s+[A-Z0-9]{2}\s+col\s+(\d+)\s+->\s+col\s+(\d+)",
    re.IGNORECASE,
)


def detect_reversal(state: dict, chosen_idx: int) -> bool:
    lm = state.get("legalMoves") or []
    if not (0 <= chosen_idx < len(lm)):
        return False
    chosen = lm[chosen_idx]
    if not (chosen.get("type") or "").startswith("tableau_to_tableau"):
        return False
    desc = (chosen.get("describe") or "").lower()
    for entry in reversed(state.get("recentMoves") or []):
        text = entry if isinstance(entry, str) else str(entry)
        m = _RECENT_TT_RE.search(text)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        needles = (
            f"from column {dst} to column {src}",
            f"col {dst} to col {src}",
            f"column {dst} to column {src}",
        )
        return any(n in desc for n in needles)
    return False


# ---- Choice extraction -----------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def extract_move_index(raw: str) -> int | None:
    if not raw:
        return None
    for cand in reversed(_JSON_BLOCK_RE.findall(raw)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fd = obj.get("final_decision")
        if isinstance(fd, dict) and isinstance(fd.get("move_index"), int):
            return fd["move_index"]
        if isinstance(obj.get("move_index"), int):
            return obj["move_index"]
    return None


# ---- Main pipeline ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="../data/dataset/training.jsonl",
                    help="path to input training.jsonl")
    ap.add_argument("--out", dest="out",
                    default="../data/dataset/training_shuffle_filtered.jsonl",
                    help="path to output filtered training.jsonl")
    ap.add_argument("--dump-dropped", default=None,
                    help="optional path to write the dropped rows for inspection")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        sys.exit(f"input not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)
    dropped_path = Path(args.dump_dropped) if args.dump_dropped else None

    stats = {
        "input_rows": 0,
        "unparseable_prompt": 0,
        "no_move_index": 0,
        "kept": 0,
        "dropped_reversal": 0,
    }
    dropped_examples = []

    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.rstrip()
            if not line:
                continue
            stats["input_rows"] += 1
            row = json.loads(line)
            prompt = row.get("prompt") or ""
            raw = row.get("rawResponse") or ""

            state = parse_state(prompt)
            if state is None:
                stats["unparseable_prompt"] += 1
                fout.write(line + "\n")
                stats["kept"] += 1
                continue

            chosen_idx = extract_move_index(raw)
            if chosen_idx is None:
                stats["no_move_index"] += 1
                fout.write(line + "\n")
                stats["kept"] += 1
                continue

            if detect_reversal(state, chosen_idx):
                stats["dropped_reversal"] += 1
                if dropped_examples is not None and len(dropped_examples) < 10:
                    lm = state["legalMoves"]
                    dropped_examples.append({
                        "id": row.get("id"),
                        "sessionId": row.get("sessionId"),
                        "turnIndex": row.get("turnIndex"),
                        "chosen_describe": lm[chosen_idx].get("describe"),
                        "last_recent_move": (state.get("recentMoves") or [None])[-1],
                    })
                if dropped_path is not None:
                    dropped_path.parent.mkdir(parents=True, exist_ok=True)
                continue

            fout.write(line + "\n")
            stats["kept"] += 1

    if dropped_path is not None and dropped_examples:
        with dropped_path.open("w") as f:
            for ex in dropped_examples:
                f.write(json.dumps(ex) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"\nwrote {stats['kept']} rows -> {out}")
    if stats["dropped_reversal"]:
        print(f"\nsample dropped rows ({len(dropped_examples)} of {stats['dropped_reversal']}):")
        for ex in dropped_examples[:5]:
            print(f"  session={ex['sessionId']} turn={ex['turnIndex']}")
            print(f"    chose: {ex['chosen_describe']}")
            print(f"    after: {ex['last_recent_move']}")


if __name__ == "__main__":
    main()
