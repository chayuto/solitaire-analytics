"""Score the raw subagent responses (Phase 1.5).

Reads raw/<arm>/<state_id>/run<N>.response.txt, parses JSON, classifies
the chosen move via the shared tier scale, writes raw/.../run<N>.scored.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR.parents[1] / "scripts"))
from ab_test_prompt_formats import classify_pick, TIER_RANK  # noqa: E402

JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def extract_move_index(text: str) -> int | None:
    """Pull move_index from any JSON object in the response.

    Try the new schema (final_decision.move_index) first, then the legacy
    flat key, scanning candidates last-to-first so trailing JSON wins over
    any incidentally-quoted snippet in board_analysis prose.
    """
    candidates = JSON_BLOCK_RE.findall(text)
    for cand in reversed(candidates):
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


def detect_chose_reversal(cg: dict, chosen_idx: int | None) -> bool:
    if chosen_idx is None:
        return False
    lm = cg.get("legalMoves") or []
    if chosen_idx < 0 or chosen_idx >= len(lm):
        return False
    desc = (lm[chosen_idx].get("describe") or "").lower()
    for entry in reversed(cg.get("recentMoves") or []):
        m = re.match(
            r"move\s+[A-Z0-9]{2}\s+col\s+(\d+)\s+->\s+col\s+(\d+)",
            entry.strip(),
            re.IGNORECASE,
        )
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        needles = (f"from column {dst} to column {src}", f"col {dst} to col {src}")
        return any(n in desc for n in needles)
    return False


def parse_state_from_prompt(prompt: str) -> dict:
    """Build a minimal current_game shaped dict from the rendered prompt text.

    We only need .legalMoves (with .describe and .type) for classify_pick.
    Foundation field is not used in scoring — left empty for safety.
    """
    legal_moves: list[dict] = []
    in_block = False
    for line in prompt.splitlines():
        if line.startswith("LEGAL MOVES (respond"):
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r"\s*\[(\d+)\]\s+(\S+)\s+(.+)$", line)
        if m:
            legal_moves.append({
                "type": m.group(2),
                "describe": m.group(3).strip(),
            })
        elif legal_moves and not line.strip():
            break  # blank line after first move = end of block
        elif legal_moves and line.startswith(("PROGRESS", "PRIOR REASONING", "Now choose")):
            break
    return {"legalMoves": legal_moves, "recentMoves": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(EXP_DIR / "raw"))
    parser.add_argument("--bench", default=str(EXP_DIR / "bench.json"))
    parser.add_argument("--prompts", default=str(EXP_DIR / "prompts"))
    args = parser.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    states_by_id = {s["state_id"]: s for s in bench["states"]}

    raw_dir = Path(args.raw)
    scored_rows: list[dict] = []
    for resp_path in sorted(raw_dir.glob("*/*/*.response.txt")):
        arm = resp_path.parent.parent.name
        state_id = resp_path.parent.name
        run_n = int(re.match(r"run(\d+)", resp_path.stem).group(1))

        text = resp_path.read_text()
        state = states_by_id.get(state_id)
        if not state:
            continue

        # Build current_game-shaped struct from the rendered prompt that the
        # subagent actually saw (not the bench's full_prompt — though they
        # share the LEGAL MOVES block, this future-proofs against renderers
        # that reorder moves).
        rendered = Path(args.prompts) / arm / state_id / "prompt.txt"
        prompt_text = rendered.read_text() if rendered.exists() else state["full_prompt"]
        cg = parse_state_from_prompt(prompt_text)

        idx = extract_move_index(text)
        tier = classify_pick(cg, idx) if idx is not None else "illegal"
        chose_reversal = detect_chose_reversal(cg, idx)
        chosen_desc = ""
        if idx is not None and 0 <= idx < len(cg["legalMoves"]):
            chosen_desc = cg["legalMoves"][idx].get("describe", "")

        row = {
            "arm": arm,
            "state_id": state_id,
            "category": state["category"],
            "has_foundation_move": state["has_foundation_move"],
            "run": run_n,
            "chosen_index": idx,
            "chosen_describe": chosen_desc,
            "tier": tier,
            "tier_score": TIER_RANK.get(tier, 0),
            "json_valid": idx is not None,
            "chose_reversal": chose_reversal,
            "response_chars": len(text),
        }
        out = resp_path.with_suffix(".scored.json")
        out.write_text(json.dumps(row, indent=2))
        scored_rows.append(row)

    print(f"scored {len(scored_rows)} responses")

    from collections import defaultdict
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in scored_rows:
        by_arm[r["arm"]].append(r)
    print(f"\n{'arm':<5}{'n':>4}{'mean_tier':>11}{'json_ok':>10}{'rev%':>7}{'resp_chars':>12}")
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        n = len(rows)
        mean_tier = sum(r["tier_score"] for r in rows) / n
        json_ok = sum(1 for r in rows if r["json_valid"]) / n
        rev = sum(1 for r in rows if r["chose_reversal"]) / n
        chars = sum(r["response_chars"] for r in rows) / n
        print(f"{arm:<5}{n:>4}{mean_tier:>11.2f}{json_ok:>10.0%}{rev:>6.0%} {chars:>11.0f}")


if __name__ == "__main__":
    main()
