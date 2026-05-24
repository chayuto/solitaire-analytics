"""Score the raw response files dropped by subagent workers.

Walks raw/<arm>/<state_id>/run<N>.response.txt, parses each, classifies
the chosen move, writes raw/<arm>/<state_id>/run<N>.scored.json, and prints
a quick summary.

Idempotent: re-running re-scores from .response.txt files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parents[1] / "scripts"))

from ab_test_prompt_formats import classify_pick, TIER_RANK  # noqa: E402

JSON_BLOCK_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def extract_move_index(text: str) -> int | None:
    candidates = JSON_BLOCK_RE.findall(text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if isinstance(obj.get("move_index"), int):
                return obj["move_index"]
            fd = obj.get("final_decision")
            if isinstance(fd, dict) and isinstance(fd.get("move_index"), int):
                return fd["move_index"]
    return None


def detect_chose_reversal(cg: dict, chosen_idx: int | None) -> bool:
    if chosen_idx is None:
        return False
    lm = cg.get("legalMoves") or []
    if chosen_idx < 0 or chosen_idx >= len(lm):
        return False
    desc = (lm[chosen_idx].get("describe") or "").lower()
    for entry in reversed(cg.get("recentMoves") or []):
        m = re.match(r"move\s+[A-Z0-9]{2}\s+col\s+(\d+)\s+->\s+col\s+(\d+)",
                     entry.strip(), re.IGNORECASE)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        needles = (f"from column {dst} to column {src}", f"col {dst} to col {src}")
        return any(n in desc for n in needles)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(EXP_DIR / "raw"))
    parser.add_argument("--bench", default=str(EXP_DIR / "bench.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    states_by_id = {s["state_id"]: s for s in bench["states"]}

    raw_dir = Path(args.raw)
    scored_rows = []
    for resp_path in sorted(raw_dir.glob("*/*/*.response.txt")):
        arm = resp_path.parent.parent.name
        state_id = resp_path.parent.name
        run_n = int(re.match(r"run(\d+)", resp_path.stem).group(1))

        text = resp_path.read_text()
        state = states_by_id.get(state_id)
        if not state:
            continue
        cg = state["current_game"]

        idx = extract_move_index(text)
        tier = classify_pick(cg, idx) if idx is not None else "illegal"
        chose_reversal = detect_chose_reversal(cg, idx)
        chosen_desc = ""
        if idx is not None and 0 <= idx < len(cg.get("legalMoves", [])):
            chosen_desc = cg["legalMoves"][idx].get("describe", "")

        row = {
            "arm": arm,
            "state_id": state_id,
            "category": state["category"],
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
    if args.quiet:
        return

    # quick per-arm summary
    from collections import defaultdict
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in scored_rows:
        by_arm[r["arm"]].append(r)
    print(f"\n{'arm':<5}{'n':>4}{'mean_tier':>11}{'json_ok':>10}{'reversal_pct':>15}{'resp_chars':>12}")
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        n = len(rows)
        mean_tier = sum(r["tier_score"] for r in rows) / n
        json_ok = sum(1 for r in rows if r["json_valid"]) / n
        rev = sum(1 for r in rows if r["chose_reversal"]) / n
        chars = sum(r["response_chars"] for r in rows) / n
        print(f"{arm:<5}{n:>4}{mean_tier:>11.2f}{json_ok:>10.0%}{rev:>14.0%} "
              f"{chars:>11.0f}")


if __name__ == "__main__":
    main()
