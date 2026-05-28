#!/usr/bin/env python3
"""Post-hoc tier scorer for the Gemma 4 E2B rung 2 baseline run.

Reads gemma4_finetune/baseline_n20_gemma4/baseline_n20_gemma4.json (move
indices already extracted by the runner), parses each prompt to recover the
legalMoves list, applies classify_pick + TIER_RANK from the shared scoring
library, and writes back an enriched JSON plus a one-line summary that lines
up with the Gemma 3n baseline numbers in baseline_n20/baseline_n20.json.

Designed to be run from the solitaire-analytics .venv (the one that already
has the scoring library on the path), not the mlx-vlm sibling venv.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from ab_test_prompt_formats import TIER_RANK, classify_pick  # noqa: E402

PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
RUN_DIR = REPO / "gemma4_finetune/baseline_n20_gemma4"
SUMMARY_PATH = RUN_DIR / "baseline_n20_gemma4.json"
TEACHER_LOOKUP = REPO / "gemma4_finetune/teacher_picks_n20.json"
BASELINE_3N = REPO / "gemma4_finetune/baseline_n20/baseline_n20.json"


def parse_state_from_prompt(prompt: str) -> dict:
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
            legal_moves.append({"type": m.group(2), "describe": m.group(3).strip()})
        elif legal_moves and not line.strip():
            break
        elif legal_moves and line.startswith(("PROGRESS", "PRIOR REASONING", "Now choose")):
            break
    return {"legalMoves": legal_moves, "recentMoves": []}


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text())
    teacher = json.loads(TEACHER_LOOKUP.read_text())
    enriched = []
    for row in data["results"]:
        sid = row["state_id"]
        prompt = (PROMPTS_DIR / sid / "prompt.txt").read_text()
        cg = parse_state_from_prompt(prompt)
        idx = row.get("e2b_move_index")
        tier = classify_pick(cg, idx) if idx is not None else "illegal"
        row["g4_tier"] = tier
        row["g4_tier_score"] = TIER_RANK.get(tier, 0)
        teacher_idx = teacher.get(sid)
        if teacher_idx is not None:
            teacher_tier = classify_pick(cg, teacher_idx)
            row["teacher_tier"] = teacher_tier
            row["teacher_tier_score"] = TIER_RANK.get(teacher_tier, 0)
        else:
            row["teacher_tier"] = None
            row["teacher_tier_score"] = None
        row["n_legal"] = len(cg["legalMoves"])
        enriched.append(row)

    n = len(enriched)
    mean_g4 = sum(r["g4_tier_score"] for r in enriched) / n
    mean_teacher = sum(
        r["teacher_tier_score"] for r in enriched if r["teacher_tier_score"] is not None
    ) / sum(1 for r in enriched if r["teacher_tier_score"] is not None)
    illegal = sum(1 for r in enriched if r["g4_tier"] == "illegal")
    foundation_states = [r for r in enriched if r.get("teacher_tier") == "foundation"]
    foundation_recovery = sum(1 for r in foundation_states if r["g4_tier"] == "foundation")

    data["results"] = enriched
    data["mean_tier_g4"] = round(mean_g4, 2)
    data["mean_tier_teacher"] = round(mean_teacher, 2)
    data["gap_vs_teacher"] = round(mean_g4 - mean_teacher, 2)
    data["illegal_count"] = illegal
    data["foundation_states"] = len(foundation_states)
    data["foundation_recovery"] = foundation_recovery
    SUMMARY_PATH.write_text(json.dumps(data, indent=2))

    print(f"Gemma 4 E2B (untuned, 4-bit, mlx-vlm) on Phase 1.5 N=20:")
    print(f"  json_valid: {data['json_valid_count']}/{n}")
    print(f"  illegal:    {illegal}/{n}")
    print(f"  agreement:  {data['agreement_count']}/{n}")
    print(f"  mean tier:  {mean_g4:.2f}  (teacher {mean_teacher:.2f}, gap {mean_g4 - mean_teacher:+.2f})")
    print(f"  foundation: {foundation_recovery}/{len(foundation_states)} states recovered")

    if BASELINE_3N.exists():
        b3n = json.loads(BASELINE_3N.read_text())
        b3n_tier = sum(r["e2b_tier_score"] for r in b3n["results"]) / len(b3n["results"])
        b3n_illegal = sum(1 for r in b3n["results"] if r["e2b_tier"] == "illegal")
        b3n_foundation_states = [r for r in b3n["results"] if r.get("teacher_tier") == "foundation"]
        b3n_foundation = sum(
            1 for r in b3n_foundation_states if r["e2b_tier"] == "foundation"
        )
        print()
        print(f"Side-by-side (untuned):")
        print(f"  {'metric':<22} {'gemma-3n-E2B':>14} {'gemma-4-E2B':>14}")
        print(f"  {'json_valid':<22} {b3n['json_valid_count']:>10}/{len(b3n['results']):<3} {data['json_valid_count']:>10}/{n:<3}")
        print(f"  {'illegal':<22} {b3n_illegal:>10}/{len(b3n['results']):<3} {illegal:>10}/{n:<3}")
        print(f"  {'agreement':<22} {b3n['agreement_count']:>10}/{len(b3n['results']):<3} {data['agreement_count']:>10}/{n:<3}")
        print(f"  {'mean tier':<22} {b3n_tier:>14.2f} {mean_g4:>14.2f}")
        print(f"  {'gap vs teacher':<22} {b3n_tier - mean_teacher:>+14.2f} {mean_g4 - mean_teacher:>+14.2f}")
        print(f"  {'foundation recovery':<22} {b3n_foundation:>10}/{len(b3n_foundation_states):<3} {foundation_recovery:>10}/{len(foundation_states):<3}")
        print(f"  {'peak GB':<22} {b3n['overall_peak_gb']:>14.2f} {data['overall_peak_gb']:>14.2f}")
        print(f"  {'mean call sec':<22} {b3n['mean_call_seconds']:>14.2f} {data['mean_call_seconds']:>14.2f}")


if __name__ == "__main__":
    main()
