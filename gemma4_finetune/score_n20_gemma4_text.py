#!/usr/bin/env python3
"""Tier scorer for the patched text-only Gemma 4 E2B baseline run."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from ab_test_prompt_formats import TIER_RANK, classify_pick  # noqa: E402

PROMPTS_DIR = REPO / "experiments/a4_phase1.5_2026_05_24/prompts/C0"
RUN_DIR = REPO / "gemma4_finetune/baseline_n20_gemma4_text"
SUMMARY_PATH = RUN_DIR / "baseline_n20_gemma4_text.json"
TEACHER_LOOKUP = REPO / "gemma4_finetune/teacher_picks_n20.json"
BASELINE_3N = REPO / "gemma4_finetune/baseline_n20/baseline_n20.json"
BASELINE_G4_MM = REPO / "gemma4_finetune/baseline_n20_gemma4/baseline_n20_gemma4.json"


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
        row["g4t_tier"] = tier
        row["g4t_tier_score"] = TIER_RANK.get(tier, 0)
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
    mean = sum(r["g4t_tier_score"] for r in enriched) / n
    teacher_scored = [r["teacher_tier_score"] for r in enriched if r["teacher_tier_score"] is not None]
    mean_teacher = sum(teacher_scored) / len(teacher_scored)
    illegal = sum(1 for r in enriched if r["g4t_tier"] == "illegal")
    foundation_states = [r for r in enriched if r.get("teacher_tier") == "foundation"]
    foundation_rec = sum(1 for r in foundation_states if r["g4t_tier"] == "foundation")

    data["results"] = enriched
    data["mean_tier_g4t"] = round(mean, 2)
    data["mean_tier_teacher"] = round(mean_teacher, 2)
    data["gap_vs_teacher"] = round(mean - mean_teacher, 2)
    data["illegal_count"] = illegal
    data["foundation_states"] = len(foundation_states)
    data["foundation_recovery"] = foundation_rec
    SUMMARY_PATH.write_text(json.dumps(data, indent=2))

    print("Gemma 4 E2B text-only (patched, int4, mlx-lm) on Phase 1.5 N=20:")
    print(f"  json_valid: {data['json_valid_count']}/{n}")
    print(f"  illegal:    {illegal}/{n}")
    print(f"  agreement:  {data['agreement_count']}/{n}")
    print(f"  mean tier:  {mean:.2f}  (teacher {mean_teacher:.2f}, gap {mean - mean_teacher:+.2f})")
    print(f"  foundation: {foundation_rec}/{len(foundation_states)} states recovered")

    if BASELINE_3N.exists() and BASELINE_G4_MM.exists():
        b3n = json.loads(BASELINE_3N.read_text())
        bg4 = json.loads(BASELINE_G4_MM.read_text())
        b3n_t = sum(r["e2b_tier_score"] for r in b3n["results"]) / len(b3n["results"])
        bg4_t = sum(r["g4_tier_score"] for r in bg4["results"]) / len(bg4["results"])
        b3n_ill = sum(1 for r in b3n["results"] if r["e2b_tier"] == "illegal")
        bg4_ill = sum(1 for r in bg4["results"] if r["g4_tier"] == "illegal")
        b3n_fs = [r for r in b3n["results"] if r.get("teacher_tier") == "foundation"]
        bg4_fs = [r for r in bg4["results"] if r.get("teacher_tier") == "foundation"]
        b3n_fr = sum(1 for r in b3n_fs if r["e2b_tier"] == "foundation")
        bg4_fr = sum(1 for r in bg4_fs if r["g4_tier"] == "foundation")
        print()
        print("Three-way side-by-side (all untuned):")
        print(f"  {'metric':<22} {'gemma-3n-E2B':>14} {'gemma-4 MM':>14} {'gemma-4 text*':>16}")
        print(f"  {'json_valid':<22} {b3n['json_valid_count']:>10}/{len(b3n['results']):<3} {bg4['json_valid_count']:>10}/{len(bg4['results']):<3} {data['json_valid_count']:>12}/{n:<3}")
        print(f"  {'illegal':<22} {b3n_ill:>10}/{len(b3n['results']):<3} {bg4_ill:>10}/{len(bg4['results']):<3} {illegal:>12}/{n:<3}")
        print(f"  {'agreement':<22} {b3n['agreement_count']:>10}/{len(b3n['results']):<3} {bg4['agreement_count']:>10}/{len(bg4['results']):<3} {data['agreement_count']:>12}/{n:<3}")
        print(f"  {'mean tier':<22} {b3n_t:>14.2f} {bg4_t:>14.2f} {mean:>16.2f}")
        print(f"  {'gap vs teacher':<22} {b3n_t - mean_teacher:>+14.2f} {bg4_t - mean_teacher:>+14.2f} {mean - mean_teacher:>+16.2f}")
        print(f"  {'foundation recovery':<22} {b3n_fr:>10}/{len(b3n_fs):<3} {bg4_fr:>10}/{len(bg4_fs):<3} {foundation_rec:>12}/{len(foundation_states):<3}")
        print(f"  {'peak GB':<22} {b3n['overall_peak_gb']:>14.2f} {bg4['overall_peak_gb']:>14.2f} {data['overall_peak_gb']:>16.2f}")
        print(f"  {'mean call sec':<22} {b3n['mean_call_seconds']:>14.2f} {bg4['mean_call_seconds']:>14.2f} {data['mean_call_seconds']:>16.2f}")
        print(f"  {'max_tokens':<22} {512:>14} {512:>14} {2048:>16}")
        print()
        print("  * text-only ran at max_tokens=2048 because thinking-mode reasoning")
        print("    chains are longer than the multimodal model's at the same prompts.")


if __name__ == "__main__":
    main()
