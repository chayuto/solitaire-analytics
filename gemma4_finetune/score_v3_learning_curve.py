#!/usr/bin/env python3
"""Aggregate v3 LoRA learning-curve evals alongside v2 and v1 baselines.

Reads the same posttune_at*.json files the v2 scorer reads, plus the new
posttune_v3_at*.json files produced by sweep_v3_checkpoints.sh.

Highlights oscillation-state behaviour explicitly: the v3 hypothesis predicts
that shuffle-filter-trained checkpoints should preserve untuned-baseline
behaviour on the 7 oscillation states (4/7 foundation recovery + zero
oscillation regressions) while still gaining on the 13 non-oscillation states.
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
RUN_DIR = REPO / "gemma4_finetune/baseline_n20_gemma4_text"
TEACHER_LOOKUP = REPO / "gemma4_finetune/teacher_picks_n20.json"
BASELINE_3N = REPO / "gemma4_finetune/baseline_n20/baseline_n20.json"
POSTTUNE_3N_AT750 = REPO / "gemma4_finetune/baseline_n20/posttune_at750.json"


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


def score_file(p: Path, teacher: dict) -> dict:
    if not p.exists():
        return {"missing": True, "path": str(p)}
    data = json.loads(p.read_text())
    n = len(data["results"])
    tier_sum = 0
    illegal = 0
    foundation_states = 0
    foundation_recovery = 0
    osc_n = 0
    osc_correct = 0
    osc_tier_sum = 0
    for row in data["results"]:
        sid = row["state_id"]
        prompt = (PROMPTS_DIR / sid / "prompt.txt").read_text()
        cg = parse_state_from_prompt(prompt)
        idx = row.get("e2b_move_index")
        tier = classify_pick(cg, idx) if idx is not None else "illegal"
        tier_sum += TIER_RANK.get(tier, 0)
        if tier == "illegal":
            illegal += 1
        t_idx = teacher.get(sid)
        if t_idx is not None:
            t_tier = classify_pick(cg, t_idx)
            if t_tier == "foundation":
                foundation_states += 1
                if tier == "foundation":
                    foundation_recovery += 1
        if sid.startswith("oscillation"):
            osc_n += 1
            osc_tier_sum += TIER_RANK.get(tier, 0)
            if row.get("agreement"):
                osc_correct += 1
    return {
        "n": n,
        "json_valid": data["json_valid_count"],
        "agreement": data["agreement_count"],
        "illegal": illegal,
        "mean_tier": round(tier_sum / n, 2),
        "foundation_states": foundation_states,
        "foundation_recovery": foundation_recovery,
        "osc_n": osc_n,
        "osc_correct": osc_correct,
        "osc_mean_tier": round(osc_tier_sum / max(osc_n, 1), 2),
        "peak_gb": data.get("overall_peak_gb"),
        "mean_call_sec": data.get("mean_call_seconds"),
    }


def main() -> None:
    teacher = json.loads(TEACHER_LOOKUP.read_text())
    rows = {}
    rows["v2 untuned"] = score_file(RUN_DIR / "baseline_n20_gemma4_text.json", teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v2 iter {ckpt}"] = score_file(RUN_DIR / f"posttune_at{ckpt}.json", teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v3 iter {ckpt}"] = score_file(RUN_DIR / f"posttune_v3_at{ckpt}.json", teacher)
    rows["v1 untuned (3n)"] = score_file(BASELINE_3N, teacher)
    rows["v1.1 iter 750 (3n)"] = score_file(POSTTUNE_3N_AT750, teacher)

    teacher_mean = 3.42
    print(f"{'config':<22} {'json':>6} {'ill':>5} {'agr':>5} {'tier':>5} {'gap':>6} {'fnd':>5} {'osc_agr':>8} {'osc_t':>6} {'peak':>6}")
    print("-" * 88)
    for name, r in rows.items():
        if r.get("missing"):
            print(f"{name:<22} (missing: {r['path']})")
            continue
        gap = r["mean_tier"] - teacher_mean
        print(
            f"{name:<22} {r['json_valid']:>3}/{r['n']:<2} "
            f"{r['illegal']:>2}/{r['n']:<2} "
            f"{r['agreement']:>2}/{r['n']:<2} "
            f"{r['mean_tier']:>5.2f} {gap:>+6.2f} "
            f"{r['foundation_recovery']:>2}/{r['foundation_states']:<2} "
            f"{r['osc_correct']:>3}/{r['osc_n']:<2}   "
            f"{r['osc_mean_tier']:>5.2f} "
            f"{(r['peak_gb'] or 0):>5.2f}G"
        )

    (RUN_DIR / "learning_curve_v3.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RUN_DIR / 'learning_curve_v3.json'}")

    # Three-way promotion check
    v11 = rows.get("v1.1 iter 750 (3n)", {})
    v2u = rows.get("v2 untuned", {})
    v11_tier = v11.get("mean_tier", 0)
    v2u_tier = v2u.get("mean_tier", 0)
    v2u_osc = v2u.get("osc_correct", 0)
    print(f"\nReferences: v1.1={v11_tier:.2f}, v2-untuned={v2u_tier:.2f}, v2-untuned oscillation={v2u_osc}/{v2u.get('osc_n',7)}")
    best_v3 = None
    best_v3_tier = -1
    for name, r in rows.items():
        if name.startswith("v3 iter") and not r.get("missing") and r["mean_tier"] > best_v3_tier:
            best_v3_tier = r["mean_tier"]
            best_v3 = name
    if best_v3:
        r = rows[best_v3]
        print(f"v3 best:    {best_v3} mean tier {r['mean_tier']:.2f}, "
              f"foundation {r['foundation_recovery']}/{r['foundation_states']}, "
              f"oscillation {r['osc_correct']}/{r['osc_n']}")
        # Two distinct success criteria for v3
        beats_v2u = r["mean_tier"] > v2u_tier
        keeps_osc = r["osc_correct"] >= v2u_osc
        beats_v11 = r["mean_tier"] > v11_tier
        print(f"  beats v2 untuned tier? {beats_v2u} ({r['mean_tier']:.2f} vs {v2u_tier:.2f})")
        print(f"  keeps v2 untuned osc? {keeps_osc} ({r['osc_correct']} vs {v2u_osc})")
        print(f"  beats v1.1?           {beats_v11} ({r['mean_tier']:.2f} vs {v11_tier:.2f})")
        if beats_v11:
            print("\nPROMOTE: v3 beats v1.1.")
        elif beats_v2u and keeps_osc:
            print("\nPARTIAL: v3 fixes the corpus issue but doesn't beat v1.1; ship v3 as cleaner companion to v1.1.")
        elif keeps_osc:
            print("\nINCONCLUSIVE: v3 preserves oscillation but doesn't beat untuned overall; review which states moved where.")
        else:
            print("\nHOLD: v3 did not fix the regression; the corpus filter is not the bottleneck.")


if __name__ == "__main__":
    main()
