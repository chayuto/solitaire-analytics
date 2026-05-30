#!/usr/bin/env python3
"""Aggregate v4-A LoRA learning-curve evals against the v1.1 baseline.

v4-A differs from v1.1 (the deployed gemma-3n student) in exactly one variable:
the training corpus is reversal-FILTERED instead of raw. Same base, hyperparameters,
patch, runner, scorer, chat template. So the comparison that matters is
v4-A best checkpoint vs v1.1 (and vs v1 untuned as the floor).

Reads posttune_v4a_at*.json produced by sweep_v4a_checkpoints.sh (under
baseline_n20_v4a/), plus the v1 references that already live under baseline_n20/.

Pre-registered bench predictions (locked, see
docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md section 3.3):
  BP1 (primary): best checkpoint mean tier >= 3.15
  BP2:           foundation recovery >= 6/7
  BP3:           oscillation agreement >= 4/7
  BP4:           teacher agreement >= 11/20

Decision gates (section 3.4) combine bench AND full-game (seed 3263196305):
  PROMOTE = full-game fc > 3 AND bench tier >= 3.15
  PARTIAL = full-game fc == 3 AND bench tier >= 3.15
  HOLD    = full-game fc < 3 OR bench tier < 3.15
This scorer evaluates the BENCH half only. If bench tier < 3.15 the gate is
already HOLD regardless of full-game. If bench tier >= 3.15 the final verdict
waits on the full-game fc from play_deck_with_student.py.
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
RUN_DIR = REPO / "gemma4_finetune/baseline_n20_v4a"
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
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    teacher = json.loads(TEACHER_LOOKUP.read_text())
    rows = {}
    rows["v1 untuned (3n)"] = score_file(BASELINE_3N, teacher)
    rows["v1.1 iter 750 (3n)"] = score_file(POSTTUNE_3N_AT750, teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v4a iter {ckpt}"] = score_file(RUN_DIR / f"posttune_v4a_at{ckpt}.json", teacher)

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

    (RUN_DIR / "learning_curve_v4a.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RUN_DIR / 'learning_curve_v4a.json'}")

    # Pre-registered bench predictions vs v1.1.
    v11 = rows.get("v1.1 iter 750 (3n)", {})
    v11_tier = v11.get("mean_tier", 0)
    best_v4a = None
    best_v4a_tier = -1
    for name, r in rows.items():
        if name.startswith("v4a iter") and not r.get("missing") and r["mean_tier"] > best_v4a_tier:
            best_v4a_tier = r["mean_tier"]
            best_v4a = name
    if not best_v4a:
        print("\nNo v4-A checkpoints scored yet. Run sweep_v4a_checkpoints.sh first.")
        return

    r = rows[best_v4a]
    print(f"\nReference: v1.1={v11_tier:.2f} (locked bench baseline tier 3.15, agreement 11/20)")
    print(f"v4-A best: {best_v4a} tier {r['mean_tier']:.2f}, "
          f"foundation {r['foundation_recovery']}/{r['foundation_states']}, "
          f"oscillation {r['osc_correct']}/{r['osc_n']}, agreement {r['agreement']}/{r['n']}")

    bp1 = r["mean_tier"] >= 3.15
    bp2 = r["foundation_recovery"] >= 6
    bp3 = r["osc_correct"] >= 4
    bp4 = r["agreement"] >= 11
    print(f"  BP1 best tier >= 3.15:        {bp1} ({r['mean_tier']:.2f})")
    print(f"  BP2 foundation recovery >= 6: {bp2} ({r['foundation_recovery']}/{r['foundation_states']})")
    print(f"  BP3 oscillation agree >= 4:   {bp3} ({r['osc_correct']}/{r['osc_n']})")
    print(f"  BP4 teacher agreement >= 11:  {bp4} ({r['agreement']}/{r['n']})")

    print()
    if not bp1:
        print("HOLD: bench tier < 3.15. The gate triggers HOLD regardless of full-game.")
        print("      Close the corpus-filter program; pivot to harvester-side levers.")
    else:
        print("BENCH CLEARS BP1 (tier >= 3.15). Final verdict pends full-game fc on seed 3263196305:")
        print("      fc > 3  -> PROMOTE (ship v4-A as canonical, supersede v1.1)")
        print("      fc == 3 -> PARTIAL (v1.1 stays canonical; documented tie)")
        print("      fc < 3  -> HOLD")
        print("      Run: .venv/bin/python gemma4_finetune/play_deck_with_student.py \\")
        print("             --deck-seed 3263196305 \\")
        print("             --model-id mlx-community/gemma-3n-E2B-it-text-4bit-dwq \\")
        print(f"             --adapter-path gemma4_finetune/adapters_v4a/_eval_stage  # stage the {best_v4a} ckpt")


if __name__ == "__main__":
    main()
