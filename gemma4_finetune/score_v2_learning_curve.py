#!/usr/bin/env python3
"""Aggregate v2 LoRA learning-curve evals into a single comparison table.

Reads:
  baseline_n20_gemma4_text/baseline_n20_gemma4_text.json   (untuned v2 base)
  baseline_n20_gemma4_text/posttune_at{250,500,750,1000}.json  (each ckpt)

For each row, recomputes the tier score via the shared classify_pick. Emits
a side-by-side table plus a written summary at:
  baseline_n20_gemma4_text/learning_curve.json
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
    return {
        "n": n,
        "json_valid": data["json_valid_count"],
        "agreement": data["agreement_count"],
        "illegal": illegal,
        "mean_tier": round(tier_sum / n, 2),
        "foundation_states": foundation_states,
        "foundation_recovery": foundation_recovery,
        "peak_gb": data.get("overall_peak_gb"),
        "mean_call_sec": data.get("mean_call_seconds"),
    }


def main() -> None:
    teacher = json.loads(TEACHER_LOOKUP.read_text())
    rows = {}
    rows["v2 untuned"] = score_file(RUN_DIR / "baseline_n20_gemma4_text.json", teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v2 iter {ckpt}"] = score_file(RUN_DIR / f"posttune_at{ckpt}.json", teacher)
    rows["v1 untuned (3n)"] = score_file(BASELINE_3N, teacher)
    rows["v1.1 iter 750 (3n)"] = score_file(POSTTUNE_3N_AT750, teacher)

    teacher_mean = 3.42
    print(f"{'config':<22} {'json':>6} {'illegal':>8} {'agree':>6} {'tier':>6} {'gap':>7} {'fnd':>5} {'peak':>6} {'sec':>6}")
    print("-" * 78)
    for name, r in rows.items():
        if r.get("missing"):
            print(f"{name:<22} (missing: {r['path']})")
            continue
        gap = r["mean_tier"] - teacher_mean
        print(
            f"{name:<22} {r['json_valid']:>3}/{r['n']:<2} "
            f"{r['illegal']:>3}/{r['n']:<2}  "
            f"{r['agreement']:>3}/{r['n']:<2} "
            f"{r['mean_tier']:>6.2f} {gap:>+7.2f} "
            f"{r['foundation_recovery']:>2}/{r['foundation_states']:<2} "
            f"{(r['peak_gb'] or 0):>5.2f}G "
            f"{(r['mean_call_sec'] or 0):>5.1f}s"
        )

    (RUN_DIR / "learning_curve.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RUN_DIR / 'learning_curve.json'}")

    # Promotion check vs v1.1
    v11 = rows["v1.1 iter 750 (3n)"]
    if not v11.get("missing"):
        v11_tier = v11["mean_tier"]
        v11_fr = v11["foundation_recovery"]
        v11_fs = v11["foundation_states"]
        print(f"\nv1.1 reference: mean tier {v11_tier:.2f}, gap {v11_tier - teacher_mean:+.2f}, "
              f"foundation {v11_fr}/{v11_fs}")
        best_name = None
        best_tier = -1
        for name, r in rows.items():
            if name.startswith("v2 iter") and not r.get("missing") and r["mean_tier"] > best_tier:
                best_tier = r["mean_tier"]
                best_name = name
        if best_name:
            best = rows[best_name]
            print(f"v2 best:         {best_name} mean tier {best['mean_tier']:.2f}, "
                  f"gap {best['mean_tier'] - teacher_mean:+.2f}, "
                  f"foundation {best['foundation_recovery']}/{best['foundation_states']}")
            if best["mean_tier"] > v11_tier:
                print(f"\nPROMOTE: v2 strictly beats v1.1 by {best['mean_tier'] - v11_tier:+.2f} mean tier.")
            else:
                print(f"\nHOLD: v2 best ({best['mean_tier']:.2f}) does not beat v1.1 ({v11_tier:.2f}). "
                      f"Delta {best['mean_tier'] - v11_tier:+.2f}.")


if __name__ == "__main__":
    main()
