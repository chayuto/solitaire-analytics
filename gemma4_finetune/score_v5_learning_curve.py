#!/usr/bin/env python3
"""Aggregate v5 (won-games-only, Gemma 4 E2B) learning-curve evals against the
v2 baselines. Clone of score_v3_learning_curve.py structure; reads the
posttune_v5_at*.json the v5 sweep writes into baseline_n20_gemma4_text/.

v5 differs from v2 in one variable: corpus = won-games-only vs all-success.
The pre-registered question (docs/reports/20260530_v5_wononly_preregistration.md)
is whether the game-level win filter PRESERVES untuned oscillation discipline
(which v2 training broke) without costing tier.

Pre-registered bench predictions (locked):
  BP1 (primary): best-checkpoint oscillation agreement >= 6/7  (preserve v2-untuned)
  BP2:           best-checkpoint mean tier >= 2.55             (no regression vs v2-untuned)
  BP3:           best-checkpoint mean tier > 2.45              (beats v2-best-trained)
  BP4:           best-checkpoint JSON validity >= 19/20

Decision gates: PROMOTE = BP1 and BP2; PARTIAL = BP1 only; HOLD = not BP1.
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
    rows["v2 untuned"] = score_file(RUN_DIR / "baseline_n20_gemma4_text.json", teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v2 iter {ckpt}"] = score_file(RUN_DIR / f"posttune_at{ckpt}.json", teacher)
    for ckpt in (250, 500, 750, 1000):
        rows[f"v5 iter {ckpt}"] = score_file(RUN_DIR / f"posttune_v5_at{ckpt}.json", teacher)

    teacher_mean = 3.42
    print(f"{'config':<16} {'json':>6} {'ill':>5} {'agr':>5} {'tier':>5} {'gap':>6} {'fnd':>5} {'osc_agr':>8} {'osc_t':>6} {'peak':>6}")
    print("-" * 84)
    for name, r in rows.items():
        if r.get("missing"):
            print(f"{name:<16} (missing: {Path(r['path']).name})")
            continue
        gap = r["mean_tier"] - teacher_mean
        print(
            f"{name:<16} {r['json_valid']:>3}/{r['n']:<2} "
            f"{r['illegal']:>2}/{r['n']:<2} "
            f"{r['agreement']:>2}/{r['n']:<2} "
            f"{r['mean_tier']:>5.2f} {gap:>+6.2f} "
            f"{r['foundation_recovery']:>2}/{r['foundation_states']:<2} "
            f"{r['osc_correct']:>3}/{r['osc_n']:<2}   "
            f"{r['osc_mean_tier']:>5.2f} "
            f"{(r['peak_gb'] or 0):>5.2f}G"
        )

    (RUN_DIR / "learning_curve_v5.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RUN_DIR / 'learning_curve_v5.json'}")

    v2u = rows.get("v2 untuned", {})
    v2u_tier = v2u.get("mean_tier", 0)
    v2u_osc = v2u.get("osc_correct", 0)
    best_v5, best_tier = None, -1
    for name, r in rows.items():
        if name.startswith("v5 iter") and not r.get("missing") and r["mean_tier"] > best_tier:
            best_tier = r["mean_tier"]
            best_v5 = name
    if not best_v5:
        print("\nNo v5 checkpoints scored yet. Run sweep_v5_checkpoints.sh first.")
        return

    r = rows[best_v5]
    print(f"\nReferences: v2-untuned tier {v2u_tier:.2f}, oscillation {v2u_osc}/7")
    print(f"v5 best: {best_v5} tier {r['mean_tier']:.2f}, oscillation {r['osc_correct']}/{r['osc_n']}, "
          f"foundation {r['foundation_recovery']}/{r['foundation_states']}, json {r['json_valid']}/{r['n']}")

    bp1 = r["osc_correct"] >= 6
    bp2 = r["mean_tier"] >= 2.55
    bp3 = r["mean_tier"] > 2.45
    bp4 = r["json_valid"] >= 19
    print(f"  BP1 oscillation >= 6/7:   {bp1} ({r['osc_correct']}/7)   [PRIMARY: preserve v2-untuned]")
    print(f"  BP2 tier >= 2.55:         {bp2} ({r['mean_tier']:.2f})")
    print(f"  BP3 tier > 2.45:          {bp3} ({r['mean_tier']:.2f})")
    print(f"  BP4 json >= 19/20:        {bp4} ({r['json_valid']}/20)")

    print()
    if bp1 and bp2:
        print("PROMOTE: won-games filter preserved oscillation AND held tier. Ship v5 trained.")
    elif bp1:
        print("PARTIAL: oscillation preserved but tier slipped. Untuned stays ship; filter validated, needs more won games.")
    else:
        print("HOLD: oscillation < 6/7. The won-only corpus does not preserve discipline either;")
        print("      conclude Gemma 4 distillation does not beat untuned by any corpus route; pivot to prompt track + ship untuned.")


if __name__ == "__main__":
    main()
