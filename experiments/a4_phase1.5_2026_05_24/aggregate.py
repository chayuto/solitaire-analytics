"""Aggregate Phase 1.5 scored runs into per-arm, per-state, per-category breakdowns.

Two arms only (C0 vs A4) — produces paired bootstrap CIs for the four
pre-registered hypotheses (H1/H2/H3/H4).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).parent


def load_scored(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*/*/*.scored.json"))]


def paired_bootstrap_ci(
    control: list[float], treatment: list[float], n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    if len(control) != len(treatment) or not control:
        return 0.0, 0.0, 0.0
    diffs = [t - c for c, t in zip(control, treatment)]
    rng = random.Random(seed)
    n = len(diffs)
    boot = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(s) / n)
    boot.sort()
    return sum(diffs) / n, boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(EXP_DIR / "raw"))
    parser.add_argument("--out-json", default=str(EXP_DIR / "results.json"))
    parser.add_argument("--out-md", default=str(EXP_DIR / "results.md"))
    args = parser.parse_args()

    rows = load_scored(Path(args.raw))
    print(f"loaded {len(rows)} scored runs")

    arms = sorted({r["arm"] for r in rows})
    states = sorted({r["state_id"] for r in rows})
    categories = {r["state_id"]: r["category"] for r in rows}
    has_found = {r["state_id"]: r["has_foundation_move"] for r in rows}

    by_arm_state: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_arm_state[(r["arm"], r["state_id"])].append(r)

    # === Per-arm overall stats ===
    arm_summary = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        tiers = [r["tier_score"] for r in ar]
        chars = [r["response_chars"] for r in ar]
        arm_summary[arm] = {
            "n": len(ar),
            "tier_mean": statistics.mean(tiers),
            "tier_std": statistics.stdev(tiers) if len(tiers) > 1 else 0.0,
            "json_valid_pct": sum(r["json_valid"] for r in ar) / len(ar),
            "reversal_pct": sum(r["chose_reversal"] for r in ar) / len(ar),
            "response_chars_mean": statistics.mean(chars),
            "response_chars_std": statistics.stdev(chars) if len(chars) > 1 else 0.0,
        }

    # === Aligned paired vectors ===
    def aligned_tiers(arm: str, state_ids: list[str]) -> list[float]:
        out = []
        for sid in state_ids:
            runs = sorted(by_arm_state[(arm, sid)], key=lambda r: r["run"])
            out.extend([r["tier_score"] for r in runs])
        return out

    # H1: A4 Δ tier overall
    overall_c0 = aligned_tiers("C0", states)
    overall_a4 = aligned_tiers("A4", states)
    h1_delta, h1_lo, h1_hi = paired_bootstrap_ci(overall_c0, overall_a4)

    # H2: A4 Δ tier on oscillation states
    osc_states = [s for s in states if categories[s] == "oscillation"]
    osc_c0 = aligned_tiers("C0", osc_states)
    osc_a4 = aligned_tiers("A4", osc_states)
    h2_delta, h2_lo, h2_hi = paired_bootstrap_ci(osc_c0, osc_a4)

    # H3: A4 endgame replication — but no endgame states in this bench
    h3 = {"note": "no endgame states in post-cutover bench; H3 untestable"}

    # H4: A4 vs C0 output size delta
    a4_chars = [r["response_chars"] for r in rows if r["arm"] == "A4"]
    c0_chars = [r["response_chars"] for r in rows if r["arm"] == "C0"]
    h4_delta_pct = (
        100 * (statistics.mean(a4_chars) - statistics.mean(c0_chars)) /
        statistics.mean(c0_chars)
    )

    # === Per-category breakdown ===
    per_category = {}
    cat_names = sorted({categories[s] for s in states})
    for arm in arms:
        per_category[arm] = {}
        for cat in cat_names:
            cr = [r for r in rows if r["arm"] == arm and r["category"] == cat]
            if not cr:
                continue
            per_category[arm][cat] = {
                "n": len(cr),
                "tier_mean": statistics.mean(r["tier_score"] for r in cr),
                "reversal_pct": sum(r["chose_reversal"] for r in cr) / len(cr),
            }

    # === Per-state picks ===
    per_state = {}
    for arm in arms:
        per_state[arm] = {}
        for sid in states:
            runs = by_arm_state[(arm, sid)]
            if not runs:
                continue
            tiers = [r["tier_score"] for r in runs]
            per_state[arm][sid] = {
                "tier_mean": statistics.mean(tiers),
                "picks": sorted(r["chosen_index"] for r in runs),
                "tiers": [r["tier"] for r in runs],
                "category": categories[sid],
                "has_foundation_move": has_found[sid],
            }

    # === Oscillation-with-foundation focus (smoking gun analog) ===
    osc_found_states = [s for s in osc_states if has_found[s]]
    osc_found_c0 = aligned_tiers("C0", osc_found_states)
    osc_found_a4 = aligned_tiers("A4", osc_found_states)
    osc_found_delta, osc_found_lo, osc_found_hi = paired_bootstrap_ci(
        osc_found_c0, osc_found_a4
    )

    aggregate = {
        "n_total": len(rows),
        "arms": arm_summary,
        "hypotheses": {
            "H1_overall_delta": {
                "delta": h1_delta, "ci_lo": h1_lo, "ci_hi": h1_hi,
                "passes_g1": h1_delta >= 0.3 and h1_lo > 0,
            },
            "H2_oscillation_delta": {
                "delta": h2_delta, "ci_lo": h2_lo, "ci_hi": h2_hi,
                "passes_safety": h2_delta >= -0.3,
            },
            "H3_endgame": h3,
            "H4_output_size_delta_pct": {
                "pct": h4_delta_pct,
                "passes_sanity": abs(h4_delta_pct) < 5,
            },
            "oscillation_with_foundation_delta": {
                "n_states": len(osc_found_states),
                "delta": osc_found_delta, "ci_lo": osc_found_lo, "ci_hi": osc_found_hi,
            },
        },
        "per_category": per_category,
        "per_state": per_state,
        "states": [
            {"state_id": s, "category": categories[s], "has_foundation_move": has_found[s]}
            for s in states
        ],
    }
    Path(args.out_json).write_text(json.dumps(aggregate, indent=2))
    print(f"wrote {args.out_json}")

    # === Markdown report ===
    md = []
    md.append("# A4 Phase 1.5 — results")
    md.append("")
    md.append(f"**N**: {len(rows)} scored runs ({len(states)} states × {len(arms)} arms × 3 runs)")
    md.append("**Proxy**: Claude Haiku 4.5 via subagent simulation")
    md.append("**Date**: 2026-05-24")
    md.append("")
    md.append("## Per-arm overall")
    md.append("")
    md.append("| arm | n | mean tier ±σ | JSON ok | rev% | resp chars ±σ |")
    md.append("|---|---:|---|---:|---:|---|")
    for arm in arms:
        s = arm_summary[arm]
        md.append(
            f"| {arm} | {s['n']} | {s['tier_mean']:.2f} ± {s['tier_std']:.2f} | "
            f"{s['json_valid_pct']:.0%} | {s['reversal_pct']:.0%} | "
            f"{s['response_chars_mean']:.0f} ± {s['response_chars_std']:.0f} |"
        )
    md.append("")
    md.append("Tier scoring: foundation=6, reveal=5, waste_play=4, shuffle=2, draw=1, recycle=1, illegal=0.")
    md.append("")

    md.append("## Pre-registered hypotheses")
    md.append("")
    h1 = aggregate["hypotheses"]["H1_overall_delta"]
    h2 = aggregate["hypotheses"]["H2_oscillation_delta"]
    h4 = aggregate["hypotheses"]["H4_output_size_delta_pct"]
    of = aggregate["hypotheses"]["oscillation_with_foundation_delta"]
    md.append("| H | Test | Result | Threshold | Pass? |")
    md.append("|---|---|---|---|---|")
    md.append(
        f"| H1 | A4 Δ tier overall | {h1['delta']:+.2f} [95% CI {h1['ci_lo']:+.2f}, {h1['ci_hi']:+.2f}] | Δ ≥ +0.3 AND CI excludes 0 | {'**PASS**' if h1['passes_g1'] else '**FAIL**'} |"
    )
    md.append(
        f"| H2 | A4 Δ tier on oscillation | {h2['delta']:+.2f} [95% CI {h2['ci_lo']:+.2f}, {h2['ci_hi']:+.2f}] | Δ ≥ −0.3 (safety) | {'**PASS**' if h2['passes_safety'] else '**FAIL**'} |"
    )
    md.append("| H3 | A4 endgame ≥ 5.0 | n/a — no endgame states | (untestable) | — |")
    md.append(
        f"| H4 | output-size sanity | {h4['pct']:+.1f}% | |Δ| < 5% | {'**PASS**' if h4['passes_sanity'] else '**FAIL**'} |"
    )
    md.append("")
    md.append("### Oscillation-with-foundation focus")
    md.append("")
    md.append(
        f"Restricted to the {of['n_states']} oscillation states with `tableau_to_foundation` available (the H2 test ground):"
    )
    md.append("")
    md.append(
        f"- A4 Δ vs C0: **{of['delta']:+.2f}** tier [95% CI {of['ci_lo']:+.2f}, {of['ci_hi']:+.2f}]"
    )
    md.append("")

    md.append("## Per-category breakdown (mean tier)")
    md.append("")
    md.append("| arm | " + " | ".join(cat_names) + " |")
    md.append("|---|" + "|".join(["---:"] * len(cat_names)) + "|")
    for arm in arms:
        row = [arm]
        for cat in cat_names:
            if cat in per_category[arm]:
                row.append(f"{per_category[arm][cat]['tier_mean']:.2f}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Per-state picks")
    md.append("")
    md.append("| state | cat | found? | C0 picks | C0 tier | A4 picks | A4 tier |")
    md.append("|---|---|:---:|---|---:|---|---:|")
    for sid in states:
        c0s = per_state["C0"][sid]
        a4s = per_state["A4"][sid]
        md.append(
            f"| `{sid[:18]}` | {categories[sid][:5]} | "
            f"{'Y' if has_found[sid] else 'N'} | "
            f"{c0s['picks']} | {c0s['tier_mean']:.2f} | "
            f"{a4s['picks']} | {a4s['tier_mean']:.2f} |"
        )
    md.append("")

    Path(args.out_md).write_text("\n".join(md))
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
