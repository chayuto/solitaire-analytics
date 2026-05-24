# Prompt audit experiment — results

**N**: 75 scored runs (5 states × 5 arms × 3 runs)
**Proxy model**: Claude Haiku 4.5 (subagent simulation)
**Date**: 2026-05-24

## Per-arm overall

| arm | n | mean tier ±σ | JSON ok | rev % | resp chars ±σ |
|---|---:|---|---:|---:|---|
| A1 | 15 | 3.13 ± 2.07 | 100% | 7% | 1495 ± 295 |
| A2 | 15 | 3.20 ± 2.01 | 100% | 7% | 87 ± 2 |
| A3 | 15 | 3.53 ± 2.03 | 100% | 7% | 86 ± 2 |
| A4 | 15 | 3.93 ± 2.02 | 100% | 7% | 1456 ± 225 |
| C0 | 15 | 3.67 ± 2.23 | 100% | 0% | 1615 ± 324 |

Tier scoring: foundation=6, reveal=5, waste_play=4, shuffle=2, draw=1, recycle=1, illegal=0.

## Δ tier-score vs C0 (paired bootstrap, 5000 resamples, 95% CI)

| arm | mean Δ | 95% CI lo | 95% CI hi | crosses 0? |
|---|---:|---:|---:|---|
| A1 | -0.53 | -1.67 | +0.53 | yes |
| A2 | -0.47 | -1.53 | +0.60 | yes |
| A3 | -0.13 | -0.73 | +0.27 | yes |
| A4 | +0.27 | -0.53 | +1.13 | yes |

## Per-category breakdown (mean tier)

| arm | early | endgame | midgame | oscillation |
|---|---:|---:|---:|---:|
| A1 | 1.33 | 4.67 | 5.00 | 2.33 |
| A2 | 1.00 | 6.00 | 5.00 | 2.00 |
| A3 | 1.67 | 4.67 | 5.00 | 3.17 |
| A4 | 2.00 | 6.00 | 5.00 | 3.33 |
| C0 | 1.00 | 4.67 | 5.00 | 3.83 |

## Per-state picks (5 states × 5 arms × 3 runs)

### `early-bddc98b1eaf9` (early)

| arm | picks | unique | tiers | mean tier |
|---|---|---:|---|---:|
| A1 | [1, 2, 2] | 2 | draw,draw,shuffle | 1.33 |
| A2 | [2, 2, 2] | 1 | draw,draw,draw | 1.00 |
| A3 | [1, 1, 2] | 2 | draw,shuffle,shuffle | 1.67 |
| A4 | [0, 2, 2] | 2 | waste_play,draw,draw | 2.00 |
| C0 | [2, 2, 2] | 1 | draw,draw,draw | 1.00 |

### `endgame-0f5f9dce20df` (endgame)

| arm | picks | unique | tiers | mean tier |
|---|---|---:|---|---:|
| A1 | [0, 4, 4] | 2 | foundation,foundation,shuffle | 4.67 |
| A2 | [4, 4, 7] | 2 | foundation,foundation,foundation | 6.00 |
| A3 | [0, 4, 4] | 2 | foundation,shuffle,foundation | 4.67 |
| A4 | [4, 4, 7] | 2 | foundation,foundation,foundation | 6.00 |
| C0 | [2, 4, 4] | 2 | foundation,shuffle,foundation | 4.67 |

### `midgame-5150510b6037` (midgame)

| arm | picks | unique | tiers | mean tier |
|---|---|---:|---|---:|
| A1 | [3, 3, 3] | 1 | reveal,reveal,reveal | 5.00 |
| A2 | [3, 3, 3] | 1 | reveal,reveal,reveal | 5.00 |
| A3 | [3, 3, 3] | 1 | reveal,reveal,reveal | 5.00 |
| A4 | [3, 3, 3] | 1 | reveal,reveal,reveal | 5.00 |
| C0 | [3, 3, 3] | 1 | reveal,reveal,reveal | 5.00 |

### `oscillation-1f5b47ba5d96` (oscillation)

| arm | picks | unique | tiers | mean tier |
|---|---|---:|---|---:|
| A1 | [1, 3, 4] | 3 | draw,shuffle,shuffle | 1.67 |
| A2 | [4, 4, 4] | 1 | shuffle,shuffle,shuffle | 2.00 |
| A3 | [0, 0, 3] | 2 | shuffle,shuffle,draw | 1.67 |
| A4 | [4, 4, 4] | 1 | shuffle,shuffle,shuffle | 2.00 |
| C0 | [1, 3, 4] | 3 | shuffle,shuffle,draw | 1.67 |

### `oscillation-bffb5d9cc178` (oscillation)

| arm | picks | unique | tiers | mean tier |
|---|---|---:|---|---:|
| A1 | [1, 2, 3] | 3 | draw,foundation,shuffle | 3.00 |
| A2 | [3, 4, 4] | 2 | shuffle,shuffle,shuffle | 2.00 |
| A3 | [2, 2, 3] | 2 | shuffle,foundation,foundation | 4.67 |
| A4 | [2, 2, 3] | 2 | shuffle,foundation,foundation | 4.67 |
| C0 | [2, 2, 2] | 1 | foundation,foundation,foundation | 6.00 |

## Headline observations

- **Schema slim wins (A2, A3): output size 87 vs C0 1615 bytes (5% of control).** Confirms anti-pattern #6 finding: prose CoT fields balloon output ~19× without changing the structured decision.
- **Best mean tier**: A4 at 3.93. But Δ vs C0 paired CI usually crosses 0 — see deltas table.
- **JSON validity 100% across all arms** — the schema simplification (A2/A3) doesn't compromise validity; Haiku handles both schemas cleanly.