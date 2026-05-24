# A4 Phase 1.5 — results

**N**: 120 scored runs (20 states × 2 arms × 3 runs)
**Proxy**: Claude Haiku 4.5 via subagent simulation
**Date**: 2026-05-24

## Per-arm overall

| arm | n | mean tier ±σ | JSON ok | rev% | resp chars ±σ |
|---|---:|---|---:|---:|---|
| A4 | 60 | 2.40 ± 2.01 | 100% | 0% | 1623 ± 356 |
| C0 | 60 | 2.52 ± 2.04 | 100% | 0% | 1523 ± 324 |

Tier scoring: foundation=6, reveal=5, waste_play=4, shuffle=2, draw=1, recycle=1, illegal=0.

## Pre-registered hypotheses

| H | Test | Result | Threshold | Pass? |
|---|---|---|---|---|
| H1 | A4 Δ tier overall | -0.12 [95% CI -0.45, +0.18] | Δ ≥ +0.3 AND CI excludes 0 | **FAIL** |
| H2 | A4 Δ tier on oscillation | -0.24 [95% CI -0.76, +0.10] | Δ ≥ −0.3 (safety) | **PASS** |
| H3 | A4 endgame ≥ 5.0 | n/a — no endgame states | (untestable) | — |
| H4 | output-size sanity | +6.6% | |Δ| < 5% | **FAIL** |

### Oscillation-with-foundation focus

Restricted to the 4 oscillation states with `tableau_to_foundation` available (the H2 test ground):

- A4 Δ vs C0: **-0.42** tier [95% CI -1.33, +0.17]

## Per-category breakdown (mean tier)

| arm | early | midgame | oscillation |
|---|---:|---:|---:|
| A4 | 2.53 | 1.79 | 3.00 |
| C0 | 2.60 | 1.83 | 3.24 |

## Per-state picks

| state | cat | found? | C0 picks | C0 tier | A4 picks | A4 tier |
|---|---|:---:|---|---:|---|---:|
| `early-1dbcd96c5df6` | early | N | [1, 1, 1] | 1.00 | [1, 1, 1] | 1.00 |
| `early-3687a40eda7b` | early | Y | [0, 0, 0] | 2.00 | [0, 0, 3] | 3.33 |
| `early-81deee72436d` | early | N | [0, 2, 2] | 2.00 | [2, 2, 2] | 1.00 |
| `early-acd9af4ae639` | early | N | [0, 1, 1] | 2.00 | [0, 0, 1] | 3.00 |
| `early-e6291973dd07` | early | Y | [2, 2, 2] | 6.00 | [0, 2, 2] | 4.33 |
| `midgame-031d9c9e3f` | midga | N | [0, 0, 0] | 2.00 | [0, 0, 0] | 2.00 |
| `midgame-0cef3a609d` | midga | N | [0, 0, 0] | 2.00 | [0, 0, 0] | 2.00 |
| `midgame-0d463176c4` | midga | N | [0, 0, 0] | 1.00 | [0, 0, 0] | 1.00 |
| `midgame-230df7b716` | midga | N | [1, 1, 1] | 1.00 | [1, 1, 1] | 1.00 |
| `midgame-4ab5735a4f` | midga | Y | [0, 1, 1] | 4.33 | [0, 1, 1] | 4.33 |
| `midgame-81dc0fb023` | midga | N | [1, 1, 1] | 1.00 | [1, 1, 1] | 1.00 |
| `midgame-823116ccc0` | midga | N | [1, 1, 1] | 2.00 | [1, 1, 1] | 2.00 |
| `midgame-a658537fe2` | midga | N | [0, 1, 1] | 1.33 | [1, 1, 1] | 1.00 |
| `oscillation-026f31` | oscil | Y | [0, 0, 0] | 6.00 | [0, 0, 0] | 6.00 |
| `oscillation-21cc52` | oscil | N | [0, 0, 0] | 1.00 | [0, 0, 0] | 1.00 |
| `oscillation-30700e` | oscil | Y | [0, 0, 0] | 6.00 | [0, 0, 0] | 6.00 |
| `oscillation-a774c0` | oscil | Y | [1, 1, 3] | 1.67 | [1, 1, 3] | 1.67 |
| `oscillation-bfb84a` | oscil | Y | [1, 1, 1] | 6.00 | [1, 1, 2] | 4.33 |
| `oscillation-d0ff55` | oscil | N | [0, 0, 0] | 1.00 | [0, 0, 0] | 1.00 |
| `oscillation-d729a3` | oscil | N | [3, 3, 3] | 1.00 | [3, 3, 3] | 1.00 |
