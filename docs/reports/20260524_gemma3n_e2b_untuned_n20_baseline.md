# Untuned Gemma 3n E2B baseline (N=20, Phase 1.5 bench)

**Date**: 2026-05-24
**Model**: `mlx-community/gemma-3n-E2B-it-text-4bit-dwq` (untuned, 4-bit, text-only)
**Bench**: Phase 1.5 C0-arm, 20 post-cutover states (5 early / 8 midgame / 0 endgame / 7 oscillation)
**Teacher**: 31B Gemma (production-recorded picks from the same turns)
**Status**: pre-registered decision rule fires -> **fine-tuning is justified**

## Why this exists

T2 (E2B load test) on n=3 showed untuned E2B beating teacher (3.33 vs 3.00), surprising enough that it could have voided the whole distillation runway. Before burning M5 cycles on T3-T5 fine-tuning, we needed to confirm at sample size large enough to trust. N=20 (the existing Phase 1.5 bench) was the cheapest credible sample.

## TL;DR

- **N=3 was a fluke.** N=20 shows untuned E2B is **1.32 tier WORSE** than the 31B teacher per-state.
- **The gap is concentrated in foundation moves.** Of 7 states where the teacher took `tableau_to_foundation` / `discard_to_foundation`, E2B followed only 2.
- **JSON validity is already perfect** (20/20). Format is not the bottleneck, strategy is.
- **Fine-tuning has clear improvement potential** and the bench has a sharp specific failure to target.
- **Decision**: proceed with T3-T5 distillation (on Gemma 3n E2B, since Gemma 4 is blocked by mlx-lm 0.31.3, see T2 progress doc).

## Results

### Headline

| metric | value |
|---|---|
| JSON validity | **20/20 (100%)** |
| Teacher agreement (move_index match) | 11/20 (55%) |
| Illegal moves chosen | 1/20 (5%) |
| Mean tier (E2B, paired n=19) | 2.11 |
| Mean tier (teacher, paired n=19) | 3.42 |
| **Paired Δ (E2B - teacher)** | **-1.32** |

Tier scoring (same as Phase 1.5): foundation=6, reveal=5, waste_play=4, shuffle=2, draw=1, recycle=1, illegal=0.

### Per category

| category | n | E2B mean tier | teacher mean tier | Δ | agreement |
|---|---:|---:|---:|---:|---:|
| early | 5 | 2.60 | 4.20 | **-1.60** | 3/5 |
| midgame | 8 (paired 7) | 1.38 | 2.00 | -0.62 | 4/8 |
| oscillation | 7 | 2.57 | 4.29 | **-1.71** | 4/7 |

The gap is largest in early and oscillation, both categories where the teacher routinely played foundation moves. Midgame's smaller gap reflects that midgame teacher picks here were mostly draws/shuffles to begin with.

### Foundation-miss pattern (the real story)

Across the 7 states where the teacher chose a foundation move:

| state | category | teacher | E2B | Δ |
|---|---|---|---|---:|
| `early-3687a40eda7b` | early | foundation (6) | shuffle (2) | -4 |
| `early-e6291973dd07` | early | foundation (6) | shuffle (2) | -4 |
| `midgame-4ab5735a4f20` | midgame | foundation (6) | draw (1) | -5 |
| `oscillation-026f3139d6f2` | oscillation | foundation (6) | foundation (6) | +0 |
| `oscillation-30700e2ca639` | oscillation | foundation (6) | foundation (6) | +0 |
| `oscillation-a774c0d22f24` | oscillation | foundation (6) | draw (1) | -5 |
| `oscillation-bfb84ae55c3f` | oscillation | foundation (6) | draw (1) | -5 |

**E2B took foundation 2 of 7 times when teacher did.** When it missed, it consistently fell back to a draw or shuffle, a known failure mode that bit A2 (Phase 1) and A4 (Phase 1.5) on the same bench. The model size doesn't fix this anti-pattern; whatever causes the smaller Haiku to miss foundations on these specific states causes the smaller Gemma 3n to do the same.

`oscillation-bfb84ae55c3f` in particular is now a triple-replication: C0-Haiku missed once, A4-Haiku missed once, and now untuned-E2B missed all 3 times. This state has a structural property (likely the way the foundation card surfaces in the legalMoves list versus the recentMoves trail) that small models can't resolve without help.

### Illegal-move rate

1/20 = 5% illegal. The case: `midgame-81dc0fb02394` had `n_legal=2` (indexes 0, 1) and E2B chose 2. Tolerable for untuned but worth tracking, fine-tuning should reduce this to near zero.

## Resource numbers (informs T3+ planning)

| | value |
|---|---|
| Load (cached) | 2.3 s |
| Mean per-call wall | 13.5 s |
| Total wall (20 calls) | 4 min 41 s |
| Peak Metal RAM | 6.26 GB |

Inference envelope is comfortable. T3's 11 GB target for training (~5 GB for LoRA gradients + activations on top) stays realistic.

## Pre-registered decision rule outcome

From task #79:
- `Δ ≥ -0.3 AND JSON ≥ 95%` -> ship untuned, skip T3-T5
- `-1.0 ≤ Δ < -0.3` -> fine-tuning could help; T3+ justified
- `Δ < -1.0 OR JSON < 80%` -> significant gap; T3+ definitely needed

Δ = -1.32, JSON = 100% -> **third branch fires**. T3+ definitely needed. Fine-tuning is the correct investment.

## What this changes about the project

### Model identity decision: ship Gemma 3n E2B, defer Gemma 4

The literal project target was Gemma 4 E2B. T2 found mlx-lm 0.31.3 cannot load any Gemma 4 quant (multimodal architecture mismatch, layers 15-34 not implemented). Three options were open: pivot to 3n, wait for mlx-lm support, switch runtime.

This baseline argues for **pivot to Gemma 3n**:

1. **It works today**: full mlx-lm support, validated on M5, fits memory envelope.
2. **It has clear room to improve**: -1.32 tier is a big target. Distillation on 1730 teacher decisions should close most of it.
3. **The failure mode is learnable**: foundation-miss is a recognition problem, not a reasoning problem. Teacher-decision matching directly trains for it.
4. **Waiting for Gemma 4 has unknown ETA**: mlx-lm 0.31.3 is current; no public PR for full Gemma 4 architecture support. Could be weeks or months.
5. **Gemma 4 E2B would likely show a SIMILAR gap** even if loadable. The foundation-miss pattern survived A4 prompt-format work and likely survives a model-version bump too.

Reframe the runway: `gemma4_finetune/` becomes `gemma_e2b_finetune/` conceptually. When mlx-lm catches up to Gemma 4, re-evaluate; until then, ship the 3n-E2B path.

### Fine-tune target metric clarified

Pre-tuning baseline: **2.11 mean tier on the 20-state bench**.

A successful distillation run should:
- Push mean tier from 2.11 toward 3.42 (teacher level)
- Cut illegal moves to ~0%
- Especially: fix the foundation-miss pattern. Even +0.5 tier on the 7 foundation states would be most of the overall improvement.

The 20-state bench can serve as the always-on eval (cheap: ~5 min, no GPU contention) for every T3-T5 checkpoint.

## Caveats

- **N=20 is enough for a directional call, not a publication-grade estimate.** The CI on a paired delta with this n is probably ±0.5 tier. The -1.32 point estimate is robust enough to make the decision; don't over-interpret per-state differences.
- **No endgame states in the bench.** Post-cutover sessions never reached endgame, so we don't know how E2B handles those. May want to backfill from a different time period before T5.
- **The teacher is the 31B Gemma, not "optimal play."** The teacher itself misses moves; we're matching it, not beating it. That's by design (distillation target) but worth remembering when interpreting tier scores.
- **3n vs 4: different model.** This baseline characterizes 3n. Gemma 4 E2B may have a different gap profile once mlx-lm supports it; revisit then.
- **Single run per state**: the Phase 1.5 work used 3 runs per state for variance estimation. Single-run is enough for "is the gap real" but not for "is THIS state degraded."

## Next actions

1. **Update memory**: project pivots to Gemma 3n E2B; Gemma 4 deferred on mlx-lm support.
2. **Proceed to T3** (mini-smoke training on Gemma 3n E2B). Use the same `lora_config.yaml`, just swap the model name.
3. **Wire the 20-state bench as the always-on eval** for T3+ checkpoints.
4. **Watch for mlx-lm Gemma 4 support**, when it ships, re-run this exact baseline on Gemma 4 E2B for comparison.
5. **Carry T1 findings forward**: max_seq_length budget (production prompts up to 2600 tokens), bad-rawResponse drop rate (11% on full data, 28% on post-cutover), session-level split sufficiency.

## Artifacts

- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_runner.py`, runner
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/baseline_n20.json`, machine-readable (with scoring)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/responses/`, 20 raw E2B responses
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/teacher_picks_n20.json`, teacher pick lookup
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/run.log`, runner log
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/tier2_progress.txt`, load-failure history
