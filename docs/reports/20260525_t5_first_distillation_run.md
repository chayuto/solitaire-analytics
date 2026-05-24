# T5 — First Gemma 3n E2B distillation run + N=20 eval

**Date**: 2026-05-25
**Adapter**: `gemma4_finetune/adapters_t5/` (1000-iter LoRA, 4 checkpoints)
**Base model**: `mlx-community/gemma-3n-E2B-it-text-4bit-dwq`
**Training data**: 1279 train + 126 valid (from full 1730-row training.jsonl, 25 sessions)
**Eval**: Phase 1.5 N=20 bench (post-cutover, C0 / passthrough arm)
**Status**: end-to-end fine-tuning pipeline validated; tier gap to teacher cut by half on first run

## TL;DR

- **Mean tier 2.10 → 2.75** on the N=20 eval bench. Gap to 31B teacher: **−1.32 → −0.67** (halved on first run).
- **2 of 5 missed-foundation states recovered**, including the triple-replicated `oscillation-bfb84ae55c3f` that had stumped C0-Haiku, A4-Haiku, and untuned 3n-E2B.
- **Illegal moves eliminated** (1 → 0).
- **JSON validity stayed at 100%** (already perfect pre-tune).
- **Teacher agreement unchanged at 11/20**, but the *composition* shifted. Some new disagreements are E2B picking a HIGHER-tier move than the teacher (model learning good play, not just memorising).
- Memory & time envelope held — pipeline is shippable; the question now is iter count and data freshness, not whether the runway works.

## Training run

| | value |
|---|---|
| Iters | 1000 (4 saved checkpoints @ 250/500/750/1000) |
| Wall time | ~95 min |
| Per-iter | 5.2 s avg |
| Peak MLX memory | 11.49 GB |
| Peak system RAM | 15.25 GB (of 17 GB total) |
| Trainable params | 11.27 M of 4.46 B (0.253%) |
| Adapter size | 45 MB per checkpoint |
| Val loss curve | 6.365 (iter 1) → 0.426 (iter 100) → **0.369 (iter 1000)** |
| Train loss final | 0.222 |
| Train/val gap final | 0.147 — mild memorisation but val still improving |

Loss collapsed in the first 100 iters (most of the learning) and then crept down slowly. The val curve is still trending down at iter 1000 — there is probably another 10–20% improvement available with 2–3× more iters, but with diminishing returns.

## Pre/post eval — headline

| metric | untuned | tuned | teacher | Δ tuned vs untuned |
|---|---:|---:|---:|---:|
| JSON valid | 20/20 | 20/20 | — | 0 |
| Teacher agreement | 11/20 | 11/20 | — | 0 |
| Illegal moves | 1/20 | **0/20** | — | **−1** |
| Mean tier (all 20) | 2.10 | **2.75** | 3.42 | **+0.65** |
| Gap to teacher (Δ) | −1.32 | **−0.67** | — | **+0.65** |

The agreement count being flat hides the real story — the tier-score change is what matters.

## Pre/post — per category

| category | n | untuned | tuned | teacher | Δ |
|---|---:|---:|---:|---:|---:|
| early | 5 | 2.60 | 3.20 | 4.20 | +0.60 |
| midgame | 8 | 1.38 | 1.75 | 2.00 | +0.38 |
| **oscillation** | 7 | 2.57 | **3.57** | 4.29 | **+1.00** |

Oscillation gained the most. That's where the foundation-miss anti-pattern lived, so it tracks with the foundation recovery (see below).

## Foundation-miss recovery (the headline win)

5 states had a teacher-chosen foundation move that untuned E2B missed. Outcomes after 1000 iters:

| state | untuned pick | tuned pick | teacher pick | verdict |
|---|---|---|---|---|
| `early-3687a40eda7b` | shuffle (2) | **foundation (6)** | foundation, idx 3 | **RECOVERED** (different foundation index, same tier) |
| `early-e6291973dd07` | shuffle (2) | draw (1) | foundation (6) | regressed by 1 tier |
| `midgame-4ab5735a4f20` | draw (1) | draw (1) | foundation (6) | unchanged |
| `oscillation-a774c0d22f24` | draw (1) | shuffle (2) | foundation (6) | slight improvement |
| **`oscillation-bfb84ae55c3f`** | draw (1) | **foundation (6)** | foundation (6) | **RECOVERED — full agreement** |

**2 of 5 missed foundations recovered.** The `oscillation-bfb84a` recovery is significant: that state was a triple-replicated failure mode (C0-Haiku missed 1/3, A4-Haiku missed 1/3, untuned 3n-E2B missed 3/3). After fine-tuning, it's now solved.

The other 3 still-missed foundation states give us a concrete next-iteration target. They probably need either more training iters or more training examples that emphasise foundation moves.

## "Lost agreements" that are actually wins

Two states where the tuned model now *disagrees* with the teacher but picks a HIGHER-tier move:

| state | untuned (= teacher) | tuned | tier change |
|---|---|---|---|
| `midgame-031d9c9e3fe7` | shuffle (2) | waste_play (4) | **+2** (better play than teacher) |
| `oscillation-d0ff552ed744` | draw (1) | shuffle (2) | +1 (better play than teacher) |

The teacher isn't oracle-optimal — sometimes the tuned model picks a strategically stronger move. This means raw teacher-agreement underrates the tuned model's quality. Tier score captures it; agreement count doesn't.

## Illegal-move elimination

Untuned: `midgame-81dc0fb02394` chose index 2 when only indexes 0–1 were legal. 5% illegal rate.
Tuned: chose index 0 (legal shuffle). 0% illegal across N=20.

This is exactly the kind of structural learning that fine-tuning is good at — and it matches `evaluate.py`'s "legal move" metric being a sensible secondary criterion.

## Resource numbers (informs T5+ planning)

T5 (training):
- Wall: 95 min for 1000 iters at seq_len=2048, num_layers=16, rank=16
- Peak MLX: 11.49 GB (within 13 GB budget)
- Peak system: 15.25 GB (within 17 GB total; ~30s of swap pressure)

Post-tune eval (same as untuned N=20):
- Wall: 4 min 29 s for 20 inferences
- Peak: 6.37 GB

For comparison, the T4 synthetic smoke at the same config was 50 iters in 3:05 (~3.7 s/iter). Real data is ~40% slower per iter because of variable length + truncation overhead.

## What this changes about the project

### Pipeline validated end-to-end on real data

We now have a working pipeline:
```
prepare_dataset.py → mlx_lm.lora (1000 iters) → posttune_n20_runner.py → tier scoring
```
Every step ran without error on real production data. The runway is no longer hypothetical.

### First quantitative target hit

Pre-tuning the gap was −1.32 tier. The pre-registered decision rule said "fine-tuning definitely needed." After one 95-min training run, the gap is −0.67. We're halfway to the teacher, with no hyperparameter tuning, no data quality pass, no eval-set-driven checkpoint selection.

### The 20-state bench is the right eval

It catches:
- Format wins (illegal rate)
- Strategy wins (foundation recovery)
- Genuine improvements that don't agree with teacher (tier > agreement)

And it runs in ~5 min on M5. Make this the always-on eval for every future checkpoint.

## What to try next

Ranked by expected ROI per compute minute:

1. **Eval the intermediate checkpoints (250 / 500 / 750)** to find the optimal stopping point. ~15 min total compute. If iter 500 is as good as iter 1000, future runs can be ~half as long. Already saved — just need to run posttune_n20_runner.py 3 more times.

2. **Train 2000–3000 iters on the same data** (~3–5 hours). Val loss was still falling at 1000; there's headroom. Direct path to closing more of the −0.67 gap.

3. **Wait for more post-cutover data, then re-train on post-cutover-only.** The current run trained on heterogeneous templates (965 of 1279 rows are pre-cutover legacy format). Training only on the matching template should help bench performance, but we only have 351 post-cutover rows today — need more sessions before this is the right call.

4. **Targeted foundation-move data augmentation.** The 3 still-unrecovered foundation states have specific signatures. We could synthetically generate more foundation-decision examples and bias the training mix toward them. Higher effort, possibly bigger win.

5. **Rank 32 instead of 16.** More LoRA capacity for the same iter count. Halves training time per unit of learning, possibly. Cheap to try.

## Caveats

- **Train loss 0.222 vs val 0.369** — train/val gap of 0.147 is starting to widen. The model is starting to memorise. Probably safe to push to 2000 iters, but not 5000 without data augmentation.
- **N=20 with single run per state** — variance estimate is noisy. The +0.65 tier delta is large enough to be real, but per-state changes (especially the 1-state foundation gains) should not be overinterpreted as "this exact state is fixed forever."
- **Trained on heterogeneous templates, evaluated on post-cutover** — there's a small template-shift confound. The fact that we got +0.65 anyway suggests the model is learning general Solitaire reasoning, not template-specific patterns.
- **Teacher itself isn't optimal**. Tier-score improvements where the model disagrees with teacher (e.g., +2 on midgame-031d9c9e3fe7) suggest the teacher leaves money on the table sometimes. Our distilled model has now picked up some of that money.
- **No production game-play eval** — we evaluated on captured states, not by running the model through whole games. The right next-level eval is "let the tuned E2B play 5 games end-to-end and compare win rate / progress vs untuned."

## Track B status — all 5 tiers cleared

| # | Tier | Status | Note |
|---|---|---|---|
| 73 | T0 Inventory | ✅ | |
| 74 | T1 Proxy pipeline | ✅ | |
| 75 | T2 E2B baseline | ✅ | found mlx-lm Gemma 4 blocker → pivoted to 3n |
| 79 | N=20 untuned baseline | ✅ | Δ=−1.32, fine-tuning justified |
| 76 | T3 Mini-smoke | ✅ | fixed LoRA `keys:` for Gemma 3n's `altup` |
| 77 | T4 Full smoke | ✅ | 8.99 GB peak, well under 13 GB budget |
| 78 | **T5 Real training + eval** | ✅ | **Δ=−0.67 (halved). Pipeline shippable.** |

## Artifacts

- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/adapters_t5/` — 4 LoRA checkpoints (45 MB each)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/baseline_n20.json` — untuned scoring
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/posttune.json` — tuned scoring
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20/posttune_responses/` — 20 raw tuned responses
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/t5_training.log` — full mlx_lm.lora log
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/t5_memlog.txt` — system RAM sampler
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/posttune_n20_runner.py` — reusable eval runner
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260524_gemma3n_e2b_untuned_n20_baseline.md` — pre-tune baseline (companion doc)
