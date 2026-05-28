# Pre-registered design: v3 distillation with per-turn shuffle filter

**Date drafted**: 2026-05-26 (before any v3 training run)
**Status**: scaffold complete, training not yet executed
**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.7)

> This document is intentionally written *before* v3 training is run, so its
> hypotheses cannot be retroactively adjusted to match the result. The lab
> log at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v2_gemma4_distillation_lab_log.md`
> records the v2 findings that motivate this experiment.

## 1. Motivating finding

The v2 distillation (1000-iter LoRA on `mlx-community/Gemma4-E2B-IT-Text-int4` with the v1.1 recipe) regressed on all 5 oscillation states it should have left alone, dropping mean tier from 2.55 (untuned) to 2.45 (best checkpoint). Per-state inspection localised the failure to doom-loop confabulation patterns transferred from the 31B teacher's responses in the training corpus.

## 2. Hypothesis

> **H_v3**: The cause of the v2 regression is the presence of teacher doom-loop turns in the training corpus, specifically interactions where the teacher chose a tableau-to-tableau move that directly reverses the most recent tableau-to-tableau move in `recentMoves`. Removing those rows before training will produce a v3 LoRA that:
>
> 1. Preserves untuned-baseline oscillation behaviour (>= 4/7 foundation recovery, agreement on all 7 oscillation states matching or beating v2-untuned's), AND
> 2. Retains v2-untuned's mean tier or improves on it (>= 2.55), AND
> 3. Adds new value on non-oscillation states (mean tier > 2.55 driven by improvement on early-game and midgame categories).

If H_v3 holds at all three conditions, v3 LoRA is the v2 we wanted in the first place. If it holds at (1) and (2) but not (3), the corpus is correctly de-poisoned but the remaining signal is too thin to drive a usable lift. If (1) fails, the bottleneck is not corpus poisoning and another hypothesis is needed (see Section 9).

## 3. Filter design

Implemented at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/filter_shuffles.py`.

For each interaction row in `data/dataset/training.jsonl`:

1. Parse the prompt to extract `legalMoves` and `recentMoves`. Both prompt formats handled:
   - **Legacy** (`CURRENT GAME (JSON):` blob, 1379 of 1635 corpus rows): brace-balanced JSON parse, read the two fields directly.
   - **Hybrid-v1** (de7dc06+ plain-text blocks, 256 rows): regex on the `LEGAL MOVES (respond ...):` and `RECENT MOVES (oldest -> newest...):` headers.
2. Parse `rawResponse` JSON to extract `final_decision.move_index`.
3. Apply the reversal predicate:
   - The chosen move's `type` must start with `tableau_to_tableau`.
   - At least one entry in `recentMoves` (search newest-first) must match the regex `move XX col N -> col M`.
   - The chosen move's `describe` must contain `from column M to column N` or `col M to col N` or `column M to column N` (i.e., the reverse direction).
4. If the predicate holds, drop the row. Otherwise keep it.

**Conservative-by-design**: rows we can't parse, rows with no move_index, rows where `recentMoves` is empty, all get KEPT. False-negative bias is acceptable for this first pass; false positives on this signature are extremely rare because the move-text patterns are mechanically generated.

**Known limitations**:
- Only detects direct one-back reversals. A three-cycle (col1→col2, col2→col3, col3→col1) would not match.
- Does not catch confabulations that aren't reversals (e.g., the teacher rationalising a King move that doesn't undo anything but is still strategically pointless).
- Does not consider confidence as a signal (which the dataset card flags as miscalibrated to 0.91 mean).

These limitations are intentional. A more aggressive filter risks dropping legitimate setup moves; a more conservative one risks not addressing the root cause. The chosen predicate matches the documented anti-pattern as closely as the prompt schema allows.

## 4. Pre-registered predictions

Quantitative predictions made *before* running the filter or training:

| prediction | predicted value | what disconfirmation looks like |
|---|---|---|
| **P1**: Dropped row count | Between 80 and 250 (5% to 15% of 1635) | Outside this range suggests either the predicate is mis-tuned or the corpus is unexpectedly clean/dirty |
| **P2**: v3 best mean tier | Strictly greater than v2-best (2.45) | <= 2.45 means filtering didn't help |
| **P3**: v3 best oscillation agreement | >= 5/7 (v2-untuned was 5/7 oscillation-correct including the foundation states) | < 5/7 means filter didn't preserve the strength |
| **P4**: v3 best foundation recovery | >= 4/7 (no regression vs v2-untuned) | < 4/7 means the filter removed too much signal |
| **P5**: v3 vs v1.1 mean tier | v3 best within 0.30 of v1.1's 3.15 | Wider gap = problem deeper than corpus content; narrower or beating = ship v3 |
| **P6**: Training resource budget | Wall <= 90 min (smaller dataset, same iters), peak memory <= 8.5 GB | Major divergence = something unexpected |
| **P7**: Loss curve shape | Val loss reaches 0.35-0.45 by iter 1000 (matching v2's 0.36) | Higher floor would suggest the filter removed important signal not just noise; lower floor would suggest cleaner signal |

If P1 and P2 both hold, v3 is the right path forward. If P1 holds but P2 fails, the corpus contains more failure-mode content than this filter catches (consider broader filters: confidence threshold, shuffle-fraction over a window, etc.).

## 5. Procedure

```bash
cd /Users/chayut/repos/solitaire-analytics

# 1. Apply the per-turn shuffle filter
.venv/bin/python gemma4_finetune/filter_shuffles.py \
    --in  data/dataset/training.jsonl \
    --out data/dataset/training_shuffle_filtered.jsonl \
    --dump-dropped data/dataset/training_dropped_shuffles.jsonl

# 2. Game-level split (same script as v2)
cd gemma4_finetune
venv/bin/python prepare_dataset.py \
    --log ../data/dataset/training_shuffle_filtered.jsonl \
    --out dataset_v3

# 3. Train (background; ~85 min)
venv/bin/python train_v2.py --config lora_config_v3.yaml \
    > /tmp/v3_train.log 2>&1 &

# 4. After training completes, run the sweep + score
bash sweep_v3_checkpoints.sh 2>&1 | tee /tmp/v3_sweep.log
```

All v2 artifacts (adapters_v2/, dataset_v2/, posttune_at*.json) are preserved in place; v3 outputs land in adapters_v3/, dataset_v3/, and posttune_v3_at*.json.

## 6. Files of record (scaffold, all created 2026-05-26)

- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/filter_shuffles.py` (the filter, 188 lines)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/lora_config_v3.yaml`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/sweep_v3_checkpoints.sh`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/score_v3_learning_curve.py`
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md` (this doc)

The `train_v2.py` wrapper is reused unchanged because it already forwards `--config` to `mlx_lm.lora.main()`.

## 7. Success criteria for promotion

Same gate as v2 had (`best v3 mean tier > 3.15`), with an additional companion-ship path:

- **PROMOTE as canonical**: v3 best > 3.15 (beats v1.1). Replace v1.1 as the recommended model.
- **PARTIAL (ship as companion)**: v3 best in [2.55, 3.15], oscillation >= 5/7. Publish v3 LoRA alongside v1.1 with a README that picks based on use case.
- **HOLD**: v3 best < 2.55 or oscillation < 5/7. The corpus filter is not the bottleneck; pursue a different hypothesis next.

## 8. What this experiment isolates

v3 differs from v2 in exactly one variable: training corpus content. Hyperparameters, base model, patch, training script, eval bench, scoring function, and chat template are all held constant. A v3-v2 comparison thus directly tests whether corpus content was the cause of the v2 regression.

It does *not* isolate the deliberation-format hypothesis (whether keeping `<|channel>thought` blocks in the training labels matters). That is a separate experiment for a later run; running both changes at once would confound the analysis.

## 9. Hypotheses for if v3 fails

If v3 best mean tier < v2-untuned (2.55) or oscillation < 5/7:

- **H_v4a**: Deliberation format matters. Re-render completions to include thinking-mode reasoning before the JSON, retrain. Estimated cost: corpus regeneration + 85 min train.
- **H_v4b**: Learning rate is too aggressive for a base this close to the target. Drop to 5e-5 (10x lower) and retrain. Estimated cost: 85 min train.
- **H_v4c**: The training set is too small for the additional signal-removal headroom. Wait for >= 8 more post-cutover sessions and >= 1000 more post-cutover training rows (the existing v1.1 trigger), then re-filter and retrain. Estimated cost: wait for harvester data.
- **H_v4d**: The 2.3B-active student's capacity is genuinely the bottleneck for the harder midgame states; oscillation strength was the only headroom we had. In that case, v2 untuned IS the ship (apache-2.0 companion to v1.1), and further LoRA effort on this base is mis-targeted.

Pre-committing to which hypothesis to try first depends on the v3 numbers; revisit then.
