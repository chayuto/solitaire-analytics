# Next compute window: pre-registered plan (v4-A + temperature probe)

**Date drafted**: 2026-05-28 (BEFORE any run this window)
**Compute budget**: 4-5 hours, single M5, serial GPU queue
**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.7)

> Written before the window opens so the predictions cannot be retrofitted to
> the result. Supersedes the "9.4 Next compute window plan" stub in
> `docs/reports/20260527_full_game_play_compute_window_report.md`, with two
> deliberate revisions justified in section 1.

## 0. Companion documents

- `docs/reports/20260527_full_game_play_compute_window_report.md` (the window this builds on)
- `docs/reports/20260526_v3_experiment_design.md` (v3 pre-registration; the filter)
- `data/benchmarks/winnable_decks.json` (regenerated 2026-05-28: now 4 unique decks)

## 1. What changed since the last window, and the two revisions

The last window's pre-registered next step was **v4-A**: train gemma-3n (the only
base that plays meaningfully) on the same reversal-filtered corpus that produced
v3, then sweep and full-game-play on seed 3263196305. That stays the anchor of
this window. Two things observed since justify revising the plan around it:

1. **More winnable decks.** `winnable_decks.json` now holds 4 unique solver-confirmed
   decks (seeds 2853966634, 2967897202, 3263196305, plus one seedless) instead of
   the single deck the full-game findings rested on. N=1 was the weakest point in
   the prior evidence. Revision: spend leftover time on deck-generality with the
   known-good v1.1 arm.
2. **The QS doom-loop was deterministic at temperature 0.0.** A bit-deterministic
   294-turn limit cycle is the signature of a greedy-decoding fixed point. The
   curated `known_outcomes` for seed 2967897202 records the 31B teacher itself
   producing BOTH a win and a stall on the SAME deck under its production
   temperature of **0.3**, diverging at turn 15 as "a clean stochastic divergence
   under temperature 0.3." That is direct prior evidence that the doom-loop vs win
   outcome is partly stochastic under temperature. Revision: add a cheap
   temperature micro-probe at the front of the queue, before committing 95 min to
   training.

What this window does NOT touch: the harvester-side prompt track (v1.3 anti-undo
hole, v1.4 INTENT) is gated on the harvester team producing 31B-on-v1.3 traces,
not on local compute. The 26B-a4b traces are excluded from training by the
`TEACHER_MODEL=gemma-4-31b-it` filter and are not part of any arm here.

## 2. The queue (serial; flex items drop first if the window runs short)

| # | Step | GPU | Drop order |
|---|---|---|---|
| 0 | Deck pool regenerated + this pre-registration locked | done, no GPU | n/a |
| 1 | Temperature micro-probe (section 4) | ~10 min | keep |
| 2 | v4-A training: gemma-3n + frozen 1832-row filtered corpus | ~95 min | keep |
| 3 | v4-A 20-state bench sweep | ~25 min | keep |
| 4 | v4-A best checkpoint full-game on seed 3263196305 | ~50 min | keep |
| 5 | Flex: v1.1 generality on 2853966634 / 2967897202; or full-game temp run if step 1 showed escape | variable | first to drop |

## 3. v4-A: the anchor experiment

### 3.1 What it isolates

v4-A differs from **v1.1** (the deployed student) in exactly one variable: the
training corpus is reversal-filtered instead of raw. Base (gemma-3n), hyperparameters,
patch, trainer, bench, scorer, chat template all held constant.

v4-A differs from **v3** in exactly one variable: the base is gemma-3n instead of
Gemma 4 E2B. The corpus is the SAME frozen snapshot
(`data/dataset/training_shuffle_filtered.jsonl`, 1832 rows, produced 2026-05-26).

CRITICAL: do NOT re-run `filter_shuffles.py` on the grown corpus (now 2844 rows).
Re-filtering would confound the base-model comparison with corpus growth and with
the two teacher won-games (c05ad4, aca45a) that landed after v3. The won-games
corpus retrain is a deliberately separate experiment the operator declined for
this window.

### 3.2 Hypothesis

> **H_v4A**: Training gemma-3n on the reversal-filtered corpus yields a LoRA that
> (1) preserves v1.1's full-game competence, (2) improves the single-turn bench
> above v1.1, and (3) extends the competent-play window before doom-loop onset.

### 3.3 Pre-registered predictions (locked)

Reference baselines: v1.1 (gemma-3n, unfiltered) bench tier 3.15, agreement 11/20,
foundations 6/7, oscillation agreement 4/7, full-game seed 3263196305 fc=3 with a
JD col4/col7 loop onset near turn 35. v3 (Gemma 4, filtered) bench best 2.85,
full-game fc=0 with a QS col5/col7 loop from turn 6.

Training process:
- **TP1**: dataset after `prepare_dataset.py` game-level split reproduces v3's split
  (same frozen input): roughly 1279 train / 144 valid / 168 test. Off by more than a
  few rows means the input is not the frozen snapshot.
- **TP2**: val loss reaches 0.30-0.45 by iter 1000 (v1/v2/v3 all landed near 0.36).
- **TP3**: wall <= 100 min, peak memory <= 12 GB (v1.1 T5 peaked 11.49 GB).

Single-turn bench (20-state):
- **BP1 (primary)**: best checkpoint mean tier >= 3.15. Disconfirm: < 3.15 means the
  filter hurt the base that actually plays.
- **BP2**: foundation recovery >= 6/7 (no regression vs v1.1).
- **BP3**: oscillation agreement >= 4/7 (v1.1 and v3 both 4/7).
- **BP4**: teacher agreement >= 11/20.

Full-game (seed 3263196305):
- **FP1 (primary)**: final foundation count >= 3 (matches v1.1). Disconfirm: < 3.
- **FP2 (stretch)**: doom-loop onset turn >= 40, OR no doom-loop (fc climbs past the
  v1.1 plateau).
- **FP3**: if it doom-loops, the pattern is a gemma-3n-style loop (like v1.1's JD
  col4/col7), NOT the Gemma-4 QS col5/col7 loop. Confirms the loop is base-specific.

### 3.4 Decision gates (pre-committed)

- **PROMOTE** (filter helps the playing base): full-game fc > 3 AND bench tier >= 3.15.
  Ship v4-A as the new canonical LoRA, superseding v1.1.
- **PARTIAL** (filter neutral): full-game fc == 3 AND bench tier >= 3.15. v1.1 stays
  canonical; v4-A is a documented tie. The filter is clean on the bench but does not
  move full-game play even on the base that plays.
- **HOLD** (filter hurts): full-game fc < 3 OR bench tier < 3.15. Conclusion: reversal
  corpus-filtering does not help full-game play on EITHER base. Close the corpus-filter
  program; pivot to harvester-side levers (resign + state-repetition annotation) and/or
  the won-games corpus retrain.

### 3.5 Commands

```bash
cd /Users/chayut/repos/solitaire-analytics
# v4-A uses the FROZEN filtered corpus; do NOT re-run filter_shuffles.py.
cd gemma4_finetune
./venv/bin/python prepare_dataset.py \
    --log ../data/dataset/training_shuffle_filtered.jsonl \
    --out dataset_v4a
# train: gemma-3n base + frozen filtered corpus (lora_config_v4a.yaml to be written:
# copy lora_config_v3.yaml, swap model -> the gemma-3n base, data -> dataset_v4a,
# adapter path -> adapters_v4a)
./venv/bin/python train_v2.py --config lora_config_v4a.yaml
# sweep + full-game per the v3 procedure, swapping v3 -> v4a paths
```

## 4. Temperature micro-probe (cheap, front-loaded)

### 4.1 Question

Is the Gemma-4 QS col5/col7 limit cycle a greedy-decoding artefact that sampling
escapes, or a genuine attention/representation attractor that survives temperature?

### 4.2 Method

1. Load the exact board state at the QS-loop onset from
   `gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/turns.jsonl`.
2. With untuned Gemma 4 E2B (text patch), sample the decision N=20 at each of
   temperature 0.4, 0.7, 1.0 on that single board.
3. Record the distribution of chosen move_index. The QS swap is the loop move; any
   mass on a non-swap legal move is an escape.

### 4.3 Pre-registered predictions

- **MP1**: at temperature 0.0 the model is deterministic on this board (sanity check;
  it must reproduce the QS swap).
- **MP2 (primary)**: at temperature 0.7, escape probability (mass on non-loop moves)
  > 0.20. Prior: the teacher escaped-or-not stochastically on a different deck at
  temperature 0.3, so a higher temperature on this board should show non-trivial escape.
- **MP3**: if MP2 holds, one full-game run of untuned Gemma 4 at the lowest escaping
  temperature is warranted (flex step 5). If escape probability is ~0 even at
  temperature 1.0, the loop is not a sampling artefact and no full-game temp run is
  run this window.

### 4.4 Why this is worth 10 minutes before training

If sampling escapes the loop, the cheapest possible intervention (an inference-time
temperature, no retraining) addresses the headline pathology, and it reframes the
entire LoRA-vs-prompt debate. If sampling does not escape, that is a clean negative
that strengthens the "base-model-deep" finding and costs almost nothing.

## 5. Flex: deck generality (only if time remains)

Run the deployed v1.1 LoRA (the only arm that plays meaningfully) on the new decks at
temperature 0.0:
- seed 2853966634 (won by the teacher in 418 moves under v1.2; a hard two-plateau
  recovery deck, the most stringent generality test we have)
- seed 2967897202 (won by the teacher in 194; the curated known_outcomes show this deck
  is win-or-stall stochastic under temperature 0.3)

Question: is "competent for ~35 turns then doom-loop" a property of v1.1, or an
artefact of the single deck 3263196305? No predictions locked (exploratory); classify
each run per the full-game report's revised rules.

## 6. What we will NOT do this window

- Re-run filtered Gemma 4 arms on 3263196305 (two identical doom-loops is enough).
- Re-filter the grown corpus for v4-A (confounds base comparison).
- Won-games corpus retrain (declined for this window; tracked as the next candidate).
- Any harvester-side prompt change (gated on harvester traces, not local compute).
