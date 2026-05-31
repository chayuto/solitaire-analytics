# v5: won-games-only Gemma 4 E2B distillation -- pre-registration

**Date drafted**: 2026-05-30 (BEFORE the training run)
**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.8)
**Compute**: single M5, 16 GB unified memory; ~95 min train + ~25 min sweep
**Base**: `mlx-community/Gemma4-E2B-IT-Text-int4` (Gemma 4 E2B Text, apache-2.0)

> Written before the run so predictions cannot be retrofitted. Results section at
> the bottom is intentionally left as TODO and will be filled ONLY from measured
> output after the producing run exits (per the integrity rule learned 2026-05-30).

## 1. Why v5, and what it isolates

Three Gemma-family distillation results so far, all HELD:
- **v2** (Gemma 4 E2B, ALL raw success turns): oscillation discipline regressed; the
  corpus contains the 31B teacher's own lost-game doom-loop turns, and training on
  them transferred the failure. Ship untuned.
- **v3** (Gemma 4 E2B, MOVE-level reversal filter): recovered some bench tier (2.85)
  but doom-looped identically to untuned in full play.
- **v4-A** (gemma-3n, same MOVE-level filter): best tier 2.80 < v1.1's 3.15, fc=0
  full-game with the Gemma-4-style QS col5/col7 loop. Corpus-filter program closed.

The move-level reversal filter failed on both bases. The untried lever is a
**game-level** filter: instead of guessing which individual moves are bad, drop
entire games the teacher LOST and keep entire games it WON. This directly targets
the v2 poisoning finding (lost-game doom-loops are the poison) at the granularity
where the win/loss label actually exists.

v5 differs from v2 in exactly one variable: corpus = won-games-only instead of
all-success. Base, hyperparameters (rank 16 / scale 2.0 / lr 2e-4 / 1000 iters /
num_layers 16), patch, trainer, bench runner, scorer, chat template all identical
to `lora_config_v2.yaml`. So v5-vs-v2 is a clean corpus-content comparison.

## 2. Corpus construction (already built + verified 2026-05-30)

- Source: `data/dataset/training.jsonl` = 3758 turns, 100% gemma-4-31b-it, 100% success, 45 sessions.
- Win labels: `data/raw/solitaire-win-*.json`, each with `gameSessionId` + `gameWon: true`.
- Join: a training turn is WON iff its `sessionId` equals some win record's
  `gameSessionId` (the game-level key). Built by `gemma4_finetune/build_wononly_corpus.py`.
  NOTE: an earlier seed-level join returned 1923 turns but was confounded -- it
  mislabels lost replays of a winnable seed as won. The correct game-level join is
  the gameSessionId match below.
- Result: **1513 won turns from 10 won games** (40.3% of the 3758-row corpus). Of
  those, `prepare_dataset.py` keeps the 892 whose rawResponse is a JSON object with
  all three required keys (board_analysis / strategic_plan / final_decision) and drops
  621 that fail `json.loads`. ROOT CAUSE (characterised 2026-05-30): all 621 drops are
  `json_decode_error`, not missing-keys -- the dropped rawResponses are valid teacher JSON
  wrapped in a markdown ```json ... ``` code fence, which json.loads rejects. The teacher
  content is fine; the splitter has no fence-stripping preprocessor. This is the SAME
  behaviour v2/v3 trained under, so v5-vs-v2 stays a fair comparison, but it means ~621
  usable won-game examples are being silently discarded for a wrapper.
  IMPORTANT: the drop is UNEVEN by seed, so it SKEWS the corpus (it does not preserve
  balance). Keep rates: 2967897202 94% (137/145), 3263196305 63% (390/618), 2853966634
  58%, 1388178981 60%, 549440324 46%, 4161700176 43%, 2003817730 40%. Net effect on the
  892 kept: 3263196305 ~44% (390) and 2967897202 ~15% (137) dominate; the four low-keep
  seeds are squeezed. A fence-stripping fix (v5b) would recover ~621 examples and
  materially rebalance; deferred so as not to change the splitter mid-experiment.
- Usable examples after split (`prepare_dataset.py --log training_wononly.jsonl
  --out dataset_v5`, game-level on sessionId, 0.8/0.1/0.1): **892 total = 666 train /
  137 valid / 89 test across 10 games.**
- Per-seed share of the 892 usable: 3263196305 ~42% (376), then 549440324 (105),
  2853966634 (102), 4161700176 (91), 2967897202 (91), 2003817730 (85), 1388178981 (77).

## 3. Pre-registered limitation (the main threat to this experiment)

Only **10 distinct won games**, and ONE seed (3263196305) is **~42% of the 892 usable
examples**; the other six seeds are 8-12% each. Consequences, acknowledged up front:
- Low game diversity dominated by one board: the student may overfit to seed
  3263196305 rather than learn general winning play. (That seed is also the doom-loop
  anchor, so v5 is in effect heavily trained on won trajectories of the exact board
  every prior arm looped on -- which is either the best possible signal or pure
  memorization fuel, and the full-game probe cannot distinguish them since that seed
  is in-corpus.)
- The split is game-level so valid (137) and test (89) are held-out GAMES, but only a
  few games each, so val loss is a noisy small-sample signal; do not over-read small
  val-loss differences.
- Volume (892) is smaller than the 1832-row filtered set v3/v4-A used; both diversity
  AND volume are limited here. 666 training examples is thin for 1000 iters (~1.5
  epochs-worth per the usual sizing), raising overfit risk further.

If v5 shows overfit signatures, the fix is NOT a hyperparameter change but MORE won
games, which is gated on the harvester producing more wins (data collection), not on
local compute. A per-seed cap (v5b) would rebalance but shrink the set further.

## 4. Hypothesis

> **H_v5**: training Gemma 4 E2B on won-games-only turns yields a LoRA that (1)
> preserves untuned oscillation discipline (the thing v2 broke) because no lost-game
> doom-loop turns are in the corpus, and (2) does not regress bench tier vs v2-untuned.

## 5. Pre-registered predictions (locked)

Reference baselines (measured, from `learning_curve_v3.json`): v2-untuned bench tier
2.55, foundation 4/7, oscillation agreement 6/7, oscillation tier 4.00; v2-best-trained
(iter500) tier 2.45, oscillation 3/7; v3-best (iter750) tier 2.85, oscillation 4/7.

Training process:
- **TP1**: dataset_v5 split = 666 / 137 / 89 (already verified; re-confirm at load).
- **TP2**: val loss reaches 0.30-0.50 by iter 1000 (wider band than usual because the
  val set is small and few-game). Watch for an overfit signature: val loss bottoming
  early then rising while train loss keeps falling -- if so the best checkpoint is an
  early one, not iter1000.
- **TP3**: wall <= 100 min, peak <= 12 GB. (Smaller corpus than v2, so likely faster.)

Single-turn bench (20-state, 512 tokens):
- **BP1 (primary)**: best-checkpoint oscillation agreement >= 6/7. This is the whole
  point -- v5 must PRESERVE untuned oscillation discipline where v2 dropped it to 3/7.
  Disconfirm: < 6/7 means the won-only corpus still erodes oscillation.
- **BP2**: best-checkpoint mean tier >= 2.55 (no regression vs v2-untuned).
- **BP3**: best-checkpoint mean tier > 2.45 (beats v2-best-trained; i.e. the game-level
  filter beats no-filter training).
- **BP4**: JSON validity >= 19/20 (won-game turns are well-formed teacher outputs).

Full-game (seed 3263196305 -- NOTE this seed is IN the training set, so this is a
memorization probe, not generalization; a held-out winnable seed would be cleaner but
all 4 winnable-deck seeds except one are in the won set):
- **FP1**: report fc and loop identity descriptively. No pass/fail threshold, because
  the only winnable decks available are largely in-corpus now. Interpret cautiously:
  high fc on an in-training seed is consistent with memorization, not generalization.

## 6. Decision gates (pre-committed)

- **PROMOTE**: BP1 (osc >= 6/7) AND BP2 (tier >= 2.55). The game-level filter preserved
  the strength v2 broke without costing tier. Ship v5 as the trained Gemma 4 student;
  supersede the untuned recommendation.
- **PARTIAL**: BP1 holds but BP2 fails (osc preserved, tier slips). Documented; untuned
  stays the ship recommendation but the game-level filter is validated as the right
  direction, pursue with more won games.
- **HOLD**: BP1 fails (osc < 6/7). The won-only corpus does not preserve oscillation
  discipline either; conclude that Gemma 4 E2B distillation does not beat untuned by any
  corpus route tried, and pivot fully to the harvester-side prompt track + untuned ship.

## 7. Commands (launch when the shell channel is confirmed stable)

```bash
cd /Users/chayut/repos/solitaire-analytics/gemma4_finetune
# corpus + split already built (build_wononly_corpus.py -> prepare_dataset.py -> dataset_v5)
./venv/bin/python train_v2.py --config lora_config_v5.yaml > /tmp/v5_train.log 2>&1 &
# checkpoints -> adapters_v5/000{0250,0500,0750,1000}_adapters.safetensors
# then sweep on the Gemma 4 TEXT runner (v5 shares the Gemma 4 base, so reuse the
# v3 path), pointing the sweep at adapters_v5 / posttune_v5_* outputs.
```

The v3/v2 sweep uses `posttune_n20_gemma4_text_runner.py` (Gemma 4 text base). A v5
sweep needs the same runner with adapters_v5 staged; budget ~10 min to clone
`sweep_v3_checkpoints.sh` -> `sweep_v5_checkpoints.sh` (swap v3->v5 paths) before the
sweep, OR run the runner directly per checkpoint.

## 8. Results (filled 2026-05-31 from measured output)

### Training process

- **TP1 PASS**: dataset_v5 = 666 train / 137 valid / 89 test (10 games), as built.
- **TP2 FAILED, and diverged**: val loss bottomed at **1.576 at iter 100**, then rose
  MONOTONICALLY: 1.73 (200) / 1.87 (300) / 1.94 (400) / 2.10 (500) / 2.09 (600) /
  2.57 (700) / 2.49 (800) / 2.68 (900) / **2.80 (1000)**, while train loss fell to 0.133.
  Textbook overfit on the small, single-seed-dominated corpus -- exactly the §3 risk.
  Never reached the 0.30-0.50 band (best 1.576, vs v2/v3 ~0.36 on larger corpora).
- **TP3 PASS**: peak 8.47 GB; wall ~80 min; final weights saved; 4 checkpoints written.

### Bench (20-state, 512 tokens, temp 0.0) -- MEASURED, from learning_curve_v5.json

| config | tier | osc agree | osc tier | foundation | teacher agree | illegal | json |
|---|---|---|---|---|---|---|---|
| v2 untuned (ref) | 2.55 | 6/7 | 4.00 | 4/7 | 12/20 | 0/20 | 20/20 |
| v2 best trained (i500) | 2.45 | 3/7 | 2.43 | 4/7 | 10/20 | 2/20 | 20/20 |
| v3 best (i750, filtered) | 2.85 | 4/7 | 3.57 | 5/7 | 11/20 | 0/20 | 20/20 |
| v5 iter250 | 2.35 | 3/7 | 3.00 | 5/7 | 6/20 | 5/20 | 20/20 |
| v5 iter500 | 2.45 | **6/7** | 4.14 | 6/7 | 10/20 | 6/20 | 20/20 |
| v5 iter750 | 2.20 | 2/7 | 2.29 | 4/7 | 8/20 | 4/20 | 20/20 |
| **v5 iter1000** | **2.70** | 5/7 | 3.14 | 6/7 | **14/20** | 2/20 | 20/20 |

### Pre-registered predictions (best = highest tier = iter1000, per the scorer)

- **BP1 (primary) FAILED**: oscillation 5/7 < 6/7. -> mechanical verdict HOLD.
- **BP2 PASS**: tier 2.70 >= 2.55 (beats v2-untuned).
- **BP3 PASS**: tier 2.70 > 2.45 (beats v2-best-trained).
- **BP4 PASS**: json 20/20.

### Verdict: HOLD (mechanical), but the most promising trained-Gemma-4 result yet

The pre-committed gate triggers HOLD because the highest-tier checkpoint (iter1000) does
not preserve oscillation at 6/7. Reported honestly, three things complicate the clean HOLD:

1. **iter1000 is the first trained Gemma 4 to beat untuned on tier (2.70 vs 2.55),
   teacher agreement (14/20 vs 12/20), and foundation (6/7 vs 4/7)** while nearly matching
   oscillation (5/7 vs 6/7). On overall bench quality it is the best Gemma 4 student to date
   except v3's tier (2.85), and v3 doom-looped in play.
2. **A checkpoint DID hit BP1**: iter500 preserved oscillation 6/7 AND foundation 6/7 (osc
   tier 4.14, above untuned's 4.00). The pre-registered "best-checkpoint" was operationalized
   as highest-tier, which is iter1000; under "did any checkpoint preserve 6/7," iter500 passes.
   But iter500 also has 6/20 illegal picks, so it is not cleanly shippable.
3. **Two regressions temper the optimism**: (a) v5 introduced illegal-move picks (2-6/20 vs
   untuned 0/20), a move-validity degradation from training on the tiny corpus; (b) iter1000's
   strong agreement co-occurs with the WORST val loss (2.80), i.e. the best teacher-match is
   the most overfit checkpoint, so part of the 14/20 is memorized teacher patterns, not
   learned skill. The bench states are not the training games, so the bench gain is partly
   real, but the val divergence means it is inflated.

Interpretation: the game-level won-only filter is the FIRST corpus route to move trained
Gemma 4 above untuned on the headline metrics, but it overfit hard on 892 examples (44% one
seed) and did not cleanly clear the oscillation bar. This is a "promising but not promotable"
result, materially better than v2 (all-raw, HELD) and not directly comparable to v3 (which
won tier but lost in play). It does NOT close Gemma 4 training the way the reversal filter
closed -- the obvious next experiment (v5b) directly addresses the binding constraint.

### Next step: v5b (corpus BUILT 2026-05-31, training not yet run)

The binding constraint v5 hit was corpus VOLUME (892 examples for 1000 iters -> val
divergence). There was free data on the floor: `prepare_dataset.py` dropped 621 of 1513 won
turns (41%) because their rawResponse is valid teacher JSON wrapped in a ```json ... ```
markdown fence that json.loads rejects (root cause characterised this session). A fence-strip
preprocessor (`strip_code_fence` in prepare_dataset.py) was added and dataset_v5b built:

- **v5b: 1513 usable (up from v5's 892, +69%, drops 621 -> 0). Split 1245 train / 145 valid
  / 123 test, same 10 games.**
- **CORRECTION to an earlier claim**: the fence-strip does NOT meaningfully rebalance the
  seed skew. Top seed 3263196305 is 41% of v5b vs 44% of v5 -- essentially unchanged. The
  skew is STRUCTURAL: 3263196305 is the longest won game (618 turns) and dominates any
  won-only corpus regardless of fencing. So v5b fixes VOLUME (the overfit driver) but NOT
  DIVERSITY (one board dominating). The memorization-toward-3263196305 risk persists.

v5b is the clean test of whether the won-only signal holds with adequate VOLUME. Expectation:
the val divergence should improve (more data, same iters), and if oscillation clears 6/7 at the
best-tier checkpoint, that flips the verdict. If v5b still fails BP1, won-only is exhausted on
the available wins, and the only further lever is MORE DISTINCT won games (harvester data
collection), not local compute -> pivot to the prompt track + ship untuned.
Inputs ready: dataset_v5b/ built; needs a lora_config_v5b.yaml (clone v5, data: dataset_v5b)
before launch.

### Full-game (FP1): not run yet

Deferred pending the v5b decision. If run, use the iter1000 checkpoint on seed 3263196305,
read STRICTLY as a memorization probe (that seed is ~44% of training): a good result would
show won-trajectory training can override the loop on a TRAINED board, but says nothing about
generalization. A cleaner generalization test needs a winnable seed held out of the won set.
With v5b HOLD (below), the full-game is not worth the compute and stays unrun.

## 9. v5b RESULT (measured 2026-05-31): HOLD, more data did not help

v5b trained clean (val divergence +0.65 vs v5's +1.22, so doubling the corpus halved the
overfit SLOPE as predicted). But the BENCH got WORSE, not better. Measured from
`learning_curve_v5b.json`:

| config | tier | osc | foundation | agree | illegal | json |
|---|---|---|---|---|---|---|
| v2 untuned (ref) | 2.55 | 6/7 | 4/7 | 12/20 | 0/20 | 20/20 |
| v5 best (iter1000, 892 ex) | 2.70 | 5/7 | 6/7 | 14/20 | 2/20 | 20/20 |
| v5b iter250 | 2.40 | 3/7 | 5/7 | 8/20 | 4/20 | 20/20 |
| v5b iter500 | 2.00 | 3/7 | 3/7 | 8/20 | 3/20 | 20/20 |
| v5b iter750 | 2.40 | 4/7 | 4/7 | 10/20 | 0/20 | 20/20 |
| v5b iter1000 (best tier) | 2.55 | 2/7 | 4/7 | 8/20 | 3/20 | 20/20 |

Pre-registered gates (best = highest-tier = iter1000): BP1 osc 2/7 < 6/7 FAIL (primary);
BP2 tier 2.55 >= 2.55 PASS; BP3 2.55 > 2.45 PASS; BP4 json PASS. NO v5b checkpoint anywhere
reached osc 6/7. Verdict: HOLD.

The decisive read: v5's apparent edge over untuned (tier 2.70, the headline of the v5 result)
did NOT survive the extra data. v5b, the better-powered run, regressed to tier 2.55 (= untuned)
and osc 2/7 (worse than untuned 6/7 and worse than v5's 5/7). So v5's 2.70 was the smaller,
more-overfit run getting lucky on a 20-state bench, not a robust won-only signal. The gentler
val curve (good) co-occurred with a worse bench (bad), one more confirmation that val loss does
not track bench quality on this task.

Conclusion: the SFT corpus line is EXHAUSTED. Four trained corpora -- v2 (raw), v3 (reversal-
filtered, Gemma 4), v4-A (reversal-filtered, gemma-3n), v5/v5b (won-only, two sizes) -- all
land at or below untuned, none preserve oscillation discipline, none fix the doom-loop. The
structural reason is established (teacher never demonstrates the loop, so imitation cannot teach
avoiding it; see memory teacher-corpus-won-lost-structure). Pivot is firm: the Gemma 4 track
ships NOTHING (user directive 2026-05-31 "skip shipping untuned"; untuned doom-loops in play so
it is not a release). Move the doom-loop fix to the prompt track (harvester recommendation
20260531_harvester_recommendation_doomloop_temporal_state.md) and/or an explicit loop-penalty
objective (preference training, blocked on no MLX DPO trainer). No further SFT corpus variant is
worth compute.
