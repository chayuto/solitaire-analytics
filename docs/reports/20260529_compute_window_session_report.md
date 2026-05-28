# Compute window session report: temperature probe + v4-A staging

**Date**: window 2026-05-28 evening to 2026-05-29 (closed early)
**Researcher**: Chayut Orapinpatipat (with Claude Opus 4.7)
**Pre-registration**: `docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md`
**Companion**: `docs/reports/20260527_full_game_play_compute_window_report.md` (the window this built on)

---

## Executive summary

The window was budgeted at 4-5 hours and closed early, after the cheap front-loaded
probe but before the expensive training arm. Two things were accomplished, one of
them a clean and decisive result:

1. **Temperature micro-probe (RAN, decisive negative).** The Gemma 4 E2B `QS col5/col7`
   doom-loop is NOT a greedy-decoding artefact. On the exact loop board (seed
   3263196305, turn 12), the untuned model picked the loop move 20/20 at temperature
   0.4, 0.7, AND 1.0, never once taking the available "draw" escape. The loop is a
   reasoning-level attractor that survives sampling. Inference-time temperature is not
   a fix.
2. **v4-A fully staged but NOT trained.** The deck pool was regenerated and a latent
   data-loss bug in the regenerator was fixed; the v4-A dataset was built and verified;
   the pre-registration and config were locked and committed. Training was queued but
   the window closed before it ran. `adapters_v4a/` does not exist.

The headline scientific contribution of the window is the temperature negative, which
materially strengthens the prior window's "doom-loop is base-model-deep" finding.

---

## 1. What ran: the temperature micro-probe

### 1.1 Question and pre-registered predictions

Pre-registered in the plan doc section 4. The deterministic (temperature 0.0) 294-turn
QS oscillation observed in the prior window is the signature of a greedy-decoding fixed
point. The probe asked: does sampling escape it?

- **MP1**: temperature 0.0 is deterministic on this board (sanity).
- **MP2 (primary)**: escape probability > 0.20 at temperature 0.7.
- **MP3**: if MP2 holds, a full-game temperature run is warranted; if escape is ~0
  even at temperature 1.0, no full-game temp run.

### 1.2 Method

`gemma4_finetune/temp_probe_qs_loop.py`. Hydrates seed 3263196305 and replays GREEDILY
(temperature 0.0) to turn 12 using `play_deck_with_student.py`'s own loop functions,
then renders the exact prompt and samples N=20 at each of temperature 0.4 / 0.7 / 1.0
(plus a temperature-0.0 sanity sample). Escape = a sampled `move_index` other than the
greedy pick.

Faithfulness self-check: the greedy replay reproduced the recorded untuned run's
`move_index` at every turn (12/12 match, 0 mismatch). The probed prompt is therefore
byte-identical to the one the original run saw at turn 12.

### 1.3 Results

The two legal moves at turn 12 were:
- `[0] Move QS from column 7 to column 5` (the loop move)
- `[1] Draw the next card from the stock onto the waste` (a genuine loop-breaking escape)

| temperature | move_index distribution (N=20) | parse failures | escape probability |
|---|---|---|---|
| 0.0 (sanity) | move 0 | 0 | deterministic |
| 0.4 | {0: 20} | 0 | 0.00 |
| 0.7 | {0: 20} | 0 | 0.00 |
| 1.0 | {0: 20} | 0 | 0.00 |

Result artefact: `gemma4_finetune/play_runs/temp_probe_qs_loop_result.json`.

### 1.4 Prediction check

| prediction | predicted | observed | verdict |
|---|---|---|---|
| MP1 | deterministic at temp 0 | deterministic, matches recorded | PASS |
| MP2 (primary) | escape > 0.20 at temp 0.7 | escape = 0.00 | FALSIFIED |
| MP3 | full-game temp run only if MP2 | escape ~0 at temp 1.0 | no temp run; option dropped |

### 1.5 Interpretation

The QS loop is robust to sampling temperature. Even at temperature 1.0, where token
sampling is maximally stochastic, 20/20 generations selected the loop move over the
available draw. The model's preference for the QS swap is not a knife-edge logit that a
little noise tips over; the entire reasoning chain reconstructs the same conclusion each
time, and the final `move_index` token is overwhelmingly constrained by that chain.

This is a reasoning-level attractor, not a decoding-level one. Two consequences:

1. **Inference-time temperature is not a fix for the doom-loop.** A cheap "just raise
   the temperature" intervention is ruled out.
2. **It strengthens the base-model-deep finding.** The prior window showed the loop
   survives every corpus intervention (untuned, v2, v3 all identical). This window shows
   it also survives sampling. The pathology is deep in the model-plus-prompt interaction.

**Nuance vs the teacher.** The curated `known_outcomes` for deck 2967897202 show the 31B
teacher diverging stochastically (win vs stall) on a different board at its production
temperature 0.3. The E2B student on the QS board does not diverge even at temperature
1.0. So the student's loop is more entrenched than the teacher's stochastic divergence
points. Whether that is a 31B-vs-E2B capacity difference or a board-specific difference
(the 2967897202 divergence may have been a genuinely balanced decision while the QS
board is not) is not resolved here.

---

## 2. What was prepped (no-GPU)

### 2.1 Deck pool regenerated and a latent regenerator bug fixed

`data/benchmarks/winnable_decks.json` was rebuilt from the 6 win-records on disk. The
naive regeneration exposed that `scripts/build_winnable_decks.py` was LOSSY: it dropped
the hand-curated `known_outcomes` field (schema v2) on every run, and it emitted one
record per win-file rather than deduping by seed. Fixed: the regenerator now dedupes by
seed and carries `known_outcomes` forward by seed (it cannot reconstruct that field, so
carry-forward is the only safe path). Result: 4 unique decks (seeds 2853966634,
2967897202, 3263196305, plus one seedless), schema v2 preserved.

### 2.2 Teacher temperature discovered

The curated `known_outcomes` for seed 2967897202 records the 31B teacher's production
**inference temperature as 0.3**, and documents the same deck producing both a win and a
stall under temperature 0.3 (divergence at turn 15 from a byte-identical 15-move prefix).
This resolves a confound the v2 lab log flagged as open ("teacher's effective inference
temperature is not known") and motivated the temperature probe.

### 2.3 v4-A staged and verified

- `gemma4_finetune/lora_config_v4a.yaml`: byte-for-byte clone of the v1.1 recipe
  (`lora_config.yaml`, gemma-3n base) with only `data` and `adapter_path` changed.
  Corpus is the single variable vs v1.1.
- `gemma4_finetune/dataset_v4a/`: game-level split of the frozen filtered corpus
  (`data/dataset/training_shuffle_filtered.jsonl`, 1832 rows). **TP1 confirmed exactly**:
  1279 train / 144 valid / 168 test, identical to v3, proving the frozen-corpus guarantee.

### 2.4 Committed

Commit `9f0e6a6`: the regenerated deck pool, the non-lossy regenerator, the v4-A
pre-registration, and `lora_config_v4a.yaml`.

---

## 3. What did NOT run, and why

The window closed before the GPU training arm. The following queue items did not run:

| step | status | reason |
|---|---|---|
| v4-A training (1000 iters, gemma-3n) | NOT RUN | window closed before launch; one launch attempt hit an internal tool error, the relaunch was stopped at window close |
| v4-A 20-state bench sweep | NOT RUN | gated on training |
| v4-A full-game play on 3263196305 | NOT RUN | gated on training |
| flex: v1.1 generality on new decks | NOT RUN | flex, lowest priority |

`adapters_v4a/` does not exist. All v4-A INPUTS are staged and verified, so the next
window can launch training immediately with no re-prep.

---

## 4. Scientific implications

1. The doom-loop is robust to temperature, not just to corpus content. Two independent
   negative controls (corpus filtering in the prior window, sampling temperature in this
   one) now point at the same conclusion: the pathology is base-model-and-prompt deep.
2. The cheapest candidate fix (inference-time temperature) is eliminated. The remaining
   live levers are: (a) the v4-A corpus-filter test on the base that actually plays
   (still un-run), (b) harvester-side resign + state-repetition annotation, (c) the
   declined won-games corpus retrain.
3. v4-A's expected value is now lower than when it was pre-registered. If corpus
   filtering changed nothing on Gemma 4 in play, and temperature changes nothing, the
   prior that filtering helps gemma-3n in play is weaker. v4-A is still worth running
   (it is the only un-tested corpus-vs-base cell and it is cheap), but the HOLD gate is
   the most likely outcome by this reasoning. Run it, but pre-commit to pivoting to the
   harvester-side levers if it HOLDs.

---

## 5. Reproducibility

```bash
# Temperature probe (ran this window)
.venv/bin/python gemma4_finetune/temp_probe_qs_loop.py
# -> gemma4_finetune/play_runs/temp_probe_qs_loop_result.json

# v4-A dataset (built this window; reproduces v3 split exactly)
cd gemma4_finetune
./venv/bin/python prepare_dataset.py \
    --log ../data/dataset/training_shuffle_filtered.jsonl --out dataset_v4a

# v4-A training (NOT run; the immediate next step)
./venv/bin/python train_v2.py --config lora_config_v4a.yaml > /tmp/v4a_train.log 2>&1 &
```

---

## 6. Honest assessment

What went right:
- Front-loading the cheap probe paid off exactly as intended: a 15-minute experiment
  produced a decisive, pre-registered negative before any expensive compute was spent.
- The deck-pool regeneration caught and fixed a real data-loss bug, and surfaced the
  teacher-temperature fact that the probe then exploited.
- v4-A is fully staged and verified, so no window time is lost to re-prep next time.

What was suboptimal:
- The training arm did not run. The window closed earlier than the 4-5 hours budgeted.
  Net: the window delivered the probe result and the prep, not the v4-A verdict.
- One training launch hit an internal tool error and produced no process; this cost a
  small amount of confusion but no real time.

What we still do not know:
- Whether v4-A (gemma-3n + filtered corpus) preserves or improves full-game competence.
  This is the single open question carried to the next window.
- Whether v1.1's "competent ~35 turns then loop" generalises beyond seed 3263196305.
