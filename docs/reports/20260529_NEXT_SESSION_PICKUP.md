# Next session pickup pointer

**Last touched**: 2026-05-29 (compute window closed early)
**Compute state at end**: idle, no active runs. v4-A training staged but NOT started.
**Branch**: `main`, last commit `9f0e6a6` (prep). One new untracked file + the report docs are uncommitted (see section 4).

## Read this first

This is the single read-on-resume document. Order to read the rest only if you need depth:

1. `docs/reports/20260529_compute_window_session_report.md` - what ran this window and the temperature-probe result
2. `docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md` - the locked pre-registration (v4-A predictions + gates)
3. `docs/reports/20260527_full_game_play_compute_window_report.md` - the prior window (full-game runner, base-model-deep finding)

## State of the world in one paragraph

The QS doom-loop is now confirmed robust to BOTH corpus filtering (prior window) AND
sampling temperature (this window: 20/20 loop move at temp 0.4/0.7/1.0). Inference-time
temperature is not a fix. The next experiment, v4-A (gemma-3n + frozen reversal-filtered
corpus), is fully staged and verified but was NOT trained before the window closed.
`adapters_v4a/` does not exist; `dataset_v4a/` and `lora_config_v4a.yaml` are ready.

## The immediate next step (turnkey, ~170 min)

v4-A is pre-registered with locked predictions and decision gates in
`docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md` section 3. Inputs are
verified ready. Just run:

```bash
cd /Users/chayut/repos/solitaire-analytics/gemma4_finetune

# 1. Train (~95 min, gemma-3n + frozen filtered corpus; dataset_v4a already built)
./venv/bin/python train_v2.py --config lora_config_v4a.yaml > /tmp/v4a_train.log 2>&1 &
# checkpoints land at adapters_v4a/000{0250,0500,0750,1000}_adapters.safetensors

# 2. After training: 20-state sweep (~25 min). NOTE: the v3 sweep script and
#    posttune_n20_gemma4_text_runner.py are hardcoded to the Gemma-4 TEXT base.
#    v4-A is gemma-3n, so you must point the runner at the gemma-3n base
#    (mlx-community/gemma-3n-E2B-it-text-4bit-dwq) WITHOUT gemma4_text_patch.
#    Cleanest path: write sweep_v4a_checkpoints.sh + a gemma-3n posttune runner
#    (adapt from the v1 posttune runner, not the gemma4_text one). This was NOT
#    done this window. Budget ~20 min of editing before the sweep runs.

# 3. v4-A best checkpoint full-game on seed 3263196305 (~50 min)
.venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed 3263196305 \
    --model-id mlx-community/gemma-3n-E2B-it-text-4bit-dwq \
    --adapter-path gemma4_finetune/adapters_v4a/<best_ckpt_staged_dir> \
    --out-dir gemma4_finetune/play_runs/v4a_seed3263196305_run1 \
    --max-turns 300 --max-tokens 2048
```

### v4-A decision gates (pre-committed)

- **PROMOTE**: full-game fc > 3 AND bench tier >= 3.15. Ship v4-A as canonical, supersede v1.1.
- **PARTIAL**: full-game fc == 3 AND bench tier >= 3.15. v1.1 stays canonical; documented tie.
- **HOLD**: full-game fc < 3 OR bench tier < 3.15. Close the corpus-filter program; pivot to
  harvester-side levers (resign + state-repetition annotation) and/or the won-games retrain.

Honest expectation (see session report section 4): given that corpus filtering changed
nothing on Gemma 4 in play AND temperature changes nothing, HOLD is the most likely
outcome. Run it anyway (only un-tested corpus-vs-base cell, cheap), but be ready to pivot.

## Open queue after v4-A

1. **Flex generality** (only if v4-A is interesting): run deployed v1.1 LoRA
   (`adapters_t5_at750`) full-game on seeds 2853966634 and 2967897202 at temp 0.0. Tests
   whether "competent ~35 turns then doom-loop" is v1.1-general or a 3263196305 artefact.
2. **Won-games corpus retrain** (declined for the prior window, candidate next): retrain
   gemma-3n on the grown 2844-row corpus that now includes the teacher's WON v1.2 games
   (c05ad4, aca45a). Winning trajectories are the multi-turn-planning signal v1.1 lacked.
   Confounds base + corpus-size + win-content; keep it separate from v4-A.
3. **Harvester-side track** (gated on harvester team producing 31B-on-v1.3 traces, NOT
   local compute): the v1.3 anti-undo design hole (see memory `v1-3-anti-undo-predicate-design-hole`)
   and the v1.4 INTENT hypothesis. Not a compute-window item.

## Confounds and caveats to remember

- The frozen filtered corpus `data/dataset/training_shuffle_filtered.jsonl` (1832 rows) is
  GITIGNORED and lives on local disk only. Do NOT delete it; v4-A and any re-run depend on
  it. Do NOT re-run `filter_shuffles.py` on the grown corpus for v4-A (confounds the base
  comparison).
- All bench/play measurements are single-run, temperature 0.0, deterministic. Phase 1.5
  noise floor is +-0.40 mean tier per single-run measurement.
- The 26B-a4b traces are excluded from training by `TEACHER_MODEL=gemma-4-31b-it`. Do not
  mix them into any v4 training set.

## Uncommitted at handover (section 4)

- `gemma4_finetune/temp_probe_qs_loop.py` (new, the probe script) - untracked
- `docs/reports/20260529_compute_window_session_report.md` (new) - untracked
- `docs/reports/20260529_NEXT_SESSION_PICKUP.md` (this file) - untracked
- `gemma4_finetune/play_runs/temp_probe_qs_loop_result.json` (the probe result)
- Pre-existing (NOT this session's work, from earlier ingest): `data/DATASET_NOTES.md`,
  `data/SUMMARY.md`, `data/index/manifest.jsonl`, `data/publish/README.md`,
  `.claude/skills/solitaire-analyst/scripts/check_winnability.py`
- `dataset_v4a/` is gitignored (regenerable in ~5s via prepare_dataset.py)

Suggested commit grouping for the next session: (1) the probe script + result + the two
report docs as one "temperature probe + session report" commit; (2) the pre-existing
ingest changes separately (they are a different logical unit).

## One-line summary if you only have time for one line

Temperature does NOT break the doom-loop (20/20 at temp 1.0); v4-A is staged but un-trained;
next step is `train_v2.py --config lora_config_v4a.yaml` then sweep (needs a gemma-3n runner) then full-game.
