#!/bin/zsh
# Phase 5 (2026-06-21): GENERALIZATION pass for the close-out recipe arms.
#
# Closes the open publish gate on volcloseout (scored in-dist only so far) and
# is the tiebreaker between volcloseout and volcloseout_v0620. Plays both on the
# SAME 12 fresh solver-confirmed-winnable decks the volume lead was scored on
# (data/benchmarks/generalization_decks.json), at the SAME settings the prior
# genTest run used (cap200, v1.6, parse-retry 0.3), into the SAME out-name so
# the comparison is apples-to-apples with the existing base/volume/wononly-gate
# results already in play_runs/genTest/.
#
# All 12 decks are winnable BY CONSTRUCTION, so any resign is a FALSE resign and
# any non-win is a failure-to-finish under the cap. Read the PAIRED delta vs
# volume (5/12, meanFC 28.5, 2 false-resigns): does the recipe add wins AND kill
# the false-resign on fresh decks the way it did in-distribution.
#
# tournament_A is resumable (skips games with an existing summary.json), so the
# already-played base/volume/wononly-gate arms are untouched; only the two new
# arms run (24 games).
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase5_generalization_recipe.log
SEEDS=9000002,9000003,9000005,9000008,9000010,9000013,9000020,9000021,9000023,9000024,9000025,9000026
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
# GPU consumers only (the prior genTest run used the same guard set minus tourA).
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep|run_phase|caffeinate" | grep -q . ; }

echo "[phase5] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/adapters_volcloseout/adapters.safetensors ]; then
  echo "[phase5] ABORT: adapters_volcloseout missing" | tee -a "$LOG"; exit 1
fi
if [ ! -s gemma4_finetune/adapters_volcloseout_v0620/adapters.safetensors ]; then
  echo "[phase5] ABORT: adapters_volcloseout_v0620 missing" | tee -a "$LOG"; exit 1
fi
if [ ! -s data/benchmarks/generalization_decks.json ]; then
  echo "[phase5] ABORT: generalization_decks.json missing" | tee -a "$LOG"; exit 1
fi

echo "[phase5] ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

echo "[phase5] eval volcloseout,volcloseout_v0620 @ cap200 on 12 fresh winnable decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py \
  --arms volcloseout,volcloseout_v0620 \
  --deck-path data/benchmarks/generalization_decks.json \
  --seeds "$SEEDS" \
  --max-turns 200 $=EVAL_FLAGS \
  --out-name genTest 2>&1 | tee -a "$LOG"

echo "[phase5] === $(date) ALL DONE ===" | tee -a "$LOG"
