#!/bin/zsh
# Phase 2 (2026-06-20): does the harvest growth (+4 new 31B wins) move the
# plain-volume baseline? Retrain `volume` on the CURRENT 31B pool and eval.
#
# Runs CHAINED behind the phase-1 volcloseout confirmation: the gpu_busy guard
# blocks until phase 1's eval frees the GPU, so only one MLX job is on the GPU
# at a time (16 GB cannot hold two). Phase-1 adjudication (CPU solver) may
# overlap phase-2 training harmlessly.
#
#   ARM   volume_v0620 = volume recipe, dataset_volume_v0620 (8115 train, the
#         9937-row 31B pool with +560 new winning rows), identical hypers.
#   READ  wins vs volume's STABLE 5/13 (measured twice). Since volume's wins did
#         not vary run-to-run, a +2 or more delta is likely real +data; +1 is
#         ambiguous under the harness parse-retry nondeterminism.
#
# The corpus (dataset_volume_v0620) was built + validated INTERACTIVELY before
# launch (no eval-deck leakage, clean +data); this script only guards it exists.
# tournament_A is RESUMABLE, so an interrupted run resumes on re-launch.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase2_volume_v0620.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
# Match the tournament_A orchestrator too (not just play_deck): it runs
# CONTINUOUSLY through all 13 games, so phase 2 cannot slip into a between-games
# gap and start training while phase 1 is still on the GPU.
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_volcloseout|run_phase2|caffeinate" | grep -q . ; }

echo "[phase2] === $(date) START ===" | tee -a "$LOG"

# guard: corpus built+validated interactively before launch
if [ ! -s gemma4_finetune/dataset_volume_v0620/train.jsonl ]; then
  echo "[phase2] ABORT: dataset_volume_v0620/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase-1 eval to free the GPU, then train
echo "[phase2] waiting for GPU (phase-1 eval finishes first)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[phase2] training adapters_volume_v0620 (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volume_v0620.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volume_v0620/adapters.safetensors ]; then
  echo "[phase2] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase2] TRAIN DONE" | tee -a "$LOG"

# 2. eval volume_v0620 @ cap300 on the 13 held-out decks
echo "[phase2] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[phase2] eval volume_v0620 @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volume_v0620 --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase2_v0620 2>&1 | tee -a "$LOG"

# 3. adjudicate non-win finals (winnable-vs-dead per stall)
echo "[phase2] adjudicating volume_v0620 finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase2_v0620/volume_v0620/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase2] === $(date) ALL DONE ===" | tee -a "$LOG"
