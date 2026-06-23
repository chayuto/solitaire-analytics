#!/bin/zsh
# Phase 9 (2026-06-22): FOOTPRINT CONTROL for the solver-grounded lever.
# Chained AFTER phase 8 (waits for its ALL DONE marker, then GPU clear).
#
# volsolver mixes 700 solver rows = 11% of train. Is its in-dist 8 the FORM
# (solver-grounded decisions) or just 11% more VOLUME? This trains volsolver_lite
# = volume + 324 solver rows (5.4%, even-stride downsample, same diversity), the
# SAME footprint as volstrategy, then evals the 13 held-out decks @ cap300.
#
#   READ  volsolver_lite vs volsolver (8): does halving the solver rows hold the win?
#         volsolver_lite vs volstrategy (7): form 3 vs form 1 at IDENTICAL 5.4% footprint.
#         volsolver_lite vs volume (5): is even 324 solver rows above plain volume?
#
# DEFAULT axis is in-distribution (tests the existing 8, valid regardless of the
# phase-8 generalization result). If phase 8 shows volsolver GENERALIZES, the
# operator may instead want this footprint check on the generalization decks;
# that is a one-line edit (add --deck-path + swap SEEDS to the 12 gen seeds).
# tournament_A is RESUMABLE.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase9_footprint_indist.log
PHASE8LOG=gemma4_finetune/play_runs/phase8_volsolver_gen.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_phase|caffeinate" | grep -q . ; }

echo "[phase9] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/dataset_volsolver_lite/train.jsonl ]; then
  echo "[phase9] ABORT: dataset_volsolver_lite/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase 8 to FULLY complete (marker), capped ~14h, then ensure GPU clear
echo "[phase9] waiting for phase 8 completion..." | tee -a "$LOG"
for i in $(seq 1 840); do
  grep -qE "ALL DONE ===|ABORT:" "$PHASE8LOG" 2>/dev/null && break
  sleep 60
done
echo "[phase9] phase 8 marker seen; ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

# 2. train the footprint-matched arm
echo "[phase9] training adapters_volsolver_lite (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volsolver_lite.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volsolver_lite/adapters.safetensors ]; then
  echo "[phase9] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase9] TRAIN DONE" | tee -a "$LOG"

# 3. eval @ cap300 on the 13 held-out decks
echo "[phase9] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done
echo "[phase9] eval volsolver_lite @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volsolver_lite --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase9_volsolver_lite 2>&1 | tee -a "$LOG"

# 4. adjudicate non-win finals
echo "[phase9] adjudicating volsolver_lite finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase9_volsolver_lite/volsolver_lite/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase9] === $(date) ALL DONE ===" | tee -a "$LOG"
