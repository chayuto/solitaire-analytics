#!/bin/zsh
# Phase 7 (2026-06-21): SOLVER-GROUNDED play-matched rows (step 4, form 3).
# Chained AFTER phase 6.
#
# Waits for phase 6 (the close-out + strategy combo) to FULLY finish (its log
# "ALL DONE ==="/"ABORT:" marker, capped), ensures the GPU is clear, then trains
# adapters_volsolver and evals the 13 held-out decks @ cap300 so the result is
# directly comparable to volume (5), volstrategy (7) and volcloseout (8), all at
# cap300.
#
#   ARM   volsolver = volume corpus + N solver-grounded rows whose targets are
#         moves on a solver-proven winning line, in the exact v1.6 play format
#         (dataset_volsolver). The form-3 escalation of the strategy lever.
#   READ  volsolver vs volstrategy (does solver-grounded form 3 beat hand-authored
#         form 1) and vs volcloseout (does grounded play approach the win lever);
#         a win-rate above the teacher's ~31% imitation would be the headline.
# Corpus built + validated INTERACTIVELY before launch; this guards it exists.
# tournament_A is RESUMABLE.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase7_volsolver.log
PHASE6LOG=gemma4_finetune/play_runs/phase6_volcombo.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_phase|caffeinate" | grep -q . ; }

echo "[phase7] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/dataset_volsolver/train.jsonl ]; then
  echo "[phase7] ABORT: dataset_volsolver/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase 6 to FULLY complete (marker), capped ~14h, then ensure GPU clear
echo "[phase7] waiting for phase 6 ALL DONE..." | tee -a "$LOG"
for i in $(seq 1 840); do
  grep -qE "ALL DONE ===|ABORT:" "$PHASE6LOG" 2>/dev/null && break
  sleep 60
done
echo "[phase7] phase 6 marker seen; ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

# 2. train
echo "[phase7] training adapters_volsolver (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volsolver.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volsolver/adapters.safetensors ]; then
  echo "[phase7] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase7] TRAIN DONE" | tee -a "$LOG"

# 3. eval @ cap300 on the 13 held-out decks
echo "[phase7] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done
echo "[phase7] eval volsolver @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volsolver --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase7_volsolver 2>&1 | tee -a "$LOG"

# 4. adjudicate non-win finals
echo "[phase7] adjudicating volsolver finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase7_volsolver/volsolver/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase7] === $(date) ALL DONE ===" | tee -a "$LOG"
