#!/bin/zsh
# Phase 6 (2026-06-21): CLOSE-OUT + STRATEGY combo. Chained AFTER phase 5.
#
# Waits for phase 5 (the generalization pass) to FULLY finish (its log
# "ALL DONE ==="/"ABORT:" marker, capped), ensures the GPU is clear, then trains
# adapters_volcombo and evals the 13 held-out decks @ cap300 so the result is
# directly comparable to volcloseout (8/13, 0 resign) and volstrategy (7/13,
# 4 correct + 1 false resign), both measured at cap300.
#
#   ARM   volcombo = volcloseout corpus + 27 strategy rows x12 (dataset_volcombo,
#         6768 train; valid/test byte-identical to volume).
#   READ  does it hold 8 wins AND add correct dead-board resigns WITHOUT
#         re-introducing volstrategy's over-resign on a winnable deck.
# Corpus built + validated INTERACTIVELY before launch; this guards it exists.
# tournament_A is RESUMABLE.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase6_volcombo.log
PHASE5LOG=gemma4_finetune/play_runs/phase5_generalization_recipe.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_phase|caffeinate" | grep -q . ; }

echo "[phase6] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/dataset_volcombo/train.jsonl ]; then
  echo "[phase6] ABORT: dataset_volcombo/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase 5 to FULLY complete (marker), capped ~14h, then ensure GPU clear
echo "[phase6] waiting for phase 5 ALL DONE..." | tee -a "$LOG"
for i in $(seq 1 840); do
  # match the completion marker "=== ... ALL DONE ===" / "ABORT:" ONLY.
  grep -qE "ALL DONE ===|ABORT:" "$PHASE5LOG" 2>/dev/null && break
  sleep 60
done
echo "[phase6] phase 5 marker seen; ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

# 2. train
echo "[phase6] training adapters_volcombo (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volcombo.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volcombo/adapters.safetensors ]; then
  echo "[phase6] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase6] TRAIN DONE" | tee -a "$LOG"

# 3. eval @ cap300 on the 13 held-out decks
echo "[phase6] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done
echo "[phase6] eval volcombo @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volcombo --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase6_volcombo 2>&1 | tee -a "$LOG"

# 4. adjudicate non-win finals
echo "[phase6] adjudicating volcombo finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase6_volcombo/volcombo/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase6] === $(date) ALL DONE ===" | tee -a "$LOG"
