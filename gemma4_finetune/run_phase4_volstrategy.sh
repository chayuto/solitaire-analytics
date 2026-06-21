#!/bin/zsh
# Phase 4 (2026-06-20): STRATEGY-TEXT probe. Chained AFTER phase 3.
#
# Waits for phase 3 to FULLY finish (its log "ALL DONE"/"ABORT" marker, capped),
# then trains, evals the 13 decks, adjudicates. Same no-GPU-collision discipline
# as phase 3: the marker gate, not a bare GPU gap.
#   ARM   volstrategy = volume + 27 strategy Q&A rows (x12, 5.4% of train).
#   READ  wins/meanFC vs volume's stable 5/13 (pure strategy effect), AND the
#         JSON parse-rescue rate (prose rows may erode JSON discipline).
# Corpus built + validated INTERACTIVELY before launch; this guards it exists.
# tournament_A is RESUMABLE.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase4_volstrategy.log
PHASE3LOG=gemma4_finetune/play_runs/phase3_volcloseout_v0620.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_volcloseout|run_phase2|run_phase3|run_phase4|caffeinate" | grep -q . ; }

echo "[phase4] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/dataset_volstrategy/train.jsonl ]; then
  echo "[phase4] ABORT: dataset_volstrategy/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase 3 to FULLY complete (marker), capped ~14h, then ensure GPU clear
echo "[phase4] waiting for phase 3 ALL DONE..." | tee -a "$LOG"
for i in $(seq 1 840); do
  # match the completion marker "=== ... ALL DONE ===" / "ABORT:" ONLY -- NOT
  # phase 3's own "waiting for phase 2 ALL DONE..." line (substring trap).
  grep -qE "ALL DONE ===|ABORT:" "$PHASE3LOG" 2>/dev/null && break
  sleep 60
done
echo "[phase4] phase 3 marker seen; ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

# 2. train
echo "[phase4] training adapters_volstrategy (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volstrategy.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volstrategy/adapters.safetensors ]; then
  echo "[phase4] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase4] TRAIN DONE" | tee -a "$LOG"

# 3. eval @ cap300 on the 13 held-out decks
echo "[phase4] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done
echo "[phase4] eval volstrategy @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volstrategy --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase4_volstrategy 2>&1 | tee -a "$LOG"

# 4. adjudicate non-win finals
echo "[phase4] adjudicating volstrategy finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase4_volstrategy/volstrategy/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase4] === $(date) ALL DONE ===" | tee -a "$LOG"
