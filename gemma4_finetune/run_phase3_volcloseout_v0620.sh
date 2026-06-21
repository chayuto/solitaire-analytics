#!/bin/zsh
# Phase 3 (2026-06-20): the SHIP-CANDIDATE arm. Gentle close-out recipe applied
# to the +data pool (recipe + data together). Chained AFTER phase 2.
#
# Waits for phase 2 to FULLY finish (its log "ALL DONE"/"ABORT" marker, capped),
# NOT just a GPU gap, so it cannot slip into phase-2's train->eval handoff and
# collide on the 16 GB GPU. Then trains, evals the 13 decks, adjudicates.
#   ARM   volcloseout_v0620 = dataset_volcloseout_v0620 (9792 train), same hypers.
#   READ  vs volume_v0620 (recipe effect on +data); vs old volcloseout (~8 on old
#         pool, does the recipe survive +data). GATE meanFC vs volume_v0620.
# Corpus built + validated INTERACTIVELY before launch; this guards it exists.
# tournament_A is RESUMABLE, so an interrupted run resumes on re-launch.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase3_volcloseout_v0620.log
PHASE2LOG=gemma4_finetune/play_runs/phase2_volume_v0620.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_volcloseout|run_phase2|run_phase3|caffeinate" | grep -q . ; }

echo "[phase3] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/dataset_volcloseout_v0620/train.jsonl ]; then
  echo "[phase3] ABORT: dataset_volcloseout_v0620/train.jsonl missing" | tee -a "$LOG"; exit 1
fi

# 1. wait for phase 2 to FULLY complete (marker), capped ~12h, then ensure GPU clear
echo "[phase3] waiting for phase 2 ALL DONE..." | tee -a "$LOG"
for i in $(seq 1 720); do
  grep -qE "ALL DONE|ABORT" "$PHASE2LOG" 2>/dev/null && break
  sleep 60
done
echo "[phase3] phase 2 marker seen; ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

# 2. train
echo "[phase3] training adapters_volcloseout_v0620 (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volcloseout_v0620.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volcloseout_v0620/adapters.safetensors ]; then
  echo "[phase3] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[phase3] TRAIN DONE" | tee -a "$LOG"

# 3. eval @ cap300 on the 13 held-out decks
echo "[phase3] waiting for GPU (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done
echo "[phase3] eval volcloseout_v0620 @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volcloseout_v0620 --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name phase3_v0620 2>&1 | tee -a "$LOG"

# 4. adjudicate non-win finals (winnable-vs-dead per stall)
echo "[phase3] adjudicating volcloseout_v0620 finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/phase3_v0620/volcloseout_v0620/seed*/ 2>&1 | tee -a "$LOG"

echo "[phase3] === $(date) ALL DONE ===" | tee -a "$LOG"
