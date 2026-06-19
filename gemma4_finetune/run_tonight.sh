#!/bin/zsh
# Overnight close-out experiment (2026-06-17). One command, unattended.
#
# Resolves the loopcompress false-resign (replay-verified: student reaches winnable
# fd=0 endgames with legal foundation plays available and resigns ~5 moves short).
# Diagnosis = emergent close-out gap, so we AUGMENT won-game close-out behaviour.
#
# Pipeline:
#   1. build dataset_closeout (skip if already built/validated)
#   2. train adapters_closeout (~80 min, iters 1000, volume-identical hypers)
#   3. eval @ cap 300 on the 13 held-out decks: closeout (experiment), then
#      loopcompress + volume (clean cap-300 baselines). tournament_A is RESUMABLE
#      and per-game timeout-bounded, so if the window ends early the experiment
#      arm (first) is done and the baselines resume next run.
#   4. adjudicate every non-win final (max_turns / resigned / stalled) with the
#      sound solver -> the false-resign verdict table.
#
# All numbers must be read from play_runs/tonightEval/<arm>/leaderboard + the
# adjudication output; do not infer wins from anything else.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/tonight_run.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep|run_tonight" | grep -q . ; }

echo "[tonight] === $(date) START ===" | tee -a "$LOG"

# 1. corpus (idempotent; keep the already-validated artifact if present)
if [ ! -s gemma4_finetune/dataset_closeout/train.jsonl ]; then
  echo "[tonight] building dataset_closeout..." | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/build_closeout_corpus.py 2>&1 | tee -a "$LOG"
else
  echo "[tonight] dataset_closeout present -> skip build" | tee -a "$LOG"
fi

# 2. train
echo "[tonight] waiting for GPU to free (train)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[tonight] training adapters_closeout (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_closeout.yaml ) 2>&1 | tee -a "$LOG"
echo "[tonight] TRAIN DONE -> gemma4_finetune/adapters_closeout" | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_closeout/adapters.safetensors ]; then
  echo "[tonight] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi

# 3. eval @ cap 300 (closeout first, then baselines; resumable)
echo "[tonight] waiting for GPU to free (eval)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[tonight] eval closeout,loopcompress,volume @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms closeout,loopcompress,volume \
  --seeds "$SEEDS" \
  --max-turns 300 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name tonightEval --prompt-version v1.6 2>&1 | tee -a "$LOG"
echo "[tonight] EVAL DONE" | tee -a "$LOG"

# 4. adjudicate every non-win final (incl. resigned -> false-resign test)
echo "[tonight] adjudicating non-win finals..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py \
  gemma4_finetune/play_runs/tonightEval/*/seed*/ 2>&1 | tee -a "$LOG"

echo "[tonight] === $(date) ALL DONE ===" | tee -a "$LOG"
