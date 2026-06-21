#!/bin/zsh
# Overnight (2026-06-19): gentle close-out on VOLUME + the cap-500 budget check.
# One command, unattended. Follows docs/reports/20260619_closeout_augmentation_eval.md
# section 7 (7.2 gentle-aug-on-volume, 7.3 cap-500 diagnostic).
#
# Last night: closeout (loopcompress + 2x fd<=2 oversample) killed the false-resign
# but regressed mid-game reach and did NOT beat volume. This run tests the lesson:
#   ARM  volcloseout = volume corpus + GENTLE (1 extra copy) close-out oversample.
#        Validated gentle: draw 44.7->41.4%, foundation 21.9->27.7% (closeout was +11.7pt).
#   GATE volcloseout must beat volume on wins AND not drop meanFC (the metric that
#        exposed closeout's mid-game regression).
#
# Pipeline (headline first, so a short window still finishes the experiment):
#   1. build dataset_volcloseout (skip if present)
#   2. train adapters_volcloseout (~80 min, volume-identical hypers)
#   3. eval volcloseout @ cap300 on the 13 held-out decks  (HEADLINE)
#   4. adjudicate volcloseout (bounded -> winnable-vs-dead per stall)
#   5. eval volume @ cap300 same window (clean same-run baseline vs volcloseout)
#   6. cap-500 diagnostic: volume on its 4 near-win winnable stalls (budget vs corpus)
#   7. adjudicate the cap300 baseline + the cap500 diagnostic
# tournament_A is RESUMABLE, so if the window ends early the headline (1-4) is done
# and 5-7 resume next run.
#
# All numbers must be read from play_runs/<out>/leaderboard + the adjudication
# output; do not infer wins from anything else.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/volcloseout_run.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
CAP500_SEEDS=3123337720,350743738,405489085,4197389931   # volume's 4 winnable cap-300 stalls
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep|run_volcloseout|run_tonight" | grep -q . ; }
adj() { .venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 "$@" 2>&1 | tee -a "$LOG"; }

echo "[volcloseout] === $(date) START ===" | tee -a "$LOG"

# 1. corpus (idempotent)
if [ ! -s gemma4_finetune/dataset_volcloseout/train.jsonl ]; then
  echo "[volcloseout] building dataset_volcloseout..." | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/build_volcloseout_corpus.py 2>&1 | tee -a "$LOG"
else
  echo "[volcloseout] dataset_volcloseout present -> skip build" | tee -a "$LOG"
fi

# 2. train
echo "[volcloseout] waiting for GPU to free (train)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[volcloseout] training adapters_volcloseout (iters 1000)..." | tee -a "$LOG"
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_volcloseout.yaml ) 2>&1 | tee -a "$LOG"
if [ ! -s gemma4_finetune/adapters_volcloseout/adapters.safetensors ]; then
  echo "[volcloseout] ABORT: training produced no adapter weights" | tee -a "$LOG"; exit 1
fi
echo "[volcloseout] TRAIN DONE" | tee -a "$LOG"

# 3. HEADLINE eval: volcloseout @ cap300
echo "[volcloseout] waiting for GPU (eval volcloseout)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[volcloseout] eval volcloseout @ cap300 on 13 decks..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volcloseout --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name tonightEval2 2>&1 | tee -a "$LOG"

# 4. adjudicate the headline arm
echo "[volcloseout] adjudicating volcloseout finals..." | tee -a "$LOG"
adj gemma4_finetune/play_runs/tonightEval2/volcloseout/seed*/

# 5. same-window volume baseline @ cap300
echo "[volcloseout] waiting for GPU (eval volume baseline)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[volcloseout] eval volume @ cap300 on 13 decks (same-window baseline)..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volume --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name tonightEval2 2>&1 | tee -a "$LOG"

# 6. cap-500 budget diagnostic: volume on its 4 near-win winnable stalls
echo "[volcloseout] waiting for GPU (cap500 diagnostic)..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done
echo "[volcloseout] cap-500 diagnostic: volume on $CAP500_SEEDS ..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volume --seeds "$CAP500_SEEDS" \
  --max-turns 500 $=EVAL_FLAGS --out-name volCap500 2>&1 | tee -a "$LOG"

# 7. adjudicate the cap300 baseline + the cap500 diagnostic
echo "[volcloseout] adjudicating baseline + cap500 finals..." | tee -a "$LOG"
adj gemma4_finetune/play_runs/tonightEval2/volume/seed*/
adj gemma4_finetune/play_runs/volCap500/volume/seed*/

echo "[volcloseout] === $(date) ALL DONE ===" | tee -a "$LOG"
