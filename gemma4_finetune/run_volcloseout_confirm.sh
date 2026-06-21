#!/bin/zsh
# Confirmation run (2026-06-20). One command, unattended.
#
# Run 1 (play_runs/tonightEval2) gave volcloseout 8 wins / meanFC 40.8 / 0 resign
# vs same-window volume 5 wins / 29.9 / 2 resign, with the +4 gains landing exactly
# on volume's weak decks (3 near-win stalls + the 4221577640 false-resign) and only
# 1 regression (1388178981 driven dead). Adjudication confirmed reach is intact (the
# non-wins are fd0 near-wins or dead, NOT closeout's early fd9-20 stalls), and
# 3841057237 was a cap-truncated winnable near-win (fc45 fd0 SOLVED).
#
# The only open risk is single-run variance. volume is already stable at 5 wins
# across two runs, so we do NOT re-run volume and we do NOT retrain (the adapter
# exists). One more volcloseout pass on the same 13 decks settles it:
#   ~8 again  -> the +3 is real, ship volcloseout (write report, commit, then
#               decide on a fresh-deck generalization pass + HF publish).
#   ~5        -> run 1 was a lucky draw; the gentle-aug result does not hold.
#
# Pipeline: eval volcloseout @ cap300 on the 13 decks -> adjudicate non-win finals.
# tournament_A is RESUMABLE, so an interrupted run resumes on re-launch.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/volcloseout_confirm.log
SEEDS=1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep|run_volcloseout|run_tonight" | grep -q . ; }

echo "[confirm] === $(date) START ===" | tee -a "$LOG"

# guard: we are NOT retraining; the adapter from run 1 must be present
if [ ! -s gemma4_finetune/adapters_volcloseout/adapters.safetensors ]; then
  echo "[confirm] ABORT: adapters_volcloseout/adapters.safetensors missing" | tee -a "$LOG"; exit 1
fi

echo "[confirm] waiting for GPU to free..." | tee -a "$LOG"
while gpu_busy; do sleep 60; done

echo "[confirm] eval volcloseout @ cap300 on 13 decks (run 2)..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py --arms volcloseout --seeds "$SEEDS" \
  --max-turns 300 $=EVAL_FLAGS --out-name volcloseoutConfirm 2>&1 | tee -a "$LOG"

echo "[confirm] adjudicating non-win finals (bounded)..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 90 \
  gemma4_finetune/play_runs/volcloseoutConfirm/volcloseout/seed*/ 2>&1 | tee -a "$LOG"

echo "[confirm] === $(date) ALL DONE ===" | tee -a "$LOG"
