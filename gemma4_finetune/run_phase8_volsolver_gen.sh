#!/bin/zsh
# Phase 8 (2026-06-22): THE DECISIVE generalization run for the solver-grounded
# lever. Eval-only (adapters_volsolver already trained in phase 7).
#
# volsolver tied the close-out recipe in-distribution (8/13). The recipe's 8
# COLLAPSED to 4 on fresh decks (overfit deck structure); volume held 5->5. The
# whole solver-as-teacher thesis is that grounding targets in solver-correct
# WINNING PLAY should GENERALIZE where corpus reweighting did not. This run tests
# exactly that on the SAME 12 fresh solver-confirmed-winnable decks volume (5/12)
# and volcloseout (4/12) were scored on, at the SAME settings (cap200, v1.6,
# parse-retry 0.3), into the SAME out-name genTest so the comparison is
# apples-to-apples and tournament_A resumes (only the 12 new volsolver games run).
#
# All 12 decks are winnable BY CONSTRUCTION: any resign is FALSE, any non-win is a
# failure-to-finish under the cap. Read the PAIRED delta vs volume 5 / volcloseout 4.
set -e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/phase8_volsolver_gen.log
SEEDS=9000002,9000003,9000005,9000008,9000010,9000013,9000020,9000021,9000023,9000024,9000025,9000026
EVAL_FLAGS="--max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 --prompt-version v1.6"
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_phase|caffeinate" | grep -q . ; }

echo "[phase8] === $(date) START ===" | tee -a "$LOG"

if [ ! -s gemma4_finetune/adapters_volsolver/adapters.safetensors ]; then
  echo "[phase8] ABORT: adapters_volsolver missing" | tee -a "$LOG"; exit 1
fi
if [ ! -s data/benchmarks/generalization_decks.json ]; then
  echo "[phase8] ABORT: generalization_decks.json missing" | tee -a "$LOG"; exit 1
fi

echo "[phase8] ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

echo "[phase8] eval volsolver @ cap200 on 12 fresh winnable decks (genTest resume)..." | tee -a "$LOG"
.venv/bin/python gemma4_finetune/tournament_A.py \
  --arms volsolver \
  --deck-path data/benchmarks/generalization_decks.json \
  --seeds "$SEEDS" \
  --max-turns 200 $=EVAL_FLAGS \
  --out-name genTest 2>&1 | tee -a "$LOG"

echo "[phase8] === $(date) ALL DONE ===" | tee -a "$LOG"
