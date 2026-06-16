#!/bin/zsh
# Volume checkpoint-selection (2026-06-16): pick the publish candidate.
#
# volume (iter 1000) is the strongest adapter (5/13 in-distribution, +12.9 on
# the fresh generalization decks) but carries a JSON-discipline regression: 34
# temp-0.3 parse-rescues across the 13 held-out decks vs base ~12. The earlier
# checkpoints may keep most of the play quality with cleaner JSON.
#
# Evals checkpoints 250/500/750 on the SAME 13 held-out benchmark decks used by
# play_runs/tourA_v16_rescue (where volume-1000 = 5 wins / meanFC 27.7), same
# settings, so the comparison is apples-to-apples. Compare each checkpoint's
# wins/meanFC AND its parse-rescue counts against the volume-1000 baseline; the
# publish candidate is the earliest checkpoint that holds the wins with the
# fewest rescues.
#
# Resumable: tournament_A resume-skips games already present under
# play_runs/volCkptSel/<arm>/seed*, so this survives a window boundary.
# One GPU job at a time (subprocess-per-game is enforced inside the harness).
set -e
cd "$(dirname "$0")/.."
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }
echo "[ckptsel] waiting for GPU to free..."
while gpu_busy; do sleep 60; done
echo "[ckptsel] volume checkpoints 250/500/750 on the 13 held-out decks..."
.venv/bin/python gemma4_finetune/tournament_A.py \
  --arms volume-250,volume-500,volume-750 \
  --seeds 1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115 \
  --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name volCkptSel --prompt-version v1.6
echo "[ckptsel] DONE"
