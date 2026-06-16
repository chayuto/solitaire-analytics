#!/bin/zsh
# Loop-compression spike EVAL (2026-06-16). Plays the loopcompress arm on the 13
# held-out decks with the SAME settings as tourA_v16_rescue / volCkptSel, so it
# compares directly against the volume-1000 baseline (5 wins / meanFC 27.7 / 34
# temp-parse-rescues).
#   loopcompress > volume -> exact doom-loop cycles were poisoning imitation.
#   loopcompress ~= volume -> exact loops are NOT the bottleneck (likely, since
#     the corpora differ by only ~55 of the ~1000 examples seen at iters=1000).
# Resumable (resume-skips completed games). Adjudicate cap-truncated finals after.
set -e
cd "$(dirname "$0")/.."
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }
echo "[loopEval] waiting for GPU to free..."
while gpu_busy; do sleep 60; done
echo "[loopEval] loopcompress on the 13 held-out decks..."
.venv/bin/python gemma4_finetune/tournament_A.py --arms loopcompress \
  --seeds 1388178981,239901548,2703165610,3123337720,3263196305,350743738,3841057237,405489085,4161700176,4197389931,4221577640,4250754298,495097115 \
  --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name loopEval --prompt-version v1.6
echo "[loopEval] DONE"
