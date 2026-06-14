#!/bin/zsh
# Volume-scaling run (2026-06-14): train on the full non-eval success pool, then
# eval on the 13 held-out decks. Follows the filter-vs-volume ablation finding
# that volume, not the won-only filter, is the lever.
#
# Trains adapters_volume (~80 min) then evals it on the same 13 held-out decks
# (~13 h) vs base/gate/allsucc already in play_runs/tourA_v16_rescue. One GPU
# job at a time; resumable (train skips if adapters exist, tournament resume-
# skips completed games).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Match only the real GPU jobs, never a zsh-hosted watcher/monitor whose command
# line happens to contain these script names (the 2026-06-14 footgun).
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }

echo "[volume] waiting for GPU to free..."
while gpu_busy; do sleep 60; done

if [ -f gemma4_finetune/adapters_volume/adapters.safetensors ]; then
  echo "[volume] adapters_volume exists, skipping train"
else
  echo "[volume] training full-volume adapter..."
  ( cd gemma4_finetune && ../.venv/bin/python train_v2.py \
      --config lora_config_volume.yaml > /tmp/volume_train.log 2>&1 )
  echo "[volume] train done: $(tail -1 /tmp/volume_train.log)"
fi

while gpu_busy; do sleep 30; done
echo "[volume] eval: volume arm on 13 held-out decks..."
$PY gemma4_finetune/tournament_A.py --arms volume \
  --seeds 495097115,1388178981,3123337720,4250754298,239901548,350743738,3263196305,2703165610,405489085,4221577640,4161700176,3841057237,4197389931 \
  --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name tourA_v16_rescue --prompt-version v1.6 \
  >> gemma4_finetune/play_runs/tourA_v16_rescue_launch.log 2>&1
echo "[volume] DONE"
