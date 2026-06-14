#!/bin/zsh
# Chained filter-vs-volume ablation for the long compute window (2026-06-13).
#
# Waits for the GPU to free (the base-temp0.3 control finishes), then:
#   1. trains the matched all-success adapter (~80 min)
#   2. evals it on the 13 held-out rescue decks vs base/gate (~13 h)
#
# Strictly one GPU job at a time: each stage waits until no play harness or
# control process is running. Resumable: training skips if adapters exist;
# the tournament resume-skips completed games. Safe to relaunch.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Match only the real GPU jobs, never a zsh-hosted watcher/monitor whose
# command line happens to contain these script names (that footgun stalled the
# 2026-06-14 run: a Monitor referencing the scripts was matched as "busy").
gpu_busy() { pgrep -af "play_deck_with_student.py|run_base_temp03_control|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }

echo "[ablation] waiting for GPU to free (control to finish)..."
while gpu_busy; do sleep 60; done
echo "[ablation] GPU free at $(ls -la --time-style=+%H:%M gemma4_finetune/play_runs/base_temp03 2>/dev/null | tail -1 | awk '{print $6}')"

# Stage 1: train (skip if already trained)
if [ -f gemma4_finetune/adapters_allsucc/adapters.safetensors ]; then
  echo "[ablation] adapters_allsucc exists, skipping train"
else
  echo "[ablation] training all-success matched adapter..."
  # train_v2.py resolves the config's `data:` relative to CWD, so run from
  # gemma4_finetune/ (where dataset_allsucc/ and adapters_allsucc/ live),
  # matching how the gate adapter was trained.
  ( cd gemma4_finetune && ../.venv/bin/python train_v2.py \
      --config lora_config_allsucc.yaml > /tmp/allsucc_train.log 2>&1 )
  echo "[ablation] train done: $(tail -1 /tmp/allsucc_train.log)"
fi

# Stage 2: eval the allsucc arm on the 13 held-out decks (base/gate already done)
while gpu_busy; do sleep 30; done
echo "[ablation] eval: allsucc on 13 held-out decks..."
$PY gemma4_finetune/tournament_A.py --arms allsucc \
  --seeds 495097115,1388178981,3123337720,4250754298,239901548,350743738,3263196305,2703165610,405489085,4221577640,4161700176,3841057237,4197389931 \
  --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name tourA_v16_rescue --prompt-version v1.6 \
  >> gemma4_finetune/play_runs/tourA_v16_rescue_launch.log 2>&1
echo "[ablation] DONE"
