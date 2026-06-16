#!/bin/zsh
# Loop-compression spike TRAINING (2026-06-16). Trains the loopcompress arm
# (exact-state cycle bodies removed, draw-safe) with volume-identical hypers so
# loopcompress-vs-volume isolates the effect of removing exact doom-loop cycles.
# ~80 min on the 16 GB machine (iters 1000, batch 1, grad-checkpoint).
# Eval afterward as a tournament arm on the 13 held-out decks (separate step).
set -e
cd "$(dirname "$0")/.."
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }
echo "[loopcompress] waiting for GPU to free..."
while gpu_busy; do sleep 60; done
echo "[loopcompress] training (iters 1000)..."
# train_v2.py resolves the config's data:/adapter_path: relative to CWD.
( cd gemma4_finetune && ../.venv/bin/python train_v2.py --config lora_config_loopcompress.yaml )
echo "[loopcompress] TRAIN DONE -> gemma4_finetune/adapters_loopcompress"
