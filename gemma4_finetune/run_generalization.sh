#!/bin/zsh
# Generalization test (2026-06-14): the DECISIVE experiment for the won-only
# gate. Plays base vs the gate on 12 FRESH solver-confirmed-winnable decks
# (data/benchmarks/generalization_decks.json) that are in no training corpus
# and not in the benchmark, so they test whether the gate learned Klondike or
# the harvester's deck distribution.
#
# Read the PAIRED base-vs-gate delta, not absolute win counts: the fresh decks
# are likely harder than the teacher-won benchmark, and they are biased toward
# easy-to-moderate winnable deals (only deals the engine solver cracked under a
# 200k-node cap were kept; see build_generalization_decks.py).
#
# Launch only after the GPU is free (e.g. the volume run finished), to keep one
# GPU job at a time. ~24 games (2 arms x 12 decks), resumable.
set -e
cd "$(dirname "$0")/.."
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py" \
             | grep -vE "zsh|pgrep|grep" | grep -q . ; }
echo "[gen] waiting for GPU to free..."
while gpu_busy; do sleep 60; done
echo "[gen] base + gate + volume on 12 fresh winnable decks..."
.venv/bin/python gemma4_finetune/tournament_A.py --arms base,wononly-gate,volume \
  --deck-path data/benchmarks/generalization_decks.json \
  --seeds 9000002,9000003,9000005,9000008,9000010,9000013,9000020,9000021,9000023,9000024,9000025,9000026 \
  --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10 \
  --out-name genTest --prompt-version v1.6
echo "[gen] DONE"
