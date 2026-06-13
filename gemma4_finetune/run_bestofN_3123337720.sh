#!/bin/zsh
# Best-of-N sampling probe on the crispest stall deck of the rescue window.
#
# seed 3123337720 stalled fc=22/fd=1 from t111 to the 200-turn cap, ON a
# position the solver wins in 103 states. Greedy is deterministic, so the
# stall is a fixed cycle; this probe measures whether temperature sampling
# breaks it (the RFT gate: pass@N > 0 means sampled wins exist to prefer).
#
# Each sample warm-starts engine-side to t111 (zero GPU for the prefix,
# drift-gated against recorded prompt lengths) and plays live from the stall
# onset at temp 0.7 with a distinct mx.random seed. Sequential: one GPU job
# at a time, subprocess isolation per sample.
#
# Run AFTER the win_banking GPU job completes:
#   zsh gemma4_finetune/run_bestofN_3123337720.sh
set -e
cd "$(dirname "$0")/.."
WS=gemma4_finetune/play_runs/tourA_v16_rescue/base/seed3123337720/turns.jsonl
for K in 1 2 3 4; do
  echo "=== sample $K (temp 0.7, mx seed $K) ==="
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed 3123337720 \
    --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
    --out-dir gemma4_finetune/play_runs/bestofN_3123337720/t07_s$K \
    --max-turns 250 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --temp 0.7 --sample-seed $K \
    --prompt-version v1.6 \
    --warm-start-from "$WS" --warm-start-until 111
done
echo "=== best-of-4 summary ==="
for K in 1 2 3 4; do
  python3 -c "import json; s=json.load(open('gemma4_finetune/play_runs/bestofN_3123337720/t07_s$K/summary.json')); print(f\"s$K: {s['outcome']} fc={s['final_foundation_cards']} fd={s['final_face_down']} turns={s['turns_played']}\")"
done
