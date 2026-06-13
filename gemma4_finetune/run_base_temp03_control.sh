#!/bin/zsh
# CONTROL for the wononly-gate confound (2026-06-13).
#
# The gate beat base on the held-out rescue decks, but the gate runs trigger
# the temp-0.3 parse-retry path 10-21x per game (won-only training degraded
# its JSON discipline ~6x), vs base's 0-2x. The best-of-N probe established
# that temperature injection alone breaks deterministic stalls. So the gate's
# advantage is confounded: learned policy vs self-induced stochasticity.
#
# This control runs the BASE (no adapter) at full --temp 0.3 on the decks the
# gate won or near-won. If base-at-temp-0.3 also wins them, the gate's edge is
# mostly stochasticity; if base-at-temp-0.3 still stalls where the gate won,
# the gate learned real policy. One sample per deck (matched to the gate's
# single sample); distinct mx seed so it is not greedy.
#
# Decisive cheap subset: the 4 outright gate wins + the fc44/n=10 near-win.
#   gate: 495097115 won, 1388178981 won, 239901548 won, 4221577640 won,
#         3123337720 fc44/fd0 SOLVED n=10 (base greedy stalled fc22 there).
#
# Run AFTER the gate eval GPU job completes:
#   zsh gemma4_finetune/run_base_temp03_control.sh
set -e
cd "$(dirname "$0")/.."
for SEED in 495097115 1388178981 239901548 4221577640 3123337720; do
  echo "=== base @ temp 0.3 (mx seed 1), deck $SEED ==="
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed $SEED \
    --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
    --out-dir gemma4_finetune/play_runs/base_temp03/seed$SEED \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 \
    --temp 0.3 --sample-seed 1 \
    --prompt-version v1.6
done
echo "=== base-temp0.3 control summary (vs gate) ==="
for SEED in 495097115 1388178981 239901548 4221577640 3123337720; do
  python3 -c "import json; s=json.load(open('gemma4_finetune/play_runs/base_temp03/seed$SEED/summary.json')); print(f\"$SEED: {s['outcome']} fc={s['final_foundation_cards']} fd={s['final_face_down']} turns={s['turns_played']}\")"
done
