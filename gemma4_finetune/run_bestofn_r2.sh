#!/bin/zsh
# GRPO round-2 Phase R: on-policy rollout harvest (2026-07-05).
#
# Pre-registered in docs/reports/20260705_grpo_round2_preregistration.md.
# Samples the CURRENT RFT policy (adapters_grpo_iter1_tr) at temp 0.7, N=8 per
# deck, on the 8-deck success-band pool (pass@5 ~20% under volume, plus the
# resign-blocked 9000207). Breadth-first (round-major) so a window-end kill
# leaves usable groups on every deck. Pure policy rollouts: NO oracle flag.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_bestofn_r2.sh \
#     > gemma4_finetune/play_runs/bestofn_r2.launch.log 2>&1 &
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/bestofn_r2.log
mkdir -p gemma4_finetune/play_runs

echo "[r2] === $(date) START round-2 rollout harvest ===" | tee -a "$LOG"

SEEDS=(9000013 9000024 9000025 9000209 9000215 9000216 9000226 9000207)
N=8
OUT=gemma4_finetune/play_runs/bestofN_r2
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
ADAPTER=gemma4_finetune/adapters_grpo_iter1_tr

deck_path_for() {
  # gate-pool decks live in generalization_decks.json, corrective-pool decks
  # in winrate_decks.json; resolve per seed.
  case $1 in
    9000013|9000024|9000025) echo data/benchmarks/generalization_decks.json ;;
    *)                       echo data/benchmarks/winrate_decks.json ;;
  esac
}

# round-major: sample k of every deck before sample k+1 of any deck
for K in $(seq 1 $N); do
  for S in $SEEDS; do
    OD=$OUT/seed$S/s$K
    if [ -s "$OD/summary.json" ]; then
      echo "[r2] skip seed$S s$K (done)" | tee -a "$LOG"; continue
    fi
    DP=$(deck_path_for $S)
    echo "[r2] === seed$S s$K (temp 0.7) $(date +%H:%M) ===" | tee -a "$LOG"
    .venv/bin/python gemma4_finetune/play_deck_with_student.py \
      --deck-seed "$S" --deck-path "$DP" \
      --model-id "$MODEL" --adapter-path "$ADAPTER" \
      --out-dir "$OD" \
      --temp 0.7 --sample-seed $((7000 + S % 1000 + K * 101)) \
      --max-turns 250 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
  done
  echo "[r2] --- round $K/$N complete $(date +%H:%M) ---" | tee -a "$LOG"
  # per-round tally so partial harvests are readable at a glance
  .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, collections
bydeck = collections.defaultdict(list)
for sm in sorted(glob.glob("gemma4_finetune/play_runs/bestofN_r2/seed*/s*/summary.json")):
    d = json.load(open(sm))
    bydeck[d["deck_seed"]].append((d.get("outcome"), d.get("final_foundation_cards")))
for seed in sorted(bydeck):
    outs = bydeck[seed]
    wins = sum(1 for o, _ in outs if o == "won")
    fcs = sorted(fc for _, fc in outs)
    print(f"[r2]   seed{seed}: n={len(outs)} wins={wins} fc={fcs}")
PY
done

echo "[r2] === $(date) ALL DONE (harvest complete) ===" | tee -a "$LOG"
