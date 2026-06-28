#!/bin/zsh
# Volume greedy generalization win-rate on a FRESH winnable deck set (2026-06-28).
#
# The on-policy SFT levers are exhausted (volume 5/12 floor uncleared; see
# docs/reports/20260628_star_iter1_oversample_ablation.md). Before committing to
# either remaining lever (lever 1 diverse harvest = ~3 windows, lever 3 RFT =
# multi-week build) or shipping volume, measure the ONE missing number that gates
# the fork: volume's TRUE greedy generalization win-rate on a broad fresh deck
# set (we only have 5/12 and 5/13, wide CIs). This window:
#   - tightens that rate on 24 fresh solver-winnable decks (seeds 9000201+, none
#     in any training corpus or the heldout/generalization eval sets),
#   - banks a DIVERSE greedy self-win pool (each win from a different fresh deck,
#     the diversity lever 1 lacked), and
#   - emits the list of fresh decks volume LOSES greedily = the exact input for a
#     future best-of-N corrective harvest or RFT.
# Greedy, one game per deck, proven harness (identical flags to the star_iter1
# eval). Resumable (skips existing summary.json).
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_volume_winrate.sh \
#     > gemma4_finetune/play_runs/volume_winrate.launch.log 2>&1 &
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/volume_winrate.log
mkdir -p gemma4_finetune/play_runs

DECKS=data/benchmarks/winrate_decks.json
EVOUT=gemma4_finetune/play_runs/volume_winrate_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
ADAPTER=gemma4_finetune/adapters_volume

[ -s "$DECKS" ] || { echo "[wr] ABORT: $DECKS missing" | tee -a "$LOG"; exit 1; }
SEEDS=($(.venv/bin/python -c "import json;print(*[d['seed'] for d in json.load(open('$DECKS'))['decks']])"))
echo "[wr] === $(date) START volume greedy win-rate on ${#SEEDS} fresh decks ===" | tee -a "$LOG"

for S in $SEEDS; do
  OD=$EVOUT/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[wr] skip seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[wr] === eval volume seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$DECKS" \
    --model-id "$MODEL" --adapter-path "$ADAPTER" \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

# ---- summary: win-rate, win list (diverse self-win pool), loss list ----
echo "[wr] === volume greedy win-rate summary ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/volume_winrate_eval"
rows=[]
for s in sorted(glob.glob(f"{EV}/seed*/summary.json")):
    o=json.load(open(s)); rows.append((o["deck_seed"], o.get("outcome"), o.get("final_foundation_cards")))
wins=[r for r in rows if r[1]=="won"]; losses=[r for r in rows if r[1]!="won"]
n=len(rows)
print(f"  played {n}/24 fresh decks")
print(f"  WINS {len(wins)}/{n} (rate {len(wins)/n:.2f} if n else 0):")
for sd,_,fc in wins: print(f"    seed{sd}: won {fc}")
print(f"  LOSSES {len(losses)} (best-of-N / RFT targets):")
for sd,oc,fc in losses: print(f"    seed{sd}: {oc} fc{fc}")
PY
echo "[wr] === $(date) ALL DONE ===" | tee -a "$LOG"
