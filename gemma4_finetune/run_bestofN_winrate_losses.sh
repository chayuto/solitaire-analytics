#!/bin/zsh
# Best-of-N corrective harvest on FRESH deep-loss decks (2026-06-28).
#
# The 24-deck fresh win-rate run (run_volume_winrate.sh) left volume losing 18 of
# 24 winnable decks greedily. Adjudication of the deep near-misses (fc>=20) split
# them into a false-resign (9000237, SOLVED n=3), a cap-truncation (9000226,
# fc40/fd0 SOLVED n=13), two stalls on winnable (9000215, 9000216), and two
# thrown boards (9000207, 9000209). Greedy is deterministic, so each is a FIXED
# failure. This extends the best-of-N gate (which only sampled the general12
# losses, solving 3/7) to a FRESH deep-loss set, and asks:
#   - does temp-0.7 sampling recover the false-resign (9000237) and the cap loss
#     (9000226)? (precedent: the gate's seed9000025 was a false-resign recovery)
#   - does it avoid the two throws (9000207, 9000209) from a different sampled
#     line, and break the two stalls (9000215, 9000216)?
# pass@N>0 banks a diverse corrective self-win trajectory for STaR/RFT; the
# per-deck recovery rate measures on-policy headroom on hard FRESH decks.
#
# Scope note: this is the DEEP-loss subset (fc>=20, the 6 most likely to recover),
# so the recovery rate is an UPPER bound on the full 18-loss set, not its mean.
# Full game from the deal (no warm-start), distinct mx seed per sample, cap 250.
# Breadth-first (round K outer, deck inner): a partial run gives pass@k across all
# 6. Resumable (existing summary.json skipped).
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/bestofN_winrate_losses.log
OUT=gemma4_finetune/play_runs/bestofN_winrate_losses
# ordered: false-resign + cap first (fast, informative), then stalls, then throws
DECKS=(9000237 9000226 9000216 9000215 9000207 9000209)
ADAPTER=gemma4_finetune/adapters_volume
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
DECKPATH=data/benchmarks/winrate_decks.json
N=5
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_bestofN|run_phase|run_volume|run_star|caffeinate" | grep -q . ; }

echo "[bonL] === $(date) START (N=$N temp0.7, 6 fresh deep-loss decks, volume) ===" | tee -a "$LOG"
[ -s "$ADAPTER/adapters.safetensors" ] || { echo "[bonL] ABORT: adapters_volume missing" | tee -a "$LOG"; exit 1; }
[ -s "$DECKPATH" ] || { echo "[bonL] ABORT: winrate_decks.json missing" | tee -a "$LOG"; exit 1; }
echo "[bonL] ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

for K in $(seq 1 $N); do
  for D in $DECKS; do
    OD=$OUT/seed$D/s$K
    if [ -s "$OD/summary.json" ]; then
      echo "[bonL] skip seed$D s$K (done)" | tee -a "$LOG"; continue
    fi
    echo "[bonL] === seed$D sample $K/$N (temp 0.7, mx seed $K) $(date +%H:%M) ===" | tee -a "$LOG"
    .venv/bin/python gemma4_finetune/play_deck_with_student.py \
      --deck-seed "$D" --deck-path "$DECKPATH" \
      --model-id "$MODEL" --adapter-path "$ADAPTER" \
      --out-dir "$OD" \
      --max-turns 250 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --temp 0.7 --sample-seed "$K" \
      --prompt-version v1.6 2>&1 | tee -a "$LOG"
  done
done

echo "[bonL] === pass@N analysis ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
OUT="gemma4_finetune/play_runs/bestofN_winrate_losses"
decks=["9000237","9000226","9000216","9000215","9000207","9000209"]
solved=0; total_wins=0; samples=0
print(f"  {'deck':14s} {'samples':7s} {'wins':4s} {'pass@N':6s} {'best_fc':7s}")
for d in decks:
    summ=sorted(glob.glob(f"{OUT}/seed{d}/s*/summary.json"))
    outs=[json.load(open(s)) for s in summ]
    wins=sum(1 for o in outs if o.get("outcome")=="won")
    fcs=[o.get("final_foundation_cards") or 0 for o in outs]
    samples+=len(outs); total_wins+=wins
    if wins>0: solved+=1
    print(f"  seed{d:10s} {len(outs):7d} {wins:4d} {'YES' if wins else 'no':6s} {max(fcs) if fcs else 0:7d}")
print(f"  -> best-of-N recovers {solved}/6 fresh deep-loss decks (greedy 0/6 on these); "
      f"{total_wins} winning trajectories from {samples} samples")
PY
echo "[bonL] === $(date) ALL DONE ===" | tee -a "$LOG"
