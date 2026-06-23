#!/bin/zsh
# Best-of-N RFT GATE probe (2026-06-23) -- the pre-registered first step of the
# on-policy program ([[orpo-loop-penalty-next-step]] final verdict 4b: "best-of-N
# probe ON THE FAITHFUL HARNESS (temp ~0.7 N~8) gates RFT"), deferred when the
# 2026-06-11..22 windows went to corpus arms. First best-of-N run ever.
#
# QUESTION: the volume student is the best fresh-deck generalizer but only 5/12
# greedy. Greedy decoding is deterministic, so a lost deck is a FIXED failure.
# Does temperature sampling (temp 0.7, N independent attempts) unlock wins on the
# decks greedy misses? pass@N > 0 on a winnable deck = sampled wins EXIST for RFT
# to reinforce; pass@N ~ 0 = the policy itself is capped and RFT has little to
# amplify (-> ship volume). Winning trajectories collected here also seed the
# first expert-iteration (STaR) corpus.
#
# TARGET: the 7 of 12 generalization decks volume LOST under greedy -- all
# winnable BY CONSTRUCTION, so every one is pure headroom (5 stalls + 2 false
# resigns: 9000021, 9000024). Sampling from the volume policy (adapters_volume),
# full game from the deal (no warm-start), temp 0.7, distinct mx.random seed per
# sample. Breadth-first (round K outer, deck inner) so a partial run still gives
# pass@k across ALL 7 decks. Resumable: existing summary.json is skipped.
set +e   # one failed game must not abort the 35-game sweep
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/bestofN_gen_volume.log
OUT=gemma4_finetune/play_runs/bestofN_gen_volume
DECKS=(9000002 9000003 9000010 9000013 9000021 9000024 9000025)
ADAPTER=gemma4_finetune/adapters_volume
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
DECKPATH=data/benchmarks/generalization_decks.json
N=5
gpu_busy() { pgrep -af "play_deck_with_student.py|train_v2.py|tournament_A.py" \
             | grep -vE "zsh|pgrep|grep|run_bestofN|run_phase|caffeinate" | grep -q . ; }

echo "[bonV] === $(date) START (N=$N temp0.7, 7 decks, volume student) ===" | tee -a "$LOG"
if [ ! -s "$ADAPTER/adapters.safetensors" ]; then
  echo "[bonV] ABORT: adapters_volume missing" | tee -a "$LOG"; exit 1
fi
if [ ! -s "$DECKPATH" ]; then
  echo "[bonV] ABORT: generalization_decks.json missing" | tee -a "$LOG"; exit 1
fi
echo "[bonV] ensuring GPU clear..." | tee -a "$LOG"
while gpu_busy; do sleep 30; done

for K in $(seq 1 $N); do
  for D in $DECKS; do
    OD=$OUT/seed$D/s$K
    if [ -s "$OD/summary.json" ]; then
      echo "[bonV] skip seed$D s$K (done)" | tee -a "$LOG"; continue
    fi
    echo "[bonV] === seed$D sample $K/$N (temp 0.7, mx seed $K) $(date +%H:%M) ===" | tee -a "$LOG"
    .venv/bin/python gemma4_finetune/play_deck_with_student.py \
      --deck-seed "$D" --deck-path "$DECKPATH" \
      --model-id "$MODEL" --adapter-path "$ADAPTER" \
      --out-dir "$OD" \
      --max-turns 250 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --temp 0.7 --sample-seed "$K" \
      --prompt-version v1.6 2>&1 | tee -a "$LOG"
  done
done

echo "[bonV] === pass@N analysis ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os
OUT="gemma4_finetune/play_runs/bestofN_gen_volume"
decks=["9000002","9000003","9000010","9000013","9000021","9000024","9000025"]
solved=0; total_wins=0; samples=0
print(f"  {'deck':14s} {'samples':7s} {'wins':4s} {'pass@N':6s} {'best_fc':7s}")
for d in decks:
    summ=sorted(glob.glob(f"{OUT}/seed{d}/s*/summary.json"))
    outs=[json.load(open(s)) for s in summ]
    wins=sum(1 for o in outs if o.get("outcome")=="won")
    fcs=[(o.get("max_fc") or o.get("final_foundation_cards") or 0) for o in outs]
    samples+=len(outs); total_wins+=wins
    passN = "YES" if wins>0 else "no"
    if wins>0: solved+=1
    print(f"  seed{d:10s} {len(outs):7d} {wins:4d} {passN:6s} {max(fcs) if fcs else 0:7d}")
print(f"  -> best-of-N SOLVES {solved}/7 winnable decks greedy LOST (greedy was 0/7 on these); "
      f"{total_wins} winning trajectories from {samples} samples")
PY
echo "[bonV] === $(date) ALL DONE ===" | tee -a "$LOG"
