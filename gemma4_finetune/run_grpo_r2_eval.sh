#!/bin/zsh
# GRPO round-2 Phase E (2026-07-19): greedy heldout12 gate eval.
# Pre-registration: docs/reports/20260705_grpo_round2_preregistration.md
#   - arm grpo_r2 = adapters_grpo_r2 (round-2 output, step-200 final)
#   - greedy, cap 200, oracle OFF, same deck set / tolerances as the cached
#     volume (5/12) and grpo_iter1_tr (5/12) baselines in star_iter1_eval/
#   - GATE: grpo_r2 >= 7 strict wins = PASS; below 7 = FAIL, RFT closes.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_grpo_r2_eval.sh \
#     > gemma4_finetune/play_runs/grpo_r2_eval.launch.log 2>&1 &
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/grpo_r2_eval.log
HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
ARM=grpo_r2
ADAPTER=gemma4_finetune/adapters_grpo_r2
mkdir -p gemma4_finetune/play_runs

echo "[r2e] === $(date) START Phase E greedy heldout12 ($ARM) ===" | tee -a "$LOG"
for S in $SEEDS; do
  OD=$EVOUT/$ARM/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[r2e] skip $ARM seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[r2e] === $ARM seed$S $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$HELDOUT" \
    --model-id "$MODEL" --adapter-path "$ADAPTER" \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

echo "[r2e] === readout ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s)); out[o["deck_seed"]]=o
    return out
r2=tally("grpo_r2")
for base_arm in ("volume","grpo_iter1_tr"):
    base=tally(base_arm)
    seeds=sorted(set(base)&set(r2))
    bw=rw=0
    print(f"\n  --- grpo_r2 vs {base_arm} (paired n={len(seeds)}) ---")
    for s in seeds:
        b=base[s]; r=r2[s]
        bw += b.get("outcome")=="won"; rw += r.get("outcome")=="won"
        flag=(" <- r2 gains" if (r.get("outcome")=="won" and b.get("outcome")!="won")
              else (" <- r2 LOSES" if (b.get("outcome")=="won" and r.get("outcome")!="won") else ""))
        print(f"  {s:<10d} {b.get('outcome'):>10s} fc{b.get('final_foundation_cards'):>2d}  "
              f"r2 {r.get('outcome'):>10s} fc{r.get('final_foundation_cards'):>2d}{flag}")
    print(f"  -> {base_arm} {bw} wins; grpo_r2 {rw} wins (delta {rw-bw:+d})")
strict=sum(1 for o in r2.values() if o.get("outcome")=="won")
print(f"\n  GATE: grpo_r2 strict wins = {strict}/12; PASS needs >= 7 "
      f"-> {'PASS' if strict>=7 else 'FAIL (RFT closes)'}")
PY
echo "[r2e] === $(date) ALL DONE ===" | tee -a "$LOG"
