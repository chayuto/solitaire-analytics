#!/bin/zsh
# Endgame-oracle A/B (2026-07-04): volume+oracle and grpo_iter1_tr+oracle on
# heldout12, paired against their cached no-oracle baselines.
#
# The oracle (endgame_oracle.solve_winning_line, probe cap 3000 nodes/turn) is
# the inference-layer lever pre-registered in 20260704_next_move_deep_research
# strand C: SOLVED plays the proven line, UNSOLVABLE folds, UNKNOWN falls back
# to the model. Validated 4/4 offline + a live warm-start smoke that converted
# volume's 9000112 fc=35 stall to a win (18 nodes, 17-move line).
#
# Arms (only the flag differs from the cached baselines; same adapters, cap
# 200, greedy, same deck set):
#   volume_oracle   = adapters_volume        + --endgame-oracle 3000
#   grpo_tr_oracle  = adapters_grpo_iter1_tr + --endgame-oracle 3000
#
# Readout: strict wins vs baseline per arm, oracle trigger stats (solved/dead/
# trigger turn/probe overhead). Decision use: quantifies the certain win-rate
# gain of the verified-advisor mode, and informs the close-vs-round-2 RFT call.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_oracle_ab.sh \
#     > gemma4_finetune/play_runs/oracle_ab.launch.log 2>&1 &
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/oracle_ab.log
mkdir -p gemma4_finetune/play_runs

echo "[oab] === $(date) START oracle A/B ===" | tee -a "$LOG"

HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4

run_arm() {
  local arm_name=$1
  local adapter=$2
  for S in $SEEDS; do
    OD=$EVOUT/$arm_name/seed$S
    if [ -s "$OD/summary.json" ]; then
      echo "[oab] skip $arm_name seed$S (done)" | tee -a "$LOG"; continue
    fi
    echo "[oab] === $arm_name seed$S $(date +%H:%M) ===" | tee -a "$LOG"
    .venv/bin/python gemma4_finetune/play_deck_with_student.py \
      --deck-seed "$S" --deck-path "$HELDOUT" \
      --model-id "$MODEL" --adapter-path "$adapter" \
      --out-dir "$OD" \
      --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --max-illegal-moves 10 --prompt-version v1.6 \
      --endgame-oracle 3000 2>&1 | tee -a "$LOG"
  done
}

run_arm volume_oracle  gemma4_finetune/adapters_volume
run_arm grpo_tr_oracle gemma4_finetune/adapters_grpo_iter1_tr

# ---- Readout: paired vs cached no-oracle baselines + oracle stats ----
echo "[oab] === oracle A/B readout ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=o
    return out
pairs=[("volume","volume_oracle"),("grpo_iter1_tr","grpo_tr_oracle")]
for base_arm, o_arm in pairs:
    base=tally(base_arm); orc=tally(o_arm)
    seeds=sorted(set(base)&set(orc))
    bw=ow=0; folds=0; probes_s=0.0
    print(f"\n  --- {o_arm} vs {base_arm} (paired n={len(seeds)}) ---")
    print(f"  {'seed':<10s} {'base':>16s} {'+oracle':>16s}  oracle")
    for s in seeds:
        b=base[s]; o=orc[s]
        bw += b.get("outcome")=="won"; ow += o.get("outcome")=="won"
        folds += o.get("oracle_result")=="dead"
        probes_s += o.get("oracle_probe_seconds") or 0
        otag = o.get("oracle_result") or "-"
        ott = o.get("oracle_trigger_turn")
        onote = f"{otag}" + (f"@t{ott}" if ott is not None else "")
        flag = " <- oracle gains" if (o.get("outcome")=="won" and b.get("outcome")!="won") else (" <- oracle LOSES" if (b.get("outcome")=="won" and o.get("outcome")!="won") else "")
        print(f"  {s:<10d} {b.get('outcome'):>10s} fc{b.get('final_foundation_cards'):>2d} "
              f"{o.get('outcome'):>10s} fc{o.get('final_foundation_cards'):>2d}  {onote}{flag}")
    print(f"  -> {base_arm} {bw} wins; {o_arm} {ow} wins (delta {ow-bw:+d}); "
          f"{folds} proven-dead folds; total probe overhead {probes_s:.1f}s")
PY

echo "[oab] === $(date) ALL DONE ===" | tee -a "$LOG"
