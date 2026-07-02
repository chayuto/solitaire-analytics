#!/bin/zsh
# GRPO / RLOO iteration-0 -- full unattended chain (2026-07-01).
#
# The on-policy pivot's SFT-on-self-wins line closed at PARITY with volume across
# every config (x4 0v3, x1 2v5, diverse 4v5; see 2026-07-01 window). SFT only ever
# REINFORCES winning trajectories. This chain tests the one lever it never had:
# the NEGATIVE gradient -- pushing the policy AWAY from its own losing rollouts on
# a deck via a group-relative advantage. That is GRPO/RLOO minus the PPO clip/KL
# (unneeded for a single on-policy epoch).
#
# Data is FREE: the two existing best-of-N pools were sampled BY the volume policy
# (temp 0.7), so continue-training volume on them is legitimate on-policy
# iteration-0. Each deck's 5 rollouts have outcome variance (deep-reach/win vs
# shallow-stall/resign) = the advantage signal. 12 usable decks (9000237 5/5-win
# is dropped as no-signal); ~12.5k decision turns; ~balanced +/- advantage.
#
#   Phase 1  extract the advantage corpus (drift-gated replay, strict JSON,
#            resign turn dropped) from BOTH best-of-N roots. CPU, seconds.
#   Phase 2  CONTINUE-TRAIN the volume adapter on the advantage-weighted
#            REINFORCE loss (subsampled to --max-rows for an overnight run,
#            gentle LR, checkpoints) -> adapters_grpo_iter1. GPU.
#   Phase 3  GREEDY eval of grpo_iter1 on the 12 fresh heldout decks
#            (9000101-9000124, disjoint from BOTH harvest sources and heldout of
#            training); volume's 12 summaries under star_iter1_eval/volume/ are
#            reused as the paired baseline (raw 5/12). Resumable. GPU.
#   Phase 4  paired held-out delta: volume vs grpo_iter1.
#   Phase 5  exact adjudication of the grpo_iter1 final positions (timeout-guarded)
#            so cap-truncated near-wins are not miscounted as losses.
#
# Decision gate: grpo_iter1 BEATS volume => the negative gradient is the lever,
# build the iterative loop (fresh rollouts each round + KL). Flat/regresses =>
# on-policy RFT (this form) also fails to beat volume, ship volume and close the
# on-policy program.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_grpo_iter1.sh \
#     > gemma4_finetune/play_runs/grpo_iter1.launch.log 2>&1 &
set +e   # one failed game must not abort the chain
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/grpo_iter1.log
mkdir -p gemma4_finetune/play_runs

abort() { echo "[grpo] ABORT: $1" | tee -a "$LOG"; exit 1; }

echo "[grpo] === $(date) START GRPO iteration-0 ===" | tee -a "$LOG"

# ---- Phase 1: extract the advantage corpus ----
echo "[grpo] --- Phase 1: extract advantage corpus ---" | tee -a "$LOG"
CORPUS=gemma4_finetune/grpo_corpus_iter1.jsonl
if [ -s "$CORPUS" ]; then
  echo "[grpo] $CORPUS exists, skipping extract" | tee -a "$LOG"
else
  .venv/bin/python gemma4_finetune/extract_grpo_corpus.py \
    --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
    --runs-root gemma4_finetune/play_runs/bestofN_winrate_losses \
    --out-corpus "$CORPUS" 2>&1 | tee -a "$LOG"
fi
[ -s "$CORPUS" ] || abort "extract produced no corpus"
echo "[grpo] corpus rows: $(wc -l < "$CORPUS")" | tee -a "$LOG"

# ---- Phase 2: continue-train volume on the advantage-weighted loss ----
echo "[grpo] --- Phase 2: RFT train adapters_grpo_iter1 (from volume) ---" | tee -a "$LOG"
if [ -s gemma4_finetune/adapters_grpo_iter1/adapters.safetensors ]; then
  echo "[grpo] adapters_grpo_iter1 exists, skipping train" | tee -a "$LOG"
else
  .venv/bin/python gemma4_finetune/train_grpo.py --train \
    --data "$CORPUS" \
    --resume-adapter gemma4_finetune/adapters_volume/adapters.safetensors \
    --adapter-path gemma4_finetune/adapters_grpo_iter1 \
    --epochs 1 --max-rows 1600 --minibatch 8 --lr 8e-5 \
    --steps-per-report 20 --steps-per-eval 50 --save-every 50 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/adapters_grpo_iter1/adapters.safetensors ] || abort "train produced no adapter"

# ---- Phase 3: greedy eval of grpo_iter1 on the 12 heldout decks ----
echo "[grpo] --- Phase 3: eval grpo_iter1 on heldout 12 (greedy) ---" | tee -a "$LOG"
HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
for S in $SEEDS; do
  OD=$EVOUT/grpo_iter1/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[grpo] skip grpo_iter1 seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[grpo] === eval grpo_iter1 seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$HELDOUT" \
    --model-id "$MODEL" --adapter-path gemma4_finetune/adapters_grpo_iter1 \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

# ---- Phase 4: paired held-out delta (volume vs grpo_iter1) ----
echo "[grpo] === held-out result (paired): volume vs grpo_iter1 ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=(o.get("outcome"), o.get("final_foundation_cards"))
    return out
vol=tally("volume"); gp=tally("grpo_iter1")
seeds=sorted(set(vol)&set(gp))   # paired only
vw=gw=0; dfc=0
print(f"  {'seed':<10s} {'volume':20s} {'grpo_iter1':20s}")
for s in seeds:
    vo=vol.get(s,("--",0)); xo=gp.get(s,("--",0))
    vw += vo[0]=="won"; gw += xo[0]=="won"
    dfc += (xo[1] or 0)-(vo[1] or 0)
    flag = " <- grpo gains" if (xo[0]=="won" and vo[0]!="won") else (" <- grpo loses" if (vo[0]=="won" and xo[0]!="won") else "")
    print(f"  {s:<10d} {str(vo):20s} {str(xo):20s}{flag}")
n=len(seeds)
print(f"  -> paired n={n}: volume {vw} wins, grpo_iter1 {gw} wins (delta {gw-vw:+d}); sum fc delta {dfc:+d}")
print(f"  (reference: SFT-on-self-wins closed at parity -- diverse 4 vs volume 5)")
PY

# ---- Phase 5: exact adjudication of grpo_iter1 finals (effective wins) ----
echo "[grpo] === adjudicate grpo_iter1 final positions (timeout 120s each) ===" | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 120 \
  $EVOUT/grpo_iter1/seed*/ 2>&1 | tee -a "$LOG"

echo "[grpo] === $(date) ALL DONE ===" | tee -a "$LOG"
