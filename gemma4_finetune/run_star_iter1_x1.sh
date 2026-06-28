#!/bin/zsh
# STaR iter-1 OVERSAMPLE ABLATION -- full unattended chain (2026-06-27).
#
# star_iter1 (x4 oversample) REGRESSED fresh-deck generalization (0 vs volume 3
# on the 7 paired held-out decks, fc down on all 7; report
# docs/reports/20260626_star_iter1_expert_iteration_eval.md). Mechanism
# hypothesis: overfit to 3 too-correlated self-win games at ~30% of the mix, NOT
# yet isolated between (a) dose too high or (b) poison at any dose. This chain
# isolates the dose: SAME 599 STaR rows re-mixed at x1 (~10% share), one fresh
# LoRA, paired vs volume's already-banked 12-deck held-out baseline.
#
#   Phase 1  extract_star_corpus.py --oversample 1 from the SAME best-of-N WON
#            trajectories (CPU, drift-gated) -> dataset_star_iter1_x1 =
#            dataset_volume train + (STaR rows x1). valid/test reused.
#   Phase 2  SFT a FRESH LoRA from the int4 base on that corpus (volume hypers,
#            iters 1000) -> adapters_star_iter1_x1.
#   Phase 3  GREEDY eval of the star_x1 arm ONLY on the 12 fresh held-out decks
#            (volume's 12 summaries already exist and are reused), resumable.
#   Phase 4  paired held-out delta: volume vs star_x1, with the 3-collapse-deck
#            early read (9000101/9000105/9000107, where x4 fell 52 -> 9/3/20).
#
# Decision gate: star_x1 recovers toward volume (collapses gone, delta ~0) =>
# the regression was the oversample dose, self-win SFT is neutral-not-poison,
# the diverse-harvest lever is justified. star_x1 still regresses => poison at
# any dose, stop SFT-on-self-wins, go reward-weighted (RFT/GRPO) or harvest
# genuinely diverse wins first.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_star_iter1_x1.sh \
#     > gemma4_finetune/play_runs/star_iter1_x1.launch.log 2>&1 &
set +e   # one failed game must not abort the chain
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/star_iter1_x1.log
mkdir -p gemma4_finetune/play_runs

abort() { echo "[x1] ABORT: $1" | tee -a "$LOG"; exit 1; }

echo "[x1] === $(date) START oversample ablation (x1) ===" | tee -a "$LOG"

# ---- Phase 1: rebuild the corpus at oversample 1 ----
echo "[x1] --- Phase 1: extract STaR corpus at x1 ---" | tee -a "$LOG"
if [ -s gemma4_finetune/dataset_star_iter1_x1/train.jsonl ]; then
  echo "[x1] dataset_star_iter1_x1 exists, skipping extract" | tee -a "$LOG"
else
  .venv/bin/python gemma4_finetune/extract_star_corpus.py \
    --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
    --deck-path data/benchmarks/generalization_decks.json \
    --base-data gemma4_finetune/dataset_volume \
    --out-corpus gemma4_finetune/star_corpus_iter1_x1.jsonl \
    --out-data gemma4_finetune/dataset_star_iter1_x1 \
    --oversample 1 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/dataset_star_iter1_x1/train.jsonl ] || abort "extract/build produced no train.jsonl"
echo "[x1] train rows: $(wc -l < gemma4_finetune/dataset_star_iter1_x1/train.jsonl)" | tee -a "$LOG"

# ---- Phase 2: SFT a fresh LoRA from base (volume hypers) ----
echo "[x1] --- Phase 2: SFT adapters_star_iter1_x1 ---" | tee -a "$LOG"
if [ -s gemma4_finetune/adapters_star_iter1_x1/adapters.safetensors ]; then
  echo "[x1] adapters_star_iter1_x1 exists, skipping train" | tee -a "$LOG"
else
  ( cd gemma4_finetune && ../.venv/bin/python train_v2.py \
      --config lora_config_star_iter1_x1.yaml ) 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/adapters_star_iter1_x1/adapters.safetensors ] || abort "SFT produced no adapter"

# ---- Phase 3: greedy eval of the star_x1 arm on 12 fresh held-out decks ----
# volume's 12 summaries already exist under .../volume/ and are reused as the
# paired baseline; only the star_x1 arm runs here.
echo "[x1] --- Phase 3: eval star_x1 on held-out 12 (greedy) ---" | tee -a "$LOG"
HELDOUT=data/benchmarks/heldout_decks.json
# 3-collapse decks first for an early read, then the rest.
SEEDS=(9000101 9000105 9000107 9000102 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
for S in $SEEDS; do
  OD=$EVOUT/star_x1/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[x1] skip star_x1 seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[x1] === eval star_x1 seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$HELDOUT" \
    --model-id "$MODEL" --adapter-path gemma4_finetune/adapters_star_iter1_x1 \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

# ---- Phase 4: paired held-out delta (volume vs star_x1) ----
echo "[x1] === held-out result (paired): volume vs star_x1 ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=(o.get("outcome"), o.get("final_foundation_cards"))
    return out
vol=tally("volume"); x1=tally("star_x1")
seeds=sorted(set(vol)&set(x1))   # paired only
vw=xw=0; dfc=0
print(f"  {'seed':<10s} {'volume':18s} {'star_x1':18s}")
for s in seeds:
    vo=vol.get(s,("--",0)); xo=x1.get(s,("--",0))
    vw += vo[0]=="won"; xw += xo[0]=="won"
    dfc += (xo[1] or 0)-(vo[1] or 0)
    flag = " <- x1 gains" if (xo[0]=="won" and vo[0]!="won") else (" <- x1 loses" if (vo[0]=="won" and xo[0]!="won") else "")
    print(f"  {s:<10d} {str(vo):18s} {str(xo):18s}{flag}")
n=len(seeds)
print(f"  -> paired n={n}: volume {vw} wins, star_x1 {xw} wins (delta {xw-vw:+d}); sum fc delta {dfc:+d}")
print(f"  (x4 baseline for reference: 0 vs volume 3 on its 7 paired decks; collapses 9000101/9000105/9000107 = 52->9/3/20)")
PY
echo "[x1] === $(date) ALL DONE ===" | tee -a "$LOG"
