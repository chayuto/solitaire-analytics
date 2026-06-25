#!/bin/zsh
# STaR expert-iteration round 1 -- full unattended chain (2026-06-25).
#
# The on-policy pivot's first TRAINING step. The best-of-N gate showed the volume
# student SAMPLES wins (temp 0.7) on winnable decks it loses greedily. This chain
# turns those self-wins into an SFT corpus, retrains, and measures whether that
# lifts generalization on FRESH held-out decks.
#
#   Phase 1  best-of-N gate resume  -> fills the 8 missing samples (clean pass@5)
#            and the final self-win trajectory set. (run_bestofN_gen_volume.sh,
#            resumable: already-done samples are skipped.)
#   Phase 2  extract_star_corpus.py -> drift-gated, clean-JSON {prompt,completion}
#            rows from every WON trajectory; build dataset_star_iter1 =
#            dataset_volume train + (STaR rows x4), volume valid/test reused.
#   Phase 3  SFT a FRESH LoRA from the int4 base on that corpus (volume hypers,
#            iters 1000) -> adapters_star_iter1.
#   Phase 4  paired GREEDY eval of volume vs star on 12 fresh solver-winnable
#            held-out decks (heldout_decks.json), resumable, then the win delta.
#
# Read the PAIRED star-vs-volume delta on identical held-out decks. Arm design
# mirrors the house ablation style: same base, same hypers, one corpus change
# (the added on-policy self-wins), so the delta isolates exactly that.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_star_iter1.sh \
#     > gemma4_finetune/play_runs/star_iter1.launch.log 2>&1 &
set +e   # one failed game must not abort the chain
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/star_iter1.log
mkdir -p gemma4_finetune/play_runs

abort() { echo "[star] ABORT: $1" | tee -a "$LOG"; exit 1; }

echo "[star] === $(date) START expert-iteration round 1 ===" | tee -a "$LOG"

# ---- Phase 1: best-of-N gate resume (clean pass@5 + final self-win set) ----
echo "[star] --- Phase 1: best-of-N gate resume ---" | tee -a "$LOG"
zsh gemma4_finetune/run_bestofN_gen_volume.sh
echo "[star] Phase 1 returned $(date)" | tee -a "$LOG"

# ---- Phase 2: extract STaR corpus + build dataset_star_iter1 ----
echo "[star] --- Phase 2: extract STaR corpus + build dataset ---" | tee -a "$LOG"
.venv/bin/python gemma4_finetune/extract_star_corpus.py \
  --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
  --deck-path data/benchmarks/generalization_decks.json \
  --base-data gemma4_finetune/dataset_volume \
  --out-corpus gemma4_finetune/star_corpus_iter1.jsonl \
  --out-data gemma4_finetune/dataset_star_iter1 \
  --oversample 4 2>&1 | tee -a "$LOG"
[ -s gemma4_finetune/dataset_star_iter1/train.jsonl ] || abort "extract/build produced no train.jsonl (no winning trajectories?)"

# ---- Phase 3: SFT a fresh LoRA from base (volume hypers) ----
echo "[star] --- Phase 3: SFT adapters_star_iter1 ---" | tee -a "$LOG"
if [ -s gemma4_finetune/adapters_star_iter1/adapters.safetensors ]; then
  echo "[star] adapters_star_iter1 exists, skipping train" | tee -a "$LOG"
else
  # train_v2.py resolves the config's data:/adapter_path: relative to CWD.
  ( cd gemma4_finetune && ../.venv/bin/python train_v2.py \
      --config lora_config_star_iter1.yaml ) 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/adapters_star_iter1/adapters.safetensors ] || abort "SFT produced no adapter"

# ---- Phase 4: paired greedy eval on 12 fresh held-out decks ----
echo "[star] --- Phase 4: paired eval volume vs star on held-out 12 ---" | tee -a "$LOG"
HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
typeset -A ARM_ADAPTER
ARM_ADAPTER=( volume gemma4_finetune/adapters_volume  star gemma4_finetune/adapters_star_iter1 )
for ARM in volume star; do
  for S in $SEEDS; do
    OD=$EVOUT/$ARM/seed$S
    if [ -s "$OD/summary.json" ]; then
      echo "[star] skip $ARM seed$S (done)" | tee -a "$LOG"; continue
    fi
    echo "[star] === eval $ARM seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
    .venv/bin/python gemma4_finetune/play_deck_with_student.py \
      --deck-seed "$S" --deck-path "$HELDOUT" \
      --model-id "$MODEL" --adapter-path "$ARM_ADAPTER[$ARM]" \
      --out-dir "$OD" \
      --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
  done
done

# ---- Phase 4 analysis: paired held-out win delta ----
echo "[star] === held-out result (paired) ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=(o.get("outcome"), o.get("final_foundation_cards"))
    return out
vol=tally("volume"); star=tally("star")
seeds=sorted(set(vol)|set(star))
vw=sw=0
print(f"  {'seed':<10s} {'volume':18s} {'star':18s}")
for s in seeds:
    vo=vol.get(s,("--",0)); so=star.get(s,("--",0))
    vw += vo[0]=="won"; sw += so[0]=="won"
    flag = " <- star gains" if (so[0]=="won" and vo[0]!="won") else (" <- star loses" if (vo[0]=="won" and so[0]!="won") else "")
    print(f"  {s:<10d} {str(vo):18s} {str(so):18s}{flag}")
print(f"  -> held-out wins: volume {vw}/{len(seeds)}  star {sw}/{len(seeds)}  (delta {sw-vw:+d})")
PY
echo "[star] === $(date) ALL DONE ===" | tee -a "$LOG"
