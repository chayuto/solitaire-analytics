#!/bin/zsh
# STaR DIVERSE retrain (Q1) -- full unattended chain (2026-06-30).
#
# The decisive on-policy test. The x1 ablation isolated the star_iter1 regression
# to TRAJECTORY DIVERSITY (3 correlated self-wins), not oversample dose. This
# chain consumes a DIVERSE self-win pool fused from BOTH best-of-N pools:
#   gate    (bestofN_gen_volume,     general12 decks): 9000013/9000024/9000025  = 3 wins / 3 decks
#   harvest (bestofN_winrate_losses, winrate24 decks): 9000237 x5, 9000209/
#           9000215/9000216/9000226 x1                                          = 9 wins / 5 decks
#   = 12 raw trajectories over 8 distinct decks. Phase 1 BALANCES to design B:
#   ONE winning sample per deck (9000237 x5 collapses to its most complete win),
#   so the corpus is 8 trajectories over 8 distinct decks with no single-deck
#   domination. Measured: 599 (gate) + harvest-balanced rows, ~21% STaR share at
#   x1 (vs the x1 ablation's 599 rows / ~10% share of 3 CORRELATED decks). The
#   held-out delta vs volume isolates the diversity lever (the x4 dose was ~30%).
#
#   Phase 1  extract each pool (--no-build, drift-gated, strict-tier only),
#            FUSE the two corpora, build dataset_star_diverse = dataset_volume
#            train + (fused STaR rows x1). CPU. Prints the STaR share.
#   Phase 2  SFT a FRESH LoRA from the int4 base on that corpus (volume hypers,
#            iters 1000) -> adapters_star_diverse. GPU.
#   Phase 3  GREEDY eval of star_diverse ONLY on the 12 fresh heldout decks
#            (9000101-9000124, clean of BOTH harvest sources); volume's 12
#            heldout summaries already exist under star_iter1_eval/volume/ and
#            are reused as the paired baseline. Resumable. GPU.
#   Phase 4  paired held-out delta: volume vs star_diverse.
#
# Decision gate: star_diverse beats volume => diversity was the blocker, on-policy
# SFT works, scale the harvest. Flat/regresses => SFT-on-self-wins is closed,
# only reward-weighted RFT/GRPO remains.
#
# Eval set heldout12 is disjoint from BOTH training sources (general 9000001-26,
# winrate 9000201-39) AND from heldout 9000101-24, so no leakage.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_star_diverse.sh \
#     > gemma4_finetune/play_runs/star_diverse.launch.log 2>&1 &
set +e   # one failed game must not abort the chain
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/star_diverse.log
mkdir -p gemma4_finetune/play_runs

abort() { echo "[div] ABORT: $1" | tee -a "$LOG"; exit 1; }

echo "[div] === $(date) START STaR diverse retrain ===" | tee -a "$LOG"

# ---- Phase 1: extract BOTH pools (--no-build), fuse, build dataset at x1 ----
echo "[div] --- Phase 1: extract + fuse diverse STaR corpus (x1) ---" | tee -a "$LOG"
if [ -s gemma4_finetune/dataset_star_diverse/train.jsonl ]; then
  echo "[div] dataset_star_diverse exists, skipping extract+fuse" | tee -a "$LOG"
else
  .venv/bin/python gemma4_finetune/extract_star_corpus.py \
    --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
    --deck-path data/benchmarks/generalization_decks.json \
    --out-corpus gemma4_finetune/star_corpus_diverse_gate.jsonl \
    --no-build 2>&1 | tee -a "$LOG"
  [ -s gemma4_finetune/star_corpus_diverse_gate.jsonl ] || abort "gate extract produced no rows"

  .venv/bin/python gemma4_finetune/extract_star_corpus.py \
    --runs-root gemma4_finetune/play_runs/bestofN_winrate_losses \
    --deck-path data/benchmarks/winrate_decks.json \
    --out-corpus gemma4_finetune/star_corpus_diverse_harvest.jsonl \
    --no-build 2>&1 | tee -a "$LOG"
  [ -s gemma4_finetune/star_corpus_diverse_harvest.jsonl ] || abort "harvest extract produced no rows"

  # Fuse the two corpora (BALANCED design B) and build the dataset via the
  # canonical build_dataset (volume base train + balanced rows x1; valid/test
  # reused from volume).
  .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "gemma4_finetune")
from extract_star_corpus import build_dataset
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
gate = load("gemma4_finetune/star_corpus_diverse_gate.jsonl")
harv = load("gemma4_finetune/star_corpus_diverse_harvest.jsonl")
rows = gate + harv
# BALANCED 8-DECK (design B): keep exactly ONE winning sample per deck, so the
# 5 copies of seed9000237 collapse to 1 and no single deck dominates the signal.
# Pick each deck's MOST COMPLETE trajectory (max strict rows; ties -> sample name).
per = defaultdict(lambda: defaultdict(int))
for r in rows:
    per[r["_seed"]][r["_sample"]] += 1
chosen = {seed: sorted(samps.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
          for seed, samps in per.items()}
bal = [r for r in rows if r["_sample"] == chosen[r["_seed"]]]
print(f"[fuse] {len(per)} distinct decks; one winning sample kept per deck:")
for seed in sorted(chosen):
    extra = f"  (collapsed from {len(per[seed])} samples)" if len(per[seed]) > 1 else ""
    print(f"[fuse]   seed{seed}: kept {chosen[seed]} ({per[seed][chosen[seed]]} rows){extra}")
base = len(load("gemma4_finetune/dataset_volume/train.jsonl"))
share = len(bal) / (base + len(bal))
print(f"[fuse] balanced STaR rows = {len(bal)} over {len(per)} decks")
print(f"[fuse] STaR share at x1 = {len(bal)}/({base}+{len(bal)}) = {share:.1%}")
# bug-catcher only (balanced expected ~21%); not a design gate.
if share > 0.45:
    raise SystemExit(f"[fuse] ABORT: STaR share {share:.1%} > 45% -- likely a bug")
build_dataset(bal, Path("gemma4_finetune/dataset_volume"),
              Path("gemma4_finetune/dataset_star_diverse"), oversample=1)
PY
fi
[ -s gemma4_finetune/dataset_star_diverse/train.jsonl ] || abort "fuse/build produced no train.jsonl"
echo "[div] train rows: $(wc -l < gemma4_finetune/dataset_star_diverse/train.jsonl)" | tee -a "$LOG"

# ---- Phase 2: SFT a fresh LoRA from base (volume hypers) ----
echo "[div] --- Phase 2: SFT adapters_star_diverse ---" | tee -a "$LOG"
if [ -s gemma4_finetune/adapters_star_diverse/adapters.safetensors ]; then
  echo "[div] adapters_star_diverse exists, skipping train" | tee -a "$LOG"
else
  ( cd gemma4_finetune && ../.venv/bin/python train_v2.py \
      --config lora_config_star_diverse.yaml ) 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/adapters_star_diverse/adapters.safetensors ] || abort "SFT produced no adapter"

# ---- Phase 3: greedy eval of star_diverse on 12 fresh heldout decks ----
# volume's 12 summaries already exist under star_iter1_eval/volume/ and are
# reused as the paired baseline; only the star_diverse arm runs here.
echo "[div] --- Phase 3: eval star_diverse on heldout 12 (greedy) ---" | tee -a "$LOG"
HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
for S in $SEEDS; do
  OD=$EVOUT/star_diverse/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[div] skip star_diverse seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[div] === eval star_diverse seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$HELDOUT" \
    --model-id "$MODEL" --adapter-path gemma4_finetune/adapters_star_diverse \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

# ---- Phase 4: paired held-out delta (volume vs star_diverse) ----
echo "[div] === held-out result (paired): volume vs star_diverse ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=(o.get("outcome"), o.get("final_foundation_cards"))
    return out
vol=tally("volume"); dv=tally("star_diverse")
seeds=sorted(set(vol)&set(dv))   # paired only
vw=dw=0; dfc=0
print(f"  {'seed':<10s} {'volume':18s} {'star_diverse':18s}")
for s in seeds:
    vo=vol.get(s,("--",0)); xo=dv.get(s,("--",0))
    vw += vo[0]=="won"; dw += xo[0]=="won"
    dfc += (xo[1] or 0)-(vo[1] or 0)
    flag = " <- diverse gains" if (xo[0]=="won" and vo[0]!="won") else (" <- diverse loses" if (vo[0]=="won" and xo[0]!="won") else "")
    print(f"  {s:<10d} {str(vo):18s} {str(xo):18s}{flag}")
n=len(seeds)
print(f"  -> paired n={n}: volume {vw} wins, star_diverse {dw} wins (delta {dw-vw:+d}); sum fc delta {dfc:+d}")
print(f"  (x1 ablation for reference: volume 5 vs star_x1 2 on the 12 heldout decks, fc delta -94)")
PY
echo "[div] === $(date) ALL DONE ===" | tee -a "$LOG"
