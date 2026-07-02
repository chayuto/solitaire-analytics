#!/bin/zsh
# GRPO / RLOO iteration-0, TRUST-REGION run (run 2) -- 2026-07-02.
#
# Run 1 (KL-free advantage-weighted REINFORCE from volume, lr 8e-5, 200 steps)
# COLLAPSED: mean logp -0.5 -> -23 monotonically from step 1, all 12 heldout decks
# parse_failure, 0 wins vs volume 5. Mechanism: the loss -A*logp is unbounded
# below for A<0, so negative-advantage turns drive their completion logp to -inf
# with no trust region to stop them, dragging the shared JSON/reasoning tokens
# down with them. The grad norm stayed ~0.3 the whole time (slow DRIFT, not
# spikes), so gradient clipping does not bind -- the fix must be a trust region.
#
# This run changes EXACTLY ONE thing vs run 1: a one-sided downward trust region
# (--trust-region-beta 2.0). A pre-pass caches each turn's reference logp under
# volume, and the loss adds beta*(ref_logp - logp)^2 whenever logp drops below
# reference, opposing the collapse direction while leaving positive learning free.
# The loss self-test proves it bounds the -advantage runaway (-3.55 -> -3.68 with
# TR vs -> -2646 without). Everything else (lr 8e-5, 1600 rows, 200 steps, volume
# resume, eval) matches run 1, so the held-out delta isolates the trust region.
#
#   Phase 1  reuse the advantage corpus (grpo_corpus_iter1.jsonl). CPU.
#   Phase 2  continue-train volume WITH the trust region -> adapters_grpo_iter1_tr.
#            A per-50-step pos/neg logp diagnostic logs whether it stays bounded
#            (the anti-collapse check) and whether pos/neg separate (lever engaged).
#   Phase 3  greedy eval on the 12 heldout decks vs cached volume (raw 5/12).
#   Phase 4  paired held-out delta: volume vs grpo_iter1_tr.
#   Phase 5  exact adjudication of the grpo_iter1_tr finals.
#
# Decision gate: beats volume => the negative-gradient lever works once collapse is
# prevented; build the full KL-GRPO loop next. Bounded but flat/regresses => even a
# trust-regioned RFT does not beat volume in one on-policy epoch; ship volume and
# close the on-policy program (RFT tried, corpus-engineering + SFT-on-self-wins
# already closed). Still collapses => the mean-logp trust region is too weak; a
# per-token KL is required.
#
# Launch:
#   nohup caffeinate -i -s zsh gemma4_finetune/run_grpo_iter1_tr.sh \
#     > gemma4_finetune/play_runs/grpo_iter1_tr.launch.log 2>&1 &
set +e
cd "$(dirname "$0")/.."
LOG=gemma4_finetune/play_runs/grpo_iter1_tr.log
mkdir -p gemma4_finetune/play_runs

abort() { echo "[grpo-tr] ABORT: $1" | tee -a "$LOG"; exit 1; }

echo "[grpo-tr] === $(date) START GRPO iteration-0 (trust region) ===" | tee -a "$LOG"

# ---- Phase 1: reuse the advantage corpus ----
CORPUS=gemma4_finetune/grpo_corpus_iter1.jsonl
[ -s "$CORPUS" ] || abort "missing $CORPUS (run extract_grpo_corpus first)"
echo "[grpo-tr] corpus rows: $(wc -l < "$CORPUS")" | tee -a "$LOG"

# ---- Phase 2: continue-train volume WITH the trust region ----
echo "[grpo-tr] --- Phase 2: RFT+TR train adapters_grpo_iter1_tr (from volume) ---" | tee -a "$LOG"
if [ -s gemma4_finetune/adapters_grpo_iter1_tr/adapters.safetensors ]; then
  echo "[grpo-tr] adapters_grpo_iter1_tr exists, skipping train" | tee -a "$LOG"
else
  .venv/bin/python gemma4_finetune/train_grpo.py --train \
    --data "$CORPUS" \
    --resume-adapter gemma4_finetune/adapters_volume/adapters.safetensors \
    --adapter-path gemma4_finetune/adapters_grpo_iter1_tr \
    --epochs 1 --max-rows 1600 --minibatch 8 --lr 8e-5 --trust-region-beta 2.0 \
    --steps-per-report 20 --steps-per-eval 50 --save-every 50 2>&1 | tee -a "$LOG"
fi
[ -s gemma4_finetune/adapters_grpo_iter1_tr/adapters.safetensors ] || abort "train produced no adapter"

# ---- Phase 3: greedy eval on the 12 heldout decks ----
echo "[grpo-tr] --- Phase 3: eval grpo_iter1_tr on heldout 12 (greedy) ---" | tee -a "$LOG"
HELDOUT=data/benchmarks/heldout_decks.json
SEEDS=(9000101 9000102 9000105 9000107 9000108 9000112 9000118 9000119 9000120 9000121 9000122 9000124)
EVOUT=gemma4_finetune/play_runs/star_iter1_eval
MODEL=mlx-community/Gemma4-E2B-IT-Text-int4
for S in $SEEDS; do
  OD=$EVOUT/grpo_iter1_tr/seed$S
  if [ -s "$OD/summary.json" ]; then
    echo "[grpo-tr] skip grpo_iter1_tr seed$S (done)" | tee -a "$LOG"; continue
  fi
  echo "[grpo-tr] === eval grpo_iter1_tr seed$S (greedy) $(date +%H:%M) ===" | tee -a "$LOG"
  .venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed "$S" --deck-path "$HELDOUT" \
    --model-id "$MODEL" --adapter-path gemma4_finetune/adapters_grpo_iter1_tr \
    --out-dir "$OD" \
    --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
    --max-illegal-moves 10 --prompt-version v1.6 2>&1 | tee -a "$LOG"
done

# ---- Phase 4: paired held-out delta (volume vs grpo_iter1_tr) ----
echo "[grpo-tr] === held-out result (paired): volume vs grpo_iter1_tr ===" | tee -a "$LOG"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
EV="gemma4_finetune/play_runs/star_iter1_eval"
def tally(arm):
    out={}
    for s in sorted(glob.glob(f"{EV}/{arm}/seed*/summary.json")):
        o=json.load(open(s))
        out[o["deck_seed"]]=(o.get("outcome"), o.get("final_foundation_cards"))
    return out
vol=tally("volume"); gp=tally("grpo_iter1_tr")
seeds=sorted(set(vol)&set(gp))
vw=gw=0; dfc=0
print(f"  {'seed':<10s} {'volume':20s} {'grpo_iter1_tr':20s}")
for s in seeds:
    vo=vol.get(s,("--",0)); xo=gp.get(s,("--",0))
    vw += vo[0]=="won"; gw += xo[0]=="won"
    dfc += (xo[1] or 0)-(vo[1] or 0)
    flag = " <- grpo+TR gains" if (xo[0]=="won" and vo[0]!="won") else (" <- grpo+TR loses" if (vo[0]=="won" and xo[0]!="won") else "")
    print(f"  {s:<10d} {str(vo):20s} {str(xo):20s}{flag}")
n=len(seeds)
print(f"  -> paired n={n}: volume {vw} wins, grpo_iter1_tr {gw} wins (delta {gw-vw:+d}); sum fc delta {dfc:+d}")
print(f"  (run 1, NO trust region: grpo 0 wins, all parse_failure, fc delta -360)")
PY

# ---- Phase 5: exact adjudication of grpo_iter1_tr finals ----
echo "[grpo-tr] === adjudicate grpo_iter1_tr final positions (timeout 120s each) ===" | tee -a "$LOG"
.venv/bin/python gemma4_finetune/adjudicate_final_position.py --timeout-s 120 \
  $EVOUT/grpo_iter1_tr/seed*/ 2>&1 | tee -a "$LOG"

echo "[grpo-tr] === $(date) ALL DONE ===" | tee -a "$LOG"
