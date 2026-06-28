# On-policy compute window: wrap-up and program state (2026-06-27 to 2026-06-29)

Author: research log, synthesized 2026-06-29.
Scope: four experiments run across one multi-night compute window on the local
16 GB M-series machine, plus the integrated findings, the revised program state,
and the gated queue. Per-experiment detail lives in the four cited reports; this
is the lab-notebook synthesis.

Standing constraints honored: one GPU job at a time, machine on mains, faithful
v1.6 harness with exact final-position adjudication, all eval decks
solver-confirmed winnable and held out of training by seed. No model was shipped
or published this window; nothing was committed (manifest in section 7).

---

## 0. Abstract

Entering this window the static-corpus line was closed for generalization (plain
"volume" SFT holds the best fresh-deck rate; every reweighting recipe collapses)
and the on-policy pivot had produced a positive-but-sparse best-of-N gate and a
regressed first STaR (expert-iteration) attempt. This window: (1) isolated WHY
STaR regressed (oversample dose vs trajectory diversity); (2) pinned volume's
true fresh-deck win-rate with a larger sample; (3) harvested a diverse on-policy
self-win pool to feed a second STaR attempt; (4) characterized volume's resign
calibration. Headline results: the STaR regression is a diversity problem, not a
dose problem (oversample tuning exhausted); volume generalizes to ~33% of fresh
winnable decks, not the ~42% two small samples implied; a corrective best-of-N
harvest recovers 5/6 fresh deep-loss decks and yields a diverse 8-deck self-win
pool; and volume's greedy resign reflex is well calibrated (only temperature
breaks it). The decisive next experiment, a STaR retrain on the diverse pool, is
built and pre-registered but not run (compute halted here by request).

---

## 1. Experiment 1: STaR oversample ablation (dose vs diversity)

Report: `20260628_star_iter1_oversample_ablation.md`.

Question: STaR iter-1 (SFT volume on its own 3 best-of-N self-wins, x4 oversample)
regressed fresh-deck generalization. Was the cause the oversample dose (3 games
over-weighted at ~30% of the mix) or the trajectories themselves (3 too-correlated
games poisoning the policy at any dose)?

Method: identical to the regressed arm except the same 599 self-win rows are mixed
at x1 (9.6% share) instead of x4 (29.7%). Fresh LoRA from the int4 base, volume
hyperparameters held identical, paired greedy eval on the 12 fresh heldout decks
against volume's existing baseline.

Result: SFT clean (iter-1 val 3.731 == x4's 3.748, since only the mix changed;
final val 0.218). Paired n=12: **volume 5 wins, star_x1 2 wins (delta -3, sum fc
delta -94)**. x1 recovered 2 of the 3 observed x4 collapses to outright wins
(9000101, 9000107) but still dropped 3 volume-wins (9000105, 9000120, 9000124),
and its wins are a strict subset of volume's. x1 (2 wins) is much better than x4
(0 wins, total collapse) but still below volume (5).

Verdict: lowering the dose helps but does not cure. Even a 9.6% dose of 3
correlated games is net-negative, and dropping the dose toward zero only
asymptotes back to volume, so no oversample setting of this pool beats volume.
Oversample tuning is exhausted; the binding constraint is self-win
diversity/count, not the mixing ratio. (Process note: an early read on the 3
collapse decks looked like a near-rescue, 2/3 recovered; the full 12 netted -3.
Do not adjudicate an ablation off a subset selected on the prior arm's failures.)

## 2. Experiment 2: volume fresh-deck win-rate

Report: `20260628_volume_fresh_winrate.md`.

Question: volume's generalization rested on two 12-deck point estimates (both
5/12 = 42%). What is the true rate on a larger fresh sample?

Method: 24 fresh decks (seeds 9000201-9000239), solver-confirmed winnable,
disjoint from all training corpora and the heldout/general eval sets. Volume
greedy, cap 200, one game per deck. Deep near-misses (fc>=20) adjudicated.

Result: **6/24 strict (25%), meanFC 24.4; effective 8/24 (33%)** after
adjudication credits two near-forced wins lost to controllable failures (9000226
cap-truncated at fc40/fd0 SOLVED n=13; 9000237 false-resigned at fc50 SOLVED n=3).
Pooled across all three fresh sets (general12 5/12 + heldout12 5/12 + this 6/24)
= **16/48 = 33%** [approx 95% CI 20-47%]. The deep losses split into two thrown
boards (UNSOLVABLE) and two stalls on winnable (SOLVED).

Verdict: volume generalizes to ~1 in 3 fresh winnable decks, below the ~42% the
small samples implied, still ~4x the untuned base (~8%). The 18-deck loss list,
with exact SOLVED/UNSOLVABLE labels on the deepest six, becomes the corrective
target set for the harvest.

## 3. Experiment 3: best-of-N corrective harvest

Report: `20260629_bestofn_corrective_harvest.md`.

Question: can temperature sampling recover the fresh decks volume loses greedily,
and does it bank a diverse self-win pool to fix the STaR diversity problem?

Method: volume, temp 0.7, N=5, cap 250, breadth-first, on the 6 deep-loss decks
(fc>=20). Resumable.

Result: **pass@5 recovers 5/6 decks, 9 win-trajectories / 30 samples.** 9000237
recovers 5/5; 9000216/9000215/9000209/9000226 recover 1/5 each; 9000207 is the
only miss (it reaches fc50 twice but resigns there). Combined with the gate's 3
general12 trajectories, the on-policy pool is now **12 trajectories over 8
distinct fresh decks** -- the diversity Experiment 1 said was missing.

Verdict: the harvest is viable and productive. It does not itself beat volume;
the STaR diverse retrain is the test.

## 4. Experiment 4: resign calibration

Folded into the harvest report. Question: a round-1 read (3/6 temp samples
resigning) suggested volume over-resigns. Is that a real greedy-policy failure?

Method: adjudicate all 4 greedy resigns from the win-rate run and the temp
resigns from the harvest.

Result: greedy volume resigns 4/24 and **3 are correct** folds on UNSOLVABLE
(genuinely thrown) boards; only 9000237 is a false resign (SOLVED n=3). The
greedy resign reflex is well calibrated. Temperature breaks it: 9/30 temp samples
resign, including deep false resigns (9000216 s1 fc41 SOLVED n=12; 9000215 s1
fc12 SOLVED n=52133).

Verdict: the "over-resign" hypothesis is refuted for deployment (greedy resign is
mostly correct, a +1-deck lever at most). It matters only as a temperature
artifact that caps the harvest, suggesting a resign-suppressed best-of-N would
yield more.

---

## 5. Integrated findings

1. **Data scaling saturated before this window and stays saturated.** The
   in-distribution size-vs-performance curve (assembled from eval summaries this
   window): base 0 rows -> meanFC 14.2 / 1 win; ~2500 rows -> ~26 / 3-4 wins;
   6859 (volume) -> 27.7 / 5; >6859 (volume_v0620, 8115) -> flat 5; recipe arms
   at 6-10k rows move in-dist wins by RECIPE (close-out, solver-grounding), not
   row count. The two largest corpora ever trained (volume_v0620 8115,
   volcloseout_v0620 9792, both ~2026-06-20/21) bought nothing over their smaller
   siblings. More clean 31B data is not the lever, and 6859 is most of the clean
   31B data that exists.

2. **Volume's real generalization is ~33%**, not ~42%, and is partly bottlenecked
   by recoverable failures (one false resign per 24, one cap truncation per 24);
   the genuine policy gaps are stalls-on-winnable (the close-out phenotype) and
   thrown boards.

3. **The on-policy SFT-by-imitation line is dose-exhausted but diversity-open.**
   x4 collapsed, x1 still -3; the missing ingredient is a diverse self-win pool,
   which the harvest now provides (8 distinct decks). Whether diversity converts
   to a generalization lift is the single open question.

4. **Volume's resign is well calibrated at greedy** (good news: not a liability
   in deployment), but temperature breaks it, which is a harvest-mechanics
   detail, not a policy flaw.

5. **Methodology held up and caught two of its own errors this window**: the
   STaR early-read trap (a favorable subset over-reported a rescue) and the
   "over-resign" overclaim (refuted by adjudication). Exact final-position
   adjudication remains the load-bearing instrument; a real bug in it (it could
   not locate fresh-set decks) was found and fixed here.

---

## 6. Program state and the gated queue

Where the project stands: the shipped/published volume student is the best
fresh-deck generalizer at ~33%, ~4x base; static corpus engineering is closed for
generalization; on-policy SFT has a built diverse pool and one decisive untried
test. The product-vs-research fork from the 2026-06-11 strategic review is still
the governing decision.

Queue (one GPU job at a time; nothing launched, by request):

- **Q1 (decisive, built not run): STaR diverse retrain.** SFT volume on the
  8-deck self-win pool (gate 3 + harvest 9, deduped) at x1 oversample, fresh
  LoRA, eval paired vs volume on heldout12 (clean of both harvest sources).
  Beats volume -> diversity was the blocker, on-policy SFT works, scale the
  harvest; flat/regresses -> SFT-on-self-wins is closed, only RFT remains.
- **Q2 (cheap mechanics): resign-suppressed best-of-N.** A --no-resign flag in
  the harness; would likely take 9000207 to 6/6 and enrich the pool. Run before
  Q1 only if a richer pool is wanted first.
- **Q3 (CPU, anytime): adjudicate the 12 un-adjudicated win-rate losses + cap-300
  re-bank of 9000226.** Completes the effective-rate record; no GPU.
- **Q4 (terminal fork): if Q1 does not beat volume**, the explicit product-vs-
  research call: commit to RFT/GRPO (the only untried mechanism, multi-week
  build, preserve full action-space coverage per the ORPO draw-starvation lesson)
  or ship volume (~33% generalizer) with the consolidated negative-results
  write-up.

---

## 7. Artifacts manifest (all uncommitted as of 2026-06-29)

Reports (new this window):
- `docs/reports/20260628_star_iter1_oversample_ablation.md`
- `docs/reports/20260628_volume_fresh_winrate.md`
- `docs/reports/20260629_bestofn_corrective_harvest.md`
- `docs/reports/20260629_onpolicy_window_wrapup.md` (this file)

Scaffolding (new):
- `gemma4_finetune/lora_config_star_iter1_x1.yaml`, `run_star_iter1_x1.sh`
- `gemma4_finetune/run_volume_winrate.sh`
- `gemma4_finetune/run_bestofN_winrate_losses.sh`
- `data/benchmarks/winrate_decks.json` (24 fresh decks; worth tracking like the
  other benchmark sets, the reports reference it)

Code fix:
- `gemma4_finetune/adjudicate_final_position.py` (locate a run's deck across all
  `data/benchmarks/*.json`; backward-compatible)

Generated/local (gitignored): `adapters_star_iter1_x1`, `dataset_star_iter1_x1`,
`star_corpus_iter1_x1.jsonl`, `play_runs/star_iter1_eval/star_x1/`,
`play_runs/volume_winrate_eval/`, `play_runs/bestofN_winrate_losses/`.

Memory updated: `star-iter1-regressed`, `volume-generalization-rate-pinned`
(new), `bestofn-rft-gate-result`, `MEMORY.md`.

## 8. Reproducibility

Every result above is read from on-disk eval summaries
(`play_runs/*/seed*/summary.json`, fields `outcome` and `final_foundation_cards`)
and solver adjudication (`adjudicate_final_position.py`, zero-drift engine replay
then `solve_winnable`). All decks are solver-confirmed winnable and seed-disjoint
across train and eval. The STaR x1 chain, the win-rate run, and the harvest are
each a single resumable shell script with the launch command in its header
comment. No result was written before it was measured.
