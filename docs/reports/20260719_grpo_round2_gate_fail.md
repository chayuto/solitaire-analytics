# GRPO round 2: gate FAIL 4/12, RFT closes permanently (2026-07-19)

Window 2026-07-18 to 2026-07-19 (operator-declared, "start/resume compute
window if plugged in"; on AC throughout). This window ran the entire remaining
pre-registered round-2 pipeline end to end: Phase R finish, Phase T extract +
train, Phase E eval + gate. Pre-registration:
`docs/reports/20260705_grpo_round2_preregistration.md`. Resume state consumed:
`docs/reports/20260707_grpo_round2_resume_state.md`. External calibration
context: `docs/reports/20260718_external_strategy_stresstest.md`.

## 1. Verdict

**GATE: grpo_r2 = 4/12 strict greedy wins on heldout12. PASS needed >= 7.
FAIL.** Paired deltas: volume 5 vs grpo_r2 4 (delta -1), grpo_iter1_tr 5 vs
grpo_r2 4 (delta -1). Per the pre-registered hard stop, **RFT CLOSES
PERMANENTLY** at parity-with-movement (round 1) followed by regression-with-
churn (round 2). No dose, LR, or beta retries. The ship path is volume plus
the knowledge-gated endgame oracle (best honest advisor measured, 7/12); the
program pivots to the inference layer and the write-up.

## 2. Phase R final harvest (completed 2026-07-18 16:50 AEST)

Resumed at 50/64 (skip-if-done stepped over all banked games, re-ran the
cleaned seed9000025/s7 slot first), finished 64/64 with 21 wins, zero
partial dirs. Per-deck pass@8 under the round-1 policy (grpo_iter1_tr,
temp 0.7), with the volume pass@5 selection prior for reference:

| deck    | volume pass@5 (prior) | r1 policy pass@8 | fc values (n=8) |
|---------|:---:|:---:|---|
| 9000013 | 1/5 | 3/8 | 11, 13, 43, 52, 52, 52 + 2 losses (round-7/8) |
| 9000024 | 1/5 | 4/8 | 11, 11, 52, 52, 52, 52 + 2 losses |
| 9000025 | 1/5 | 0/8 | all losses, fc 3 to ~18 |
| 9000207 | 0/5 | 2/8 | 3, 24, 24, 25, 52, 52 + 2 losses |
| 9000209 | 1/5 | 2/8 | 5, 5, 5, 9, 9, 37, 52, 52 |
| 9000215 | 1/5 | 1/8 | 11, 12, 21, 21, 21, 28, 49, 52 |
| 9000216 | 1/5 | 4/8 | 26, 27, 27, 28, 52, 52, 52, 52 |
| 9000226 | 1/5 | 5/8 | 10, 23, 39, 52, 52, 52, 52, 52 |
| total   | 7/40 (18%) | 21/64 (33%) | |

The round-1 policy's coverage lift over volume held at N=8 (about 1.9x the
prior rate on the band). Seven of eight decks carried win/loss contrast into
training, above the pre-registered 4-deck invalidation floor.

## 3. Phase T extract (0 drift) and train (cleanest diag yet)

Extraction (`extract_grpo_corpus.py --advantage-mode mean --traj-normalize`,
runs-root `play_runs/bestofN_r2`): **13,563 turn-rows from 8 decks, 0 drift
skips**, 5,655 positive / 7,908 negative advantage rows, corpus
`gemma4_finetune/grpo_corpus_r2.jsonl` (+ manifest sidecar). Two recorded
notes:

1. **9000025 was included, not skipped.** It finished 0/8 wins but its fc
   spread (3 to ~18) gives nonzero reward std, so the registered std-rule
   keeps it with small advantages (max +0.2). The resume doc's "0/8 becomes
   zero-variance and is skipped" prediction assumed all-tie finals; wins are
   not the reward, fc/52 is.
2. **Traj-normalize tail.** Mean rows/trajectory 226.7; two early-death loss
   trajectories (1 row at fc 1, 8 rows at fc 3) scale to |adv| 107.4 and
   12.3, i.e. 9 of 13,563 rows carry |adv| > 10. Equal-mass per trajectory
   by design; AdamW bounds the per-step kick. Proceeded as registered. This
   is the Dr.GRPO-vs-traj-normalize self-inconsistency the 2026-07-18
   stress-test flagged, now with a concrete failure shape (short-trajectory
   advantage concentration) for any future design.

Training (`train_grpo.py --train`, resume + TR anchor
`adapters_grpo_iter1_tr/adapters.safetensors`, beta 2.0, lr 8e-5, minibatch
8, 1 epoch over 1,600 stratified rows = 200 steps, seed 0, ckpt every 50):

- Subsample: 13,525 usable turns (38 dropped > 2048 tok) -> 1,600 (667 pos /
  933 neg). TR pre-pass reference logp mean -0.391, range [-0.602, -0.180].
- Loss small and oscillating (+0.029 to -0.005), gnorm steady 0.1-0.2, no
  drift signature, peak 10.68 GB, 0.03 it/s.
- **Diag separation opened monotonically and passed the pre-registered
  step-100 rule early** (pos = 24 most-positive-adv rows, neg = 24 most
  negative):

| step | pos-adv logp | neg-adv logp | gap |
|---:|---:|---:|---:|
| 0 | -0.4101 | -0.4156 | 0.006 |
| 50 | -0.4261 | -0.5165 | 0.090 |
| 100 | -0.3818 | -0.4969 | 0.115 |
| 150 | -0.3549 | -0.4944 | 0.140 |
| 200 | -0.3281 | -0.5082 | 0.180 |

Pos rose in absolute terms while neg fell: the cleanest training-side
signature of any RFT run in this project (round 1 only held pos flat).
Output: `gemma4_finetune/adapters_grpo_r2/` (step 50/100/150/200 ckpts +
final adapters.safetensors = step 200).

Operational note: the first training launch was killed about 30 minutes in by
a session-side background-task reap (both watcher tasks died simultaneously;
no operator signal). Only the 26-minute TR pre-pass was lost; the relaunch
ran detached (nohup + caffeinate, the Phase R runner pattern) and survived.
First-attempt log preserved at `gemma4_finetune/train_grpo_r2.attempt1.log`.

## 4. Phase E: greedy heldout12, oracle OFF (2026-07-19 08:37 to 11:55 AEST)

Runner `gemma4_finetune/run_grpo_r2_eval.sh` (same deck file, cap 200,
tolerances, and output tree as the cached baselines; arm
`star_iter1_eval/grpo_r2`). Paired table (readout verbatim, fc = final
foundation cards):

| deck | volume | grpo_iter1_tr | grpo_r2 | note |
|---|---|---|---|---|
| 9000101 | won 52 | won 52 | won 52 | |
| 9000102 | loss 10 | loss 22 | loss 10 | |
| 9000105 | won 52 | won 52 | loss 11 | r2 LOSES vs both |
| 9000107 | won 52 | won 52 | won 52 | |
| 9000108 | loss 7 | loss 24 | loss 7 | |
| 9000112 | loss 40 | won 52 | loss 16 | r2 gives back r1's conversion |
| 9000118 | loss 17 | loss 13 | loss 14 | final SOLVED (n=113) |
| 9000119 | loss 7 | loss 7 | loss 7 | final SOLVED (n=264) |
| 9000120 | won 52 | loss 18 | won 52 | r2 recovers r1's regression |
| 9000121 | loss 3 | loss 6 | loss 6 | |
| 9000122 | loss 16 | loss 22 | loss 41 | fd 0, final SOLVED (n=16) |
| 9000124 | won 52 | won 52 | won 52 | |
| **wins** | **5** | **5** | **4** | |
| **sum fc** | **360** | **372** | **320** | |

- Strict wins 4/12; sum fc 320, below volume (360) and iter1_tr (372).
- Zero parse-failure or illegal-move terminations in any arm; retry rescues
  grpo_r2 1 + 36 temp-rescued turns vs volume 2 + 39, iter1_tr 12 + 50
  (format fluency intact, r2 actually cleanest).
- Exact adjudication of all 8 r2 losses (zero replay drift): 5 finals
  structurally UNSOLVABLE (9000102, 9000105, 9000108, 9000112, 9000121), 3
  finals SOLVED (9000118 fc14, 9000119 fc7, 9000122 fc41 fd0 in a 16-node
  near-forced cascade). "Effective" reach 7/12, but the gate is strict wins
  and reads 4.
- Win-set churn, not lift: vs round 1 the policy dropped 9000105 and 9000112
  and recovered 9000120; vs volume it is volume's win set minus 9000105.

## 5. Reading (with the 2026-07-18 external calibration)

1. **The long-odds read was right.** The stress-test put a 7+/12 outcome at
   roughly 39 percent even under a +8-point true lift, and flagged that no
   fetched precedent converts coverage to greedy inside a 200-step, 8-prompt
   budget. Observed: not only no conversion but a net -1 churn. Per the
   pre-registration this FAIL reads "RFT at this budget and scale does not
   convert", and the hard stop executes regardless of the mechanism.
2. **Train-set separation does not transfer.** The cleanest diag trajectory
   yet (gap 0.006 to 0.180, pos rising) co-occurred with a heldout greedy
   regression. The optimizer did exactly what it was asked on the sampled
   band; the asked-for thing is not heldout greedy strength. That is the
   signal-dilution / low-decision-density regime the stress-test named as
   the best mechanistic suspect for parity-sized moves, now with a third
   data point (r1 parity, r1 pass@k double, r2 regression-with-churn).
3. **Coverage stayed real.** The round-1 policy's pass@8 on the band (21/64,
   33 percent vs volume prior 18 percent) confirms the RLVR coverage gain
   was not an artifact; it simply never became greedy wins on fresh decks.
4. **The oracle remains the only measured +2.** 9000122 (fc41, fd0, SOLVED
   n=16) is precisely the phenotype the knowledge-gated endgame oracle
   converts at inference; the shipped advisor should be volume+oracle as
   pre-registered.

## 6. Consequences (pre-registered, now in force)

- **RFT/GRPO line: CLOSED PERMANENTLY.** No round 3; the round-3 recipe
  research (GAGPO state-matched advantages, STEP allocation, routine-turn
  down-weighting) stays banked in
  `docs/reports/20260718_extresearch_round3_recipe.md` as a record, not a
  plan.
- **Ship: volume + knowledge-gated endgame oracle** (honest A/B 7/12, oracle
  13/13 when fired; `docs/reports/20260705_endgame_oracle_ab.md`).
- **Pivot: inference layer + write-up**, per the stress-tested amendments:
  constrained decoding as parse-failure fallback only, per-move solver
  veto/force with a solver-only control arm and unanimity rule, narrowed
  novelty claim ("no published LLM plays complete Klondike with a reported
  win rate, checked 2026-07-18").

## 7. Artifacts

- Rollouts: `gemma4_finetune/play_runs/bestofN_r2/seed*/s*/` (64 games)
- Corpus: `gemma4_finetune/grpo_corpus_r2.jsonl` (+ `.manifest.json`)
- Adapters: `gemma4_finetune/adapters_grpo_r2/`
- Logs: `gemma4_finetune/train_grpo_r2.log` (+ `.attempt1.log`),
  `gemma4_finetune/play_runs/grpo_r2_eval.log`,
  `gemma4_finetune/play_runs/bestofn_r2.log`
- Eval games: `gemma4_finetune/play_runs/star_iter1_eval/grpo_r2/seed*/`
- Runner: `gemma4_finetune/run_grpo_r2_eval.sh`
