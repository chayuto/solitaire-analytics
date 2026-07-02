# GRPO / RLOO iteration-0: run 1 collapse, trust-region run 2 (2026-07-01/02)

## TL;DR

The last untried lever (on-policy RFT) was un-parked and built tonight. Run 1,
the KL-free cut (advantage-weighted REINFORCE continue-trained from the volume
adapter, lr 8e-5, 200 steps), COLLAPSED: mean completion logp fell -0.5 to -23
monotonically from step 1, and the resulting adapter parse-failed on all 12
heldout decks (0 wins vs volume 5, sum fc delta -360). Mechanism diagnosed: the
loss -A*logp is unbounded below for A<0, so negative-advantage turns drive their
completion logp toward -inf and drag the shared JSON scaffold down with them;
gradient norms stayed ~0.3-0.5 throughout, so this is slow DRIFT, not spikes,
and gradient clipping cannot bind. Fix built and unit-proven: a one-sided
downward trust region beta*(ref_logp - logp)^2 that activates only when logp
drops below the volume reference. Run 2 (identical config plus the trust region,
beta 2.0) was launched 2026-07-02 09:13 AEST and was at training step 40/200,
healthy, at session wrap-up. Result lands in
`gemma4_finetune/play_runs/grpo_iter1_tr.log` (Phase 4 paired table + Phase 5
adjudication).

## 1. Why RFT, and why now

Both prior research lines are closed: static-corpus engineering (all three forms
collapse on fresh decks, 2026-06-23) and on-policy SFT-on-self-wins (x4/x1/
diverse all at-or-below volume, 2026-07-01). SFT only ever reinforces winning
trajectories; it has no gradient that says "do less of what the losing rollouts
did". Group-relative RFT supplies exactly that missing NEGATIVE gradient, and it
was the pre-registered sole remaining lever.

Iteration-0 data was free: the two existing best-of-N pools
(`bestofN_gen_volume`, `bestofN_winrate_losses`) were sampled BY the volume
policy at temp 0.7, so continue-training volume on them is legitimately
on-policy. Each deck's 5 rollouts have outcome variance (deep-reach/win vs
shallow-stall/resign) = the advantage signal, at zero new rollout cost.

## 2. What was built (all committed with this report)

- `gemma4_finetune/grpo_loss.py` - advantage-weighted REINFORCE loss
  (L = -A * length-normalised mean completion logp), reusing orpo_loss's
  response-only slice-trick verbatim (project only response positions through
  the 256k head; 16 GB-safe). Plus the one-sided trust region (section 5).
  Self-test proves the gradient sign end-to-end on a stub model: +adv logp
  rises (-3.55 to -0.0001), -adv falls (-3.55 to -2642), adv=0 is a strict
  no-op (zero loss, zero grad).
- `gemma4_finetune/extract_grpo_corpus.py` - generalises the STaR extractor to
  ALL rollouts (not just wins). R = final_fc/52; per-deck group advantage
  A = (R - mean)/std shared by every decision turn of that rollout; zero-std
  groups skipped (9000237, 5/5 wins, correctly contributes nothing). Inherits
  the drift-gated replay, the strict-JSON completion filter, and (critically)
  the break-at-resign, so a false-resign-at-fc50 rollout (seed9000207) donates
  its high-reward reach turns but never the resign action itself.
- `gemma4_finetune/train_grpo.py` - the RFT trainer: continue-trains the volume
  adapter (resume verified: 160/160 keys matched, 80 non-zero lora_b), gradient
  ACCUMULATED over a minibatch (8) before each AdamW step so advantage
  magnitude is preserved within the batch, mem-cap/grad-checkpoint/probe-mode
  inherited from train_orpo, stratified subsampling (--max-rows) because the
  ~2000-token v1.6 prompts make the full 12.5k-row corpus a ~20h epoch.
- `gemma4_finetune/run_grpo_iter1.sh` / `run_grpo_iter1_tr.sh` - the chained
  unattended runners (extract -> train -> heldout12 greedy eval -> paired delta
  -> exact adjudication), modeled on run_star_diverse.sh, idempotent/resumable.

## 3. The corpus (measured)

12,516 turn-rows from 12 decks, extracted in ~4 s with ZERO drift failures
(the byte-fidelity gate holds on losing trajectories too, not just the wins the
STaR extractor had exercised). 5,993 positive-advantage rows / 6,523 negative;
advantage range [-2.00, +1.99], mean +0.015. 359 rows over the 2048-token cap
were dropped at load (12,157 usable). Per-deck table in
`gemma4_finetune/grpo_corpus_iter1.jsonl.manifest.json` (the corpus itself is
gitignored). Sanity spot-checks: seed9000003's fc26 reach carries +1.99 while
its resign-at-fc2 rollouts carry -0.6; seed9000207's two fc50 reaches +1.22 vs
its fc24 stalls -0.82.

## 4. Run 1 (KL-free): COLLAPSE, measured

Config: volume resume, 1600 turns (763 pos / 837 neg, stratified subsample,
seed 0), minibatch 8, lr 8e-5, 200 steps (1 epoch), no trust region.
Launched 2026-07-01 21:38 AEST; chain ALL DONE 01:46 AEST.

Training diagnostic (mean logp over the 24 most-positive / 24 most-negative
advantage turns):

| step | pos-adv logp | neg-adv logp |
|---|---|---|
| 0 | -0.53 | -0.55 |
| 50 | -5.54 | -5.66 |
| 100 | -10.97 | -11.22 |
| 150 | -24.13 | -25.52 |
| 200 (final) | -22.67 | -23.96 |

Monotonic collapse from step 1; pos and neg fall TOGETHER (separation stays
~1.3), i.e. the update is not discriminating good from bad play, it is
uniformly destroying the output distribution. Loss values oscillated (+/-2)
while gnorm stayed ~0.3-0.5 the whole run.

Heldout12 paired result (cap 200, greedy, vs the cached volume baseline):
volume 5 wins, grpo_iter1 0 wins (delta -5); sum fc delta -360. All 12 decks
ended `parse_failure` at fc=0, fd 19-21, i.e. the adapter emitted unparseable
output from the very first decisions and burned the full 10-failure allowance
even with retry temp 0.3. Every checkpoint (50/100/150/200) is inside the
collapse; nothing salvageable.

## 5. Diagnosis and the fix

The loss -A*logp is ASYMMETRIC without a trust region: for A>0 the incentive is
bounded (logp -> 0), but for A<0 loss reduction is UNBOUNDED (drive logp ->
-inf). With ~half the corpus negative-advantage, the negative branch dominates
and runs away. Two measurements pin the mode:

1. A clipped 6-step smoke measured raw batch gradient norms of ~0.3 (and run
   2's live log shows 0.4-0.5): there are NO spikes to clip. The collapse is a
   slow, consistent drift, so grad clipping (the first fix attempted) cannot
   bind and was abandoned for the trust region.
2. The negative/positive completions are format-identical clean JSON differing
   only in move content, so the "shared-scaffold gradients cancel" hope was
   wrong: suppressing thousands of negative completions suppresses the JSON
   format itself.

Fix: a ONE-SIDED downward trust region. A pre-pass caches each turn's
reference logp under the unmodified volume policy (the policy currently equals
volume at step 0, so a no-grad forward suffices); the loss adds
beta * relu(ref_logp - logp)^2, penalising only downward drift past the
reference while leaving positive learning (logp above ref) untouched. The
self-test proves the bound: under pure -1.0-advantage pressure for 40 steps,
logp goes -3.55 -> -2646 without the trust region and -3.55 -> -3.68 with it
(beta 5). A double-forward bug in grpo_terms (the transformer body ran twice
per example) was found and fixed in the same edit, roughly halving step time.

Grad-norm clipping support (--clip-grad-norm) remains in the trainer but is
documented as non-binding for this failure mode.

## 6. Run 2 (trust region): IN FLIGHT at wrap-up

Launched 2026-07-02 09:13 AEST (auto-gated on the smoke releasing the GPU).
Changes exactly ONE variable vs run 1: --trust-region-beta 2.0 (same lr 8e-5,
same 1600-turn subsample, same 200 steps, same eval protocol), so the paired
delta isolates the trust region.

Measured before wrap-up: reference pre-pass cached 1600 turns in 1608 s
(ref logp mean -0.543, range [-0.879, -0.224]); training at step 40/200 with
loss -0.01/+0.11 and gnorm 0.4-0.5, peak 10.99 GB. The step-50/100/150/200
[diag] lines in the log are the anti-collapse check: bounded pos/neg logp near
-0.5 = trust region holding; falling past ~-2 = still collapsing.

The chain is detached (`nohup caffeinate -i -s`, survives the session); it
will train, eval all 12 heldout decks, print the paired table, and adjudicate
finals into `gemma4_finetune/play_runs/grpo_iter1_tr.log`. Expected finish
roughly 4-5 h from launch.

## 7. Decision gate (pre-registered)

- grpo_iter1_tr BEATS volume on heldout12 -> the negative-gradient lever works
  once collapse is prevented; next step is the full loop (fresh rollouts per
  round, per-token KL, more rounds).
- Bounded (no collapse) but flat/regresses -> one on-policy epoch of
  trust-regioned RFT does not beat volume either; with corpus-engineering and
  SFT-on-self-wins already closed, the improvement program closes on
  SHIP VOLUME.
- Still collapses -> the mean-logp trust region is too weak; a per-token KL to
  the reference policy is the required escalation before any verdict on the
  lever itself.

Adjudication note for whoever reads the result: judge run 2 on the paired
table AND the Phase 5 exact adjudication (cap-truncated near-wins count), and
check `rescued_turns` / parse-failure counts in the summaries for JSON-health
regression vs volume.

## 8. Artifacts

- Committed: the four modules + two runners above, this report, and the corpus
  manifest (`grpo_corpus_iter1.jsonl.manifest.json`).
- Gitignored/local: `grpo_corpus_iter1.jsonl` (regenerable via
  extract_grpo_corpus.py from the gitignored play_runs),
  `adapters_grpo_iter1*` (run 1's collapsed adapter + run 2's in-training),
  `play_runs/grpo_iter1*.log` + eval dirs under
  `play_runs/star_iter1_eval/grpo_iter1*/`.
- Volume baseline reused: `play_runs/star_iter1_eval/volume/` (raw 5/12,
  cap 200), unchanged since the STaR evals, so all arms stay paired.
