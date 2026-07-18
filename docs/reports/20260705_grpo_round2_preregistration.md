# GRPO round 2: pre-registration (2026-07-05)

Operator decision 2026-07-05 (after the run-2 parity-with-movement result and
the honest-oracle A/B): run ONE upgraded RFT round with a HARD STOP. This doc
pre-registers the design, the gate, and the stop rule before any compute.

## 1. Decision context (measured, in hand)

- Run 2 (TR beta 2.0, minimal iteration-0): paired heldout12 = volume 5 vs
  grpo_iter1_tr 5 (delta +0), sum fc +12, zero parse failures, first-ever
  RFT-converted deck (9000112), one loss (9000120, cap-bound SOLVED at fd=2).
  Report `20260702_grpo_rft_iter0.md` section 8.
- Honest knowledge-gated endgame oracle: volume+oracle 7/12, grpo_tr+oracle
  6/12; oracle 13/13 when fired; stacks on any policy (independent lever).
- Run 2 lacked every research-identified upgrade (20260704 research doc):
  std-normalized advantages at N=5, no variance filtering, no per-trajectory
  normalization, no curriculum, stale-pool iteration-0 data.

## 2. Hypothesis

Fresh on-policy rollouts (sampled by the CURRENT grpo_iter1_tr policy) on a
success-band deck pool, learned with mean-only advantages, per-trajectory
normalization, and the proven one-sided trust region, lift paired heldout12
performance to >= volume+2 strict wins. Mechanism: run 2 proved the negative
gradient moves play without collapse; round 2 fixes the signal-quality
defects and samples where the reward contrast lives.

## 3. Design (three phases, ~3 windows)

### Phase R: rollout harvest (~1.5-2 windows, resumable)

- Policy: adapters_grpo_iter1_tr (the round-1 output; on-policy for round 2).
- Sampling: temp 0.7, distinct mx seed per sample, N=8 per deck, cap 250,
  parse/illegal tolerance 10, NO oracle flag (pure policy rollouts).
- Deck pool (success-band selection, pass@5 under volume as the prior):
  9000013, 9000024, 9000025 (gate pool, 1/5 each), 9000209, 9000215,
  9000216, 9000226 (corrective pool, 1/5 each), 9000207 (0/5 strict but
  reaches fc50 resign-blocked = recoverable class). 8 decks x N=8 = 64 games,
  breadth-first (round-major) so a window-end kill still leaves usable
  groups. 9000237 (5/5) excluded as no-signal; 9000002/3/10/21 (0/5) excluded
  as likely all-fail under the band prior.
- Eval decks (heldout12, 9000101-24) are NEVER rolled out on.

### Phase T: extract + train (~0.5 window)

Extractor upgrades (extract_grpo_corpus.py, flag-gated, defaults preserve
iteration-0 behavior):
- `--advantage-mode mean`: A_i = R_i - mean(R_group), NO std division
  (Dr.GRPO fix; kills the small-group noise amplification).
- `--traj-normalize`: scale each row's advantage by
  (mean rows per trajectory / rows in its trajectory), so every trajectory
  contributes equal gradient mass regardless of length (removes the
  ~2.5x long-game dominance).
- Existing zero-variance group skip retained (DAPO-style; all-tie groups
  carry no signal).

Trainer (train_grpo.py, unchanged code):
- Resume from adapters_grpo_iter1_tr; TR pre-pass anchors the reference to
  that same policy (the loss penalizes downward drift below the ROUND-2
  starting policy).
- --trust-region-beta 2.0, lr 8e-5, minibatch 8, 200 steps, --max-rows 1600
  stratified, checkpoints every 50. LR held at run-2's value deliberately
  (the TR held it stable; the literature band 5e-6..1e-5 is noted, and
  halving is the FIRST knob if the diag shows TR-bind saturation or
  separation collapse, not a mid-run change).

### Phase E: eval + gate (~0.5 window)

- Greedy, cap 200, heldout12, vs the cached volume baseline (raw 5/12) and
  vs grpo_iter1_tr (5/12). Exact adjudication of finals.
- Oracle OFF for the gate (raw policy measurement); a +oracle run may follow
  for the product picture but does not enter the gate.

## 4. Pre-registered gate and HARD STOP

- **PASS: grpo_r2 >= 7 strict paired wins on heldout12 (volume+2).** Then
  RFT is a confirmed lever: continue the loop (round 3 design would add
  per-token KL and re-banded decks) under normal window discipline.
- **FAIL: anything below 7 strict wins.** RFT CLOSES PERMANENTLY at
  parity-with-movement. No further rounds, no dose/LR/beta retries. The
  program ships volume+oracle and pivots fully to the inference layer +
  write-up. This stop is unconditional; the round-2 upgrades were the
  best-evidence fixes, and if fixed-signal on-policy RFT cannot beat volume
  by 2 on its favorable bench, the lever is not worth more laptop-nights.
- Secondary readouts (recorded, not gating): vs grpo_iter1_tr delta, fc
  profile, diag trajectory (pos/neg separation), rollout pool pass@8 per
  deck (updates the band prior for any future work).

## 5. Budget and interruption

Rollouts ~16-20h GPU total (resumable, breadth-first), train ~3-4h, eval
~3.5h. Any window-end kill is cheap by construction (skip-if-done at every
phase). If the operator ends the program early, whatever groups exist with
outcome variance are usable; the gate still requires the full paired 12.

## 6. What would invalidate this round

- Rollout pool yields < 4 decks with within-group outcome variance
  (mean-only advantages then have too little contrast; record and stop
  before training rather than train on noise).
- Any drift-gate failure in extraction (fidelity bug -> fix before train).
- Diag showing pos/neg separation NOT opening by step 100 (training-side
  no-signal; stop, record, close).
