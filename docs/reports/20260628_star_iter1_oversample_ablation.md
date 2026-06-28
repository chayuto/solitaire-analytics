# STaR iter-1 oversample ablation: dose helps, diversity is the binding constraint

Date: 2026-06-28
Context: on-policy pivot, follow-up that isolates WHY STaR iter-1 regressed
(`docs/reports/20260626_star_iter1_expert_iteration_eval.md`).

## TL;DR

The STaR iter-1 regression (x4 oversample of 3 correlated self-win games) is
roughly HALVED by dropping the oversample to x1 (mix share 29.7% -> 9.6%) but NOT
cured. On the 12 fresh held-out decks, paired greedy: volume 5 wins, star_x1 2
wins (delta -3), sum fc delta -94. x1 recovers 2 of the 3 observed x4 collapses
to outright wins (9000101, 9000107) yet still drops 3 decks volume wins (9000105
fc13, 9000120 fc30, 9000124 fc27), and x1's wins are a strict subset of volume's.
Lowering the dose is necessary but not sufficient: even a 9.6% dose of 3
correlated trajectories is net-negative vs plain volume. Oversample tuning (lever
2 of the 20260626 report) is exhausted; the binding constraint is self-win
diversity/count (lever 1), not the mixing ratio. Volume 5/12 stays the floor.

## Method

One change versus the regressed `star_iter1` arm: the SAME 599 drift-gated
self-win rows (3 won best-of-N games, seeds 9000013 / 9000024 / 9000025)
re-mixed into `dataset_volume` train at `--oversample 1` instead of 4 (train 6262
= 5663 volume + 599 STaR, STaR share 9.6% vs 29.7%). A FRESH LoRA from the int4
base, volume hypers held identical (iters 1000, lr 2.0e-4, rank 16, scale 2.0, 16
layers, seq 2048), so the held-out delta isolates exactly the dose. Paired greedy
eval (temp 0, cap 200, prompt v1.6) of star_x1 vs the already-banked volume
baseline on the 12 fresh solver-winnable held-out decks (`heldout_decks.json`,
seeds 9000101..9000124). Unattended chain: `run_star_iter1_x1.sh`.

## Results

SFT health: Iter 1 val 3.731 (x4 was 3.748; same base and valid set, so only the
train mix changed), final Iter 1000 val 0.218, peak 8.41 GB. Textbook clean, the
same loss-vs-play split as before.

Paired held-out (n=12), the 5 decks volume wins shown plus a summary of the rest:

| deck | volume | star_x1 | note |
|---|---|---|---|
| 9000101 | won 52 | won 52 | x1 recovers (x4 was fc9) |
| 9000107 | won 52 | won 52 | x1 recovers (x4 was fc20) |
| 9000105 | won 52 | max_turns 13 | x1 loses (x4 was fc3) |
| 9000120 | won 52 | max_turns 30 | x1 loses (not evaluated under x4) |
| 9000124 | won 52 | max_turns 27 | x1 loses (not evaluated under x4) |
| other 7 | both lose | fc 10/7/4/6/34/6/25 | 9000102/108/112/118/119/121/122 |

-> paired n=12: volume 5 wins, star_x1 2 wins (delta -3); sum fc delta -94.

Dose comparison: x4-star scored 0 wins on its 7 finished decks with fc down on
all 7 (total collapse); x1-star is 2 wins / 12 with fc net -94. So x1 >> x4 (much
less catastrophic, recovers wins) but x1 < volume on the full set.

The early-read trap: the 3-collapse subset (9000101 / 9000105 / 9000107) showed
2 of 3 recovered and looked like a near-rescue. The full 12 reveals x1 newly
drops 9000120 and 9000124 (volume wins that x4 never evaluated), netting -3. The
favorable subset over-reported; the full paired set is the verdict. (Lesson
reinforced: do not call an ablation off a subset selected on the prior arm's
failures.)

## Mechanism

The 20260626 report named two hypotheses for the regression; this run resolves
them jointly. (a) Dose too high: CONFIRMED as a real factor (x1 halves x4's
damage and recovers 2 wins). (b) Poison at any dose: CONFIRMED in the weak sense
that a 9.6% dose is still net-negative vs volume. Reconciling read: 3 self-win
games are too few and too correlated, so imitating them warps the policy toward
their specific lines and costs general wins at ANY positive dose; raising the
dose just amplifies it. Dropping the dose toward zero only asymptotes back to
volume, so no oversample setting of THIS pool beats volume. The lever is the
pool's diversity and count, not its weight. An x2 point would interpolate between
x1's -3 and x4's collapse and is not worth running.

## Verdict and next steps

Lever 2 (oversample tuning) is exhausted: x1 is the best dose and it still loses
to volume. The on-policy pivot's path to any lift narrows to:

1. Lever 1: far more, more diverse self-wins, so the mix is not 3-game-dominated.
   Requires a harvest (greedy volume over many fresh solver-winnable decks, keep
   the ~40% it wins; or higher-N best-of-N over more decks). Multi-window;
   needs a fresh held-out eval set carved out from the harvested decks.
2. Lever 3: reward-weighted RFT / GRPO that weights advantage rather than
   imitating whole correlated games. Most setup; no scaffolding yet.

Else ship volume (5/12), the standing best generalizer and the floor every
on-policy attempt has so far failed to clear.

## Reproducibility and artifacts

- Chain: `gemma4_finetune/run_star_iter1_x1.sh`.
- Config: `gemma4_finetune/lora_config_star_iter1_x1.yaml` (clone of
  `lora_config_star_iter1.yaml`, data + adapter_path swapped).
- Corpus: same 599 rows, `gemma4_finetune/star_corpus_iter1_x1.jsonl`, built by
  `extract_star_corpus.py --oversample 1` (drift-gated re-extraction).
- Generated locally (gitignored / untracked):
  `gemma4_finetune/adapters_star_iter1_x1`,
  `gemma4_finetune/dataset_star_iter1_x1`,
  `gemma4_finetune/play_runs/star_iter1_eval/star_x1/*`.
- Eval reused volume's 12 banked summaries under
  `gemma4_finetune/play_runs/star_iter1_eval/volume/`.
- Memory: `star-iter1-regressed` (updated with this ablation).
