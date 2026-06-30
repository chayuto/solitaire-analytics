# STaR diverse retrain (Q1): diverse self-win SFT reaches parity, not lift; the SFT-on-self-wins track is closed

Date: 2026-07-01
Context: this is Q1, the decisive experiment pre-registered in the 2026-06-29
on-policy window wrap-up (`20260629_onpolicy_window_wrapup.md`, section 6). It is
the last open test of whether the student can be improved by supervised
fine-tuning on its own winning self-play trajectories (STaR / expert iteration).

## TL;DR

Training volume on a DIVERSE, balanced pool of its own best-of-N self-wins
(8 distinct fresh decks, one win each) scores **4 wins vs volume's 5** on the 12
heldout decks (paired greedy), delta -1 win, sum foundation-card delta -2. Two
readings: (1) the diversity fix WORKS as predicted, it erases the catastrophic
x1 regression (fc delta -2 here vs x1's -94) and recovers 2 of the 3 decks x1
collapsed; but (2) removing the poison only buys parity with volume, not the lift
the pre-registered gate required. Verdict: SFT-on-self-wins does not beat volume
in any configuration tried (correlated or diverse, single or multi-deck, x1 or
x4 dose). The on-policy SFT-by-imitation line is closed for generalization. The
sole remaining on-policy lever is reward-weighted RFT/GRPO; the decision taken is
to ship volume.

## Background

The expert-iteration line has three prior data points:

- **star_iter1 (x4 dose, 3 correlated wins)** regressed: 0 wins vs volume 3 on
  the 7 paired decks, fewer foundation cards on every deck
  (`20260626_star_iter1_expert_iteration_eval.md`).
- **star_iter1_x1 (x1 dose, same 3 correlated wins)** halved the damage but still
  regressed: 2 vs volume 5 on heldout12, sum fc delta -94, with three of volume's
  wins collapsing to fc 9/3/20 (`20260628_star_iter1_oversample_ablation.md`).
  Conclusion: oversample dose was not the root cause; the 3 self-wins were too
  correlated. Diversity, not dose, is the binding constraint.
- **best-of-N corrective harvest** then banked a diverse self-win pool: 12 winning
  trajectories over 8 distinct fresh decks (gate 3 + harvest 9), the diversity
  the ablation said was missing (`20260629_bestofn_corrective_harvest.md`).

Q1 consumes that diverse pool and asks the decisive question: with the
correlation removed, does SFT-on-self-wins finally beat volume?

## Method

### Corpus construction and a dose confound caught before training

Phase 1 extracted strict-tier, drift-gated {prompt, completion} rows from both
best-of-N pools (`extract_star_corpus.py --no-build`, byte-faithful prompt
reconstruction via the v1.6 renderer, strict-JSON completions only):

- gate pool (general12 decks): 9000013/9000024/9000025 = 3 wins, **599 rows**
  (matches the historical star_iter1 count exactly, confirming extractor
  fidelity).
- harvest pool (winrate24 decks): 9000209/9000215/9000216/9000226 plus 9000237
  won five times = 9 wins, **1626 rows**.

Naively fusing all 12 trajectories at x1 oversample would have been **2225 rows,
a 28.2% STaR share** of the 7888-row train set, with deck **9000237 alone
contributing 910 rows (41% of the STaR signal)**. That is two confounds: (a) a
28% share sits at the same dose as the x4 run that poisoned the policy (~30%),
and (b) five wins of one board re-introduces exactly the correlation the diverse
harvest was meant to remove. Either would make a regression uninterpretable.

The corpus was therefore built as **design B (balanced 8-deck)**: keep exactly
one winning sample per deck (9000237's five collapse to its most complete
trajectory, s4, 222 rows), so the pool is 8 trajectories over 8 distinct decks
with no single-deck domination. Result: **1537 STaR rows, 21.3% share**.

```
dataset_star_diverse: 5663 volume base + 1537 balanced STaR x1 = 7200 train
  (valid 531 / test 665 reused from dataset_volume, no STaR leakage)
  per-deck rows: 9000013 214, 9000024 158, 9000025 227, 9000209 223,
                 9000215 203, 9000216 151, 9000226 139, 9000237 222
```

Dataset validated: 0 malformed rows, every completion parses as JSON with all
three required keys, valid/test byte-identical to volume.

### Training and evaluation

Fresh LoRA from `mlx-community/Gemma4-E2B-IT-Text-int4`, volume hyperparameters
held identical (rank 16, scale 2.0, lr 2e-4, 1000 iters, 16 layers,
`lora_config_star_diverse.yaml`). Greedy paired evaluation on the 12 heldout
decks (seeds 9000101-9000124, disjoint from BOTH training sources: general
9000001-26 and winrate 9000201-39), faithful v1.6 harness, cap 200. Volume's 12
heldout summaries from the x1 ablation are reused as the paired baseline; only
the star_diverse arm ran.

## Results

SFT was clean: val loss 3.727 (iter 1) to **0.220** (iter 1000), matching x1's
0.218. No training pathology.

Paired greedy, heldout12:

| seed | volume | star_diverse | |
|---|---|---|---|
| 9000101 | won 52 | won 52 | |
| 9000102 | max_turns 10 | **won 52** | diverse gains |
| 9000105 | won 52 | max_turns 18 | diverse loses |
| 9000107 | won 52 | won 52 | |
| 9000108 | max_turns 7 | max_turns 10 | |
| 9000112 | max_turns 40 | max_turns 7 | |
| 9000118 | max_turns 17 | max_turns 44 | |
| 9000119 | max_turns 7 | max_turns 7 | |
| 9000120 | won 52 | won 52 | |
| 9000121 | max_turns 3 | max_turns 3 | |
| 9000122 | max_turns 16 | max_turns 19 | |
| 9000124 | won 52 | max_turns 42 | diverse loses |

**Paired n=12: volume 5 wins, star_diverse 4 wins (delta -1); sum fc delta -2.**

star_diverse won one deck volume could not (9000102, where volume stalled at
fc10) and lost two volume won (9000105 at fc18, 9000124 at fc42). Adjudicating
the deep losses could at most credit 9000124 as a cap-truncation and reach 5-5
parity; it cannot produce a win advantage, so the verdict is robust to
adjudication.

### The diversity fix worked, in isolation

Against the x1 regression specifically, the diverse pool did exactly what the
ablation predicted:

- Sum fc delta improved from **-94 (x1) to -2 (diverse)**: the catastrophic
  draw-down is gone.
- Of the three decks x1 collapsed (9000101/9000105/9000107, which fell 52 to
  9/3/20), star_diverse **recovered two to outright wins** (9000101, 9000107) and
  lifted the third from fc3 to fc18 (9000105).

So x1 regressed because its three self-wins were too correlated, not because of
the oversample dose. Balanced diversity removes the poison. This confirms the
oversample-ablation hypothesis cleanly.

## Verdict: negative branch, track closed

The pre-registered gate was: beats volume -> diversity was the blocker, scale the
harvest; flat or regresses -> SFT-on-self-wins is closed, only RFT/GRPO remains.

star_diverse does not beat volume (4 vs 5, a one-deck swing within noise on
n=12). Removing the correlation poison brought the policy back to PARITY with
volume but produced no lift above it. Combined with the prior runs, the full
picture is that training the student on its own wins lands at or below volume in
every configuration tried: x4 correlated (0 vs 3), x1 correlated (2 vs 5), x1
diverse balanced (4 vs 5). **The on-policy SFT-by-imitation line is exhausted for
generalization.**

The mechanism is consistent with the broader project finding that single-turn
imitation loss does not predict full-game play
(`full-game-play-reframes-bench`): the diverse adapter trains to the same loss as
volume and imitates winning moves faithfully, yet greedy full-game generalization
does not exceed the volume baseline, because the wins it imitates do not encode a
policy improvement the base corpus lacks. Imitation cannot exceed the
distribution it imitates; only a reward signal can.

## What remains, and the decision

The sole untried on-policy mechanism is reward-weighted RFT/GRPO, which optimizes
a reward rather than imitating trajectories. The best-of-N gate showed a real but
sparse reward signal (roughly 8.6% per sample on hard decks), so RFT is viable
but not cheap (multi-week build; must preserve full action-space coverage per the
ORPO draw-starvation lesson, `orpo-loop-penalty-next-step`).

Decision taken (2026-07-01): **ship volume.** Two independent research lines have
now closed on the same conclusion: static-corpus engineering does not generalize
(`solver-grounded-form3-result`), and on-policy SFT-on-self-wins does not beat
volume (this report). Volume remains the best fresh-deck generalizer at ~33%
pooled (16/48), roughly 4x the untuned base, already published to Hugging Face
(`chayuto/gemma-4-e2b-it-solitaire-advisor-lora`). RFT/GRPO is parked as the only
remaining upside lever should the project resume the chase later.

## Reproducibility and artifacts

- Runner: `gemma4_finetune/run_star_diverse.sh` (Phase 1 extract+fuse+balance,
  Phase 2 SFT, Phase 3 greedy heldout12 eval, Phase 4 paired delta). Resumable.
- Config: `gemma4_finetune/lora_config_star_diverse.yaml`.
- Corpus builder: `gemma4_finetune/extract_star_corpus.py` (unchanged; invoked
  twice with `--no-build`, fused and balanced in the runner's Phase 1).
- Decks: `data/benchmarks/heldout_decks.json` (eval),
  `data/benchmarks/generalization_decks.json` + `winrate_decks.json` (pool
  provenance).
- Generated/local (gitignored): `dataset_star_diverse/`, `adapters_star_diverse/`,
  `star_corpus_diverse_{gate,harvest}.jsonl`,
  `play_runs/star_iter1_eval/star_diverse/`.
- Run log: `gemma4_finetune/play_runs/star_diverse.log` (START 2026-06-30 20:56,
  ALL DONE 2026-07-01 02:43, exit 0).
- Every result above is read from on-disk eval summaries
  (`play_runs/star_iter1_eval/*/seed*/summary.json`) and the training log. No
  result was written before it was measured.
