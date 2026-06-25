# STaR expert-iteration round 1: eval and verdict

Date: 2026-06-26
Context: session close-out. On-policy pivot, first training step.

## TL;DR

Naive expert-iteration (SFT the volume student on its own best-of-N self-wins)
REGRESSED fresh-deck generalization. On 12 fresh solver-winnable held-out decks,
paired greedy decoding: volume wins 3, the STaR adapter wins 0 on the 7 decks
both arms finished (eval stopped at wrap-up), and STaR reached FEWER foundation
cards than volume on every one of those 7 decks. SFT loss was healthy throughout
(val 3.75 -> 0.23), so this is a loss-vs-play split, not a broken-training
artifact. Volume (5/12) remains the best generalizer. Verdict: the on-policy
pivot's cheapest probe is negative as designed.

## Background

The corpus-engineering line is closed for generalization: every static-corpus
lever (close-out reweight, strategy-text, solver-grounded) lifts in-distribution
but collapses to 3-4/12 on fresh decks, while plain volume uniquely holds 5/12
(see `20260619_closeout_augmentation_eval.md` and the solver-grounded form-3
result). The remaining lever is on-policy. Its pre-registered first step was a
best-of-N probe to check whether the volume policy even SAMPLES wins on decks it
loses greedily. This session ran the clean pass@5 of that gate and then took the
first on-policy training step: expert iteration (STaR), i.e. fine-tune the policy
on its own winning trajectories and re-measure generalization.

## Method

One unattended chain, `gemma4_finetune/run_star_iter1.sh`, four phases:

1. Best-of-N gate resume. Volume student (`adapters_volume`), temp 0.7, N=5
   independent samples (distinct mx.random seed each) on the 7 of 12
   generalization decks volume lost greedily, all winnable by construction.
   Resumable (skips existing `summary.json`).
2. STaR corpus extraction. `extract_star_corpus.py` replays each WON trajectory
   through the harness's own v1.6 renderer and pairs each decision with its saved
   raw response. Prompt fidelity is guaranteed by the harness DRIFT GATE: the
   re-rendered prompt length must equal the recorded `prompt_chars` at every
   decision turn, or the extraction aborts. Only responses that pass
   `prepare_dataset`'s filter (fence-stripped text parses as a JSON object with
   all three required keys) are emitted, so the STaR completions are
   byte-format-identical to the teacher/volume corpus. Output is mlx-lm
   `{prompt, completion}` rows.
3. SFT. Build `dataset_star_iter1` = `dataset_volume` train + (STaR rows x4
   oversample); volume valid/test reused unchanged (the STaR games are too few to
   hold out, and adding them to valid would leak). Train a FRESH LoRA from the
   int4 base with the volume hyperparameters held identical (iters 1000, lr
   2.0e-4, rank 16, scale 2.0, 16 layers, seq 2048). Following the house ablation
   style, this isolates one change versus volume: the added on-policy self-wins.
4. Paired held-out eval. Greedy (temp 0), max-turns 200, prompt v1.6, volume vs
   star, on 12 FRESH solver-confirmed-winnable decks (`heldout_decks.json`, seeds
   9000101..9000124, built this session, in no training corpus and no prior eval,
   so contamination-free). Resumable.

## Results

### Phase 1: best-of-N gate, clean pass@5 = 3/7

| deck | samples | wins | pass@5 | best fc |
|---|---|---|---|---|
| seed9000002 | 5 | 0 | no  | 13 |
| seed9000003 | 5 | 0 | no  | 26 |
| seed9000010 | 5 | 0 | no  | 15 |
| seed9000013 | 5 | 1 | YES | 52 |
| seed9000021 | 5 | 0 | no  | 6 |
| seed9000024 | 5 | 1 | YES | 52 |
| seed9000025 | 5 | 1 | YES | 52 |

3/7 winnable decks that greedy lost (greedy was 0/7) are solved by sampling: 3
winning trajectories from 35 samples, ~8.6% per sample. The resume added
seed9000025 (a deck greedy false-resigned) over the earlier partial's 2/7.
Positive but sparse, unchanged in character from the partial read.

### Phase 2: STaR corpus

| won game | strict rows | turns |
|---|---|---|
| seed9000013 s2 | 214 | 221 |
| seed9000024 s2 | 158 | 162 |
| seed9000025 s5 | 227 | 234 |

599 clean rows from 3 won games. `dataset_star_iter1` train = 5663 volume + 599
STaR x4 (2396) = 8059 rows (STaR share 29.7%). Drift gate passed on all three
games (byte-faithful prompt reconstruction).

### Phase 3: SFT health (textbook clean)

| iter | val loss |
|---|---|
| 1   | 3.748 |
| 100 | 0.316 |
| 300 | 0.254 |
| 500 | 0.286 |
| 700 | 0.246 |
| 900 | 0.211 |
| 1000 | 0.234 |

Final train loss 0.222, 1.88M trained tokens, peak 8.4 GB. Val loss decreased
monotonically in trend and never diverged. By every training metric this adapter
is healthy.

### Phase 4: held-out paired eval (greedy)

Volume completed all 12; star completed 7 before the session wrap-up stop. Paired
on the 7 decks both finished:

| deck | volume | star |
|---|---|---|
| seed9000101 | won (fc52) | lost (fc9)  |
| seed9000102 | lost (fc10) | lost (fc7) |
| seed9000105 | won (fc52) | lost (fc3)  |
| seed9000107 | won (fc52) | lost (fc20) |
| seed9000108 | lost (fc7)  | lost (fc5) |
| seed9000112 | lost (fc40) | lost (fc2) |
| seed9000118 | lost (fc17) | lost (fc5) |

Paired (n=7): volume 3 wins, star 0 wins (delta -3). STaR reached fewer
foundation cards than volume on all 7 decks, including hard collapses on the
three decks volume won outright (star fc 9 / 3 / 20 vs 52). Volume's full
12-deck baseline is 5/12 (wins: 9000101, 9000105, 9000107, 9000120, 9000124).
The 0-vs-3 with universal fc regression will not flip in the unrun 5.

## Mechanism

Healthy SFT loss, broken full-game play. This is the same loss-vs-play split
seen before (single-turn or held-out-token metrics do not predict full-game
behaviour). The most likely cause is overfitting to 3 highly-correlated full-game
trajectories at ~30% of the training mix: the policy is pulled toward the
specific move patterns of three particular games in a way that damages general
play, even as next-token loss on the volume validation distribution stays low.
This is NOT the ORPO draw-starvation failure mode, where preference pairs never
contained draws; the STaR rows are full games and span the entire action space.
The diagnosis of "overfit to too few, too correlated trajectories" is a
hypothesis, not yet isolated by an ablation.

## Verdict and next steps

Naive expert iteration as designed (SFT on own 3 self-wins, x4 oversample, mixed
into volume, fresh LoRA, volume hypers) does not lift fresh-deck generalization
and in fact regresses it. The sparse-but-positive gate did not convert into a
training win via the cheapest on-policy method.

Open levers before further on-policy spend:

1. Far more, and more diverse, self-win trajectories. Three games is too few and
   too correlated; sample many more decks (beyond the 7 gate decks) or raise N so
   the STaR mix is diverse rather than 3-game-dominated.
2. Lower oversample / smaller STaR share, to avoid warping the policy toward a
   handful of trajectories.
3. Proper RFT / GRPO with a reward signal rather than SFT-on-wins, which can
   weight advantage rather than imitate whole games.

If none pan out, ship volume (5/12), the standing best generalizer and the
existing in-distribution lead recipe's base.

## Reproducibility and artifacts

- Chain: `gemma4_finetune/run_star_iter1.sh` (committed 2db19ad).
- Extractor: `gemma4_finetune/extract_star_corpus.py` (drift-gated replay).
- Config: `gemma4_finetune/lora_config_star_iter1.yaml`.
- Held-out decks: `data/benchmarks/heldout_decks.json` (12 fresh winnable, seeds
  9000101..9000124).
- Generated locally (gitignored / untracked): `gemma4_finetune/adapters_star_iter1`,
  `gemma4_finetune/dataset_star_iter1`, `gemma4_finetune/star_corpus_iter1.jsonl`,
  `gemma4_finetune/play_runs/star_iter1*` and `.../star_iter1_eval/`.
- Eval is resumable: rerun the chain (or just the Phase 4 loop) to finish the
  last 5 star decks for a clean 12-deck paired number; finished samples are
  skipped by their `summary.json`.
- Memory: `star-iter1-regressed`, with `bestofn-rft-gate-result` updated to the
  clean pass@5 = 3/7.
