# Best-of-N corrective harvest on fresh deep-loss decks: 5/6 recovered, resign calibration characterized

Date: 2026-06-29
Context: on-policy pivot. Second best-of-N ever (the first was the RFT gate on
general12, `bestofn-rft-gate-result`). Follows the 24-deck fresh win-rate run
(`20260628_volume_fresh_winrate.md`), which left volume losing 18 of 24 winnable
decks and produced a labeled loss list.

## TL;DR

Temperature sampling (volume, temp 0.7, N=5) recovers **5 of 6** fresh deep-loss
decks that greedy lost, banking **9 winning trajectories from 30 samples**. The
one miss (9000207) is resign-blocked: it reaches fc50 on two samples but resigns
there rather than closing. A resign-calibration sub-study corrects an in-flight
"over-resign" overclaim: greedy volume's resign is in fact well-calibrated (4/24
resigns, 3 of them correct folds on UNSOLVABLE boards, 1 false on 9000237);
temperature is what breaks the calibration (9/30 temp samples resign, several on
winnable boards), which is also what caps the harvest. Net: the on-policy harvest
works and now yields a diverse self-win pool (these 9 across 5 decks + the gate's
3 across 3 decks = 12 trajectories over 8 distinct fresh decks), which is the
diversity the STaR x1 ablation identified as the missing ingredient. The STaR
diverse retrain that consumes this pool was built but not run (compute halted for
this wrap-up).

## Background

The STaR x1 ablation (`20260628_star_iter1_oversample_ablation.md`) concluded
that naive expert iteration regressed because the self-win pool was 3 highly
correlated trajectories, not because of the oversample dose; diversity is the
binding constraint. The win-rate run then measured volume's true fresh-deck rate
(~33% pooled) and, by adjudication, labeled the deep losses. This run harvests
self-wins on those labeled losses to (a) measure on-policy recovery on a fresh
hard set and (b) build a diverse pool for a second STaR attempt.

## Method

Target: the 6 fresh deep-loss decks with final foundation cards >= 20 from the
24-deck win-rate run (the decks most likely to be sampling-recoverable):
9000237, 9000226, 9000216, 9000215, 9000207, 9000209 (all in
`data/benchmarks/winrate_decks.json`, solver-confirmed winnable, in no training
corpus). Volume adapter (`adapters_volume`), full game from the deal (no
warm-start), temp 0.7, N=5 with a distinct mx.random seed per sample, cap 250,
prompt v1.6, production-faithful retry rules. Breadth-first (round-major) so a
partial run gives pass@k across all decks. Resumable. Runner
`run_bestofN_winrate_losses.sh`. Deep near-misses then adjudicated with
`adjudicate_final_position.py` (zero-drift replay + sound solver).

## Results

### pass@5 (30 samples)

| deck | pass@5 | wins | samples (outcome, fc) |
|---|---|---|---|
| 9000237 | YES | 5/5 | won x5 (the greedy false-resign deck) |
| 9000216 | YES | 1/5 | resigned41, max28, **won52**, resigned8, max28 |
| 9000215 | YES | 1/5 | resigned12, max21, **won52**, resigned4, max12 |
| 9000209 | YES | 1/5 | resigned8, max20, **won52**, max5, max9 |
| 9000226 | YES | 1/5 | max44, resigned44, resigned31, max23, **won52** |
| 9000207 | no  | 0/5 | max24, max24, resigned50, resigned50, max24 |

Recovers 5/6 decks; 9 win-trajectories / 30 samples (30% per sample). The 30%
rate is NOT comparable to the gate's 8.6% on general12: this is the deep-near-miss
subset (volume already reaches fc>=20 greedily), so recovery is intrinsically
easier. 9000237 recovers on every sample; the other three stalls/throws recover
on exactly one of five.

### The single miss is resign-blocked

9000207 reaches fc50 on two of five samples and resigns at fc50 both times (the
same false-resign shape as 9000237), and stalls at fc24 on the other three. It
never sampled a non-resign close from fc50. So it is recoverable in principle
(it gets to fc50) but the resign reflex blocked every attempt; a resign-suppressed
sample would very likely take it to 6/6.

### Resign calibration (corrects an in-flight overclaim)

A round-1 read showed 3/6 temp samples resigning and prompted an "over-resign"
hypothesis. Adjudication refuted it for the greedy (deployment) regime:

Greedy win-rate run, all 4 resigns adjudicated:

| greedy resign | final fc/fd | solver | verdict |
|---|---|---|---|
| 9000237 | 50 / 0 | SOLVED n=3 | false (near-won) |
| 9000232 | 11 / 5 | UNSOLVABLE n=104 | correct (board dead) |
| 9000238 | 8 / 15 | UNSOLVABLE n=168452 | correct (board dead) |
| 9000239 | 3 / 10 | UNSOLVABLE n=50184 | correct (board dead) |

Greedy volume resigns 4/24 and 3 are correct folds on boards it had genuinely
thrown; only 9000237 is a false resign. The greedy resign reflex is well
calibrated. Under temp 0.7 it is not: 9/30 samples resigned, including the deep
false resigns 9000216 s1 (fc41, SOLVED n=12) and 9000215 s1 (fc12, SOLVED
n=52133). So temperature, not the policy, produces the apparent over-resign, and
it is the mechanism capping this harvest (it folds winnable boards mid-sample).

## Mechanism and reading

Greedy decoding makes each lost deck a fixed failure; temperature sampling
explores alternative lines and recovers most deep near-misses (false resigns,
stalls, and even one thrown board, 9000209). The recovery is one-in-five on the
genuine stalls/throws and five-in-five on the false-resign deck, consistent with
"the policy can win these, greedy just lands on a bad deterministic line." The
resign action is double-edged: correct at greedy (folds dead boards) but
sampled too eagerly at temperature, where it discards winnable boards and caps
both the harvest yield and (on 9000207) a deck outright.

## Outcome: a diverse self-win pool

The harvest banks 9 winning trajectories across 5 distinct fresh decks (9000237
x5, 9000209, 9000215, 9000216, 9000226). Combined with the RFT gate's 3
(general12 seeds 9000013/9000024/9000025), the on-policy pool is now 12
trajectories over 8 distinct fresh decks. This is qualitatively different from
the 3 correlated trajectories that sank STaR x1, and is the input the STaR
diverse retrain needs.

## Verdict and next steps

The on-policy harvest is viable and productive: it recovers most fresh hard decks
and yields a diverse pool. It does not by itself beat volume; that is the next
experiment.

1. STaR diverse retrain (pre-registered, built not run): SFT volume on the 8-deck
   self-win pool at x1 oversample (per the x1 ablation), fresh LoRA, eval paired
   vs volume on heldout12 (clean of both harvest sources, general12 and winrate).
   Decision: beats volume -> diversity was the blocker, on-policy SFT works,
   scale; flat or regresses -> SFT-on-self-wins is dead, only RFT/GRPO remains.
2. Resign-suppressed best-of-N (cheap harness change): add a --no-resign flag that
   routes a sampled move_index -1 to the illegal-move retry, which would likely
   take 9000207 to a win (6/6) and raise the trajectory count for a richer pool.
3. If the retrain does not beat volume, the remaining lever is reward-weighted
   RFT/GRPO (multi-week build) versus shipping volume (~33% generalizer).

## Reproducibility and artifacts

- Runner: `gemma4_finetune/run_bestofN_winrate_losses.sh`.
- Decks: `data/benchmarks/winrate_decks.json` (seeds 9000201-9000239; the 6
  targets are the fc>=20 losses).
- Samples: `gemma4_finetune/play_runs/bestofN_winrate_losses/seed*/s*/` (gitignored).
- Adjudicator: `adjudicate_final_position.py` (patched this window to locate a
  run's deck across all `data/benchmarks/*.json`; the summary records only a
  generator tag, not a path).
- Memory: `bestofn-rft-gate-result` (update 2026-06-29), `volume-generalization-rate-pinned`.
