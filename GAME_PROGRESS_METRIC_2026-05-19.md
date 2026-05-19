# Proposal: A Better Game Progress Metric

Date: 2026-05-19
Audience: data harvesting team, data pipeline
Status: proposal

## Why this exists

The export reports `completionProgress`, which equals foundation cards divided
by 52. It is a correct completion number but a poor progress signal, and that
gap caused a real failure: game session `9229e2cc3adc` ran for more than 76
logged moves with `completionProgress` flat at 17%, and nothing flagged it as
stalled. The harvester kept calling the model, and the pipeline kept accepting
the resulting decisions as training rows.

## What is wrong with foundation-only progress

- It ignores the mid-game objective. Revealing a face-down tableau card is the
  main work of the early and mid game, but it moves a foundation-only metric by
  exactly zero.
- It is lumpy. The number changes only when a card reaches a foundation, so it
  reads flat through long stretches of genuine work or genuine decline.
- It cannot separate a healthy 17% from a deadlocked 17%. A single number with
  no components gives nothing to build a stall test on.

## Proposal

Keep `completionPct` for win detection and display. Add a separate progress
signal that blends the two quantities that actually advance a Klondike game:
cards locked onto foundations, and cards revealed from face-down.

```
foundationCards = sum of ranks currently on the 4 foundations    # 0 to 52
faceDownTotal   = sum of faceDownCount across the 7 columns       # 21 down to 0

completionPct = 100 * foundationCards / 52
progressScore = 100 * ( 0.65 * foundationCards / 52
                      + 0.35 * (21 - faceDownTotal) / 21 )
```

Both endpoints are clean: a fresh deal scores 0, a win scores 100. The 0.65 and
0.35 weights are tunable; foundations carry more because they are the win
condition, revealed cards carry the rest because they are the enabling work.

## Export the raw components

The blended score is for a smooth signal. The raw components are what make
stall detection trustworthy. The export should carry, per state:

- `foundationCards` (0 to 52)
- `faceDownTotal` (21 down to 0)

From those, a deadlock test is straightforward:

```
stalled = foundationCards AND faceDownTotal both unchanged for N turns
```

N of 20 to 30 is reasonable. With foundation-only data this test fires
constantly as a false positive. With both components it fires only on a real
deadlock, which makes it safe to drive an auto-terminator.

Also useful per move: `progressDelta = progressScore(now) - progressScore(prev)`.
A move with delta 0 that was not a forced draw is a candidate bad label.

## Uses

1. Live auto-termination in the harvester. End a game when `stalled` is true,
   so a hard deal does not become a several-hundred-call drain.
2. Poison-row filtering in the ingest pipeline. Decisions from a stalled game
   teach the model to loop and should be kept out of the training set.
3. Scoring teacher decisions. `progressDelta` gives a per-move signal of whether
   a decision helped, which can weight or filter training examples.

## Optional refinements

- Rank-weighted foundations. Sending a King home is much harder than an Ace.
  Weighting by rank (`sum of foundation ranks / 364`) tracks effort better than
  a linear card count.
- Tableau sortedness. Count face-up cards that sit in valid descending
  alternating runs. It captures board organization, but it rises and falls in
  healthy play, so use it as a tie-breaker, not a primary term.
- Solver distance, offline only. This repo has `ParallelSolver` and
  `analysis.calculate_progression_score`. Scoring a state by how close a beam
  search gets to a win is the gold standard for validating the cheap metric.
  It is too expensive to run per move during harvesting.

## Where to compute it

The ingest pipeline does not need to wait for the harvester. `scripts/ingest_exports.py`
already parses the board JSON out of every prompt, so it can compute
`foundationCards`, `faceDownTotal`, and `progressScore` for every interaction
now, and flag stalled games. The harvester should adopt the same formula so its
live auto-terminator and the offline pipeline agree on what counts as progress.

## Reference

Failure case: session `9229e2cc3adc`, exports `solitaire-ai-log-cc3adc-1779184594840.json`
and `solitaire-ai-log-cc3adc-1779186724024.json` in `data/raw/`. See
`GEMMA4_E2B_SCHEMA_AUDIT_2026-05-19.md` for the related schema audit.
