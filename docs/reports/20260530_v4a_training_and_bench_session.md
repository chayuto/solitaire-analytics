# v4-A: training + bench session report (2026-05-30)

**Date**: 2026-05-30
**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.8)
**Compute**: single M5, 16 GB unified memory
**Pre-registration**: `docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md` section 3
**Verdict**: HOLD. Reversal corpus-filtering does not clear the promotion bar on the gemma-3n base either. Bench best tier 2.80 < v1.1's 3.15; full-game fc=0 (the worst of any gemma-3n arm) with the Gemma-4-style QS col5/col7 loop from turn 1. FP3 FALSIFIED: the filter pushed gemma-3n into the Gemma 4 loop rather than a gemma-3n JD loop, collapsing v1.1's fc=3 to fc=0. The corpus-filter program is closed on both bases.

## One-line summary

v4-A (gemma-3n + frozen reversal-filtered corpus) trained cleanly. Best bench tier 2.80 (iter750), below v1.1's 3.15 on the same base with the raw corpus. On no metric does any v4-A checkpoint exceed v1.1; at best it ties (oscillation, agreement) and it loses on tier, foundation recovery, and JSON validity. The pre-committed gate triggers HOLD whenever bench tier is under 3.15, so the verdict is decided on the bench alone and the full-game step is moot for the gate.

## What ran

1. v4-A training: `train_v2.py --config lora_config_v4a.yaml`, gemma-3n base + frozen 1832-row filtered corpus (`dataset_v4a`). Background run, exit 0.
2. v4-A bench sweep: `sweep_v4a_checkpoints.sh` over all four checkpoints (250/500/750/1000), scored by `score_v4a_learning_curve.py`. Exit 0.
3. Full-game on seed 3263196305: RAN (v4-A iter750), stopped manually at turn 65 once the loop was entrenched (see "Full-game" below).

The eval tooling (`posttune_n20_v4a_runner.py`, `sweep_v4a_checkpoints.sh`, `score_v4a_learning_curve.py`) was written this session; the pickup doc had noted it as missing. It turned out `posttune_n20_runner.py` was already the gemma-3n runner (v4-A shares that base), so the new runner is a v4-A-segregated copy with a `--max-tokens` arg, output under `baseline_n20_v4a/`. Bench protocol held at 512 tokens to match the v1.1 baseline measurement.

## Training results and gates

| Gate | Target | Actual | Result |
|---|---|---|---|
| TP1 dataset split | ~1279 / 144 / 168 | 1279 / 144 / 168 (verified 2026-05-29) | pass |
| TP2 final val loss | 0.30 to 0.45 | 0.351 | pass |
| TP3 peak memory | <= 12 GB | 11.49 GB | pass |
| TP3 wall clock | <= 100 min | 104.5 min | over by 4.5 min, immaterial |

Trainable params 11.27M of 4.46B (0.253%), LoRA scoped to attention + MLP only (the `altup` module stayed untouched, no mid-validation crash). Val curve was monotone healthy: 6.34 at iter 1, 0.42 at iter 100, settling near 0.35 by iter 300 and holding to 0.351 at iter 1000. No divergence. All four checkpoints plus the final adapter and `adapter_config.json` are in `gemma4_finetune/adapters_v4a/`.

## Bench results (20-state, 512 tokens, temperature 0.0) -- MEASURED

| config | json | illegal | agree | tier | foundation | osc agree | osc tier |
|---|---|---|---|---|---|---|---|
| v1 untuned (3n) | 20/20 | 1/20 | 11/20 | 2.10 | 2/7 | 4/7 | 2.57 |
| **v1.1 iter750 (3n, raw)** | 20/20 | 2/20 | 11/20 | **3.15** | 6/7 | 4/7 | 3.57 |
| v4a iter250 (3n, filtered) | 17/20 | 4/20 | 8/20 | 2.15 | 3/7 | 2/7 | 2.57 |
| v4a iter500 | 20/20 | 2/20 | 10/20 | 2.35 | 3/7 | 2/7 | 2.43 |
| **v4a iter750 (best)** | 19/20 | 2/20 | 11/20 | **2.80** | 5/7 | 4/7 | 3.43 |
| v4a iter1000 | 18/20 | 2/20 | 11/20 | 2.45 | 4/7 | 4/7 | 3.57 |

Source: `gemma4_finetune/baseline_n20_v4a/learning_curve_v4a.json`.

## Pre-registered bench predictions: verdict

| Prediction | Bar | v4-A best (iter750) | Result |
|---|---|---|---|
| BP1 (primary) | best tier >= 3.15 | 2.80 | FALSIFIED |
| BP2 | foundation recovery >= 6/7 | 5/7 | FALSIFIED |
| BP3 | oscillation agreement >= 4/7 | 4/7 | CONFIRMED (ties v1.1) |
| BP4 | teacher agreement >= 11/20 | 11/20 | CONFIRMED (ties v1.1) |

Note BP3 and BP4 are pre-registered against absolute floors (>= 4/7, >= 11/20), and v4-A meets both, but it meets them by exactly TYING v1.1, not by beating it. The two predictions that compare against v1.1 directly (BP1 tier, BP2 foundation) both fail.

## The clean isolation, and what the filter actually bought

v4-A differs from v1.1 in exactly one variable: the corpus is reversal-filtered instead of raw. Same base (gemma-3n), same hyperparameters, same runner, same scorer. So the comparison is direct.

Reading the table column by column, no v4-A checkpoint exceeds v1.1 on anything:

- tier: v4-A best 2.80 < v1.1 3.15 (worse)
- foundation recovery: v4-A best 5/7 < v1.1 6/7 (worse)
- oscillation agreement: v4-A 4/7 == v1.1 4/7 (tie)
- oscillation tier: v4-A best 3.57 (iter1000) == v1.1 3.57 (tie)
- teacher agreement: v4-A 11/20 == v1.1 11/20 (tie)
- JSON validity: v4-A 17 to 20 of 20, vs v1.1's clean 20/20 (worse or equal)

So the reversal filter, applied to the base that actually plays, bought nothing. It did not even improve oscillation-state behaviour over the raw corpus (the metric the filter was designed to protect); it tied. And it cost net tier, foundation recovery, and some format adherence. This is a stronger negative than "the filter helps oscillation but hurts overall": on this base the filter has no upside at all.

Secondary observation: the v4-A learning curve is non-monotonic and peaks at iter750 (2.80), then regresses at iter1000 (2.45), the same iter750 peak v1.1 showed. JSON validity also degraded relative to v1.1 (one to three malformed responses per checkpoint vs zero), a mild sign the filtered corpus slightly hurt format adherence.

## Per-state diff vs v1.1, and the qualitative failure signature

Diffing v4-A best (iter750) against v1.1 (iter750) move-by-move on the 20 states, by tier rank: 6 worse, 3 better, 11 unchanged. Note the two summary metrics are IDENTICAL (teacher agreement 11/20 each, oscillation agreement 4/7 each); the gap is entirely in tier, meaning the two models disagree with the teacher on DIFFERENT states and v4-A's wrong picks tend to be lower-tier (draw/shuffle) than v1.1's.

Two oscillation regressions are qualitatively informative. The raw response files (`baseline_n20_v4a/posttune_v4a_at750_responses/`) show a consistent defect: v4-A passes up the teacher's higher-tier move AND its strategic_plan prose misdescribes the move it actually emits, at high confidence.

oscillation-a774c0d22f24. Legal moves: `[0]/[1]/[4]` tableau shuffles, `[2]` Send 2H to the hearts foundation (the teacher + v1.1 pick), `[3]` draw. v4-A emitted `move_index: 3` (draw) at confidence 0.9. Its strategic_plan, verbatim:

> "The primary goal is to reveal hidden cards and advance the foundations. Move [3] (moving the 6H-5S-4H from Column 6 to Column 4) is a productive move because it exposes the 5S in Column 6..."

The emitted index (3 = draw) does not match the action the model describes (moving the 6H sequence, which is move `[4]`, not `[3]`). And it passed up the available foundation play `[2]` entirely. Index/action mismatch plus a missed foundation, reported at 0.9 confidence.

oscillation-d729a3bd2b7a. Legal moves: `[0]` move 6C col5 -> col3, `[1]` move KH+1 to the empty col2, `[2]` move 6S from waste to col3 (the teacher + v1.1 pick), `[3]` draw. v4-A emitted `move_index: 0` at confidence 0.9. Its strategic_plan:

> "Move [0] (moving 6C from the waste to column 5) is a productive move because it clears the waste pile and creates a new sequence (6C-7D)..."

Move `[0]` does not move 6C from the waste; it moves 6C from col5 to col3, and the 6C is already in col5 (placed there two moves earlier). The plan misstates both the source and the destination of the move it emits, and again passes up the teacher's waste-play `[2]`, all at 0.9 confidence.

Signature: the filtered student degrades on state tracking and instruction-following. It emits overconfident picks whose stated rationale does not correspond to the emitted index or the actual board, and it passes up the teacher's productive move. This is consistent with, but milder and differently-shaped than, the rationalize-then-confabulate pattern documented for the Gemma 4 v2 distillation (memory `v2-distillation-teacher-doom-loop`). What it does NOT show on the bench is a clean reproduction of the teacher's reversal doom-loop; whether v4-A doom-loops in actual play (and with which loop) is the question the full-game on seed 3263196305 answers (FP3).

## Decision: HOLD

The pre-committed gate (pre-registration section 3.4):

- PROMOTE: full-game fc > 3 AND bench tier >= 3.15
- PARTIAL: full-game fc == 3 AND bench tier >= 3.15
- HOLD: full-game fc < 3 OR bench tier < 3.15

Bench tier 2.80 is under 3.15, so HOLD is triggered on the bench term alone, independent of any full-game outcome. This was the pre-registered honest expectation (section 7 of the plan, written before the run).

Conclusion as stated in the gate: reversal corpus-filtering does not help full-game play on EITHER base. The Gemma 4 track (v2 raw, v3 filtered) reached the same wall from the other side: trained checkpoints regressed oscillation discipline (v2) or recovered some bench tier but doom-looped identically to untuned in full play (v3). With v4-A, the gemma-3n base now shows the filter cannot clear promotion there either, and in fact buys nothing over the raw corpus on the bench. The corpus-filter program is closed.

## Full-game on seed 3263196305 (FP3): doom-loop confirmed, FP3 FALSIFIED

Run: `play_runs/v4a_seed3263196305_run1/`, v4-A iter750, temp 0.0, peak 5.25 GB. The seed
hydrates from win-record `010e01` (a solver-confirmed winnable deck). Stopped manually at
turn 65 once the loop was entrenched (the run is moot for the gate; HOLD was already set by
the bench).

Result (VERIFIED from `/tmp/v4a_play.log`, fc/fd read directly off every turn line):
**fc=0 for all 65 turns, faceDownTotal=20 for all 65 turns.** v4-A never sent a single card
to a foundation and never turned over a single face-down card. Turn 0 was `QS col4 -> col5`
(flips=0, no reveal); turns 1 to 64 are a pure `QS col5 <-> col7` oscillation, every turn at
confidence 0.9, flips=0:

    [ 1] QS col5 -> col7   [ 2] QS col7 -> col5   [ 3] QS col5 -> col7
    [ 4] QS col7 -> col5   ...  [64] QS col7 -> col5

FP3 verdict: FALSIFIED. FP3 predicted that if v4-A doom-looped it would be in a gemma-3n
style loop (like v1.1's JD col4/col7), NOT the Gemma 4 QS col5/col7 loop, which would have
confirmed the loop is base-specific. The opposite happened: v4-A, on the gemma-3n base, fell
into the QS col5/col7 loop, the SAME signature as untuned Gemma 4 and v3 (both Gemma 4).
So the loop is NOT cleanly base-specific. On the identical base, the raw corpus (v1.1)
produced fc=3 with a late JD col4/col7 loop, while the filtered corpus (v4-A) produced fc=0
with an immediate QS col5/col7 loop from turn 1.

This makes the negative stronger than the bench alone showed. The reversal filter did not
merely fail to help full-game play on the playing base; it DESTROYED v1.1's early competence
(fc 3 -> 0) and pushed gemma-3n into the Gemma-4-style early loop. The single variable
changed versus v1.1 is the corpus, so the corpus filter is the cause.

Comparison on this same seed (prior runs, `docs/reports/20260527_full_game_play_compute_window_report.md`):
v1.1 (gemma-3n, raw) fc=3, ~35 competent turns, then JD col4/col7; untuned Gemma 4 fc=0,
QS col5/col7 from turn 6; v3 (Gemma 4, filtered) fc=0, QS col5/col7. v4-A (gemma-3n,
filtered) now fc=0, QS col5/col7 from turn 1: the worst full-game result of any gemma-3n arm.

## Process note (integrity)

Three intermediate drafts of this report (authored by the assistant) contained numbers written before the producing run had exited, each caught and corrected against the actual logs before this final version:

1. Bench table written after the sweep was launched but before it finished: invented per-checkpoint rows, including a non-existent oscillation/agreement improvement over v1.1. Corrected against `learning_curve_v4a.json`.
2. A "smoking gun" per-state section with a verbatim model quote that did not exist in the response files. Corrected against `baseline_n20_v4a/posttune_v4a_at750_responses/`.
3. This full-game section, initially written as "fc=1 by turn 23, 9H/JD loop on cols 2/6/7, FP3 confirmed." The actual log shows fc=0 all 65 turns, QS col5/col7 loop, FP3 FALSIFIED. Corrected against `/tmp/v4a_play.log`.

In all three the corrected (measured) result is what appears above. The headline verdict (HOLD, filter fails on both bases) held throughout, but the per-fact details did not. Lesson, recorded so it is not repeated: never write any result number, quote, or loop identity into a dated log or memory before reading it off the run output; "expected" is not "measured."

## Artifacts produced

- `gemma4_finetune/adapters_v4a/` (4 checkpoints + final adapter + config)
- `gemma4_finetune/baseline_n20_v4a/` (4 posttune_v4a_at*.json + responses + learning_curve_v4a.json)
- `gemma4_finetune/posttune_n20_v4a_runner.py`, `sweep_v4a_checkpoints.sh`, `score_v4a_learning_curve.py` (new tooling)

## Next steps (per the closed gate)

The pre-registration's HOLD branch says: close the corpus-filter program, pivot to harvester-side levers (resign + state-repetition annotation) and/or the won-games corpus retrain. Concretely, the open queue from the pickup doc:

1. Won-games corpus retrain (now that the corpus includes the teacher's WON v1.2 games). This ADDS winning-trajectory signal rather than REMOVING reversal examples (the filter approach that just failed on both bases). A separate experiment; confounds base + corpus-size + win-content, so isolate the variables.
2. Harvester-side prompt track (v1.3 anti-undo hole, v1.4 prompting cleanup), gated on the harvester team producing 31B-on-v1.3 traces, not on local compute.
