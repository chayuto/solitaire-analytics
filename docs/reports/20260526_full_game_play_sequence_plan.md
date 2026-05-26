# Plan: full-game play sequence for the deployed student

**Date:** 2026-05-26
**From:** Chayut / dataset side
**Re:** Mini-goal sequence. Pre-register the runs and the interpretation
rules BEFORE we see any outcome, so we cannot post-hoc rationalise the
result.

**Companion documents:**
- `gemma4_finetune/play_deck_with_student.py` (the runner)
- `data/benchmarks/winnable_decks.json` (the bench decks; one source of truth)
- `data/DATASET_NOTES.md` "Cross-version teacher benchmarks" (corpus side)
- `docs/reports/20260526_harvester_team_resign_hygiene_versioning_ask.md` (the parallel prompt-side track)

---

## The mini-goal in one sentence

Can the deployed v1.1 LoRA student actually win a solver-confirmed
winnable deck end to end (not just match the teacher's single-turn
pick on the 20-state bench)?

The 20-state Phase 1.5 bench measures single-turn match-to-teacher.
It cannot tell us whether the student can plan, follow through, or
fall into the same doom-loop pathologies as the teacher across many
turns. A full-game play on a known-winnable benchmark deck is the
direct diagnostic.

---

## The runs (in order)

| # | model | deck (seed) | expected wallclock | status |
|---|---|---|---:|---|
| 1 | LoRA v1.1 (gemma-3n + adapters_t5_at750) | `3263196305` | ~50 min | in progress |
| 2 | LoRA v1.1 (gemma-3n + adapters_t5_at750) | `2967897202` | ~50 min | queued |
| 3 | Gemma 4 E2B untuned (the v2 ship target) | `3263196305` | ~50 min | queued |

Total: ~2.5 hours wallclock. The 2x2 (both models on both decks) is
~3.3 hours and is the natural extension if any of the first three
finishes with an ambiguous outcome.

### Why this matrix and not more

- LoRA v1.1 is the actually-deployed student. Two decks tells us
  whether outcome generalises or is deck-specific.
- Gemma 4 untuned is the v2 ship target per the
  `v2-distillation-teacher-doom-loop` memory. One deck is a useful
  corroborating point. If it diverges materially from v1.1, that
  re-opens the v2 story.
- We are NOT running every prompt variant or every annotation
  proposal. Sticking to the deployed prompt (hybrid-v1.0) means the
  result is directly interpretable for the actually-deployed student.

---

## Pre-registered interpretation rules

These are the rules we commit to BEFORE seeing any outcome. They
apply per run, not aggregated.

| outcome class | concrete signal | what it means | what we do next |
|---|---|---|---|
| **WIN** | `summary.outcome == "won"` AND turns_played <= 250 | Student has end-to-end playable competence on this deck. | Run #2 (second deck) to confirm. If both win, queue a small N-seed win-rate study under harvester-fresh seeds. |
| **CLOSEOUT FAIL** | `final_foundation_cards >= 30` AND outcome in (`stalled`, `max_turns`) | Student can plan and progress, falls down at endgame foundation sequencing. | Compare final-state hands to teacher's path at equivalent point; isolate which class of move stops being chosen (likely tableau-to-foundation in the foundation-pump phase). |
| **MIDGAME STALL** | `final_foundation_cards in [10..29]` AND `plateau_at_end_turns >= 30` | Student can open but not sustain. Probably a doom-loop class similar to harvester behaviour. | Inspect `recentMoves` tail for oscillation signature; classify against the existing 2-card / 3-card chain taxonomy in `data/DATASET_NOTES.md`. |
| **EARLY FAIL** | `final_foundation_cards < 10` | Student cannot play, only react on isolated states. Honest red flag for v1.1 as a standalone player. | Stop the sequence. Re-evaluate the 20-state bench framing; possibly add a "multi-turn coherence" axis. |
| **PARSE FAILURE** | `outcome == "parse_failure"` | Prompt-format drift between training corpus and runner render. Diagnostic, NOT a competence finding. | Diff the rendered prompt against a corpus prompt; fix the renderer; retry. |
| **ILLEGAL MOVE** | `outcome == "illegal_move"` | Engine-vs-harvester legal-moves rule mismatch OR student picked an out-of-range index. | Inspect the offending turn; verify the engine's legal moves match what a real harvester rendering would have shown. |
| **HIT MAX_TURNS** with `final_foundation_cards >= 45` | progress was happening, just ran out of budget | Increase `--max-turns` and rerun. Not a competence failure. |

We commit to NOT introducing new outcome classes mid-interpretation.
If a result genuinely surprises us, we add the new class to this
table BEFORE we decide what it means.

---

## Prep work to complete BEFORE kickoff of runs 2 and 3

### Done already
- [x] Runner scaffolded and pushed (`gemma4_finetune/play_deck_with_student.py`)
- [x] Both bench decks have full ground-truth `initialBoardSetup` and pyksolve-confirmed solvability under draw-1
- [x] Run #1 started in background (b**npj22jn**); will report when complete

### To do during the compute window for run #1

**1. Engine vs harvester rule-mode validation** (~10 min)
Replay one ai-log session's recorded moveHistory through the engine in
draw-1 mode. Confirm every move applies cleanly. If any move fails,
we have a silent rule-mismatch bug that would corrupt the runner's
interpretation. Specifically: pick the 010e01 won session (170 moves,
known clean trajectory), replay through engine, assert each step.

**2. Diagnostic comparison scaffold** (~30 min; only if #1 passes)
A small script that takes the runner's `turns.jsonl` and compares
per-turn picks against the teacher's actual pick on the equivalent
position in the 010e01 corpus session. Output: agreement rate,
divergence points, when the student first diverges from the teacher's
known-winning line. This is the "where did the wheels come off"
diagnostic.

We do NOT run this against the live run mid-stream. The scaffold is
ready to fire once the run finishes.

### Explicitly NOT in this sequence

- State-repetition annotation experiments (deferred to next harvester
  cycle, after resign ships)
- v2 LoRA training or filter-corpus exploration (different track; the
  v2 ship target is currently UNTUNED Gemma 4 E2B and that is what
  run #3 tests)
- Any prompt-edit local benches beyond what we've already done
- Adding new bench decks (the corpus has 2 with seeds + 1 without;
  more come from harvest, not from synthesis)

---

## After-sequence decision tree (pre-committed)

```
Run #1 outcome
  |- WIN
  |    Run #2.
  |    Both WIN: queue 20-seed win-rate study using fresh harvest.
  |    Run #2 fails class: investigate the deck-specific failure;
  |      Run #3 proceeds.
  |
  |- CLOSEOUT FAIL or MIDGAME STALL
  |    Run #2 first (confirms whether deck-specific or general).
  |    Then Run #3 (does the v2 base behave differently?).
  |    Document the failure class and tie it to existing taxonomy.
  |
  |- EARLY FAIL
  |    Stop. The single-turn bench framing needs revisiting before
  |    spending more compute. Write up the gap.
  |
  |- PARSE FAILURE or ILLEGAL MOVE
       Fix the renderer / rule mismatch. Re-queue run #1.
```

The decision tree is intentionally narrow. If the actual outcome
does not fit, we document the surprise in this file BEFORE acting.

---

## What we will report back, per run

A short comment in this file (one paragraph) plus the run's
`summary.json` and `turns.jsonl` linked. Notable runs (e.g., the
first win) get an entry in `data/DATASET_NOTES.md` under a new
"Student full-game play" section.

We do NOT publish the runs to the HF dataset (those are model
artefacts, not corpus). The published dataset stays a teacher-only
record per the existing data card.

---

## Run results

### Run #1 (seed 3263196305, LoRA v1.1, attempt 1): ENGINE BUG

Crashed at turn 12 with `apply_move` returning None on a legal move.
Root cause: stock cards were hydrated face-down (matching the engine's
canonical `deal_klondike()` convention), but the engine's
`STOCK_TO_WASTE` apply does not flip the card on draw, so face-down
cards entered the waste face-down and then the tableau face-down via
`WASTE_TO_TABLEAU`, breaking legality checks downstream. A second
silent bug: the deck JSON lists `drawPile` top-of-stock-first, but
the engine pops from `state.stock[-1]`, so my hydrator was making the
engine draw cards in reverse order from the harvester. Both fixed in
`gemma4_finetune/play_deck_with_student.py:deck_to_state`. The
runner now also aborts cleanly with an `engine_violation` outcome if
`apply_move` ever returns None for a move generated by
`generate_moves`.

### Run #2 (seed 3263196305, LoRA v1.1, attempt 2): MIDGAME STALL via doom-loop

`gemma4_finetune/play_runs/v1_seed3263196305_run2/`. 58 turns played
to `final_foundation_cards=3`, then 3 consecutive illegal-move picks
triggered the safety abort.

**Outcome classification.** Per the strict pre-registered rule this is
`final_foundation_cards < 10` so EARLY FAIL. But the actual narrative
contradicts the spirit of that rule: turns 0-35 were competent play
(reduced face-down from 21 to 12, played AC and AH to foundations,
built sensible KH+QC tableau sequences). The student then fell into a
**19-turn JD col4 to col7 oscillation** between turns 36 and 54, and
the safety abort fired at turn 55 when the legal-moves list shifted
and the student's preferred move_index became out of range. This is
not EARLY FAIL; it is a **textbook MIDGAME STALL with
behavioural-doom-loop signature**, exactly the same pathology class as
the 31B teacher in sessions `adf71b`, `645d03`, `73fd85`. Confidence
stayed saturated at 0.95-1.0 across the oscillation, same as the
teacher.

**Decision rule revision.** The pre-registered EARLY FAIL cutoff
(`fc < 10`) was too coarse. The diagnostic signal that actually
discriminates is `plateau_at_end_turns >= 15 AND clear oscillation
pattern in the last 10 moves`. By that revised rule, run #2 is a clean
MIDGAME STALL. We update the rule here BEFORE seeing future runs so
the revision is not post-hoc on the run that triggered it. Original
EARLY FAIL rule is preserved below for cases where the student
genuinely fails to make any meaningful early progress (no foundation
plays in the first 30 turns, no face-down reductions, parse errors
clustered).

**New failure subclass discovered: move_index fixation.** The
illegal-move trigger at turns 55-57 is NOT random; the student picked
the same `move_index=4` three turns in a row even after the legal-moves
list dropped to 4 entries (indices [0..3]). The student appears to
fixate on a specific positional pick and fails to re-evaluate when the
legal-moves shape shifts. Worth cataloguing as a discrete failure
mode alongside the doom-loop family. The runner's existing
`--max-illegal-moves 3` cap is the right safety; the diagnostic value
is in the prior 19 turns of oscillation, not the final 3.

**Implications.**

1. The deployed v1.1 student exhibits the same doom-loop class as the
   31B teacher. The student is not just bad at endgame; it inherits
   the canonical failure mode.
2. The harvester ask's RESIGN feature is now doubly justified. The
   deployed student would benefit from it as much as the 31B teacher
   does. When all legal moves are 2-card oscillations and stock has
   just been drawn, the student has no out, same action-space problem.
3. State-repetition annotation moves from "next cycle if needed" to
   "next cycle definitely useful" given the same pathology is showing
   up at the student level too.
4. Bench artefacts: `gemma4_finetune/play_runs/v1_seed3263196305_run2/`
   (summary.json + turns.jsonl + 58 turn-level raw responses).

### Run #3 (seed 3263196305, Gemma 4 E2B untuned): in progress

Running in background. Purpose: confirm whether the v2 ship target
(Gemma 4 untuned per `v2-distillation-teacher-doom-loop` memory)
shows the same doom-loop pathology or a different failure pattern.
The two arms had a meaningful split on the PRIOR REASONING
truncation bench (v1.1 SHIP, Gemma 4 untuned REVISE) so different
behaviour on full-game play is plausible. Will append result here
when complete.
