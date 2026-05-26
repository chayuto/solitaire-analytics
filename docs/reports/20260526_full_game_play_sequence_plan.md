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
