# Late-Game Prompt Audit: What Bleeds In and Why It Hurts

**Date**: 2026-05-27
**Status**: Findings + hypotheses, NOT YET ROLLED OUT
**Author**: Audit performed during the post-compute-window analysis session
**Audit tool**: `/tmp/audit_prompt.py` (one-off replayer, copy into repo if we keep iterating)
**Runs audited**: all four under `gemma4_finetune/play_runs/`
**Triggering question**: "why is the model so slow and stuck in loops at high turn numbers?"

## Executive summary

At steady-state (turn 25+) the hybrid-v1 runner prompt is ~7700 chars. Component-level decomposition shows ~40 percent of that prompt is dead weight or actively harmful: 5 copies of the same `strategic_plan` text, redundant move-type labels, and ~720 chars of header spec for fields scheduled for removal. The single largest non-static block is `PRIOR REASONING` at 37.3 percent of the prompt, and during a doom-loop it is byte-identical noise that positively reinforces the wrong move.

Six concrete failure modes are documented below, each paired with a falsifiable hypothesis and a pre-registered test. None of the proposed runner changes have been rolled out. They will be tested in a future compute window before any change ships.

## 1. Background

The 2026-05-27 full-game compute window (see `20260527_full_game_play_compute_window_report.md`) found that all three LoRA arms doom-loop in late game on the winnable benchmark deck `seed=3263196305`. The user asked: "deep audit actual prompt in the recent late game. where turn numbers are high. often very slow and stuck in loop. anything bleed in? too much history? records? useless info?"

This document is the answer to that audit ask, framed as hypotheses to test rather than fixes to ship.

## 2. Method

A one-off replayer at `/tmp/audit_prompt.py` was written to:

1. Load `data/benchmarks/winnable_decks.json` for the seed.
2. Hydrate the engine state via `play_deck_with_student.deck_to_state`.
3. Walk `turns.jsonl` for the run, replaying each successful turn's chosen move through `engine.apply_move`.
4. Re-render the prompt at any specified turn using the live `render_prompt` from `play_deck_with_student.py`.
5. Decompose the rendered prompt at known section markers and report byte counts per section.

Sampled turns: turn 0 (cold start), turn 30 (still productive in v1.1 run2), turn 50 (entering doom-loop in v1.1 run2), turn 250 (deep in steady-state doom-loop on both v3 iter750 and Gemma 4 untuned runs).

The replayer's accuracy was sanity-checked against the runner's logged `prompt_chars`: matches exactly for v3 iter750 turn 250 (7702 chars) and Gemma 4 untuned turn 250 (7552 chars).

## 3. Component-level character lengths

Steady-state late game (turn 250, v3 iter750 doom-loop):

| Component                          | Chars | Percent |
|------------------------------------|------:|--------:|
| STATIC HEADER (rules + JSON spec)  | 3740  | 48.6%   |
| PRIOR REASONING (last 5 entries)   | 2871  | 37.3%   |
| RECENT MOVES (last 10)             |  374  |  4.9%   |
| LEGAL MOVES                        |  206  |  2.7%   |
| TABLEAU                            |  150  |  1.9%   |
| NOTATION line                      |  122  |  1.6%   |
| PROGRESS                           |   66  |  0.9%   |
| FINAL INSTRUCTION                  |   61  |  0.8%   |
| STOCK + WASTE                      |   53  |  0.7%   |
| FOUNDATIONS                        |   45  |  0.6%   |
| CURRENT GAME header                |   14  |  0.2%   |
| SEEN IN WASTE                      |    0  |  0.0%   |
| **TOTAL**                          | **7702** | **100%** |

Growth trajectory:

| Turn | Prompt chars | Notes                                 |
|-----:|-------------:|---------------------------------------|
|    0 |         4571 | No RECENT MOVES, no PRIOR REASONING   |
|   25 |         ~7800 | All history caps full                |
|   25+ steady | ~7700 (v3 locks to exactly 7702) | History saturation |

The 3131-char growth from turn 0 to steady-state is entirely RECENT MOVES + PRIOR REASONING + SEEN IN WASTE filling to their renderer caps.

## 4. Findings: what bleeds in

### 4.1 PRIOR REASONING is a doom-loop amplifier, not a memory aid

**Observation**: at turn 250 of `v3_iter750_seed3263196305_run1` the 5 PRIOR REASONING entries are byte-identical 504-char strings. Each `why` is "The primary goal is to reveal the hidden cards in the waste pile. To do this, we must create a landing spot for the King of Diamonds (KD). The most efficient way to do this is to move the Queen of Spades (QS) from Column 5 to the King of Diamonds (KD) in Column 7..."

**Verified at the source**: turns 17, 18, 19, 20, 21 of `v1_seed3263196305_run2` all emit byte-identical 524-char `strategic_plan` strings in their JSON response. The renderer is not buggy. The MODEL emits the same plan turn after turn because it is stuck.

**Why this hurts**: the rendered prompt now contains "this is the right move" five times. That is positive feedback on the wrong action. The next turn, the model reads its own prior conviction five times and is more confident in the wrong move. This is the largest non-static block in the prompt (37.3 percent).

### 4.2 PRIOR REASONING entries are internally inconsistent

**Observation**: at turn 250, entry 2 of PRIOR REASONING reads:

```
move: Move QS from column 7 to column 5
why: ...move the Queen of Spades (QS) from Column 5 to the King of Diamonds (KD) in Column 7
```

The recorded action and the recorded justification are LITERALLY OPPOSITE directions. This happens because the runner pairs each turn's actual `move_text` with that turn's emitted `strategic_plan`. When the model loops and emits the same plan while flip-flopping its chosen move, half the entries pair an action with a justification for the opposite action.

**Why this hurts**: the model sees self-contradictory training-distribution-shaped data. We have no evidence this helps. We have a strong prior it hurts.

### 4.3 STATIC HEADER carries ~720 chars of spec for fields being dropped

**Observation**: the STATIC HEADER includes a ~600-char "confidence calibration" paragraph (5-tier probability spec) and a ~120-char `alternative_move_index` field spec. Per `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/upcoming-export-schema-drop-confidence-altmove.md` both fields are scheduled for removal from the next harvester export schema.

**Why this hurts**: ~9 percent of every prompt is teaching the model to fill fields the downstream consumer no longer reads. The model still spends tokens computing and emitting them, slowing inference.

### 4.4 RECENT MOVES at 10 entries is too long when the loop is short

**Observation**: at turn 250, the last 10 RECENT MOVES are:
```
move QS col 7 -> col 5
move QS col 5 -> col 7
[repeated 4 more times]
```

10 lines, 2 distinct signatures, 8 of them dupes. ~300 chars of redundancy.

**Why this hurts**: the model now has to scan 10 lines to extract the same information a single "you have repeated this pair 5x" annotation would carry. Worse, the repetition reinforces the same "this is what playing looks like" signal that PRIOR REASONING is also feeding.

### 4.5 LEGAL MOVES type labels are noise

**Observation**: each LEGAL MOVES line currently has shape:

```
[0] tableau_to_tableau        Move QS from column 7 to column 5
```

The middle column (move-type name, padded to width 24) duplicates information the right column already states in prose. Mapping is 1-to-1 and the prose verb identifies the type ("Move", "Draw", "Send ... foundation").

**Why this hurts**:
- ~26 chars per line of pure redundancy. With 6 to 12 legal moves per turn, 156 to 312 chars of waste per prompt.
- Two parallel labels for the same move makes `MOVE_INDEX_FIXATION` (the failure class already classified in `gemma4_finetune/analyze_play_run.py:63`) easier, not harder. The model may latch on to the category instead of the action.
- snake_case identifiers are out-of-distribution for a reasoning task. The model has to parse, ignore, and not echo them.
- The model may infer the label is something it should output. We mitigate this in the header by saying "return the [index]" but the presence of the label keeps the ambiguity alive.

### 4.6 Dead-weight tally

| Source                                    | Bytes | Notes                          |
|-------------------------------------------|------:|--------------------------------|
| 4 redundant copies of identical `why`     | ~2016 | 5 entries minus 1 unique       |
| 8 redundant RECENT MOVES lines            |  ~300 | 10 entries minus 2 distinct    |
| Confidence calibration paragraph          |  ~600 | Field being dropped            |
| `alternative_move_index` spec             |  ~120 | Field being dropped            |
| Move-type labels in LEGAL MOVES           |   ~50 | Per typical turn               |
| **Total**                                 | **~3086** | **40 percent of prompt**   |

## 5. Hypotheses, each falsifiable

Each hypothesis is a prediction. Each is paired with a metric and a pre-registered pass/fail rule. We test them BEFORE rollout, not after.

### H1: dedupe PRIOR REASONING

**Change**: in `render_prompt`, dedupe `prior_decisions` by `why` text before showing the last N distinct entries (likely N=3).

**Prediction**: full-game runs on `seed=3263196305` show a measurable reduction in plateau length and/or first-doom-loop turn.

**Pass criterion**: on v1.1 LoRA, plateau-at-end shrinks by 20 percent or more vs the current baseline (35 turns of competent play before doom-loop in run #2). On v3 iter750 or Gemma 4 untuned, the model reaches fc>=1 at least once (currently they never do).

**Fail criterion**: no change in plateau length, no fc gain, OR new doom-loop signatures appear.

**Confound to watch**: removing the rationalization echo might make the model LESS coherent across turns. We need to check that the freed budget is being used for fresh reasoning, not noise.

### H2: drop confidence + alternative_move_index from STATIC HEADER

**Change**: strip the calibration paragraph and the `alternative_move_index` spec from `STATIC_PROMPT_HEADER` (`play_deck_with_student.py:71-126`). Update the JSON schema example to a 2-key `final_decision`.

**Prediction**: ~720 chars saved per prompt; ~5-10 percent inference speedup; no regression in move quality.

**Pass criterion**: per-turn wallclock drops by at least 5 percent on the v3 iter750 arm (which is currently 4.9s/turn mean); move quality (legal-move pick rate, foundation progress) does not regress.

**Fail criterion**: move quality regresses (lower fc trajectory or higher illegal-pick rate), even if speed wins.

**Coupling to harvester**: this change should be SYNCHRONIZED with the harvester's next prompt-template version bump (already requested in `20260526_harvester_team_resign_hygiene_versioning_ask.md`). Rolling out the runner change alone creates a small train/play mismatch on the response schema.

### H3: collapse RECENT MOVES when the same pair repeats

**Change**: in `render_prompt`, detect AB / BA oscillation in `recent_moves[-10:]` and render as `<sig_A> <-> <sig_B> (repeated 5x in last 10)` instead of 10 individual lines. Same logic for AB / BC / CA 3-cycles.

**Prediction**: ~250 chars saved per doom-loop prompt; model may break the loop sooner if the explicit "repeated" annotation triggers different attention.

**Pass criterion**: doom-loop runs end in fewer total turns (model resigns or breaks out), OR plateau-at-end shrinks on at least one arm.

**Fail criterion**: model does not break the loop AND prompt savings are too small to matter (<200 chars).

**Risk**: the explicit "repeated 5x" annotation could be ignored. We need to watch whether the model's RESPONSE references the annotation. If it does not, H3 fails.

### H4: drop move-type column from LEGAL MOVES

**Change**: in `render_prompt:321`, remove `{mt_h:<24}  ` from the f-string. Line becomes `[0] Move QS from column 7 to column 5`.

**Prediction**: ~26 chars per legal move saved; lower rate of `MOVE_INDEX_FIXATION` classification on full-game runs; no other regressions.

**Pass criterion**: `MOVE_INDEX_FIXATION` rate drops on a multi-deck full-game sweep, OR move quality holds and prompt shrinks as expected.

**Fail criterion**: `MOVE_INDEX_FIXATION` rate unchanged AND model starts emitting snake_case move-type strings in its `final_decision.move_index` field (parse failures rise).

**Coupling to harvester**: same as H2. If harvester still ships the labels in its prompt, we accept a small training-distribution mismatch. Acceptable on the inference side because the model still sees an ordered indexed list; only the side annotation changes.

### H5: cumulative effect

**Prediction**: applying H1+H2+H3+H4 together drops late-game prompt from ~7700 to ~4600 chars (~40 percent reduction) and at least one arm now reaches fc>=10 on the benchmark deck (currently no arm reaches fc>=4).

**Pass criterion**: combined change beats the best single-change variant on at least one of {plateau length, peak fc reached, doom-loop turn-of-onset}.

**Fail criterion**: combined change is no better than the best single change, OR introduces a new failure mode (e.g. parse failures rise above the 3-in-a-row abort threshold).

### H6: the bench-vs-play gap is partly a prompt-shape problem

**Background**: the 2026-05-27 compute-window report's headline finding was "single-turn bench does not predict full-game competence". The bench evaluates the model on isolated states with NO PRIOR REASONING and minimal RECENT MOVES. Full-game play carries 5 copies of stale reasoning into every prompt.

**Prediction**: if H1 (dedupe PRIOR REASONING) lands the predicted effect, the bench-vs-play gap should narrow. The bench rank order of arms (Gemma 4 untuned > v1.1 LoRA on tier; v1.1 LoRA >> Gemma 4 untuned on play) should converge.

**Pass criterion**: post-fix full-game ranking of arms moves closer to the post-fix bench ranking of arms. Specifically, Gemma 4 untuned should reach fc>=1 at least once.

**Fail criterion**: the gap is unchanged. This would mean the gap is purely a model-base property, not a prompt-shape artifact.

## 6. Test plan (run before any rollout)

### 6.1 Suggested order

1. **Day A**: implement H1 + H2 + H4 (lowest risk, biggest expected win, smallest code surface).
2. **Day A run**: re-run all three arms (v1.1 LoRA, Gemma 4 untuned, v3 iter750) on `seed=3263196305`. ~3 hours wallclock.
3. **Day A grade**: re-run `analyze_play_run.py` on all three; compare per-arm vs the corresponding 2026-05-26/27 baseline.
4. **Day A go/no-go**: if H1 / H2 / H4 pass criteria are met, proceed. If any fails, root-cause before adding H3.
5. **Day B**: add H3 (RECENT MOVES collapse) and re-run only the arms that benefited in Day A.
6. **Day B**: re-run the 20-state single-turn bench (`bench_prior_reasoning_truncation.py` plus the original 20-state state set) against all three arms to validate H6.

### 6.2 What we measure

Primary metrics (full-game play, all on `seed=3263196305` benchmark deck):
- `plateau_at_end_turns` from `summary.json`
- `final_foundation_cards` (peak fc reached during the run)
- Doom-loop turn-of-onset (first plateau >= 15 turns)
- `MOVE_INDEX_FIXATION` rate from `analyze_play_run.py`
- Mean `call_seconds` per turn

Secondary metrics:
- Mean `prompt_chars` per turn (should drop by ~3000 if all hypotheses land)
- Parse-failure rate (should not rise)
- Bench tier on the 20-state set (should not drop)

### 6.3 Multi-deck expansion (if Day A wins)

If Day A shows wins, expand to a 5-deck sweep. We currently have 3 winnable decks in `data/benchmarks/winnable_decks.json`. We need 2 more solver-confirmed winnable decks before this. Easy ask: the next batch of WON exports will likely surface them.

## 7. Pre-registered scope guards

What we will NOT change in this pass:
- The STATIC HEADER rules section (~1500 chars). Not in scope. Has not been audited for content quality.
- The NOTATION line. Established convention; changing it risks corpus drift.
- The TABLEAU rendering format (`??` for face-down). Compact and on-distribution.
- The auto-flip and seen-in-waste tracking. Both are working as designed.

What we explicitly defer:
- Re-training (any LoRA) on a prompt-format that differs from this audit's prompt. Re-training amplifies the cost of getting any of H1 to H4 wrong. Validate inference-time first.
- Harvester-side prompt changes beyond what is already in `20260526_harvester_team_resign_hygiene_versioning_ask.md`.

## 8. Open questions for the test compute window

- **PRIOR REASONING N tuning**: is N=3 distinct entries the right cap, or is N=2 enough? Run a tiny sweep (N in {2, 3, 5}) on one arm to find the knee.
- **Oscillation annotation wording**: does the model respond better to "repeated 5x" or "you have done this loop 5 times, try something different"? The second is more interventionist; risk is making the model think the loop is illegal.
- **Stale rationale invalidation**: should we drop PRIOR REASONING entries whose `move_text` does not match the current move-text pattern? Cheap heuristic: if the recorded `move_text` involves a card not present in any LEGAL MOVE, drop the entry as stale.
- **Header length sensitivity**: H2 saves 720 chars on the STATIC HEADER. How much more can we trim from the rules section without losing move quality? Out of scope this pass but a candidate for the follow-up.

## 9. Reproducibility

- Audit script: `/tmp/audit_prompt.py` (move to `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/audit_prompt.py` if we iterate). Replays any turn of any run under `gemma4_finetune/play_runs/`.
- Runs audited:
  - `gemma4_finetune/play_runs/v1_seed3263196305_run2/` (v1.1 LoRA, 55 turns, doom-loop)
  - `gemma4_finetune/play_runs/v3_iter750_seed3263196305_run1/` (v3 iter750, 300 turns, locked at 7702-char prompt from turn ~25)
  - `gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/` (300 turns, oscillating between 7500 and 8500 chars)
  - `gemma4_finetune/play_runs/v1_seed3263196305_run1/` (sparse, only 13 turns)
- Renderer code under audit: `play_deck_with_student.render_prompt` at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py:256-340`.
- Related memory: `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/upcoming-export-schema-drop-confidence-altmove.md` (H2 coupling), `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/v2-distillation-teacher-doom-loop.md` (doom-loop background).

## 10. Honest assessment

The audit is precise on what is in the prompt. It is a HYPOTHESIS about why the model struggles. The strongest evidence is H1 (PRIOR REASONING dedupe): we can see, byte for byte, that 37 percent of every late-game prompt is the same string repeated 5 times. That is hard to defend as load-bearing.

What we DO NOT know:
- Whether the model is actually attending to PRIOR REASONING in late game, or if it has learned to ignore it. If the latter, H1 helps speed but not quality.
- Whether the doom-loop is base-model-fundamental (as the compute-window report concluded) and the prompt is innocent. If the prompt fixes give zero quality lift, we have evidence the pathology is base-model-internal and prompt engineering will not save us.
- Whether the harvester team's next prompt-template version (currently in flight) renders some of these changes moot.

The test plan above is designed to give a clear answer on each of these in one compute window.

## 11. Decision rule for go / no-go

After Day A + Day B runs:

- **All wins, no regressions**: roll out the runner changes. Coordinate the harvester-side H2 piece with the team's next template bump.
- **Mixed (some hypotheses pass, some fail)**: roll out only the passing hypotheses. Document failures for follow-up.
- **All fail or regressions**: keep the current prompt. Update the compute-window report's "doom-loop is base-model-specific" finding with new evidence that the prompt is not at fault.

In all cases: update `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_full_game_play_compute_window_report.md` section 9 (Way Forward) with the experimental result and the next test.
