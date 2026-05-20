# Prompt Engineering Proposal — Doom-Loop Mitigation

**Date:** 2026-05-20
**Author:** analytics-side audit, for the harvest team
**Status:** proposal, awaiting harvest-team trial
**Scope:** teacher prompt for the `gemma-4-31b-it` advisor (and any future teacher). Server, schema, and pipeline unchanged.

## TL;DR

The teacher already loses ~50% of its successful calls to doom-loops on harvestable games (current local set: **182 eligible / 757 teacher-model successes** = 24% yield; the gap is mostly stalled decisions filtered out). The dominant failure mode is a **two-card oscillation** the model itself wrote `recentMoves` evidence for, then ignored. The current prompt has one line of anti-loop guidance buried in heuristic #7 and no structural check anywhere in the response format. This doc proposes three additive prompt changes, ordered by expected ROI, that should be testable independently in a small A/B harvest.

## Evidence

Source: `data/raw/solitaire-ai-log-645d03-1779270371464.json` (session `…d46eb2645d03`, seed `3689552861`, app build `ce6afe1`). Foundations stuck at 6 cards, face-down stuck at 17 from turn 60 through turn 135 (75-turn plateau). Full session description in `data/DATASET_NOTES.md` under "Known doom-loop sessions".

### Turn 133 — the smoking gun

`recentMoves` at that turn (visible to the model in the prompt):

```json
"recentMoves":[
  "draw JS","draw 8D",
  "move 5C col 4 -> col 6","move 4D col 4 -> col 6",
  "move 5C col 6 -> col 4","move 4D col 6 -> col 4",
  "move 5C col 4 -> col 6","move 4D col 4 -> col 6",
  "move 5C col 6 -> col 4","move 4D col 6 -> col 4"
]
```

Four full cycles of the same 5C/4D pair. The model's own response that turn:

> "Move 0 (moving 5C and 4D from Column 4 to Column 6) is a strong setup move. It exposes the 6D in Column 4… by moving 4D to Column 6, it creates a destination for the 3S in Column 2. Moving 3S to Column 6 would then expose the 4H in Column 2."

It picks the same move for the fifth time, calls it a "strong setup move", reports `confidence: 0.9`, and proposes a multi-step plan that it never executes — because next turn it reverses 5C/4D again and re-proposes the same plan.

### Turn 104 — the model knows

> "The board is currently stalled."

The model explicitly recognises the stall in `board_analysis`, then continues to recommend the same shuffles a few turns later.

### Where the cards actually are

All 52 cards accounted for from the final-turn observation:

- Unblock cards the model itself named (black 7s, red 7s, red Kings): **`7C`, `7D`, `7S`, `KD` all in the seen-draw pile**, `7H` face-up on column 5. Only `KH` is in the face-down stack — and it becomes reachable once any of the seen black 7s is played onto 6D or 6H.
- This is a **bad-AI verdict, not a bad-deck verdict.** The deck is winnable in principle; the teacher failed to play the unblock cards when they reached the waste top.

## What the current prompt does well

- Rule recitation is clean and complete (~1100 chars of rules, well-structured).
- Strategy heuristics cover the right ground: prioritise revealing face-down, play Aces/2s promptly, don't empty without a King, etc.
- Confidence rubric is explicit and calibrated by intent (1.0–0.9 forced, 0.5–0.3 guess, etc.).
- Response format is strict JSON with a clear `board_analysis` → `strategic_plan` → `final_decision` order.

## Where the prompt fails

Four concrete issues found by reading the model's own responses across the 75-turn plateau:

1. **Anti-loop guidance is buried.** The only line addressing the failure mode is *"Avoid moves that simply undo a recent move or lead nowhere"* — heuristic #7 of 7 in the STRATEGY GUIDANCE block. It is stated as a heuristic, not a rule, and the model demonstrably treats it as low-priority advice it can rationalise around ("this is a setup move, not an undo").
2. **No structural check in the response format.** The required schema has `board_analysis`, `strategic_plan`, `final_decision`. Nothing requires the model to *compare its candidate move against `recentMoves`* in writing. The cycle-check is left to vibes.
3. **Per-turn statelessness.** The `strategic_plan` field is generated fresh each turn. The model proposes "move 5C/4D, then 3S to col 6, then expose 4H" — and next turn proposes the same plan again because nothing forces continuity. Turn N's plan never becomes a constraint on turn N+1's decision.
4. **Confidence rationalisation.** The model reports 0.9 confidence on the 5th repetition of a move that has empirically failed four times. The rubric does not account for "I have already tried this and the board did not change." Confidence stays saturated even mid-stall — see also the existing finding that `confidence` is miscalibrated across the whole harvest (`GEMMA4_E2B_DATA_EVALUATION.md` §calibration).

## Proposed changes (additive; cheapest first)

### Change #1 — Add a required `loop_check` field

Promote loop detection from "heuristic the model can ignore" to "structured field the model must populate, in writing, before choosing." Concrete schema addition:

```json
"loop_check": {
  "would_reverse_recent_move": <bool>,
  "matching_recent_move": <string or null>,
  "justification_if_yes": <string or null>
}
```

Insert it **between** `strategic_plan` and `final_decision` so the model commits to the check before locking the move index. Then add one line to the strategy block:

> *If `loop_check.would_reverse_recent_move` is true, you must either justify with a concrete next-step gain on the very next turn, or pick a draw/recycle action instead.*

**Why this should work.** The model already produces text describing what its candidate move does. Forcing it to also produce a structured boolean comparing the candidate against the `recentMoves` array makes the rationalisation cost go up: it now has to *write down* that this is an inverse of a recent move, and either own that or change its pick. The 645d03 transcripts show the model can identify the pattern when asked ("the board is currently stalled") — it just isn't asked.

**Cost.** Single prompt edit. No infra change. Response schema is backward-compatible if the parser tolerates the new key (or extracts only the keys it cares about).

**Risk.** Negligible — worst case the model fills the field with confabulation, in which case we still have a new training signal ("did the model correctly self-detect a loop?") that is useful downstream.

### Change #2 — Promote the anti-loop rule and ground it in an example

Move *"Avoid moves that simply undo a recent move"* out of STRATEGY GUIDANCE and into a new HARD CONSTRAINTS block placed **immediately after** the rules and **before** the strategy heuristics. Two changes:

- Promote heuristic → rule. Replace "Avoid moves that…" with "**Do not** play a move whose inverse appears in the last 8 `recentMoves` entries unless you can name the immediate next move that would not have been possible without this move."
- Add a worked counter-example, paraphrased from the 645d03 transcript, so the model has a concrete template of the failure mode:

> *Counter-example: if `recentMoves` shows ["move 5C col 4 -> col 6", "move 4D col 4 -> col 6", "move 5C col 6 -> col 4", "move 4D col 6 -> col 4"], do not propose moving 5C col 4 -> col 6 again. This pattern is a doom-loop. Recycle or draw instead.*

**Cost.** Single prompt edit, ~5 added lines. No schema change.

**Risk.** The rule cannot be applied verbatim to every situation (rare legitimate "rewind to retry a different branch" plays exist), but the worked example narrows the scope: only forbid the *cyclic* repetition, not all reversals.

### Change #3 — Plan-continuity field

Force the model to reconcile this turn's plan against last turn's plan. Add a new field at the top of the response, **before** `board_analysis`:

```json
"plan_continuity": {
  "previous_plan_summary": <string or null>,
  "executed_so_far": <string>,
  "continuing": <bool>,
  "reason_for_change": <string or null>
}
```

The harvest already has the previous turn's `strategic_plan` text (it's in the `decision` field of the prior interaction) — feed the last 2 turns' `strategic_plan` text back into this prompt as `PREVIOUS PLANS` context, and require the model to acknowledge them.

**Why this should work.** It makes plan abandonment a written decision, not a silent default. If the model wrote "I will move 5C/4D then 3S then expose 4H" on turn N and on turn N+1 it never moved 3S, it must now either *continue with 3S* or *write down* that it is abandoning the plan and why. Either result is useful: the former breaks the loop; the latter produces a clean stall signal we can train against.

**Cost.** Prompt edit + small harness change to inject `PREVIOUS PLANS` from the prior turn's response. Schema gains one field.

**Risk.** Adds prompt tokens (~200–400 chars per turn). May hurt latency on the timeout-prone 31B model.

## Recommended experiment

**Phase 1 — Change #1 only.** Run on 5 fresh games at the same difficulty and information level as the existing harvest. Compare against the existing 12 sessions (per `data/SUMMARY.md`) on:

- Stall ratio: (stalled-game decisions) / (success decisions).
- Mean longest plateau length (consecutive turns with no foundation/faceDown change).
- Loop-detect rate inside the harvest: % of `play_move` calls where `loop_check.would_reverse_recent_move` was `true` AND the chosen move was the inverse. (This is a self-consistency metric — if it's > 0, the model is admitting cycles and still committing to them; useful failure signal.)
- Win rate, foundation-progress curve. These are secondary; the primary target is "did the doom-loop pathology shrink?"

Acceptance gate to proceed to Change #2: **stall ratio drops by ≥ 30%** vs the current 12-session baseline. If it doesn't, Change #1 is wishful thinking and we layer #2 + #3.

**Phase 2 — Layer #2 if Phase 1 falls short.** Same metrics, same baseline.

**Phase 3 — Layer #3 if Phase 2 falls short.** Same metrics. After this, if the stall ratio has not dropped, the conclusion is "prompt engineering is not the right lever for this teacher" and we escalate to either a stronger teacher or a solver-augmented approach (see the open `terminate vs continue` thread in chat for solver-augmented options).

## Open questions for the harvest team

1. Is `recentMoves` actually 10 entries, or does it window? Confirm the buffer length and whether `move_count` is the underlying counter. (Change #1's `last 8 recentMoves` rule depends on this.)
2. Is `reasoningTrail` reliably populated? Some turns in 645d03 had it and some didn't. Change #3 needs the prior `strategic_plan` text to be retrievable.
3. Does the export schema have a place to put `loop_check` and `plan_continuity` without breaking the existing ingest? Suggest extending `decision` with a `meta` dict for future structured fields rather than adding top-level keys each time.

## Out of scope (deliberately)

- **Solver-based veto / move masking.** Those are server-side mechanical fixes discussed in chat; they fight a different fight (preventing the bad move from being applied) rather than improving teacher decisions. They contaminate the harvest signal — if the server overrides the teacher, the row no longer represents what the teacher would have done, and the distillation target is corrupted.
- **Replacing the teacher entirely.** Bigger lever, separate decision; this doc is the cheap-experiment path.
- **Confidence recalibration.** Real issue (see `GEMMA4_E2B_DATA_EVALUATION.md`) but orthogonal — confidence drops out of training if treated as suspect, independent of loop fixes.
