# Klondike Per-Turn Strategy + Prompt Engineering — Deep Dive

**Date:** 2026-05-20
**Author:** analytics-side, for the harvest team
**Status:** companion to `PROMPT_ENGINEERING_DOOM_LOOP_2026-05-20.md` — that doc proposes three minimal patches; this doc is the full rebuild rationale and recipe.
**Scope:** how a single Klondike turn should actually be decided, what prompt-engineering practices apply to that decision, and how to encode both in the teacher prompt.

## Part 0 — Why this doc exists

The current teacher prompt is a *role + rules + heuristics + format* sandwich, ~7500 chars, with strategy guidance written as a flat list of soft heuristics. That structure has two problems:

1. **It's not an algorithm.** Klondike per-turn decision-making is a small but strict *priority cascade* — there is a near-canonical "what to check, in what order" sequence that human experts and Klondike solvers both converge on. The prompt presents the priorities as parallel suggestions, so the model treats them as parallel too.
2. **It hasn't absorbed prompt-engineering practice for sequential decisions.** Modern best practice for "pick the best move from a list" tasks combines structured CoT, per-priority traversal, anti-rationalisation fields, and stateful continuity. The current prompt uses one of these (structured CoT via `board_analysis` → `strategic_plan` → `final_decision`); the others are absent.

This doc fills in both gaps. Part 1 lays out the Klondike per-turn decision tree as a strict algorithm. Part 2 walks through the prompt-engineering practices that map onto that algorithm. Part 3 stitches them together into a concrete proposed prompt structure with example text. Part 4 is the experiment.

## Part 1 — The Klondike per-turn decision algorithm

This is the priority cascade. Apply in order; the first priority that has a candidate move wins, ties broken by sub-rules within the priority.

### Priority 0 — Forced / always-correct moves

These are moves where the optimal-play literature (Bjarnason et al. 2009; Yan & Sturtevant solvers) agrees there is no scenario in which not playing them is better. Always check first.

- **Ace → foundation.** Always.
- **2 → foundation** when the corresponding Ace is already on the foundation. Always — 2s have essentially no tableau utility once the Ace is placed.
- **A move that flips the LAST face-down card of a column** if that move does not also bury a card needed by foundations within the next few plies. (Edge case: "last face-down" almost always passes the "doesn't bury" check, since you're shortening, not lengthening, a column.)

### Priority 1 — Reveal a face-down card

The single most important goal in Klondike. 21 face-down cards in the starting deal; winning requires flipping all of them. Every face-down still hidden is a chance for the game to be unwinnable.

- Move the top face-up card (or runnable sequence) off a column with face-down cards onto a legal destination.
- **Sub-rule 1a — Prefer flipping the column with the MOST face-down cards** when multiple priority-1 moves exist (deeper columns are bigger blockers).
- **Sub-rule 1b — Reject if the move buries a card needed soon.** "Needed soon" = a card the next foundation rank will require within the next 1–2 ply (e.g. don't move 5C onto 6H burying the 4D if the 3D just went up). Use the `metrics.completionProgress` and the foundations array.
- **Sub-rule 1c — Reject if the move leaves the column empty without a King ready.** Empty columns are valuable real estate; wasting one on no King is a major loss of optionality.

### Priority 2 — Build foundation without trapping

Foundation builds advance progress directly. But they trap if the card was needed in tableau to receive its opposite-color one-lower neighbour.

- Send the card up if and only if **the same-rank opposite-color card of the suits below is no longer needed**, i.e. there is no chance of needing this card to receive a one-lower card from waste/tableau.
- Concrete rule used by strong human players: **once both 2s of the same colour are on foundations, the 3s of the opposite colour become safe to send up; and so on by induction.** Encode as: a card `r` of colour `c` is *safe to send up* if both `r-1` cards of colour ¬c are already on foundations.
- **Sub-rule 2a — Aces and 2s bypass this rule** (Priority 0; always safe).

### Priority 3 — Setup move toward Priority 1 or 2

A move that does not itself reveal a face-down or build a foundation, but creates a destination or unlocks a card that will, in 1–2 plies, do one of those.

- **Sub-rule 3a — Must name the immediate next move.** A setup move without a named follow-up is a doom-loop in disguise (this is precisely the failure pattern in `645d03`).
- **Sub-rule 3b — Reject if the same setup was tried within the last N moves and the follow-up was not executed.** Mechanically: if `recentMoves` contains the candidate move and its proposed follow-up was not subsequently played, do not retry the candidate.

### Priority 4 — Draw from stock

Only when no priority 0–3 move exists. Drawing is "information acquisition" — it has no direct progress value, only sets up future Priority 0–3 moves.

- **Sub-rule 4a — Before drawing, verify exhaustion.** If `legalMoves` includes any move that flips a face-down, advances a foundation, or creates an empty column with a king ready, that move dominates a draw.

### Priority 5 — Recycle waste

Only when stock is empty, waste is non-empty, and no priority 0–3 move exists.

- **Sub-rule 5a — Bounded.** In draw-3 Klondike, recycle count is finite by rule (typically 0–3 redeals). In draw-1, it is unlimited but each recycle without intervening progress is a tighter signal of unwinnability.

### Anti-moves (NEVER play these, regardless of legality)

These are not separate priorities; they are *vetoes* applied to candidates from priorities 1–3:

- **Anti-loop.** A move whose inverse appears in the last 8 `recentMoves` entries, unless explicit Priority 3 sub-rule 3a is satisfied with a never-tried follow-up.
- **Anti-bury-needed.** A move that places a card on top of one needed for foundation within the next 1–2 ply.
- **Anti-empty-without-king.** Emptying a column with no King immediately playable to it.
- **Anti-foundation-trap.** A foundation send that violates Priority 2's safety rule (e.g. sending 5C up while 4H is still in waste).

### What the algorithm doesn't capture (intentionally)

- **Multi-ply lookahead beyond 2.** Klondike does have deeper tactical plays (3-ply setups, deferred King placements). The 1-turn algorithm above approximates a strong amateur; full optimal play requires search. For a teacher LLM, 1-turn correctness is the right ambition.
- **Stock-order memory.** A skilled human tracks seen-but-not-played waste cards by position. The model has access to `seenDrawPileCards` and `recentMoves`; we can lean on that without requiring full position tracking.
- **Endgame transitions.** Late game (foundations > ~40 cards) sometimes flips Priority 2's safety rule — burying becomes less risky because there is less coming. Out of scope for the prompt; the model can infer from `completionProgress`.

### Failure modes this priority cascade catches

| Observed failure | Caught by |
|---|---|
| Doom-loop oscillation (`645d03` 5C/4D) | Anti-loop veto + Priority 3 sub-rule 3a (named follow-up required) |
| Premature foundation build trapping tableau | Priority 2 safety rule |
| Emptying column with no King | Anti-empty-without-king veto + Priority 1 sub-rule 1c |
| Premature draw with productive move available | Priority 4 sub-rule 4a |
| Saturated confidence on bad moves | Indirect — algorithm-conformant moves should report higher confidence; non-conformant moves must justify, which lowers reported confidence honestly |

## Part 2 — Prompt-engineering principles that map onto this algorithm

Selected and ranked by direct relevance to "pick one move from a numbered list, given a game state and a strict decision policy."

### 2.1 Structure as algorithm, not prose

Heuristics in a flat bullet list are read as *parallel*. Priorities in a numbered cascade are read as *sequential*. Restructure the strategy block to mirror Part 1's cascade exactly — "check Priority 0, then 1, then 2…" — and put the anti-moves into a distinct VETO block.

### 2.2 Per-priority structured CoT

The current `board_analysis` → `strategic_plan` → `final_decision` cascade is *generic* CoT. Replace with a *per-priority* CoT where each priority gets its own field and the model fills them in order. This is the prompt-engineering equivalent of converting a `if/elseif/else` into explicit `if` checks the model is forced to write:

```json
{
  "priority_0_check": { "candidate": <move_index|null>, "rationale": <string> },
  "priority_1_check": { "candidate": <move_index|null>, "rationale": <string> },
  "priority_2_check": { "candidate": <move_index|null>, "rationale": <string> },
  "priority_3_check": { "candidate": <move_index|null>, "rationale": <string>, "next_move_committed": <string|null> },
  "veto_checks": { "loop": <bool>, "bury": <bool>, "empty_no_king": <bool>, "foundation_trap": <bool> },
  "final_decision": { "move_index": <number>, "priority_selected": <number>, "confidence": <number> }
}
```

This is more verbose, but it converts every step of Part 1 into a written commitment. The model cannot skip a priority check; it has to write `null` to opt out, which is itself a structured signal we can mine later.

### 2.3 Anti-rationalisation via veto block

The 645d03 transcripts show the model can rationalise around heuristic-form rules ("it's a setup move, not an undo"). Vetoes need to be *binary booleans the model has to set*. If `veto_checks.loop` is true, the model has just admitted in writing that its candidate is a loop, and the prompt rule says: *"If any `veto_checks` field is `true`, the chosen move must be from Priority 4 or 5."* Hard constraint.

### 2.4 Stateful continuity across turns

Inject the previous turn's `priority_3_check.next_move_committed` (if any) into the new prompt as `PREVIOUS COMMITMENT`. Require the model to:

- Acknowledge it.
- Either execute it (the committed move is now this turn's candidate), or
- Explicitly abandon it with a written reason.

The harvest already has the prior turn's response stored; this is a harness change, not a model-side requirement.

### 2.5 Few-shot grounding with worked examples

Two examples are enough. One example each of:

1. **A correct turn** — board, legalMoves, the right Priority-1 reveal, why it's chosen, why Priority 2 was checked and skipped.
2. **A trap turn** — the exact 645d03 turn 133 board with the 5C/4D loop, showing the correct response: `veto_checks.loop = true` → forced to Priority 4 (draw) or 5 (recycle).

Few-shot examples cost ~600–1000 tokens per example. The current prompt has zero. For a model the size of gemma-4-31b-it making sequential decisions, this is a high-ROI insertion.

### 2.6 Calibrated confidence anchored to priority

Re-anchor the confidence rubric to *which priority was selected*:

- Priority 0 selection: `0.95+` (forced or near-forced).
- Priority 1 selection: `0.8–0.95` (clear progress).
- Priority 2 selection: `0.7–0.9` (depends on safety check).
- Priority 3 selection: `0.5–0.8` (depends on follow-up plausibility).
- Priority 4 or 5: `0.4–0.7` (no immediate progress; honest uncertainty).

Plus a hard rule: *"If any veto fired and you still played the move (i.e. the safe-Priority-4-or-5 escape was not chosen), confidence must be ≤ 0.3."* This punishes the rationalisation-then-commit pattern numerically.

### 2.7 Token economy

The current rules block (~1100 chars) is wasted on a 31B model that already knows Klondike. Compress it ruthlessly:

- "Rules summary" → one sentence per non-obvious rule (variant-specific: this variant's draw count, this variant's column count, this variant's recycle rule).
- Eliminate restated standard rules.
- Spend the recovered budget on the priority cascade, vetoes, and worked examples.

Expected new prompt budget: 6500–8500 chars (similar order; better-allocated).

### 2.8 What we are explicitly NOT doing

- **No tree-of-thoughts / multi-branch search prompting.** ToT-style prompts have the model explore multiple branches and merge. Per Part 1, single-turn decision is a priority cascade, not a search; ToT adds tokens without adding signal here.
- **No self-critique loop within one turn.** A second "critique your answer" pass doubles cost and the few studies on its value for game decisions are inconclusive. The veto block is a cheaper, structured equivalent.
- **No external tool calls / function calling.** The model is meant to be the decision-maker. Calls to a solver from inside the prompt blur the harvest signal (see `PROMPT_ENGINEERING_DOOM_LOOP_2026-05-20.md` "Out of scope").

## Part 3 — Concrete proposed prompt structure

Sketched as a template; harvest team fills in exact text. Order matters.

```
[01] ROLE (1 line)
   "You are an expert Klondike Solitaire decision-maker. You apply a strict
   priority cascade to choose exactly one move per turn."

[02] VARIANT NOTE (3-4 lines, this-variant specifics only)
   - draw-1 or draw-3
   - recycle policy
   - column indexing (1-based)

[03] PRIORITY CASCADE (the algorithm from Part 1, numbered 0..5)
   Each priority gets:
     - One-sentence statement
     - Sub-rules as bulleted constraints
   No "heuristics"; everything is a rule.

[04] VETO BLOCK
   Four binary vetoes, each one sentence. Hard rule: ANY veto true => move to
   Priority 4 or 5.

[05] WORKED EXAMPLES (2)
   - Correct Priority-1 reveal example
   - 645d03 turn 133 trap example with the correct veto + escape

[06] RESPONSE SCHEMA
   The per-priority structured-CoT JSON from §2.2.

[07] CONFIDENCE RUBRIC (priority-anchored, from §2.6)

[08] PREVIOUS COMMITMENT (injected by harness, may be empty)
   "Last turn you committed to: <prior committed move or 'none'>."
   "Either execute it this turn (set move_index accordingly) or explicitly
    abandon it in the strategic notes."

[09] CURRENT GAME (JSON, as today)
   Adds: nothing structural. The model now consumes the same observation but
   through the cascade lens.

[10] CLOSING INSTRUCTION
   "Walk the cascade in order. Stop at the first priority that yields a
    candidate not vetoed. Emit the JSON object only — no prose outside it."
```

### Estimated character budget

| Block | chars |
|---|---|
| Role | 100 |
| Variant note | 250 |
| Priority cascade | 1800 |
| Veto block | 400 |
| Worked examples | 1500 |
| Response schema | 700 |
| Confidence rubric | 350 |
| Previous commitment | 150 (variable) |
| Current game (per turn) | 3000–4000 |
| Closing | 150 |
| **Total** | **~8500** |

Versus the current ~7500. Slightly larger; the gain comes from converting prose to algorithm + adding examples.

## Part 4 — Experimental design

This subsumes Phase 1/2/3 in the doom-loop doc. Run the new prompt as a **single intervention** rather than three patches, with the same metrics:

| Metric | Definition | Target vs baseline |
|---|---|---|
| Stall ratio | `stalled-game decisions` / `success decisions` | ↓ ≥ 50% |
| Mean longest plateau | turns of no foundation/faceDown change | ↓ ≥ 40% |
| Loop self-admission rate | % calls with any `veto_checks` field `true` | report; no target |
| Loop-while-vetoed | % calls where a veto fired AND the chosen move was the vetoed one | should be near-zero by construction |
| Priority distribution | histogram of `priority_selected` | sanity check; Priority 1 should dominate mid-game |
| Win rate | wins / completed games | ↑ any amount is a success |
| Foundation curve | mean `foundationCards` at turn 50, 100, 150 | ↑ at each marker |

### Sample size

Current baseline is 12 sessions over varied seeds. Re-run **the same seeds** with the new prompt to control for deal difficulty. If seeds are unavailable, run 12 fresh sessions at the same difficulty distribution. Pilot before scaling: **3 sessions first.** If Stall Ratio doesn't drop at all on the pilot, the prompt rebuild is wrong — stop and iterate.

### Failure mode

If Stall Ratio drops but Win Rate doesn't, the rebuild has fixed the doom-loop but exposed a different failure mode (probably premature foundation trapping or stock-card mis-tracking). Diagnose with the new `priority_selected` histogram — if Priority 2 selections jumped, the safety rule needs tightening; if Priority 4 selections jumped without progress, the cascade is being short-circuited.

## Part 5 — Open questions for the harvest team

1. **Variant.** Is the live harvest draw-1 or draw-3? Different recycle behaviour, different stock-cycling memory expectations.
2. **`seenDrawPileCards` semantics.** Is it ordered by appearance, or by current waste-pile-top? Affects whether the model can reason about "I've seen card X but it cycled past."
3. **Token budget.** What is the 31B teacher's max input tokens? At ~8500 chars + game JSON, we're near 3000 tokens. If we're close to the timeout cliff, drop a worked example.
4. **Prior-turn injection cost.** Hooking `PREVIOUS COMMITMENT` requires the harness to read the prior decision's `priority_3_check.next_move_committed`. Schema change vs ad-hoc parse?
5. **Whether to also store the per-priority CoT.** Adds bytes to each row but is a goldmine for downstream analysis (where exactly in the cascade does the model think it is?).

## Part 6 — How this relates to the doom-loop doc

`PROMPT_ENGINEERING_DOOM_LOOP_2026-05-20.md` proposes three *minimal additive patches* — keep the existing prompt structure, add `loop_check`, promote one heuristic to a rule, add `plan_continuity`. That doc is the right move if the harvest team wants a small, controlled experiment that doesn't touch the rest of the prompt.

This doc is the *rebuild* path — restructure the strategy block as a priority cascade, add per-priority CoT, add vetoes, add worked examples. It is higher ceiling, higher risk, and requires more harness work. If the minimal patches in the doom-loop doc fail their Phase 1 acceptance gate, this doc is the recommended next step.

The two are not mutually exclusive: the rebuild absorbs all three minimal patches (loop_check becomes `veto_checks.loop`; the promoted rule becomes part of the priority cascade; `plan_continuity` becomes `PREVIOUS COMMITMENT` + Priority 3 sub-rule 3a). Running the patches first lets us pin down which specific change moves the metric.
