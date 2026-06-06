# v1.6 Harvester Ask: fix the DRAW TIMELINE `{NOW}` prose, and log the recycle in RECENT MOVES

**Date:** 2026-06-07 | **Target:** `gemma-4-31b-it` (applies to every client) | **Stamp:** `hybrid-v1.6`. | **Origin:** the v1.5 prompt audit (`docs/reports/20260606_v1_5_prompt_bug_audit.md`) plus a cross-turn read of `#6eb393`.

v1.5 has shipped (`hybrid-v1.5`, build `6810750`), so this is the next iteration, v1.6. A full single-turn audit of the rendered prompt (every section, all 145 turns of `#6eb393`) was clean: the v1.5 stall counts, completion math, and reveal tags are all correct; RECENT MOVES is actions-only; the output schema is a clean three keys; and the 36% of errored turns were provider timeouts ("did not respond within 240s"), not prompt or parse failures. A cross-turn read across the recycle boundaries then surfaced a move-history gap (item 2). Both ask items are state-not-logic: they render what a human at the table sees, and inject no heuristic or predicate.

**The ask (pass these two to the harvester team):**

| # | Change | Kind | Severity |
|---|---|---|---|
| 1 | Delete the phantom `{NOW}`; describe the braced waste-top marker instead | prose | low-med (clarity) |
| 2 | Record `recycle stock` as an entry in RECENT MOVES | render | medium (history omission, false-loop signal) |

Two further low-severity fidelity findings (DRAW TIMELINE dropped on empty-waste turns; the `??` vs `???` notation overlap) are recorded in the **Internal appendix** at the end and are **deliberately NOT part of this ask**. Observed-impact note: on this strong 31B win the model coped with all of these (at the recycle it used the timeline to plan "2C is at the bottom, recycle to reach it", then drew); they matter more for the weaker E2B student and messier games.

## Item 1: the `{NOW}` phantom marker (prose)

The DRAW TIMELINE explanation (verbatim from `#6eb393`, build `6810750`):

> The DRAW TIMELINE block renders the stock and waste piles as one linear sequence of card identifiers. **The current waste top is wrapped in {curly braces}.** Tokens **LEFT of {NOW}** are cards that will be drawn next in this stock cycle: the immediate next draw is the token **directly left of {NOW}**, the draw after that is the token two positions left, and so on. Tokens **RIGHT of {NOW}** are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top. The token `???` marks a card whose identity has not yet been observed. `???` can only appear during the first stock cycle ...

The data it describes (verbatim, mid-game):

```
??? ??? ??? {KD} JH 4C 4H 9S 2C
```

The explanation **defines the marker as `{curly braces}`, then immediately refers to it three times as `{NOW}`** — a token that never appears in the rendered data. The reader has to infer that the braced card (`{KD}`) is the `{NOW}` position. The paragraph is internally inconsistent with itself.

### Deep analysis

- **It is a naming inconsistency, not a data bug.** The renderer is correct and consistent: across 124 timeline-bearing turns of `#6eb393` there is **always exactly one braced token** (the waste top), and never zero. The marker is unambiguous in the data; only the prose names it wrongly.
- **Pre-existing, not a v1.5 regression.** Sampled one prompt per version: the data marker is a braced card and the prose says `{NOW}` in all three.

  | Version | Session | Prose says `{NOW}` | Data marker |
  |---|---|---|---|
  | v1.5 | `#6eb393` | yes | `{KD}` |
  | v1.4 | `#136236` | yes | `{JC}` |
  | v1.3 | `#404d11` | yes | `{2S}` |

- **The model copes, so severity is bounded.** It reads the timeline correctly in practice: `#6eb393` turn 208 reasoned "the immediate next draw is KD, the subsequent draw is 7D", and the corpus has wins under this notation across all three versions. The cost is an extra inference step every turn and a misread risk that grows for weaker models (the E2B student) and at edge cases.

### Keep the braces, fix the prose (and do not change the marker to `??`)

1. **The braces work.** There is no evidence they cause misreads; the bug is the prose. Fix the prose, not the thing that works.
2. **The data is already unambiguous** (always exactly one braced token = the waste top). Nothing structural to fix on the render side.
3. **Changing the marker is unjustified risk.** A new marker (arrow, `NOW:` label, square brackets) would need re-validation, inject a fresh misread chance, and **make the new corpus inconsistent with the 15k+ existing rows** that use braces. A prose fix clears the operator bar (simple, fixes an existing error); a marker change does not.
4. **Do NOT change the marker to `??`.** `???` already means "unobserved card identity". Reusing `??`/`???` for the position marker would collide with that meaning. The two notations must stay distinct: **`{braces}` = current position, `???` = unknown identity.**

### The fix (prose only)

Replace the three `{NOW}` references with references to the braced token:

**From:**

> The current waste top is wrapped in {curly braces}. Tokens LEFT of {NOW} are cards that will be drawn next in this stock cycle: the immediate next draw is the token directly left of {NOW}, the draw after that is the token two positions left, and so on. Tokens RIGHT of {NOW} are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top.

**To:**

> The current waste top is wrapped in {curly braces}; it marks the current position in the cycle. Tokens to its LEFT are cards that will be drawn next in this stock cycle: the immediate next draw is the token directly to its left, the draw after that is the token two positions to its left, and so on. Tokens to its RIGHT are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top.

This removes the phantom `{NOW}`, ties LEFT/RIGHT to the braced token actually rendered, injects no decision logic, and matches the state-not-logic principle (`prompt-closes-info-gap-not-logic`). Leave the `???` sentence and everything else in the paragraph unchanged.

## Item 2: record the recycle in RECENT MOVES (render)

`recycle stock` is chosen and executed, but it is never written to RECENT MOVES: across all 145 turns of `#6eb393`, **0** RECENT MOVES blocks contain a recycle entry. The effect at turn 171 (just after the turn-170 recycle):

```
   8. move 3C col 5 -> col 3
   9. draw 7D
  10. draw 7D
```

Two identical `draw 7D` with nothing between them, because the recycle that separated them is invisible. The model is left seeing what looks like the same card drawn twice in a row, exactly the repeat the anti-undo bullet ("do not undo your own work") warns against, with no recycle marker to explain it. **Fix:** append a `recycle stock` line to RECENT MOVES like any other action, so the history reads `... draw 7D / recycle stock / draw 7D`. This records an action the player actually performs (state, not logic).

(Related logging note for whoever implements this: the recycle and the following draw currently share a `turnIndex` at each boundary, so `turnIndex` is not a unique key; order interactions by `(turnIndex, id)`. This is internal and needs no prompt change.)

## Validation and scope (items 1 and 2)

- Item 1 is prose-only (no state, no logic, no renderer); risk is minimal. Item 2 is a render change (the move-history logger).
- Validate item 1 by rendering a prompt under the new template and confirming: (a) no `{NOW}` remains in the prose, (b) the braces still wrap the waste top, (c) the timeline data line is byte-identical to before (only the explanation text changed).
- Validate item 2 by recycling in a game and confirming RECENT MOVES shows the `recycle stock` entry between the surrounding draws, with no other move types altered.
- No behavioural A/B is strictly required for item 1; for item 2 a light read of reasoning across a recycle on one or two seeds confirms the model reads the recycle correctly.
- Ship with the next build; stamp the version fields `hybrid-v1.6` and bump `promptTemplateHash`; distinguish builds by templateHash per the usual rule.

---

# Internal appendix (NOT part of the v1.6 harvester ask)

Found during the audit and kept here for our own tracking. These are low severity, the 31B coped, and we are deliberately **not** asking the harvester team to change them now. Revisit if a future audit shows they bite, especially on the weaker E2B student.

## Deferred A: show the DRAW TIMELINE on empty-waste turns (render)

The timeline is dropped on every empty-waste turn (20 of them in `#6eb393`, 0 exceptions), because the position marker is the braced waste top and there is no card to brace. On the turn right after a recycle the stock is full and, this being cycle 2+, every card is a known identity, so the order would be maximally useful, yet it is omitted (e.g. turn 170-B: STOCK 7 cards, no timeline). The model recovers by drawing and the timeline returns the next turn, so impact is small. A fix would render the timeline with an empty-position marker when the waste is empty (show the known stock order, brace an empty slot). Tied to the marker design; deferred.

## Deferred B: name the `???` marker in NOTATION (prose)

The tableau uses `??` (two marks) for a face-down card; the DRAW TIMELINE uses `???` (three marks) for a not-yet-observed stock card. Both render consistently (across `#6eb393`: tableau `??` 995x and never three; timeline `???` 676x and never two) and the model showed no confusion. But they differ by a single character for related-but-different things, and the RULES NOTATION line defines only `??`. A one-clause fix would define both in NOTATION ("`??` = a face-down tableau card; `???` (three marks) = a stock card not yet observed"). Pure clarity; deferred because the render is consistent and no misread was observed.

## Considered and rejected: the timeline's left-to-right order is correct

The order mirrors the physical board: the stock (cards drawn next) sits to the LEFT, drawn cards go to the waste on the RIGHT, so rendering `[next draws ... {current top} ... already-drawn waste]` left-to-right is exactly the spatial layout the player sees. That is "render the state a human has" working as intended, not a convention to teach. No change; no labeled-line redesign warranted.
