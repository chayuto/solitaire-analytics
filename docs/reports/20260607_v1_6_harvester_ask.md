# v1.6 Prompt Ask: fix the DRAW TIMELINE `{NOW}` phantom marker

**Date:** 2026-06-07 | **Target:** `gemma-4-31b-it` (the fix applies to every client) | **Scope:** prompt explanation text only, one paragraph. No state renderer, no logic. | **Stamp:** `hybrid-v1.6`. | **Severity:** low-to-medium (clarity). | **Origin:** found in the v1.5 prompt audit (`docs/reports/20260606_v1_5_prompt_bug_audit.md`), confirmed pre-existing.

v1.5 has shipped (`hybrid-v1.5`, build `6810750`; see `docs/reports/20260606_v1_5_harvester_ask.md`), so this carries forward as the next iteration, v1.6, not a v1.5 patch. v1.6 is deliberately a single focused clarity fix: the v1.5 audit found the rest of the rendered prompt bug-free, so there is nothing else to change this round.

## The bug

The DRAW TIMELINE explanation (verbatim from `#6eb393`, build `6810750`):

> The DRAW TIMELINE block renders the stock and waste piles as one linear sequence of card identifiers. **The current waste top is wrapped in {curly braces}.** Tokens **LEFT of {NOW}** are cards that will be drawn next in this stock cycle: the immediate next draw is the token **directly left of {NOW}**, the draw after that is the token two positions left, and so on. Tokens **RIGHT of {NOW}** are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top. The token `???` marks a card whose identity has not yet been observed. `???` can only appear during the first stock cycle ...

The data it describes (verbatim, mid-game):

```
??? ??? ??? {KD} JH 4C 4H 9S 2C
```

The explanation **defines the marker as `{curly braces}`, then immediately refers to it three times as `{NOW}`** — a token that never appears in the rendered data. The reader has to infer that the braced card (`{KD}`) is the `{NOW}` position. The paragraph is internally inconsistent with itself.

## Deep analysis

- **It is a naming inconsistency, not a data bug.** The renderer is correct and consistent: across 124 timeline-bearing turns of `#6eb393` there is **always exactly one braced token** (the waste top), and never zero. The marker is unambiguous in the data; only the prose names it wrongly.
- **Pre-existing, not a v1.5 regression.** Sampled one prompt per version: the data marker is a braced card and the prose says `{NOW}` in all three.

  | Version | Session | Prose says `{NOW}` | Data marker |
  |---|---|---|---|
  | v1.5 | `#6eb393` | yes | `{KD}` |
  | v1.4 | `#136236` | yes | `{JC}` |
  | v1.3 | `#404d11` | yes | `{2S}` |

- **The model copes, so severity is bounded.** It reads the timeline correctly in practice: `#6eb393` turn 208 reasoned "the immediate next draw is KD, the subsequent draw is 7D", and the corpus has wins under this notation across all three versions. The cost is an extra inference step every turn and a misread risk that grows for weaker models (the E2B student) and at edge cases.

## The design question: keep the curly braces, or change the marker?

**Recommendation: keep the curly braces. Fix the prose only. Do not change the marker, and specifically do not change it to `??`.**

Reasoning:

1. **The braces work.** There is no evidence they cause misreads; the bug is the prose. Fix the prose, not the thing that works.
2. **The data is already unambiguous** (always exactly one braced token = the waste top). Nothing structural to fix on the render side.
3. **Changing the marker is unjustified risk.** A new marker (arrow, `NOW:` label, square brackets) would need re-validation, would inject a fresh misread chance, and would **make the new corpus inconsistent with the 15k+ existing rows** that use braces, hurting the dataset's uniformity. Per the operator doctrine (keep it simple, fix existing errors, high confidence), a prose fix clears the bar and a marker change does not.
4. **Do NOT change the marker to `??`.** `???` already means "unobserved card identity". Reusing `??`/`???` for the position marker would collide with that meaning and create real ambiguity (is `??` the current position or an unknown card?). The two notations must stay distinct: **`{braces}` = current position, `???` = unknown identity.**

## The fix (prose only)

Replace the three `{NOW}` references with references to the braced token. Exact change to the explanation paragraph:

**From:**

> The current waste top is wrapped in {curly braces}. Tokens LEFT of {NOW} are cards that will be drawn next in this stock cycle: the immediate next draw is the token directly left of {NOW}, the draw after that is the token two positions left, and so on. Tokens RIGHT of {NOW} are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top.

**To:**

> The current waste top is wrapped in {curly braces}; it marks the current position in the cycle. Tokens to its LEFT are cards that will be drawn next in this stock cycle: the immediate next draw is the token directly to its left, the draw after that is the token two positions to its left, and so on. Tokens to its RIGHT are cards drawn earlier in this cycle, still sitting in the waste pile beneath the current top.

Leave the `???` sentence and everything else unchanged. This removes the phantom `{NOW}`, ties LEFT/RIGHT to the braced token actually rendered, injects no decision logic, and matches the state-not-logic principle (`prompt-closes-info-gap-not-logic`).

## On the timeline's left-to-right order (justified; do not change)

The order is correct and should stay. It mirrors the physical board: the stock (the cards drawn next) sits to the LEFT, and drawn cards go to the waste on the RIGHT, so rendering `[next draws ... {current top} ... already-drawn waste]` left-to-right is exactly the spatial layout the player sees. That is the "render the state a human has" principle working as intended, not a convention the model has to be taught. No change here, and no labeled-line redesign is warranted; v1.6 is only the `{NOW}` prose fix. (An earlier draft of this note wrongly called the order counterintuitive; it is board-faithful.)

## Validation and scope

- Clarity-only prose change: no state, no logic, no data-renderer change. Risk is minimal.
- Validate by rendering a prompt under the new template and confirming: (a) no `{NOW}` remains in the prose, (b) the braces still wrap the waste top, (c) the timeline **data line is byte-identical** to before (only the explanation text changed).
- No behavioural A/B is strictly required; a light read of timeline-reasoning on one or two seeds is enough to confirm it is unchanged-or-clearer.
- Ship with the next build; stamp the version fields `hybrid-v1.6` and bump `promptTemplateHash`; distinguish builds by templateHash per the usual rule.
