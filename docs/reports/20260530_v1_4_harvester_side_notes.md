# v1.4 Prompting Notes: One Simple Fix, Tested Small

**Date**: 2026-05-30
**Status**: NOTES. Not committed. Test small before any deploy.
**Scope**: prompting system only (the prompt the harvester builds, both state and wording). Out of scope: model, inference-config, training, and operational harness logic.
**Priority**: a high-confidence change, or a fix to an existing prompt error. Keep it simple, do not add features we cannot justify.

## The one change: clean up the STRATEGY GUIDANCE block

One block, three small text fixes. All fix existing errors. None add a new feature or a new restriction, so regression risk is low.

1. **Rank the seven bullets as a numbered priority list.** Today reveal-priority (listed first) and anti-undo (listed last) contradict each other with no tie-break. That is the documented cause of the 49b05f override (anti-patterns 1 and 5.3).
2. **Fix the horizon mismatch.** Anti-undo says "last 5 moves" but RECENT MOVES renders up to 10 (verified 7 to 10 across 222 turns). Change it to "last 10 moves" so the rule matches what is shown.
3. **Anchor reveal-priority to the objective tag.** Change "exposes a face-down tableau card" to "a move tagged (reveals a hidden card)". The model co-opts the looser wording to justify non-revealing toggles; the tag is already in the move list and is 100% accurate. This is the fix most aligned with the 31B data.

After (replaces the current block):

```
STRATEGY GUIDANCE (priority order; when two rules conflict, the lower number wins):
1. Play any available Ace or 2 to its foundation immediately.
2. Do not move a card to a tableau column it occupied in the last 10 moves shown
   in RECENT MOVES. This overrides rule 3.
3. Prefer a legal move tagged "(reveals a hidden card)". When several are tagged,
   prefer the one in the column with the most face-down cards remaining.
4. Draw from the stock when no legal move is tagged "(reveals a hidden card)" or
   advances a foundation.
5. Do not send a higher card to a foundation if it may still be needed to receive
   a tableau card.
6. Do not empty a column unless a King is ready to occupy it.
```

Confidence: HIGH that these fix real errors and are low-risk (pure text, no new restriction). MEDIUM that they move 31B outcomes, because the model already takes offered reveals about 90% of the time. Fix 3 is the one most likely to matter on 31B. If we want to attribute the effect we can ship fix 3 alone first; otherwise the three are one coherent cleanup.

## Temporal awareness: one piece earns confidence, one does not

In scope, judged by the confidence bar:

- **Longer move-history window** (show more than 10 recent moves): SOME confidence, for a mechanical reason. The anti-undo rule cannot fire on a loop longer than the window, so extending the window is what makes the rule actually work on long-period oscillation. But it is a harness render change, not just wording, so it is the SECOND step, taken only if the guidance fix does not calm the messy loops.
- **Stall indicator** ("foundations unchanged for N moves"): NOT enough confidence to include. It relies on the model reacting to a new signal, and the data this session shows the model under-uses signals it already has. Skip for now.

## Test small before deploy

1. Sanity check on the local 20-state bench that move-selection does not regress. Cheap and single-turn, but note it does NOT predict full-game behavior, so it is only a "did not break it" check.
2. Real check: run the modified prompt on a few full-game seeds we already have v1.3 data for, the anchor 3263196305, the kill seed 1792828001, and one or two fresh, and compare reveal pass-up and oscillation against v1.3.
3. Deploy to the full harvest only if those show no regression and ideally an improvement.

## Not now

Stall indicator (low confidence), INTENT / plan-continuity (targets the rarer plan-abandonment mode), and anything outside the prompt (model, inference-config, training, auto-terminate tooling).
