# v1.5 Prompt Ask: delete the draw-directive, surface the stall counts

**Date:** 2026-06-06 | **Target:** `gemma-4-31b-it` | **Scope:** prompt + state renderer only.

Two state-side changes. We remove one verdict from the guidance and add two facts to the state. No new decision logic. Stamp both fields `hybrid-v1.5`.

## The ask

**1. Delete the draw-directive** (STRATEGY GUIDANCE, the bullet that begins "Drawing from the stock is the correct action when...").

```
remove:
- Drawing from the stock is the correct action when no legal move is tagged "(reveals a hidden card)" and no legal move advances a foundation.
```

Leave every other bullet, including the v1.4 reveal-tag anchoring and the anti-undo fix, exactly as is.

**2. Add two true counts to the PROGRESS line** (names your call, the requirement is real counts with no threshold attached):

```
PROGRESS: foundation=11/52, face-down remaining=15, completion=21%, turns since foundation grew: 63, turns since a card was revealed: 47
```

## What is actually happening

The stall is one bullet running unbounded. The draw-directive says drawing is correct whenever nothing is tagged as a reveal and nothing advances a foundation. On a locked board that condition is true *every turn*, so the model draws every turn until the cap. From `#136236` turn 470, with `CYCLE: 29` sitting in the prompt (the stock had been cycled 29 times):

> "Move [1] does not advance a foundation ... is not tagged ... Therefore, Move [0] (draw_card) is the correct choice."

It is not confused. It is obeying our rule, and it never connects the rendered `CYCLE: 29` or the fully-known DRAW TIMELINE to "drawing is doing nothing." Across the four most recent sessions:

| Session | Prompt | Late draw rate | Emitted resign | Said "resign" in thinking |
|---|---|---:|---:|---:|
| #404d11 | v1.3 | 33% | 0 | never |
| #c36b7b | v1.3 | 24% | 0 | never |
| #136236 | v1.4 | 56% | 0 | never |
| #3e91a0 (26B, won) | v1.3 | 9% | 0 | never |

The resign action that already works is never reached, because the draw-directive always supplies a "valid" move so the resign condition is never tested. And v1.4 made it worse-shaped: anchoring the reveal phrases to the tag (right for reveals) also sharpened the draw-directive, so the v1.4 stall is a clean 56%-draw loop with no card oscillation, which means the corpus oscillation detector cannot see it.

## Why this is not rigging (state, not logic)

This is the line we want held.

- **Change 1 is removing logic, not adding it.** The draw-directive is a piece of decision procedure we wrote in ("do X when Y"). Deleting it pushes the model toward nothing. `draw_card` is still a legal move every turn; the model now has to decide from the state whether it is worth it, the way a human does. We are not biasing it to resign and not biasing it to draw. We are handing the choice back.

- **Change 2 is a fact, not a directive.** Turns-since-progress is a count of true past events, the thing a human feels ("I have been cycling for ages"). Our RECENT MOVES window shows only 10 moves, so a 40-draw dry spell is invisible; the count restores it. Same category as `foundation=11/52`. We render the number and stop. No "resign if N", no "exhausted after one cycle".

- **The rule we are deliberately NOT writing.** The correct logic is plain: after one cycle the stock is fully known, so a draw only repositions you in a sequence you have already seen, and is worth it only to reach a specific card you can see coming and can play; otherwise you are stuck. We are not putting that in the prompt, because the model already does it. The lone escape in this batch, the 26B win `#3e91a0`, drew through a dry spell and named the line itself: "the next card is 8C, which can be played to C ... then 6S unlocks the column-5 run." Same DRAW TIMELINE every session sees. The dead boards could not produce that sentence because no such card existed, not because the prompt failed to instruct the look-ahead. The draw-directive was suppressing that reasoning by answering "just draw" before the model got there.

The boundary that keeps this clean: the stock and waste are *known* after one cycle (seen, in memory, already rendered as the DRAW TIMELINE), so reasoning over them is the model's job and is fair. The face-down tableau stays *unknown* (`??`), and we touch nothing there. v1.5 adds no information a human would not already have.

## The honest risk

The draw-directive is fatal on a dead board, but "keep drawing when nothing else helps" is how slow boards get won: you draw through a dry spell until the card surfaces. The 26B win did that, and so did `#2c84bac05ad4` (won after a 94-turn plateau), `#3ced34aca45a` (after a ~290-move grind), and `#a11e74` (after a 119-turn plateau). Deleting bullet 6 risks a timid model under-drawing and losing boards it currently wins. So this is not free. If the slow-win set regresses, the directive was load-bearing, and the fallback is to replace it with a pure observation that keeps the fact without the verdict ("After the first stock cycle every stock card is known; a draw only changes which already-seen card is on top") and re-test.

Scope honesty: every stall we diagnosed was a structurally dead deck. v1.5 buys termination and clean labels (stop burning 250 to 380 turns, kill the silent draw-loop), not win rate. Dead decks stay losses.

## Test gate

Run v1.5 against v1.4 on `gemma-4-31b-it`, same seed, adjudicate non-wins with `check_winnability.py --solver engine`.

- **Must still win, no premature resign** (the floor): 3263196305, 2853966634, 601852437, plus the v1.4 14-seed regression set.
- **Should stop earlier** (correct early resign is the win): 3255629335 (#404d11), 3602844246 (#c36b7b).
- **Silent-loop check**: 2945049884 (#136236) longest consecutive-draw run should fall from the hundreds.

Ship only if the slow-win floor holds.

## Guardrails

Do not, in v1.5: add a reasoning-trail or prior-rationale block (keep RECENT MOVES as actions only); replace bullet 6 with a different directive; add a resign predicate; or attach a threshold or recommended action to the new counts. Render the state, leave the decision to the model.

Evidence: the turn-470 quote and all thinking text are in the export `prompt`/`thinkingText` fields; session write-ups in `data/DATASET_NOTES.md`; F2 was parked in `docs/reports/20260604_v1_4_harvester_ask.md`, which v1.5 un-parks alongside the deletion.
