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

## Why this is not rigging: state, not logic

This is the spine of the ask, and it holds three ways.

**1. Deleting the draw-directive removes injected logic; it adds none.** The draw-directive is a piece of decision procedure we wrote in ("do X when Y"). Deleting it biases the model toward *nothing*: `draw_card` is still a legal move every turn, and the model now has to decide from the state whether it is worth it, the way a human does. We are not biasing it to resign and not biasing it to draw. We hand the choice back.

**2. The two PROGRESS counts are facts, not directives.** Turns-since-progress are counts of true past events, the thing a human feels ("I have been cycling for ages"). Our RECENT MOVES window shows only 10 moves, so a 40-draw dry spell is invisible; the counts restore exactly what the window hides. Same category as `foundation=11/52`. We render the numbers with no threshold and no action attached: no "resign if N", no "exhausted after one cycle".

**3. The look-ahead/resign rule we could write is the one we refuse to — because the model already does it unaided** once the directive stops pre-empting it. The correct logic is plain: after one cycle the stock is fully known, so a draw only repositions you in a sequence you have already seen, and is worth it only to reach a specific card you can see coming and can play; otherwise you are stuck. We do not put that in the prompt, because the model produces it on its own. The lone escape in this batch, the 26B win `#3e91a0` (seed 3169322146), drew through a dry spell and named the line itself: "the next card is 8C, which can be played to C ... then 6S unlocks the column-5 run" — the same DRAW TIMELINE every session sees. The dead boards could not produce that sentence because no such card existed, not because the prompt failed to instruct the look-ahead; the draw-directive was suppressing the reasoning by answering "just draw" before the model got there. The boundary that keeps this fair: stock and waste are *known* after one cycle (seen, in memory, already rendered as the DRAW TIMELINE), so reasoning over them is the model's job. The face-down tableau stays *unknown* (`??`), and we touch nothing there. v1.5 adds no information a human would not already have.

## The honest risk

The draw-directive is fatal on a dead board, but "keep drawing when nothing else helps" is how slow boards get won: you draw through a dry spell until the card surfaces. Deleting bullet 6 risks a timid model under-drawing and losing boards it currently wins, so this is not free. The three slow wins that prove the risk are exactly the gate floor below: `#2c84bac05ad4` (seed 3263196305, won after a 94-turn plateau), `#3ced34aca45a` (seed 2853966634, after a ~290-move grind), and `#a11e74` (seed 601852437, after a 119-turn plateau); the 26B win drew through one too. If that set regresses, the directive was load-bearing — and the fallback is to soften, not delete: replace it with a pure observation that keeps the fact without the verdict ("After the first stock cycle every stock card is known; a draw only changes which already-seen card is on top") and re-test.

Scope honesty: every stall we diagnosed was a structurally dead deck. v1.5 buys termination and clean labels (stop burning 250 to 380 turns, kill the silent draw-loop) — that is efficiency, not win rate. Dead decks stay losses.

## Test gate

Run v1.5 against v1.4 on `gemma-4-31b-it`, same seed, adjudicate non-wins with `check_winnability.py --solver engine`.

- **Must still win, no premature resign** (the floor — the three slow-win stress seeds the deletion most endangers): 3263196305 (`#2c84bac05ad4`, 94-turn plateau), 2853966634 (`#3ced34aca45a`, ~290-move grind), 601852437 (`#a11e74`, 119-turn plateau), plus the v1.4 14-seed regression set.
- **Should stop earlier** (correct early resign is the win): 3255629335 (#404d11), 3602844246 (#c36b7b).
- **Silent-loop check**: 2945049884 (#136236) longest consecutive-draw run should fall from the hundreds.

Ship only if the slow-win floor holds.

## Guardrails

Do not, in v1.5: add a reasoning-trail or prior-rationale block (keep RECENT MOVES as actions only); replace bullet 6 with a different directive; add a resign predicate; or attach a threshold or recommended action to the new counts. Render the state, leave the decision to the model.

Evidence: the turn-470 quote and all thinking text are in the export `prompt`/`thinkingText` fields; session write-ups in `data/DATASET_NOTES.md`; F2 was parked in `docs/reports/20260604_v1_4_harvester_ask.md`, which v1.5 un-parks alongside the deletion.
