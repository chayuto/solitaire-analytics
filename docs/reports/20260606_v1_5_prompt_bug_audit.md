# v1.5 generated-prompt bug audit (across a full game)

**Date:** 2026-06-06 | **Session:** `#6eb393` (full `019e99f9-05a8-7eb3-9d21-76995f6eb393`) | **Build:** `6810750` | **Prompt:** `hybrid-v1.5` (templateHash `8a46ca22…`) | **Source:** `data/raw/solitaire-ai-log-6eb393-1780743062479.json`, all 145 rendered prompts.

Goal: audit the prompt the harvester renders, turn by turn across one whole game, for rendering and correctness bugs, with emphasis on the new v1.5 surface (the deleted draw-directive and the two added PROGRESS counts), which is the highest-risk place for a regression.

## Summary

- **No v1.5-introduced bug found.** The new counts, the deletion, and the surrounding render are all correct on every one of the 145 turns.
- **One real bug found, but PRE-EXISTING** (identical in v1.3, v1.4, v1.5): the DRAW TIMELINE explanation refers to a `{NOW}` marker that never appears in the rendered data. Low-to-medium severity. Fix is a one-line clarity reword.

## Checks run and results

Each check was computed over all 145 successful-turn prompts (ordered by `turnIndex`), parsing the PROGRESS line and the chosen legal move.

| Check | Result |
|---|---|
| `turns since foundation grew` resets to 0 exactly when `foundationCards` rises, increments otherwise | **0 anomalies / 145** |
| `turns since a card was revealed` resets to 0 exactly when `faceDownTotal` drops (a real reveal), increments otherwise | **0 mismatches / 145** |
| `completion%` == `round(foundationCards / 52 * 100)` | **0 errors / 145** |
| Reveal-tag accuracy: a move tagged `(reveals a hidden card)` actually drops `faceDownTotal` next turn | **13 / 13 correct** |
| PROGRESS line present, counts never missing or negative | **145 / 145 present, 0 negative** |
| Unrendered template placeholders (`{{ }}`, `${}`, `undefined`, `NaN`, `null`, `[object`) | **none** |
| Blank-line gap / dangling text from the draw-directive deletion | **none** (STRATEGY GUIDANCE is 6 clean bullets; no 4+ newline runs) |
| Leftover draw-directive language anywhere in the prompt | **none** (clean deletion) |
| All sections present every turn (TABLEAU, DRAW TIMELINE, RECENT MOVES, LEGAL MOVES, PROGRESS, STRATEGY GUIDANCE, CYCLE) | **yes** |

Representative correct PROGRESS line (mid-game):

```
PROGRESS: foundation=13/52, face-down remaining=6, completion=25%, turns since foundation grew: 6, turns since a card was revealed: 29
```

Count behaviour sample (turnIndex: foundationCards / sinceGrew / sinceRev), showing reset-on-progress:

```
turn 11: fnd=2 grew=4 rev=0      turn 13: fnd=3 grew=0 rev=0   <- foundation grew, grew-count reset
turn 24: fnd=3 grew=8 rev=0      turn 25: fnd=4 grew=0 rev=1   <- foundation grew, grew-count reset
```

Note (design, not a bug): once `faceDownTotal` reaches 0, no card can be revealed, so `turns since a card was revealed` only climbs (observed 0..11 over the 12 face-down=0 turns). The number stays well-defined; it just stops being actionable late game.

## The one bug: DRAW TIMELINE `{NOW}` marker is documented but never rendered

**What the explanation says** (static instruction text, 3 references to `{NOW}` per prompt):

> The current waste top is wrapped in {curly braces}. Tokens LEFT of {NOW} are cards that will be drawn next in this stock cycle: the immediate next draw is the token directly left of {NOW} ... Tokens RIGHT of {NOW} are cards drawn earlier in this cycle ...

**What the data actually renders** (the marker is the waste-top card wrapped in braces, not a literal `{NOW}`):

```
??? ??? ??? ??? ??? ??? {8S} KS TH JH 4C 4H 9S 2C
```

So `{NOW}` appears only in the prose; the position is actually marked by `{8S}` (the braced waste top). The reader must infer that the braced card IS the `{NOW}` point. The explanation introduces two notations (`{curly braces}` = waste top, and `{NOW}` = position) and never reconciles them.

**Pre-existing, not a v1.5 regression.** Verified by sampling a prompt from each version; the data marker is a braced card and the explanation says `{NOW}` in all three:

| Version | Session | Explanation says `{NOW}` | Data marker |
|---|---|---|---|
| v1.5 | `#6eb393` | yes | `{8H}` |
| v1.4 | `#136236` | yes | `{JC}` |
| v1.3 | `#404d11` | yes | `{2S}` |

**Severity: low-to-medium.** The model evidently infers the mapping (it reads the timeline correctly, e.g. #6eb393 turn 208 names the exact next draws, and the corpus has wins across all versions). But it is a genuine doc/data mismatch that costs the model an inference step on every turn and risks a misread.

**Recommended fix (clarity only, no logic injection, per the state-not-logic principle in `prompt-closes-info-gap-not-logic`):** reword the explanation to describe the marker that is actually rendered, e.g.

> The current waste top is wrapped in {curly braces} and marks the current position in the stock cycle. Tokens to its LEFT are drawn next this cycle (the immediate next draw is the token directly to its left); tokens to its RIGHT were drawn earlier and sit in the waste beneath the top.

This removes the phantom `{NOW}` token and points the reader at the `{...}` braces that are actually present. It changes no state and injects no decision rule.

## Conclusion

v1.5's new rendering is correct across the whole game. The only single-turn prompt bug is the long-standing `{NOW}` documentation mismatch in the DRAW TIMELINE, which predates v1.5.

## Addendum: cross-turn read (recycle boundaries)

A follow-up read of the prompts in sequence across the two recycle boundaries (turns 170 and 203) found two render-side fidelity gaps that single-turn checks cannot see:

- The `recycle stock` action is never written to RECENT MOVES (0 of 145 blocks), so post-recycle the model sees two identical draws with the recycle invisible (a false-loop signal next to the anti-undo bullet).
- The DRAW TIMELINE is dropped on every empty-waste turn (20 of 20), including the post-recycle moment when the stock order is fully known.

Both are state-render fixes (not prose) and had low observed impact on this 31B win (it coped: it used the timeline to plan the recycle, then drew). The recycle-in-RECENT-MOVES finding is **item 2 of the v1.6 ask** (`docs/reports/20260607_v1_6_harvester_ask.md`); the `{NOW}` fix is item 1. The timeline-on-empty-waste finding is **deferred to that doc's internal appendix** (operator decision 2026-06-07: not passed to the harvester team).
