# Next change request to harvester team

**Date:** 2026-05-27
**From:** Chayut / dataset side
**Re:** Prompt v1.2: close the post-recycle "discovery" reasoning loophole. One block rename + format change, one inline counter, one heuristic rewrite. No new response fields.

**Companion documents:**
- `docs/reports/20260526_harvester_team_resign_hygiene_versioning_ask.md` (previous cycle: confidence drop + resign output were SHIPPED in build `20a825f`; semantic versioning request was NOT shipped and is re-raised below)
- `docs/reports/20260527_prompt_v1_2_candidate_spec.md` (full technical spec)
- `docs/reports/20260527_conversation_log_v1_1_discovery_and_ingest.md` section 7 (v1.1 discovery and the c99da9 case study that prompted this)

---

## TL;DR

1. **The problem (quantified)**: across 13 sessions with >=30 moves and >=1 recycle in `data/raw/`, **1129 of 1153 post-recycle draws (97.9 percent) use "discover / reveal / find / introduce / unlock" language** to justify drawing, even though every stock card has by then been observed at least once. The teacher is hallucinating that drawing might reveal new information when it cannot. The one WON session in the set (010e01, seed 3263196305) sits at 46.2 percent and the reasoning is qualitatively different (deterministic "the 9C will trigger a chain reaction" vs the failed runs' speculative "such as a red 7 or a black 9 or potentially...").

2. **Root cause**: the current prompt has TWO gaps that prevent the model from doing deterministic stock reasoning:
   - During the first stock cycle, no `SEEN IN WASTE` list is rendered at all. The only memory of what was drawn is `RECENT MOVES (last 10)`. A 24-card cycle drops 14 cards off the back of that window before the cycle ends.
   - After the first recycle, a `SEEN IN WASTE THIS CYCLE` block appears, but its label is misleading: the contents are actually "cards currently in the stock pile that have been observed before". The list SHRINKS as the model draws. A naive read of the label is backwards.

3. **The ask** (three small changes, no schema break):
   - **DRAW TIMELINE block** replaces `SEEN IN WASTE THIS CYCLE`. Single linear render of stock + waste in draw order, with `{NOW}` marker around the current waste top and `???` placeholders for unobserved stock positions (only possible in cycle 1).
   - **CYCLE counter** added to the existing STOCK line as `STOCK: N cards   CYCLE: K   WASTE top: X   recycle stock: yes/no`.
   - **Strategy heuristic update**: change the "Drawing from the stock is reasonable when no productive tableau/foundation move exists" line to a timeline-aware version with a hard `ONLY IF` predicate.

4. **Re-raise**: semantic prompt versioning (`promptLayoutVersion: hybrid-v1.2`) was asked last cycle and not shipped. The build `20a825f` ships a new template under the same `hybrid-v1` label, distinguished only by `promptTemplateHash`. Please bump on this cycle's change.

5. **Not asking this cycle**:
   - Tightening the resign trigger. Per our reframe (resign on a winnable deck is failure, not success), the existing conservative trigger is correct.
   - Re-introduction of edit 4 from last cycle (PRIOR REASONING truncation). Still on hold.
   - State-repetition annotation. Same scope as last cycle.

---

## Hard constraint: symmetry to human play

Every information item in v1.2 must be obtainable by a human playing the same deck. Memory aids are permitted; card-x-ray vision is not.

| Item | Human can observe? | v1.2 renders? | Symmetric? |
|---|---|---|---|
| Current waste top | Yes (face-up on waste) | Yes (`{NOW}`) | YES |
| Cards already drawn this cycle | Yes (visible in waste pile beneath top) | Yes (right of `{NOW}`) | YES |
| Identity of stock cards in cycle 1 | NO (face-down, not yet drawn) | `???` placeholder | YES |
| Identity of stock cards in cycle 2+ | Yes IF perfect recall of cycle 1 | Yes (full identities) | YES (perfect-recall human) |
| Number of cards in stock | Yes (count is visible) | Yes (STOCK line) | YES (unchanged from v1.1) |
| Identity of face-down tableau cards | NO | `??` placeholders | YES (unchanged from v1.1) |
| Current cycle number | Yes (human knows how many times they recycled) | Yes (CYCLE counter) | YES |

**Caveat we are flagging explicitly**: the cycle 2+ render gives the model perfect recall over cycle 1's draw order, whereas a typical recreational human has imperfect memory. This is within game rules (memory aids are not forbidden) but it does mean the model is playing as "perfect-recall human", not "typical human". We accept this trade-off because the alternative (asking the model to remember 24 draws in working memory) is what the current prompt does and is the source of the 97.9 percent discovery-rationale failure.

---

## The three changes (precise specifications)

### Change 1: DRAW TIMELINE block

**Replaces**: the existing `SEEN IN WASTE THIS CYCLE: <list>` block (rendered only post-first-recycle).

**Two pieces, in two different locations in the prompt:**

**Piece A, in the RULES preamble** (immediately after KLONDIKE SOLITAIRE RULES, before THE GOAL):

```
INTERPRETING THE DRAW TIMELINE (when present in the game state):
The DRAW TIMELINE block renders the stock and waste piles as one linear sequence
of card identifiers. The current waste top is wrapped in {curly braces}. Tokens
LEFT of {NOW} are cards that will be drawn next in this stock cycle: the
immediate next draw is the token directly left of {NOW}, the draw after that is
the token two positions left, and so on. Tokens RIGHT of {NOW} are cards drawn
earlier in this cycle, still sitting in the waste pile beneath the current top.
The token ??? marks a card whose identity has not yet been observed. ??? can
only appear during the first stock cycle. After the first recycle every position
is a known identity, because every stock card has passed through the waste at
least once.
```

**Piece B, in the CURRENT GAME block** (replacing the position where SEEN IN WASTE THIS CYCLE currently goes):

```
DRAW TIMELINE:
  ??? ??? ??? 8D QH 3C {7H} 4S KH 2D
```

No re-explanation in the data block; the orientation rule is in the preamble.

**Render rules**:
- During cycle 1: leftmost positions render as `???` until the model has seen that card pass through the waste at least once.
- After first recycle: every slot is a known identity.
- When stock is empty (right before recycle is taken): `{NOW}` sits at the leftmost position; only waste-side cards render to its right.
- When the run has no draws yet (turn 0): SKIP the block entirely.
- Maximum render length: 24 cards plus the `{NOW}` token.
- Whitespace-separated, no `|` separator. The `{}` braces are sufficient delimiter.

### Change 2: stock-cycle counter inline in STOCK line

**Replaces**: nothing. Adds a `CYCLE:` key to the existing STOCK line.

Before:
```
STOCK: 14 cards   WASTE top: 2H   recycle stock: no
```

After:
```
STOCK: 14 cards   CYCLE: 2   WASTE top: 2H   recycle stock: no
```

Semantics: `CYCLE = 1 + (number of times recycle has fired)`. Cycle 1 starts at game start. Cycle 2 begins the moment the first recycle action is applied. The cycle counter increments on the recycle event, so the prompt that immediately follows a recycle shows the NEW cycle number.

Same colon-key pattern as the existing STOCK line fields. No new line, no new heading.

### Change 3: strategy-heuristic rewrite

**Replaces** the existing line in the STRATEGY GUIDANCE block:

```
- Drawing from the stock is reasonable when no productive tableau/foundation move exists.
```

**With**:

```
- Drawing from the stock is reasonable ONLY IF the DRAW TIMELINE shows an
  upcoming card (a token left of {NOW}) that will be playable to a foundation
  or to a tableau column once drawn. If every remaining stock card has already
  been seen and none unlock the board, drawing wastes turns and you should
  commit to a tableau move instead, even an imperfect one.
```

This is what closes the 97.9 percent loophole. The old line is the rule the failed sessions are following correctly: "no productive move? draw." The new line adds the missing predicate: "no productive move AND the timeline shows an upcoming card that will help".

---

## Prompt-budget impact

| Block | v1.1 chars (typical) | v1.2 chars (typical) | Delta |
|---|---:|---:|---:|
| SEEN IN WASTE THIS CYCLE | ~70 | 0 (removed) | -70 |
| INTERPRETING DRAW TIMELINE paragraph (in RULES) | n/a | ~580 | +580 |
| DRAW TIMELINE data line | n/a | ~90 | +90 |
| STOCK line CYCLE field | n/a | ~10 | +10 |
| Updated strategy heuristic | ~95 | ~245 | +150 |
| **Net** | | | **+760** |

The v1.1 cleanup (drop confidence + alternative_move_index + calibration paragraph) freed about 1200 chars. v1.2 spends 760 of that. Net prompt is still about 440 chars SMALLER than v1.0.

---

## Verification on the harvester side (suggested)

Before sharing v1.2 broadly, please run one game on a known-winnable benchmark seed (e.g. seed `3263196305` which has solver-confirmed solvability) and check:

1. **No information leak in cycle 1**: dump the rendered prompt at cycle-1 turn 5. Confirm that all `???` positions in the DRAW TIMELINE correspond to cards NOT in `RECENT MOVES` draws. Any `???` slot showing a card identity is a bug.

2. **Cycle counter correctness**: at the prompt rendered immediately after the first recycle action, confirm `CYCLE: 2`. The session-level moveCount stat should also include the recycle as a move.

3. **Reading-direction sanity**: at any cycle-2 turn, take the token immediately left of `{NOW}` and confirm it matches the card that gets drawn at the NEXT turn's `WASTE top:`. The token two positions left should match the draw after that.

A small render-only unit test would catch all three. We can supply our local Python implementation as a reference if helpful.

---

## Status of standing asks

| | Ask | Status | Notes |
|---|---|---|---|
| 1 | Resign output (`move_index: -1`) | shipped in `20a825f` | Working. Used 0 times in c99da9 so far; conservative trigger is correct. |
| 2 | Drop confidence + alternative_move_index from response schema | shipped in `20a825f` | Working. |
| 3 | Drop confidence calibration paragraph from static header | shipped in `20a825f` | Working; freed ~600 chars. |
| 4 | Move NOTATION line to static rules block | shipped in `20a825f` | Working. |
| 5 | Edit 4 (truncate PRIOR REASONING to move text only) | HELD | Re-test after v2 LoRA training; do not ship this cycle. |
| 6 | Semantic prompt versioning (`promptLayoutVersion: hybrid-v1.x`) | NOT shipped | Re-raised this cycle. `20a825f` is a new template under the same `hybrid-v1` label; we have had to compute and track templateHash mappings manually. |
| NEW | DRAW TIMELINE block (Change 1) | proposed | This cycle. |
| NEW | STOCK CYCLE counter (Change 2) | proposed | This cycle. |
| NEW | Timeline-aware strategy heuristic (Change 3) | proposed | This cycle. |

---

## Evidence appendix: the 97.9 percent measurement

Per-session post-recycle draw-rationale analysis (audit script details in `docs/reports/20260527_prompt_v1_2_candidate_spec.md`):

| session tail | seed | outcome | mv | first recycle @ idx | post-recycle draws | discovery-language | percent |
|---|---|---|---:|---:|---:|---:|---:|
| fef598 | 4208249311 | incomplete | 355 | 41 | 236 | 236 | 100.0 |
| 9f22c2 | 885853979 | incomplete | 297 | 48 | 165 | 165 | 100.0 |
| c98e59 | 3976147086 | incomplete | 281 | 40 | 159 | 159 | 100.0 |
| 359e1f | 2114734045 | incomplete | 277 | 41 | 159 | 153 | 96.2 |
| a0d9fe | 1236111563 | incomplete | 232 | 53 | 137 | 137 | 100.0 |
| 359e1f | 2114734045 | incomplete | 216 | 41 | 112 | 108 | 96.4 |
| adf71b | 114946100 | incomplete | 210 | 41 | 44 | 40 | 90.9 |
| db1804 | 2967897202 | stalled | 210 | 45 | 35 | 34 | 97.1 |
| 6459d9 | 2325626768 | stalled | 127 | 40 | 32 | 31 | 96.9 |
| 010e01 | 3263196305 | **won** | 170 | 63 | 13 | 6 | **46.2** |

Detection regex: `\b(reveal|discover|uncover|find|introduce|new card|might find|see what|might reveal|unlock|unknown|search for|hunt for|look for)\b` against `decision.reasoning` of post-recycle draw_card interactions.

Two representative quotes (both `gemma-4-31b-it` on hybrid-v1, in cycle 2 of their respective games):

**Failed session db1804 ti=209 (speculative)**:
> "The board is currently in a deadlock regarding the reveal of Column 7. The only way to break this deadlock is to obtain the 8C or 9D from the stock. Obtaining the 8C would allow the (7H, 6S, 5H, 4C) sequence to be moved, exposing the 8S. Obtaining the 9D would allow the (8S, 7H, 6S, 5H, 4C) sequence to be moved if a black 10 (like the TS in Column 2 or TC in Column 6) is available. Since no other tableau moves reveal hidden cards or create the necessary slots for the 8S, drawing from the stock is the only productive action."

**WON session 010e01 ti=163 (deterministic)**:
> "The only way to progress is to draw the 9C from the stock. Once the 9C is placed on the Clubs foundation, it will trigger a chain reaction: TC (Col 7) moves to foundation, then JC (Col 1), then QC (Col 4), which finally exposes KD (Col 4) to move to the Diamonds foundation, and lastly KC (Col 6) to the Clubs foundation. Moving Kings or sequences between empty columns is unnecessary as there are no hidden cards to reveal."

The failed model is HYPOTHESIZING what card it wants to find ("such as", "or", "if"); the winning model has DEDUCED what card is actually in the stock and traced the consequences. The same model is capable of both modes. The difference is whether the prompt makes the relevant stock-contents information legible.

---

## Summary

Three small changes. One block rename + format. One inline counter. One heuristic rewrite. Net prompt size is still smaller than v1.0. No response-schema break. Rule-symmetric to human play. Targets a documented 97.9 percent failure pattern in the existing corpus. Implementation effort on the harvester side is small; verification is mechanical.

Ready to discuss any of the three independently if you want to ship a subset first. Our preference is to ship all three together, because Change 3 (the heuristic rewrite) without Changes 1 and 2 would just confuse the model (it would reference a DRAW TIMELINE that does not exist).
