# Next change request to harvester team

**Date:** 2026-05-27
**From:** Chayut / dataset side
**Re:** Prompt v1.2: close the post-recycle "discovery" reasoning loophole. Two changes: one block rename + format change, one inline counter. No heuristic edits, no new response fields.

**Design principle for this ask**: the prompt's job is to render the state and observations a human player would have through memory. It is NOT to inject decision logic, heuristics, or thinking procedures. That is the LLM's training's job. Every change below is information that a human player at the table can observe; nothing prescribes what the model should do with it.

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

3. **The ask** (two small changes, no schema break):
   - **DRAW TIMELINE block** replaces `SEEN IN WASTE THIS CYCLE`. Single linear render of stock + waste in draw order, with `{NOW}` marker around the current waste top and `???` placeholders for unobserved stock positions (only possible in cycle 1).
   - **CYCLE counter** added to the existing STOCK line as `STOCK: N cards   CYCLE: K   WASTE top: X   recycle stock: yes/no`.

4. **Re-raise**: semantic prompt versioning (`promptLayoutVersion: hybrid-v1.2`) was asked last cycle and not shipped. The build `20a825f` ships a new template under the same `hybrid-v1` label, distinguished only by `promptTemplateHash`. Please bump on this cycle's change.

5. **Not asking this cycle (deliberately)**:
   - **No strategy-heuristic rewrites of any kind.** An earlier draft proposed a "drawing is reasonable ONLY IF the timeline shows..." rewrite of the existing STRATEGY GUIDANCE bullet. Dropped on the principle that the prompt should close information gaps, not prescribe thinking procedures. The model's training is what teaches it to use the timeline.
   - Tightening the resign trigger. Per our reframe (resign on a winnable deck is failure, not success), the existing conservative trigger is correct.
   - Re-introduction of edit 4 from last cycle (PRIOR REASONING truncation). Still on hold.
   - State-repetition annotation. Same scope as last cycle.

**Separately requested out-of-band**: drop the existing "Drawing from the stock is reasonable when no productive tableau/foundation move exists" bullet from STRATEGY GUIDANCE. This is a deletion (no replacement), aligned with the same principle. Requested manually by Chayut; not part of this written ask. If your team prefers to bundle it into this cycle for atomicity, it is welcome here.

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

## The two changes (precise specifications)

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

---

## Prompt-budget impact

| Block | v1.1 chars (typical) | v1.2 chars (typical) | Delta |
|---|---:|---:|---:|
| SEEN IN WASTE THIS CYCLE | ~70 | 0 (removed) | -70 |
| INTERPRETING DRAW TIMELINE paragraph (in RULES) | n/a | ~580 | +580 |
| DRAW TIMELINE data line | n/a | ~90 | +90 |
| STOCK line CYCLE field | n/a | ~10 | +10 |
| **Net** (this ask) | | | **+610** |

The v1.1 cleanup (drop confidence + alternative_move_index + calibration paragraph) freed about 1200 chars. v1.2 as proposed here spends 610 of that. Net prompt is still about 590 chars SMALLER than v1.0. If the out-of-band heuristic deletion lands in the same cycle, net savings grow by another ~95 chars.

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
| (out-of-band) | Drop existing "Drawing from stock is reasonable when..." heuristic from STRATEGY GUIDANCE | manually requested | Pure deletion, no replacement. Welcome to bundle into this cycle. |

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

## Evidence appendix B: token-budget analysis across three same-seed attempts

Seed `2967897202` now has three teacher runs on file (`gemma-4-31b-it`, temperature 0.3), one per prompt template generation. Same model, same deck, three different prompts. The token-usage trajectories show the v1.0 to v1.1 change is not just a prompt size delta; it changes the model's thinking distribution.

### Per-session totals

| Session | Build | Prompt | Turns | Wallclock | Thought tokens | Total tokens | Outcome |
|---|---|---|---:|---:|---:|---:|---|
| `…044461` | `7f01833` | json-format | 164 | 4.88 h | 542,209 | 929,146 | won 52/52 in 194 moves |
| `…db1804` | `de7dc06` | hybrid-v1.0 | 115 | 4.06 h | 464,324 | 744,746 | stalled at fc=15 |
| `…c99da9` | `20a825f` | hybrid-v1.1 | 111 | 3.24 h | 373,002 | 612,348 | stalled at fc=15 |

### Thought-token trajectory (20-turn rolling means)

```
ti range         WON     v1.0    v1.1     interpretation
  0-19         1,793    2,188   1,305     v1.1 starts cheapest
 20-39         1,868    3,248   1,466
 40-59         2,333    4,258   1,845     v1.0 already thinking hard
 60-79         2,675    4,366   1,962
 80-99         4,955    3,573   3,368     WON spikes at a hard decision
100-119        4,884    2,728   5,696     v1.1 SUDDEN spike to 5.7K
120-139        4,383    5,939   6,376     v1.0 doom-loop active
140-159        2,948    5,146   6,361     v1.1 sustained 6K+
160-179        4,179    5,721   6,399
180-199        2,198    5,215   4,558     WON finalising the win
```

### Three distinct thinking styles

- **WON** (json-format): variable, responsive to game complexity. Spikes at hard decisions (ti 80-99) then recovers as the path forward becomes clear. Thinks hard where it matters, easy elsewhere. Earns the win.
- **STALL v1.0** (hybrid-v1.0): heavy thinking from the start. The 600-char confidence-calibration paragraph (since dropped in v1.1) appears to have anchored the model toward uniformly careful reasoning. Never had an "easy phase".
- **STALL v1.1** (hybrid-v1.1): adaptive thinking. Efficient through ti 99 (~50-100 sec/turn, ~1500-3400 thought tokens). At ti 100+, thought tokens roughly doubled and duration nearly tripled (50 sec/turn to 190 sec/turn). The model correctly recognized "stuck" and ramped up. **But the extra thinking found nothing**, because the prompt does not surface the information needed for the deduction.

### What this contributes to the v1.2 case

The v1.1 model is willing to spend ~6,000 thought tokens per turn on the doom-loop. Across 80 doom-loop turns, that is ~480,000 wasted thought tokens per session. The model is hypothesizing "8C might be in the stock" turn after turn even though the `SEEN IN WASTE THIS CYCLE` block at the same turn lists the stock contents and 8C is not among them. The information is in the prompt but unreachable through the current label and format.

This is not a model-capacity problem. The same model in the WON run (lower per-turn budget!) successfully reasoned chains like "draw 9C, then TC, then JC, then QC, then KD, then KC". The capability is there. The v1.2 ask is to surface the stock state in a form the model can actually integrate into that reasoning.

If v1.2 lands and even half the v1.1 late-game thought-token budget converts to productive reasoning (e.g. "8C is not in the timeline, must commit to a tableau move"), the marginal value per session is large in compute and potentially decisive in outcome.

---

## Summary

Two small changes, pure information. One block rename + format. One inline counter. No heuristic edits, no decision-rule injection. Net prompt size is still smaller than v1.0. No response-schema break. Rule-symmetric to a perfect-recall human player. Targets a documented 97.9 percent failure pattern in the existing corpus by closing the information gap the model needs to reason deterministically; what the model does with the new information is left to its training.

The two changes go together (Change 2 references the cycle concept introduced alongside Change 1). Ready to discuss either independently if your team prefers to stage them.
