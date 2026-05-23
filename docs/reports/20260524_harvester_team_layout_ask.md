# Next change request to harvester team — prompt layout

**Date:** 2026-05-24
**From:** Chayut / dataset side
**Re:** Single next change — switch `CURRENT GAME (JSON)` to a hybrid ASCII layout
**Companion documents:**
- `docs/reports/20260523_prompt_format_ab_status.md` (the work-in-progress that produced this ask)
- `docs/reports/20260523_training_data_options.md` (pattern-specific decode-incoherence finding; oscillation 33% take-rate)
- `docs/reports/20260523_harvester_team_followup_ask.md` (previous cycle — closed with no change, T=0.3 baseline)

---

## TL;DR

1. **One ask: replace the `CURRENT GAME (JSON)` block** in the per-turn
   prompt with a hybrid ASCII layout — an ASCII tableau, numbered
   `RECENT MOVES` adjacent to numbered `LEGAL MOVES`, all other state
   compressed to one-line blocks. Preserves every field the JSON
   carries, just renders them in a shape the model can read at a glance.
2. **Why now**: 4–5× fewer tokens per turn, decision-relevant patterns
   (the model's just-made moves vs the moves the legal-moves list is
   offering) become visually adjacent instead of structurally split.
3. **What we are NOT asking** this round: don't touch `reasoningTrail`.
   We considered it; the data doesn't support a blanket drop.

---

## Status of standing asks

| | Ask | Status | Notes |
|---|---|---|---|
| 1 | Per-interaction `promptTemplateHash` + `promptTemplateFinalisedAt` | ✅ shipped | Working as expected. Build attribution unblocked. |
| 2 | Resignation output (`move_index = -1` → `outcome: resigned`) | ⏳ standing | Still queued; defer until layout change has had one harvest cycle. |
| 3 | Same-seed cross-build experiment on seed `3263196305` | ⏳ standing | Still queued; same reason. |
| ✨ NEW | This round: hybrid layout for `CURRENT GAME` block | ⏳ proposed | Detailed below. |

---

## This round's ask: hybrid layout

### The change

Replace the `CURRENT GAME (JSON):\n{ ... }` block in the current prompt
with the layout below. All fields currently inside the JSON object are
preserved — they're just re-rendered. The only field we propose to
**hold back from the per-turn prompt** is `reasoningTrail`, and we're
explicitly NOT asking for that yet (see "What we're not asking" below).

### Spec (before / after)

**Before** (current, abbreviated for readability):
```
CURRENT GAME (JSON):
{
  "notation": "Cards: rank then suit ...",
  "foundations": { "hearts": "4H", "diamonds": null, "clubs": null, "spades": "2S" },
  "tableau": [
    { "column": 1, "faceDownCount": 0, "faceUp": ["8H","7S","6D","5C"] },
    { "column": 2, "faceDownCount": 0, "faceUp": ["TD"] },
    ...7 entries...
  ],
  "discardTop": "KD",
  "drawPileCount": 16,
  "canRecycleStock": false,
  "legalMoves": [
    { "index": 0, "type": "tableau_to_tableau", "describe": "Move 7S plus 2 more from column 1 to column 5" },
    { "index": 1, "type": "draw_card",          "describe": "Draw the next card from the stock onto the waste" },
    { "index": 2, "type": "tableau_to_tableau", "describe": "Move JD plus 2 more from column 7 to column 3" }
  ],
  "metrics": { "foundationCards": 6, "faceDownTotal": 12, ... },
  "recentMoves": ["draw 6C","draw JC", ..., "move 9D col 3 -> col 7","draw KD"],
  "seenDrawPileCards": ["9H","5S","KS","JC","6C","KC","KH","TS","3C","7H","QH","2C","3D","5D","8C","8S"],
  "reasoningTrail": [...]
}
```

**After** (proposed, full):
```
NOTATION: rank+suit (A 2-9 T J Q K; H D C S). ?? = face-down.
In each column the top of the stack is the rightmost card.

FOUNDATIONS:   H: 4H   D: --   C: --   S: 2S
STOCK: 16 cards   WASTE top: KD   recycle stock: no

TABLEAU:
  col1: 8H 7S 6D 5C
  col2: TD
  col3: ?? QS
  col4: 7D 6S 5H 4C
  col5: ?? ?? 8D
  col6: ?? ?? ?? ?? JS TH
  col7: ?? ?? ?? ?? ?? QC JD TC 9D

RECENT MOVES (oldest -> newest; review before picking, do not undo your own work):
   1. draw 6C
   2. draw JC
   3. draw KS
   4. draw 5S
   5. draw 9H
   6. draw KD
   7. move JD col 3 -> col 7
   8. move TC col 3 -> col 7
   9. move 9D col 3 -> col 7
  10. draw KD

SEEN IN WASTE THIS CYCLE: 9H 5S KS JC 6C KC KH TS 3C 7H QH 2C 3D 5D 8C 8S

LEGAL MOVES (respond with the index of your chosen move):
  [0] tableau_to_tableau    Move 7S plus 2 more from column 1 to column 5
  [1] draw_card             Draw the next card from the stock onto the waste
  [2] tableau_to_tableau    Move JD plus 2 more from column 7 to column 3

PROGRESS: foundation=6/52, face-down remaining=12, completion=12%
```

### Layout rules (for porting)

- **TABLEAU**: one line per column. Face-down cards rendered as `??`
  space-separated; face-up cards rendered as their notation, also
  space-separated. Face-down come first in the line (deeper in the
  stack), face-up come after. Top of stack = rightmost token. Empty
  column renders as `  colN: <empty>`.
- **RECENT MOVES**: numbered list 1..N, oldest first, padded so the
  numbers right-align. We deliberately surface this immediately before
  `LEGAL MOVES` so the model sees what it just did against what it's
  being offered.
- **LEGAL MOVES**: numbered list, `[index]` + 24-char type column +
  describe text. Index is the canonical move identifier (response
  shape stays `{"final_decision":{"move_index": N, ...}}`).
- **FOUNDATIONS** / **STOCK** / **PROGRESS** / **SEEN IN WASTE** /
  **NOTATION**: one line each, plain English labels.
- Preserve **everything** the current JSON carries except
  `reasoningTrail` (see below). No information loss.

A reference Python renderer (`render_hybrid` in
`scripts/compare_prompt_formats.py`) ports straightforwardly to TS —
the layout rules above are the contract; the implementation language
doesn't matter.

### Why this is the right next ask

Three independent signals point at the layout as a meaningful lever:

1. **Token cost is objective and large.** Across six oscillation-sample
   snapshots, the hybrid layout was consistently **0.18× – 0.24×** the
   character count of the JSON form. Roughly 4-5× fewer tokens per turn
   for the same information. That's compute and latency back even before
   any decision-quality change.

2. **Decision-relevant patterns become visually adjacent.** Today the
   oscillation trap pattern (a `recentMove` whose exact reversal is in
   `legalMoves`) is split across ~30 lines of nested JSON — `recentMoves`
   sits near the bottom of the object, `legalMoves` sits in the middle.
   In the hybrid layout the numbered `RECENT MOVES` block sits
   immediately above the numbered `LEGAL MOVES` block; the same trap
   shows up as a one-glance comparison.

3. **Cross-model qualitative pick test confirms the direction.** Ran
   the same 3 oscillation samples × 2 formats through two LLMs as Gemma
   proxies:
   - **Claude Sonnet** (stronger model) picked identically on JSON and
     hybrid — too strong to reproduce the failure mode either way.
   - **Claude Haiku** (weaker model, closer to Gemma's capability) showed
     a clear sample-02 result: under JSON it picked the move that
     continues a JS/TH/9C/8D oscillation loop documented in the recent
     moves; under hybrid it picked a productive break-out move (move TC+5
     from col 3 to col 5). The format change was load-bearing for that
     pick.

   We do not yet have a Gemma 4 31B A/B (free-tier API quota was
   exhausted before we could run it). What we have is a directionally
   consistent qualitative read across multiple models, plus an
   objective token reduction.

### What we're NOT asking this round (re: reasoningTrail)

The `reasoningTrail` array — the prior turns' written justifications
that the current prompt includes inside `CURRENT GAME (JSON)` — was
flagged in a parallel investigation as a likely anchor: when the trail
encodes the strategy that started a doom-loop, presenting it
immediately before the next pick biases the model toward continuing
that loop.

The Haiku data complicated that story. On a different sample the
trail was *correct* (it pointed the model toward `draw`, which was
right); stripping it left the model to misread the board and pick a
useless move. So the trail is **high-variance** — helpful when prior
reasoning was right, harmful when wrong — rather than uniformly
harmful.

That's a real intervention worth considering, but not in this round.
We want to ship the layout change first, observe, and bring
`reasoningTrail` back as a separate single-ask once we have cleaner
data (and ideally a Gemma A/B) on whether to drop it, reposition it,
or re-label it.

The hybrid layout above **carries `reasoningTrail` forward** (you
can append it after `PROGRESS:` as a final block titled
`PRIOR REASONING (may be obsolete; verify against current state):` if
you want a soft de-emphasis, or just leave it identical to today's
position). Either is fine for this ask — the change is about the
board / moves layout, not the trail.

---

## What we should see

If the layout change is load-bearing:

- **Doom-loop turn counts drop.** Sessions that today reach 50+ turns
  of pure oscillation either don't enter the loop or exit it within a
  few turns when the visible adjacency of recent / legal moves makes
  the trap obvious.
- **Token usage per turn drops 4-5×.** This one is essentially
  guaranteed since the layout itself is smaller; the interesting
  question is what the model does with the saved attention budget.
- **No change in `boardAnalysis` quality.** We don't expect this to
  affect the reasoning section, only the action.

If the layout change is NOT load-bearing:

- Win rate is unchanged, oscillation count is unchanged.
- We've still saved 4-5× tokens for free, and we've ruled out layout
  as the bottleneck — which sharpens the case for the
  `reasoningTrail` follow-up ask.

Either outcome is informative.

---

## The experiment

Minimum-viable observation cycle:

- Switch the prompt block on the next harvest cycle. **No other
  changes**: same model, same temperature (0.3), same template SHA, same
  build, same seed pool.
- The dataset side will compare the resulting harvest against the
  immediately prior cycle on:
  - Win rate
  - Mean plateau / doom-loop turn count
  - Per-session oscillation count (recentMove reversal-pick rate)
  - Average prompt token count (sanity check on the 4-5× saving)
- Two-week decision: keep, revert, or escalate to the
  `reasoningTrail` ask.

---

## Open coordination questions

1. **Are you happy with the layout spec above**, or do you want to
   adjust formatting choices (e.g. spacing, label wording, where
   `SEEN IN WASTE` sits)? The substance is "ASCII tableau + adjacent
   numbered moves"; the typography is negotiable.
2. **Do you want a TS port of the renderer**, or is the Python
   reference + the layout rules enough? We can produce TS if it
   speeds you up.
3. **Confirm we're keeping `reasoningTrail` in the prompt as-is this
   round** (not dropped, not relocated). We can revisit it as a
   separate ask later.
4. **Will the new `inferenceParams` field land on the same cycle as
   this layout change?** If so, the dataset side will key the
   comparison report on (`promptTemplateHash`, `inferenceParams`,
   layout-version) so we can attribute lift cleanly.

---

## Cadence

Same as last round: **one ask, one observation cycle, one decision.**
Asks 2 (resignation output) and 3 (same-seed cross-build) remain on
the queue and will return as single-ask documents after this cycle
concludes.

---

## Where to find the evidence

- `docs/reports/20260523_prompt_format_ab_status.md` — the working
  document; includes the Haiku/Sonnet pick table, subagent qualitative
  read, and what's still blocked on Gemma quota.
- `data/dataset/demos/prompt_format_compare/sample-*/` — six concrete
  before/after pairs (`original.txt` / `hybrid.txt`) from real corpus
  snapshots. Open any pair side-by-side to see the format change on a
  real prompt.
- `scripts/compare_prompt_formats.py` — reference renderer
  (`render_hybrid` function). Layout rules above are the contract;
  this is one possible implementation.
- `docs/reports/20260523_training_data_options.md` §
  "Pattern-specific decode-incoherence" — the 33% oscillation
  take-rate measurement that motivated this whole line of work.
