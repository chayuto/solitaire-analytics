# Dead-deal signature catalogue

The user's #1 ask of this skill: **be the early-warning signal**. Every
additional turn on a dead deal burns retry budget under the current
~75% provider error rate. After session 502768 ran ~150 retry calls on
a structurally lost position, the user made it explicit — flag the dead
deal in the first analysis pass, not the postmortem.

This catalogue lists the signatures, with worked examples from the corpus.
Check them in order; the first hit is usually enough.

## 1. Missing low foundation card buried with no reveal path

The classic dead deal. The board needs an Ace or 2 of some suit to make
*any* foundation progress in that suit. If that card is face-down in a
column whose face-up tower can't be cleared, the suit is locked.

**Check:**
1. Find suits whose foundation is empty (`foundations.<suit> = null`) or
   needs only A/2/3.
2. For each such suit, search `seenDrawPileCards` and every column's `faceUp`
   for the missing rank+suit. If absent, the card is face-down somewhere.
3. For each column with `faceDownCount > 0`, check whether the face-up tower
   above it can be peeled off via valid tableau moves to another column or
   to a foundation. The tower is unpeelable when:
     * the bottom face-up card is, e.g., a 5 of a colour with no 6 of the
       opposite colour in play and no path to the foundation
     * an empty column is needed and none is available / can be created
4. Sum it up: if no column with face-down cards can be unburied, the missing
   low card is permanently locked → dead deal.

**Real example — 1779012106344 (legacy export, fc=1, fd=16, 83-turn plateau):**
foundations stuck at clubs=AC, hearts/diamonds/spades empty. Across 200 turns
the missing Aces never surfaced. Honest-hunt at first, but with no reveal
path after the early-game peels, the deal was permanently locked.

## 2. Small face-down concentrated in 1–2 columns, no reveal path

When `faceDownTotal` is low (≤8) but stuck for many turns, it's almost
always a structural lock — the few hidden cards are pinned under sequences
that can't be moved coherently.

**Check:**
1. Identify which 1–2 columns hold all the face-down cards.
2. For each, confirm there's no in-play card that can be placed on the
   bottom face-up card to chain it elsewhere.
3. Confirm no empty column is available (and none can be created with the
   current foundation state).

**Real example — 502768 (fc=12, fd=6, 26-turn plateau):**
Six face-down cards concentrated in cols 5 and 7 with face-up bottoms that
required a black 8 (not in play) to relocate. drawPileCount=0,
canRecycleStock=True — the agent had been recycling the stock for 26 turns
looking for cards that were already known not to be there.

## 3. Stock fully known, no productive waste card

If the agent has seen every stock card across multiple recycles AND none of
the seen cards can productively place on a tableau or foundation, drawing
further is futile.

**Check:**
1. `len(seenDrawPileCards) + drawPileCount ≈ 24 - foundationCards_drawn_from_stock`
   suggests the stock is fully observed.
2. Cross-reference each `seenDrawPileCards` entry against the current
   tableau and foundation tops. If none can place anywhere useful (and
   discardTop also can't), the draw mechanic has nothing left to offer.

**Real example — 5061b71279a3 (Aces hunt):**
seenDrawPileCards had cycled through ≥3 full passes. The missing AH and AS
never appeared. With both Aces face-down and no peel path, the session
was hunting cards that couldn't come.

## 4. Foundation suit blocked by buried chain

The next-needed foundation card (e.g. `5D` when foundations.diamonds = `4D`)
exists face-up on the tableau but is buried under a sequence longer than
the available empty-column slack can disassemble.

**Check:**
1. For each foundation needing rank R+1 of suit S, find R+1 of S on the
   tableau.
2. Count the cards on top of it.
3. Empty columns provide K-headed sequence parking. If `empty_columns ×
   max_movable_run < cards_on_top`, the chain can't be unbundled.

## Distinguishing dead-deal from behavioural doom-loop

This matters because the fixes are different. A behavioural doom-loop on
a *winnable* board needs a prompt fix (anti-oscillation, reveal-priority
rules). A dead deal needs a harness-side stall auto-terminator OR a
prompt-side resign rule — no prompt rephrasing solves an unsolvable board.

| Signal | Behavioural | Dead-deal |
|---|---|---|
| `recentMoves` shows mostly tableau shuffles | yes | sometimes |
| `recentMoves` shows mostly draws + recycles | sometimes | usually |
| Named card the agent wants is in `faceUp` or `seenDrawPileCards` | yes (look harder) | no (truly unreachable) |
| Plateau length | varies, often 30–80 | often shorter at detection (15–30) |
| Agent's `boardAnalysis` self-aware about deadlock | sometimes | usually (it sees the lock but has no resign action) |
| Solver `--samples 5` solve rate | high | zero |

Honest-hunt is a third category: agent genuinely names a specific missing
card it's drawing for, makes varied moves (not just oscillation), and the
plateau hasn't yet hardened. Watch for 20 more turns before killing.

## Calibration history

These signatures came from analysing the corpus on 2026-05-20/21:
- 645d03 (5C/4D loop on solvable board) → behavioural
- 73fd85 (TS/9D loop with 5 dominating alternatives) → behavioural
- 502768 (recycling on locked-low-card structure) → dead-deal
- 5061b71279a3 (Aces hunt across recycles) → honest-hunt → dead-deal
- 29a7f5 (same as 645d03 baseline, comparison arm) → behavioural

Update this file when a new failure pattern shows up in the corpus.
