# Dead Deal Analysis: Seed `4153653383`

Date: 2026-05-20
Audience: harvesting team, data pipeline, distillation project
Status: case study

## Scope

Two new advisor exports landed on 2026-05-20, both build `ec38c03`, both on the
same seed `4153653383`:

- `/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-481557-1779218621204.json`
  -- session `019e3fd1-9770-79ce-a8d3-78e0b5481557` (imperfect info, the normal
  game), 200 interactions, turns 229 to 360.
- `/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-7381e0-1779218617358.json`
  -- session `019e3fcb-2113-7004-ba71-3cfcbb7381e0` (perfect information),
  200 interactions, turns 141 to 254.

Both look stuck. This document explains why, drawing on a reachability search
over the perfect-info game and a behavioural read of the model's reasoning.

## Headline

The same-seed pair is a perfect-info versus imperfect-info head-to-head. Both
games died at exactly the same place: foundations stuck at 2 (AH, AC),
18 face-down cards in the tableau, completion 4%. Across the two files, not
one face-down card was ever flipped and no further card ever reached a
foundation.

The cause is threefold:

1. The deal is brutal. The position handed to the perfect-info model at turn
   141 is a *provably total deadlock* -- no foundation move is reachable from
   that state under any sequence of legal moves.
2. The model does not recognise terminal positions. It executes a confident
   multi-turn plan around an *illegal* move (placing a tableau card onto the
   waste).
3. There is no stall auto-terminator, so the model is allowed to burn 200
   API calls per game on a board with a 0% win chance.

Per-seed: two independent sessions of the same deal reached the same dead
position. This is strong evidence about the deal, not just one bad playthrough.

## Method

The perfect-info export carries `faceDown` identities on every tableau column
and `drawPileOrder` for the stock, so the full 52-card state is recoverable.
The waste-and-stock cycle is 16 cards and never loses a card (the model never
plays a `discard_to_tableau` in either game), so its membership is invariant.
At turn 152 a fresh recycle exposes the full 16-card ring in
`drawPileOrder`.

To check winnability I ran a complete reachability search from the turn-141
state under a *generous over-approximation*: I let the search play any of the
16 cycle cards at will, ignoring the draw-3 ordering constraint. This relaxes
the real game's rules. If the relaxation is unwinnable, the real game is too.

Engine rules in the search:

- Tableau builds down, alternating colour. Valid runs move as units. Only a
  King (or run headed by a King) may occupy an empty column.
- Foundations build up by suit from the Ace.
- Cycle cards may be placed on a valid tableau target or sent to a foundation.
- Face-down cards flip automatically when exposed.

## Result of the search

```
nodes explored: 5749        distinct states: 1656
RESULT: PROVEN UNWINNABLE -- full reachable space exhausted, no win
best foundation cards reachable from turn 141: 2 of 52
```

Zero additional cards can ever reach a foundation. Not "hard to win" -- the
state space contains no foundation move at all. The empirical data confirms it:
across turns 141 to 254 the 18 face-down cards are byte-identical, and across
turns 229 to 360 in the imperfect-info twin the same signature holds.

Because the real draw-3 game is strictly more constrained than my
over-approximation, the real position is unwinnable too.

## Why it is locked: the interlock

The unrecovered Aces are AD (buried in col3 under the 7S) and AS (col6, under
three cards). All progress depends on revealing a face-down card. Every column
is mutually blocked:

- **col3**: `7S` sits on the AD. 7S can only move onto a red 8 in the tableau.
  The red 8s are `8H` and `8D`, both in the stock-and-waste cycle. The cycle
  is not in the tableau.
- A cycle card can only *enter* the tableau by landing on a black 9 (`9C` or
  `9S`). Both black 9s are face-down (col5, col7).
- Digging col5 or col7 to those 9s requires moving their face-up runs onto a
  black 8 (`8S` or `8C`). `8S` is pinned under `7H`; `8C` is buried face-down
  in col6.
- No column is empty and none can ever be emptied, so the cycle Kings (`KC`,
  `KD`) cannot be placed and the buried `KS` cannot head a pile.

Every unlock depends on a card that is itself locked. Classic Klondike
interlock, perfectly closed.

## The model behaviour

The model is not flailing. It is **confidently executing an illegal plan**.
Its own reasoning at turn 143:

> "Once recycled, the QD will be drawn first, followed by the 8D. The 8D can
> then be used to move the 7S in column 3, revealing the AD... This will create
> a chain of progress."

This plan is rules-illegal. You cannot move a tableau card (7S) *onto a waste
card* (8D). The waste only hands cards out; it is never a destination. For 7S
to land on 8D, the 8D must be face-up in a tableau column -- and as shown
above, 8D can never enter the tableau anyway. The model's strategy rests on a
move that does not exist, and chases it through 100 draws and 6 recycles.

Across the 105 success decisions in the perfect-info game:

| Signal | Value |
|---|---|
| `draw_card` chosen | 94 |
| `recycle_stock` | 6 |
| Tableau moves | 5 -- oscillating, e.g. `6S+3` col2 to col5 then col5 to col2, net zero |
| Confidence: min / mean / max | 0.80 / 0.91 / 1.00 -- never once below 0.80 |
| "deadlock" / "stall" in reasoning | 19 / 40 |
| "lost" / "unwinnable" / "give up" | 1 / 0 / 0 |
| Mean call duration | 113 s (max 208 s) |

The model names the deadlock 59 times but never concludes the game is lost,
never drops its confidence, and never stops. It treats a terminal dead end as a
temporary obstacle it has a plan for.

The imperfect-info twin (`481557`) shows the same family of behaviour. It
cannot see the hidden cards, so it draws forever searching for the missing
Aces. The two oscillating tableau moves in that file also net to zero
(`6C+3` col2 to col5, then col5 to col2).

## Answers to the three hypotheses

- **Bad seed / difficult deck?** Yes, decisively for the position from turn 141
  onward, and very likely for the original deal. Two independent sessions of
  the same seed both died at foundations equal to 2 with 18 face-down. The
  original deal (turn 0) is not directly reproducible (the harness RNG is not
  the repo's), but the joint evidence is unambiguous: this deal is at minimum
  savage and probably unwinnable from start.
- **Model not capable?** Not of winning *this* -- nothing could. The model
  does have two real capability gaps: (a) an incorrect rules model in which a
  waste card can be a tableau destination, and (b) no recognition of terminal
  dead ends. The loss is the deck; the wasted 200 calls are the model.
- **Deadlock cause:** the closed interlock above, plus a model that plans
  around an illegal move instead of resigning.

## Implications

For the harvest team:

1. The auto-terminator (handover P1 #3) is the cost-critical fix. These two
   files cost roughly 400 advisor calls on a position with 0% win chance, the
   second instance after session `9229e2cc3adc`.
2. The rules prompt should state explicitly that the waste is never a move
   destination. The model is generating multi-turn plans that depend on a
   non-existent move; tightening the rules block costs nothing and removes a
   confident-but-illegal failure mode.
3. Seed-replay across configs is a great experiment design (perfect-info vs
   imperfect-info on the same deck). On this seed, perfect information did
   not save the deal -- a useful baseline result, worth keeping.

For the data pipeline:

4. The ingest pipeline now needs a stall filter so doom-loop decisions stay
   in the store and publish set as a baseline, but do not pollute the LOCAL
   training set. The filter is implemented in `scripts/ingest_exports.py`
   alongside this document. Without it these 239 stalled rows poison the
   `data/dataset/training.jsonl` (71% of the local set was poison before the
   filter).

For the distillation:

5. Every stalled row teaches E2B two defects at once: the illegal "move onto
   the waste" plan, and infinite drawing on a dead board. Filtering them out
   is not optional. Keeping them in the publish set is fine; they remain as a
   research artefact for stall and dead-end detection.

## Files referenced

- The two exports listed at the top of this document.
- `GAME_PROGRESS_METRIC_2026-05-19.md` -- progress metric and stall test.
- `HARVEST_TEAM_HANDOVER_2026-05-19.md` -- live priority list, with the
  auto-terminator at P1 #3.
- `GEMMA4_E2B_SCHEMA_AUDIT_2026-05-19.md` -- v4 schema audit.
- `scripts/ingest_exports.py` -- the ingest pipeline, now stall-aware.
- `data/DATASET_NOTES.md` -- pipeline layout.
