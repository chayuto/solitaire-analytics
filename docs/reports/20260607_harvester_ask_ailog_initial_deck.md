# Harvester ask: log the full initial deal inside the ai-log (`initialBoardSetup`)

**Date:** 2026-06-07 | **Target:** every client / model (applies to 31B and 26B) | **Nature:** logging-only. This is NOT a prompt change and is independent of the prompt-version track (v1.5 shipped, v1.6 in flight). It can ship with any build; it does not touch the prompt, the schema the model sees, or play behaviour.

## The ask (one line)

Add the complete turn-0 deal as a new top-level key, `initialBoardSetup`, to every `solitaire-ai-log-*.json` export, using the exact same object you already emit in `solitaire-win-*` and `solitaire-game-*` files.

## Why

1. **Trace completeness.** Today an ai-log records every decision but not the deck it was played on. Embedding the deal makes the ai-log a self-contained, replayable record: from the ai-log alone we can reconstruct the exact starting board and re-derive any later state.
2. **It is the only way we get the deck for LOSSES.** Won and abandoned games emit a `solitaire-win-*` / `solitaire-game-*` file that carries `initialBoardSetup`, but a plain loss produces only an ai-log, so the losing deck is currently unrecoverable (the seed is not reproducible to a deck on our side). This single field closes that gap.
3. **It unlocks analysis we cannot do now:** for any loss we could (a) solve the real board to classify dead-deal vs winnable-but-lost definitively, (b) compare won-vs-lost deck difficulty head-to-head, and (c) reconstruct exact boards for training-data work. All of this is blocked today solely because losses carry no deck.

## What to put inside (exact schema, reuse what you already emit)

Use the identical `initialBoardSetup` object already present in the win/game files, so there is one canonical schema and zero new computation. Verified shape (from 25 existing win/game files, 1300 cards):

```json
"initialBoardSetup": {
  "drawPile":   [ { "suit": "clubs", "rank": "9", "faceUp": false, "id": "clubs-9" }, ... ],   // the 24 stock cards, in draw order (index 0 drawn first)
  "discardPile": [],                                                                            // empty at deal
  "foundations": { "hearts": [], "diamonds": [], "clubs": [], "spades": [] },                   // empty at deal
  "tableau": [                                                                                  // 7 columns, bottom-to-top
    [ { "suit": "clubs", "rank": "J", "faceUp": true, "id": "clubs-J" } ],                       // column 1: 1 card, face-up
    [ { "suit": "diamonds", "rank": "8", "faceUp": false, "id": "diamonds-8" },
      { "suit": "hearts", "rank": "3", "faceUp": true,  "id": "hearts-3" } ],                    // column 2: 2 cards, top face-up
    ...                                                                                          // ... through column 7 (7 cards)
  ]
}
```

Card object: exactly four keys.
- `suit`: one of `"clubs"`, `"diamonds"`, `"hearts"`, `"spades"` (spelled out, lowercase).
- `rank`: one of `"A"`, `"2"`..`"9"`, `"10"`, `"J"`, `"Q"`, `"K"` (ten is the two-character string `"10"`, not `"T"`).
- `faceUp`: boolean.
- `id`: `"<suit>-<rank>"`, e.g. `"diamonds-10"`, `"clubs-A"`.

Critical requirements:
- It must be the **complete 52-card deal**: 24 in `drawPile` plus 28 in `tableau` (columns of size 1 through 7).
- It must carry the **true identities of the face-down cards** (that is the entire point); `faceUp` reflects the deal (each column's top card true, the rest false), but every card's `suit`/`rank` is the real one.
- `drawPile` order must be the **draw order** (index 0 is the first card drawn), matching the order cards surface via `draw_card`.

## Placement

Top-level key on the ai-log object (it currently has `exportedAt`, `appCommit`, `appBuildTime`, `session`, `count`, `interactions`), mirroring where it already lives in the win/game files. Nesting it inside the existing `session` object is also acceptable if that fits your schema better; we will read either location.

## Cost and risk

- **Size:** `initialBoardSetup` serializes to about 3.6 KB. A typical ai-log is already about 5.5 MB (it carries every turn's rendered prompt), so this is roughly a 0.06% increase. Negligible.
- **No new file, no new type.** This is one extra key on an existing artifact, not a new export. Our ingest reads the ai-log as a dict and will simply pick up the new key.
- **Additive and behaviour-neutral.** No prompt text changes, no change to what the model sees, no change to move selection. Purely a logging addition.

## Consistency notes

- The deal is constant for a session, so emit the **same** `initialBoardSetup` in every export of that session, including mid-game re-exports. (We dedup sessions, so repetition across re-exports is fine.)
- For sessions that also produce a win/game file, the ai-log copy will be redundant with it. That is harmless and gives a built-in cross-check.

## Validation (how to confirm it is correct)

- The 52 ids are unique and cover all suit/rank combinations exactly once.
- `len(drawPile) == 24` and tableau column sizes are `[1,2,3,4,5,6,7]`.
- For a session that also has a `solitaire-win-*`/`solitaire-game-*` file, the ai-log `initialBoardSetup` is byte-identical to that file's.
- The `draw_card` reveals over the game match `drawPile` in order (first draw equals `drawPile[0]`).

## Versioning

This is a logging change, not a prompt-template change, so it does **not** need a `hybrid-vX` bump or a new `promptTemplateHash`. The `appCommit` / `appBuildTime` will change as usual. We detect the field by its presence, and treat it as nullable for older logs that predate it.

## Effort

Trivial on the harvester side: the object is already constructed and written to the win/game exports; this just also writes it (or a reference to it) into the ai-log export. No new computation, no new file path.
