# Ingest triage: `solitaire-game-a11e74-1780215783804.json`

## Verdict (answer to "should I log this as a loss?")

**No. Do NOT log this as a loss.** Record class: **PENDING-SNAPSHOT** (not a terminal record).

The filename prefix is `solitaire-game-`, which is a mid-play snapshot, not a
terminal record. Its `gameWon: false` means "not won *yet*", not "lost". You do
not record any outcome from this file: no win, no loss. Ingest it (the operator
ingests the full stream), but leave the outcome uncatalogued until the terminal
`solitaire-win-a11e74-*.json` arrives.

This is, in fact, the canonical proof-case for the snapshot trap: session
**#a11e74** is the session where a 76.92% `gameWon: false` snapshot preceded the
actual win. Logging it as a loss would record a loss for a game that wins.

## Attribution (model / build / seed)

Read from the file's `aiConfig` and top-level fields:

- **Session:** #a11e74 (full `019e7aaa-0dfa-722b-9fd9-03ff87a11e74`)
- **Model:** `gemma-4-31b-it` (provider `gemini`, preset `standard`,
  `seeHiddenCards: false` so imperfect-information play)
- **Build:** `appCommit` 262774b, `appBuildTime` 2026-05-30T07:08:17Z
- **Seed:** 601852437
- **Prompt template:** no template hash in this file (state files do not carry
  one reliably; the hash lives in the `solitaire-ai-log-*` file, which is not in
  this drop)

`gemma-4-31b-it` is the default `TEACHER_MODEL`, so this session's interactions
are training-eligible in principle (subject to the stall filter). It is not a
26B / non-teacher cohort, so no training-mix exclusion applies on model grounds.

## Why this is a snapshot, not a loss (the reasoning)

The pipeline cannot make this call for you, and it gets it wrong if trusted:

- I ran the live classifier directly: `scripts/ingest_exports.py`'s
  `classify_file(doc)` returns **`win_record`** for this file, and the ingest
  loop then copies `gameWon` (`False`) verbatim into the manifest. To the
  pipeline, this snapshot is **indistinguishable from a terminal loss.** The
  only thing that tells them apart is the `solitaire-game-` filename, which the
  pipeline ignores and this skill keys on.

- The board state reads as a deep mid-game position, not a stalled dead end:
  - Foundations hold **40 of 52** cards (hearts up to 8, diamonds to J, clubs to
    J, spades to 10). 40/52 = 76.92%, which is exactly the reported
    `completionProgress`.
  - **0 face-down cards** remain across all 7 tableau columns (tableau is nearly
    emptied: column counts 3/4/2/1/0/0/1).
  - `drawPile: 1`, `discardPile: 0`, `recycleCount: 26`, `moveHistory: 461
    moves`. The game is still in active progress at export time
    (`autoPlayInProgress: false`, `replayMode: false`), not abandoned.
  - `initialBoardSetup` is present, so the deal is fully reproducible regardless.

  A board with everything unburied and 40 cards home is a near-win position, not
  a doom-loop carcass. That is consistent with the documented #a11e74 history
  where this exact snapshot was taken roughly an hour before the session won.

- The skill's read-only triage helper independently agrees. Its verdict on this
  file:

  > `>> PENDING-SNAPSHOT` — NOT terminal. Do NOT record an outcome (no win, no
  > loss) from this file. ... Await the terminal `solitaire-win-*` export before
  > cataloging.

## How to ingest it (the plan)

1. **Triage first (done).**

   ```
   .venv/bin/python .claude/skills/solitaire-ingest/scripts/triage_export.py \
       /path/to/solitaire-game-a11e74-1780215783804.json
   ```

   Result: 1 session, class PENDING-SNAPSHOT, model `gemma-4-31b-it`, build
   262774b, seed 601852437. Pass the `solitaire-ai-log-a11e74-*.json` alongside
   it if/when you have it, since that file carries the prompt-template hash and
   the full per-turn interaction rows (this snapshot has only a trailing
   `aiDecisionLog` sample, not the full log).

2. **Ingest the file (full-stream).** The operator ingests everything the
   harvester makes, so this snapshot does get ingested. It is safe to ingest a
   snapshot; what you must not do is record its outcome.

   ```
   mv solitaire-game-a11e74-1780215783804.json data/raw/   # mv, never cp
   .venv/bin/python scripts/ingest_exports.py              # idempotent, sha256-deduped
   ```

   Verify after: the file shows up as one new row in `data/index/manifest.jsonl`
   (it will be typed `win_record` with `gameWon: false` — that is the pipeline's
   known mislabel, harmless as long as you do not read an outcome off it), and
   `data/SUMMARY.md` file counts move by the expected amount.

3. **Cataloging / outcome — this is the important part.** Do **not** write a
   terminal verdict for #a11e74 from this file. Do not add it to
   `## Won sessions` and do not add it to `## Known doom-loop sessions`.
   - If you note it at all in `data/DATASET_NOTES.md`, mark it explicitly as a
     mid-game snapshot (76.92%, 40/52 foundations, 0 face-down) awaiting the
     terminal export, leading the entry with `#a11e74` and the full UUID once in
     parentheses.
   - When the terminal `solitaire-win-a11e74-*.json` later arrives, that file is
     canonical and supersedes this snapshot; the snapshot is then just an
     archived predecessor. (Per the documented history, that terminal file is
     `gameWon: true`, 100%, a win — but you confirm the outcome from that file,
     not from this one.)

4. **Win-rate accounting:** count this session toward the win rate only from its
   terminal `solitaire-win-*` file, never from this snapshot. Counting a
   snapshot as a loss is exactly what corrupts the within-version win rate.

## Hand-off

If this session is still **live** and the question were "kill or continue?",
that is the **solitaire-analyst** skill's job (winnability + doom-loop
diagnosis via its `check_winnability.py` / `load_export.py`), not this skill's.
Given the position here (40/52 home, board fully unburied), it reads as a strong
continue, but the terminal export is what settles it. This ingest skill's role
is only to bring the result into the corpus correctly, which here means:
**ingest yes, log-as-loss no, await the terminal win-file before cataloging the
outcome.**
