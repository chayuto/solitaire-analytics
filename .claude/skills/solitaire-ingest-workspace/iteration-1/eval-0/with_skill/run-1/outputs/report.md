# Ingest triage report: #62f09b

## Triage verdict (read this first)

**#62f09b -> TERMINAL-WIN -> INGEST.** The game is over and won, so there is no
kill-or-continue decision and nothing to hand to the analyst. Ingest both files
full-stream and catalog the session under `## Won sessions` in
`data/DATASET_NOTES.md`.

One session, two files, both terminal-consistent. No heterogeneity, no pending
snapshot, no non-teacher model. Clean win.

## Per-session attribution

Both dropped files share session id `019e7d96-d8d9-7317-985a-e73b1c62f09b`
(`gameSessionId` in the win-file == `session.sessionId` in the ai-log), so they
are confirmed to be one session.

| Field | Value | Source |
|---|---|---|
| Session | `#62f09b` (full `019e7d96-d8d9-7317-985a-e73b1c62f09b`) | both |
| Model | `gemma-4-31b-it` (provider `gemini`, preset `standard`, temp 0.3) | `aiConfig` / ai-log |
| Build | `f5c3870` | both (`appCommit`) |
| Build time | `2026-05-31T10:27:59.429Z` | both (`appBuildTime`) |
| Prompt template | `hybrid-v1.3`, hash `7d9ecda4cb415ec2335b3e970421d297730773f541b523666332dd024e9772bb` (finalised `2026-05-28`) | ai-log interactions |
| Seed | `3590201206` | both |
| Record class | TERMINAL-WIN | win-file `gameWon: true` |

Attribution notes:

- **Model is the default teacher** (`gemma-4-31b-it`), so these rows land in the
  default training set; the `TEACHER_MODEL=gemma-4-31b-it` filter does not
  exclude them. No 26B/MoE or Gemini-proper traces in this drop.
- **Template is stable across the whole session.** All 232 ai-log interactions
  carry the single hash `7d9ecda4...` / version `hybrid-v1.3`; there was no
  mid-session template swap, so the entire session is attributable to one prompt.
- The ai-log `exportedAt` is `2026-05-31T19:06:37.973Z`; the win-file has no
  `exportedAt`. The ai-log timestamp (`...397973`) is ~70s after the win-file
  timestamp (`...328065`), consistent with the log being flushed just after the
  terminal record on the same finished game.

### Why TERMINAL-WIN (not a snapshot, not a verify)

The deciding file is `solitaire-win-62f09b-1780254328065.json` (a
`solitaire-win-*`, which is a terminal record by construction, unlike a
`solitaire-game-*` snapshot). Its terminal state is a fully solved board, so it
passes the TERMINAL-WIN check (`progress == 100` and `faceDown == 0`) with no
VERIFY caveat:

- `gameWon: true`, `completionProgress: 100`
- `foundations`: hearts 13, diamonds 13, clubs 13, spades 13 (all 52 cards up)
- `faceDown: 0`, `drawPile: 0`, `recycleCount: 5`
- `moveHistory`: 259 moves; `initialBoardSetup` present (full deck, reproducible)
- ai-log `session`: `outcome: "won"`, `finalProgress: 100`, `moveCount: 259`
- ai-log artifacts: 232 interactions, 200 `success` / 32 `error`; max successful
  `turnIndex` 258. (The ~14% error rate is the known background
  provider-unavailable/timeout pattern; not flagged as a concern per standing
  guidance.)

The 259 `moveCount` (win-file `moveHistory`) vs 232 ai-log interactions is the
normal gap between applied game moves and model interactions (multi-move
decisions, retries, auto-play), not a discrepancy.

## Ingest plan

Full-stream ingest (the operator ingests everything the harvester makes, which
keeps the within-version win rate true rather than selection-biased). Both files
go to `data/raw/` and the pipeline is run idempotently.

### Files -> destination

| File | Becomes (manifest `type`) | Destination |
|---|---|---|
| `solitaire-win-62f09b-1780254328065.json` | `win_record` (259 moves, `gameWon: true`) | `data/raw/` |
| `solitaire-ai-log-62f09b-1780254397973.json` | `ai_log` (232 interactions, teacher rows) | `data/raw/` |

`classify_file` (`scripts/ingest_exports.py:203`) keys on shape: the ai-log
becomes `ai_log`, and the win-file becomes a `win_record` because it carries a
`moveHistory`. For this session that label is correct (it really is terminal);
the snapshot-vs-terminal trap does not bite here because there is no
`solitaire-game-*` file in the drop.

### Commands I WOULD run (NOT run in this dry run)

```bash
# raw exports are gitignored; only manifest/store/dataset/publish are tracked
mv /Users/chayut/Downloads/solitaire-win-62f09b-1780254328065.json \
   /Users/chayut/Downloads/solitaire-ai-log-62f09b-1780254397973.json \
   /Users/chayut/repos/solitaire-analytics/data/raw/

.venv/bin/python scripts/ingest_exports.py   # idempotent, sha256-deduped
```

### Verification I WOULD do after ingest

- `data/index/manifest.jsonl` gains 2 rows: one `type: win_record`
  (file `solitaire-win-62f09b-1780254328065.json`, `moves: 259`,
  `gameWon: true`) and one `type: ai_log`
  (file `solitaire-ai-log-62f09b-1780254397973.json`).
- `data/SUMMARY.md` file counts move by exactly +1 ai_log and +1 win_record;
  the `gemma-4-31b-it` teacher decision count rises by this session's rows.
- Pipeline also refreshes `data/store/`, `data/dataset/`, `data/publish/`.

### Catalog location

`data/DATASET_NOTES.md` -> **`## Won sessions`** (TERMINAL-WIN routing). One new
entry, leading with `#62f09b`. No same-seed experiment block applies (seed
`3590201206` is not one of the tracked validation seeds in this drop). Do not
reformat surrounding entries.

## Draft DATASET_NOTES.md entry (`## Won sessions`)

> - Session `#62f09b` (full `019e7d96-d8d9-7317-985a-e73b1c62f09b`), seed
>   `3590201206`, model `gemma-4-31b-it`, app build `f5c3870`
>   (built 2026-05-31T10:27:59Z), prompt template `hybrid-v1.3`
>   (hash `7d9ecda4...`). Ingested via
>   `solitaire-win-62f09b-1780254328065.json` (terminal win-record, 259 moves,
>   full deck in `initialBoardSetup`) plus
>   `solitaire-ai-log-62f09b-1780254397973.json` (canonical interaction log, 232
>   interactions: 200 success / 32 provider errors). Final stored state:
>   `gameWon: true`, `moveCount: 259`, `finalProgress: 100%`, all four
>   foundations complete (hearts/diamonds/clubs/spades = 13), faceDown 0,
>   drawPile 0, `recycleCount: 5`, max successful turn 258. Clean full solve
>   under the standard 31B teacher config (temp 0.3, imperfect info /
>   `seeHiddenCards: false`); template hash stable across all 232 interactions
>   (no mid-session prompt change). Counts toward the hybrid-v1.3 / build
>   `f5c3870` win rate.

## Batch summary

- classes: `{TERMINAL-WIN: 1}`
- models: `{gemma-4-31b-it: 1}` (default teacher; included in training set)
- builds: `{f5c3870: 1}` (homogeneous)
- templates: `{7d9ecda4... (hybrid-v1.3): 1}` (homogeneous)
- win-rate contribution: 1/1 terminal record won (counted from the terminal
  win-file, not from a snapshot or a stall detector)
- caution: none. No PENDING-SNAPSHOT in this drop, so there is no
  outcome-deferral and nothing to hand to the solitaire-analyst skill.

Note: only the named `#62f09b` pair was triaged, per the request. The staging
fixtures directory also contains files for other sessions (#5c25ad, #bcd6cf,
#a11e74), but those were not part of this drop and were left untouched.
