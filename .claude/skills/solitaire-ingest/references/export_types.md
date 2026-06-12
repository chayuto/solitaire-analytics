# Harvester export types and ingest routing

Reference for the `solitaire-ingest` skill. Read this when you need the
field-level shape of a file, the exact classification rules, or the cataloging
routing.

## The three files, field by field

### `solitaire-ai-log-<id>-<ts>.json`
The per-interaction decision log. Top-level shape:
- `session`: `{ sessionId, seed, model, outcome, finalProgress, moveCount }` —
  the terminal summary, if the session ended by export time.
- `interactions`: list of per-turn rows, each `{ turnIndex, timestamp, outcome,
  prompt, decision, ... }`. The `prompt` embeds the `CURRENT GAME` board; the
  `decision` carries `boardAnalysis` / `strategicPlan` / `reasoning`.
- `appCommit`, `appBuildTime`, `exportedAt`.
- Prompt-template fields (`promptLayoutVersion`, `promptTemplateVersion`,
  `templateHash`) live here, per interaction. **This is the only file that
  reliably carries the template hash.**

This is the solitaire-analyst skill's primary input. For ingest, it is the
canonical interaction log and contributes the training rows.

### `solitaire-win-<id>-<ts>.json`
A terminal full-state record. Top-level shape:
- `gameWon` (bool — true OR false), `completionProgress` (int),
  `recycleCount`, `seed`, `gameSessionId`, `appCommit`, `appBuildTime`.
- `foundations`, `tableau`, `drawPile`, `discardPile` — the final board.
- `moveHistory` — every move of the game (this is what makes `classify_file`
  call it a `win_record`).
- `initialBoardSetup` — the full deal (the deck), so the game is reproducible.
- `aiConfig` — `{ model, provider, preset, seeHiddenCards, ... }`. The model
  lives here, not in a `session` block.
- `aiDecisionLog` — a trailing sample of decisions (usually ~30), NOT the full
  per-turn log, and usually WITHOUT the template hash.

Despite the name, this is a terminal record of *any* outcome. `gameWon: true` is
a win; `gameWon: false` is a true terminal loss.

### `solitaire-game-<id>-<ts>.json`
Same shape as the win file (`moveHistory`, `initialBoardSetup`, `aiConfig`,
`gameWon`, `completionProgress`, full board), but captured **mid-play**. It is a
SNAPSHOT, not a terminal record. `classify_file` still labels it a `win_record`
because it has a `moveHistory`, which is the trap below.

## The snapshot-vs-terminal trap (#a11e74)

`scripts/ingest_exports.py` does not distinguish a snapshot from a terminal
record. It keys on the presence of `moveHistory` and copies `gameWon`. So a
`solitaire-game-*` snapshot taken at `completionProgress: 77, gameWon: false`
gets ingested as a `win_record` with `gameWon: false` — which reads exactly like
a terminal loss.

Session **#a11e74** (`019e7aaa-0dfa-722b-9fd9-03ff87a11e74`) is the canonical
case: its `solitaire-game-a11e74-1780215783804.json` snapshot was 77% with
`gameWon: false`, captured ~1 hour before the session actually WON (the terminal
`solitaire-win-a11e74-1780219323762.json`, `gameWon: true`, 474 moves). If you
trust the pipeline's label, you record a loss for a won game and corrupt the
win-rate.

Rule: a `solitaire-game-*` file never settles an outcome. Treat it as
PENDING-SNAPSHOT and wait for the terminal `solitaire-win-*`.

## Classification decision table

Decide from the file that best settles the session (prefer a win file, then a
game file, then the ai-log):

| Signal | Class | Action |
|---|---|---|
| `solitaire-win-*`, `gameWon: true`, progress 100, faceDown 0 | TERMINAL-WIN | Ingest; catalog under `## Won sessions`. No kill decision. |
| `solitaire-win-*`, `gameWon: false` | TERMINAL-LOSS | Ingest; catalog under `## Known doom-loop sessions`. Live twin = kill. |
| `solitaire-win-*`, `gameWon: true` but progress < 100 or faceDown > 0 | VERIFY | Unexpected; inspect the board before cataloging. |
| `solitaire-game-*` (any `gameWon`) | PENDING-SNAPSHOT | Ingest is fine; do NOT record an outcome. Await terminal export. |
| ai-log only, `session.outcome == "won"` | AI-LOG-ONLY (win) | Ingest; catalog as a win; terminal win-file may arrive later. |
| ai-log only, other / unset outcome | AI-LOG-ONLY | Kill/continue judgement -> hand to solitaire-analyst. |
| user reports a pending session KILLED | OPERATOR-KILL | The latest ai-log IS the terminal record. Catalog as loss-by-kill; adjudicate the killed board exactly (`true_world_winnability.py`); count resigns. |

`triage_export.py` implements this table and prints the verdict per session plus
a batch summary. It also prints, per ai-log, the session liveness line
(outcome, moveCount, finalProgress, resigns, last activity) and flags
RE-EXPORTS (sessions already in `data/index/manifest.jsonl`).

## Re-exports and operator kills

Two session-lifecycle events arrive without any new file type:

- **Re-export of a known session.** Ingest dedups interactions by UUIDv7 and
  unions across exports, so a re-export is always safe to ingest and can
  EXTEND a session: #92762f's first export ended at move 195 in a 503 wall
  (read at the time as "died to provider errors"); the re-export added 1800
  rows showing it recovered and reached move 422. Update the existing
  DATASET_NOTES entry rather than writing a duplicate, and re-check liveness:
  a session presumed finished may need a fresh kill call.
- **Operator kill.** When the user says the pending sessions were killed, no
  terminal `solitaire-win-*` will ever arrive; the already-ingested ai-log is
  the terminal record and the session counts as a loss in its cohort
  denominator. On deck-logging builds (`2af3ae5` 2026-06-07 and later) run
  the analyst skill's exact solver on each killed board: STRUCTURALLY DEAD
  means the kill was correct (and extends the proven-dead no-fold tally if
  resigns=0); WINNABLE means a behavioural stall killed for budget, whose
  seed is a best-of-N replay candidate. The 2026-06-13 batch (3 dead, 2
  winnable-killed, 0 resigns in 6 sessions) is the precedent.

## Heterogeneity and the training mix

Read `model`, `appCommit` / `appBuildTime`, and the template hash PER FILE. A
single drop can span builds (e.g. `262774b` and `f5c3870` in one batch) or
models. Consequences:

- **`TEACHER_MODEL=gemma-4-31b-it`** is the default training teacher. The
  `gemma-4-26b-a4b-it` MoE cohort (active-4B, top-8 of 128 experts) is ingested
  and catalogued like any model, but the default filter keeps it out of the
  training set. Training-mix decisions over the 26B rows stay open.
- **Win-rate is read from terminal win-files only** (`gameWon: true`), never from
  a snapshot and never from a stall detector. The operator ingests the full
  stream, so within-version win rate off the corpus is true rather than
  selection-biased — but only if snapshots are not miscounted as losses.
- Same prompt template across a build delta means the build change is
  harness-side, not a prompt change. Worth stating explicitly when it happens,
  because it isolates what a same-seed comparison is actually testing.

## Cataloging routing

`data/DATASET_NOTES.md` is hand-maintained interpretation; `data/SUMMARY.md` is
auto-generated stats. Sections:

- `## Won sessions` — TERMINAL-WIN and confirmed AI-LOG-ONLY wins.
- `## Known doom-loop sessions (kept; flagged by stall filter)` — TERMINAL-LOSS
  and behavioural doom-loops.
- `## Same-seed validation experiments` — comparison arms re-running a known
  seed under a different build/prompt.

The entry shape (lead with `#<6-char id>`, full UUID once in parens, per-file
attribution, terminal stats, failure/oscillation signature quoted exactly) is
specified in the solitaire-analyst skill's `references/dataset_notes_format.md`.
A PENDING-SNAPSHOT gets no terminal entry; when its terminal export arrives, the
terminal record is canonical and supersedes the snapshot (note the snapshot as
an archived predecessor with its row count).
