---
name: solitaire-ingest
description: >-
  Triage and ingest the Klondike harvester's terminal and snapshot export files
  (solitaire-win-*.json and solitaire-game-*.json, usually paired with a
  solitaire-ai-log-*.json) into the training corpus. Use this whenever the user
  drops one or more harvester export files and wants them ingested, cataloged, or
  added to the dataset / corpus / manifest, or says things like "ingest these
  wins", "ingest WINS", "add this harvest", "catalog this game", or pastes
  solitaire-win- / solitaire-game- paths. It triages FIRST (terminal-win vs
  terminal-loss vs mid-game pending-snapshot) before any deep analysis, reads
  model / build / prompt attribution per file because the harvester mixes models
  and builds across time, and routes each session to the right ingest and
  DATASET_NOTES action. ALSO use this skill when the user reports that pending
  sessions were killed (e.g. "all killed", "I killed those") after an ingest:
  the operator-kill flow turns the already-ingested ai-logs into terminal
  loss-by-kill records and adjudicates each killed board. For a kill-or-continue
  verdict on a single still-running session's ai-log (is this a doom-loop, is
  the board winnable), prefer the solitaire-analyst skill; this skill is for
  bringing finished or snapshot results into the corpus.
---

# solitaire-ingest

The harvester drops export files into `/Users/chayut/Downloads/`. This skill
brings them into the corpus correctly: triage what each file actually is, read
its provenance, run the ingest pipeline, and catalog the result in the right
place. Lead with the one-line verdict per session; save the long-form for
`data/DATASET_NOTES.md`.

## Why triage comes before ingest

Two facts make a blind `mv + ingest` unsafe:

1. **The pipeline cannot tell a terminal record from a mid-game snapshot.**
   `scripts/ingest_exports.py`'s `classify_file` labels anything carrying a
   `moveHistory` as a `win_record` and copies `gameWon` verbatim. A
   `solitaire-game-*` snapshot captured at 77% with `gameWon: false` is, to the
   pipeline, indistinguishable from a terminal loss. Session **#a11e74** is the
   standing proof: its 77% `gameWon: false` snapshot preceded the actual win by
   an hour. Recording that as a loss would be wrong. You have to make the
   distinction the pipeline can't.

2. **The harvester runs different models and builds across time.** A single drop
   can mix builds (this really happens: in one batch #bcd6cf was on build
   `262774b` while #5c25ad and #62f09b were on `f5c3870`, same prompt). Never
   assume a batch shares one model / build / prompt. Read attribution per file,
   because the training-mix filter (`TEACHER_MODEL=gemma-4-31b-it`) and the
   per-version win-rate analysis both depend on getting it right.

So: triage first, ingest second, catalog third.

## The three file types

| File | What it is | Terminal? |
|---|---|---|
| `solitaire-ai-log-<id>-<ts>.json` | Per-interaction decision log (every turn's prompt + reasoning). The analyst's input. | Reflects the session at export time; may be mid-game. |
| `solitaire-win-<id>-<ts>.json` | Full terminal state + deck + `moveHistory`. `gameWon` is true OR false (the name is a misnomer). | **Yes** — a terminal record. |
| `solitaire-game-<id>-<ts>.json` | Full state + deck, captured mid-play. | **No** — a SNAPSHOT. Its `gameWon: false` is "not won *yet*", not "lost". |

Files for one session share the 6-char id token (e.g. `a11e74`).

## Workflow

### 1. Triage first — always, before any deep analysis

Run the bundled triage helper on whatever was dropped. It groups files by
session, reads attribution per file, classifies each record, and prints a
one-line verdict. It is read-only.

```bash
.venv/bin/python .claude/skills/solitaire-ingest/scripts/triage_export.py <files...>
# e.g. ~/Downloads/solitaire-*.json
```

Pass the ai-log alongside the state file when you have it — the prompt-template
hash lives in the ai-log, not the win/game file, so attribution is richer with
both.

Then tell the user the verdict, one line per session, before doing anything
else. The classes and what they mean:

- **TERMINAL-WIN** (`solitaire-win-*`, `gameWon: true`, progress 100, faceDown 0)
  — ingest it; the game is over so there is no kill decision; it belongs under
  `## Won sessions`.
- **TERMINAL-LOSS** (`solitaire-win-*`, `gameWon: false`) — ingest as a loss
  under `## Known doom-loop sessions`. If a twin of this session is still
  running, that is a kill. (None exist in the corpus yet, but the type is real.)
- **PENDING-SNAPSHOT** (`solitaire-game-*`) — NOT terminal. Do not record an
  outcome (no win, no loss) from this file. If the session is still live, this
  is a kill-or-continue call, which is the **solitaire-analyst** skill's job
  (winnability + doom-loop diagnosis). Wait for the terminal `solitaire-win-*`
  before cataloging the outcome.
- **AI-LOG-ONLY** (no state file in the drop) — if `session.outcome == "won"`,
  ingest and catalog as a win; otherwise this is a kill/continue judgement, so
  hand it to **solitaire-analyst**.

This triage-first ordering is the point of the skill: the user wants to know
"ingest, kill, or wait?" up front, not after a page of analysis.

**Re-exports of known sessions.** The triage helper flags sessions that already
have files in `data/index/manifest.jsonl`. A re-export is not a duplicate to
skip: ingest dedups interactions by UUIDv7 id and UNIONS across exports, so a
re-export can carry hundreds of new rows and reveal that a session thought
finished is in fact STILL ALIVE and progressing. Standing proof is #92762f: the
first export ended at move 195 in a wall of 503s ("died to provider errors");
a re-export three days later added 1800 new rows showing the session had
recovered and ground on to move 422 on a board already proven structurally
dead. When triage flags a known session, grep `data/DATASET_NOTES.md` for its
id: if an entry exists, APPEND an update to that entry (the entry's claims
about how the session ended may now be wrong) instead of writing a new one,
and surface the liveness change to the user, since it can revive a kill
decision everyone thought was moot.

### 2. Confirm identity and surface heterogeneity

Confirm the files in a drop that share an id are one session (same `gameSessionId`
/ `session.sessionId`). Then state the per-file attribution: model, build +
build time, prompt template, seed. If the batch summary flags a heterogeneous
build or template, call it out — it changes how the rows are attributed. If a
non-`gemma-4-31b-it` model appears (e.g. the `gemma-4-26b-a4b-it` MoE cohort),
note that it is catalogued but excluded from the default training set by the
`TEACHER_MODEL` filter.

### 3. Ingest (full-stream)

The operator ingests everything the harvester makes (this keeps within-version
win-rate true rather than selection-biased), so ingest all files in the drop,
including a pending snapshot. Ingesting a snapshot is fine; what you must not do
is *record its outcome* as a win or loss.

```bash
# raw exports are gitignored; only the manifest / store / dataset are tracked
mv <files...> data/raw/                          # mv, never cp
.venv/bin/python scripts/ingest_exports.py       # idempotent, sha256-deduped
```

Verify: the new files appear in `data/index/manifest.jsonl` (one row per file,
`type` = `ai_log` or `win_record`), and `data/SUMMARY.md`'s file counts moved by
the expected amount. The pipeline also refreshes `data/store/`, `data/dataset/`,
and `data/publish/`.

### 4. Catalog in `data/DATASET_NOTES.md`

Route by class:

- TERMINAL-WIN → `## Won sessions`
- TERMINAL-LOSS / behavioural doom-loop → `## Known doom-loop sessions`
- PENDING-SNAPSHOT → do **not** write a terminal verdict. If you note it at all,
  mark it explicitly as a mid-game snapshot awaiting the terminal export, citing
  the #a11e74 precedent. When the terminal `solitaire-win-*` later arrives, that
  becomes the canonical entry and supersedes the snapshot.

Entry shape, section conventions, and the short-`#id` style are documented in
the **solitaire-analyst** skill's `references/dataset_notes_format.md` — follow
it rather than restating it here. Lead each entry with `#<6-char id>` and keep
the full UUID once in parentheses. Quote specific cards / columns / counts; do
not paraphrase. Plain prose, no em-dashes or emojis (the user reads these).

### 5. Operator kill follow-up ("all killed")

After an ingest surfaces pending AI-LOG-ONLY sessions, the user often kills
them and reports back with something terse like "all killed". That message is
a corpus event, not just chat: each killed session's already-ingested ai-log
is now its terminal record (no further export will arrive), and the outcome
must be recorded or the cohort win-rate denominators silently undercount
losses. The flow, first run on the 2026-06-13 kill batch (6 sessions):

1. **Record each kill as a terminal loss-by-kill** in DATASET_NOTES (a batch
   subsection under `## Known doom-loop sessions` works well). Quote per
   session: final `moveCount`, `finalProgress`, rows + success/error split,
   last-activity time, and the resign count. If a killed session already has
   an entry (it usually does, that is why it was pending), also append a
   short "Resolved: operator killed <date>" note to that entry.
2. **Adjudicate every killed board exactly.** All builds since `2af3ae5`
   (2026-06-07) log the full deck in the ai-log, so run the analyst skill's
   exact solver on each killed session:
   ```bash
   .venv/bin/python .claude/skills/solitaire-analyst/scripts/true_world_winnability.py <ai-log>
   ```
   The verdict splits kills into two very different records:
   - **STRUCTURALLY DEAD** = the kill was correct (and a resign would have
     been too). Counts toward the proven-dead no-fold tally when the session
     never resigned.
   - **WINNABLE at the killed position** = a behavioural stall killed for
     budget, not a dead deal. These go in the stalls-on-winnable bucket and
     their seeds are best-of-N replay candidates; say so in the entry.
3. **Count resigns** (a success row whose `decision.move_index == -1`; the
   triage helper prints this). The resign-lever reliability record is a
   tracked research thread, so three more proven-dead no-folds is a finding,
   not a footnote.
4. **Update the cohort tallies** the kills touch (26B, flash-lite, v1.6
   resign record) in memory and DATASET_NOTES, then commit.

In the 2026-06-13 batch this flow found 3 dead boards (kill correct), 2
winnable-killed behavioural stalls, and 0 resigns across all 6, in one pass.

### 6. Hand-offs

This skill owns *bringing results into the corpus*. It does not do the deep
winnability or doom-loop diagnosis. When triage lands on a live session
(PENDING-SNAPSHOT still running, or AI-LOG-ONLY mid-game), hand to the
**solitaire-analyst** skill, which has the briefing tool
(`load_export.py`) and the Monte-Carlo solver (`check_winnability.py`). The
exact deck-logged solver (`true_world_winnability.py`) also lives there; this
skill borrows it for the operator-kill flow above.

## Reference

- `references/export_types.md` — the full file-type taxonomy, the snapshot-vs-
  terminal trap in detail, the heterogeneity / training-mix rules, and the
  cataloging routing table.
- `scripts/triage_export.py` — the read-only triage helper used in step 1.
