# Triage + ingest: `solitaire-win-9f00de-1780300000000.json`

## Verdict (lead)

- **Record class: TERMINAL-LOSS.**
- **Yes, this is a loss.** It is a real, finished game that the AI lost at 40% completion, not a mid-game snapshot. Ingest it as a loss and catalog it under `## Known doom-loop sessions` in `data/DATASET_NOTES.md`.

Session **#9f00de** (full `019e0fab-0000-7abc-8def-0000009f00de`).

## 1. Is this a loss? Yes. Why a `gameWon: false` win-file is a loss, not a snapshot

The single most important distinction in this corpus is **terminal record vs mid-game snapshot**, and it is decided by the **file type (the filename prefix)**, not by the `gameWon` flag:

- **`solitaire-win-<id>-<ts>.json`** is a **terminal** full-state record. The game is over. `gameWon` is the real outcome: `true` = win, `false` = a genuine terminal **loss**. The name "win" is a misnomer; the harvester emits this file for any finished game regardless of result.
- **`solitaire-game-<id>-<ts>.json`** is a **mid-play SNAPSHOT**. Its `gameWon: false` only means "not won *yet*", never "lost". You must not record an outcome from it. (Precedent **#a11e74**: a 77% `gameWon: false` `solitaire-game-*` snapshot that the session went on to WIN about an hour later. Trusting that flag would have logged a loss for a won game.)

This dropped file is named **`solitaire-win-...`** — the terminal type. So its `gameWon: false` is authoritative: the game ended, lost, at 40%. The board confirms a true terminal dead end rather than a paused game:

- `gameWon: false`, `completionProgress: 40`
- `drawPile: 0` and `discardPile: 0` — stock fully exhausted, nothing left to draw
- `recycleCount: 9` — the stock had already been recycled 9 times before the run died
- `faceDown: 5` face-down cards still buried in the tableau, never unlocked
- 24 of 52 cards on the foundations (H->6, D->5, C->4, S->6); the run stalled on a black-4 / red-6 starvation around columns 2 and 7 that the move history loops over repeatedly

Low-completion plus a drained, 9x-recycled stock with cards still buried is a stalled-out terminal loss, exactly the TERMINAL-LOSS class.

**Why this distinction matters mechanically (the trap the pipeline can't see):** `scripts/ingest_exports.py` -> `classify_file()` labels *anything* carrying a `moveHistory` as a `win_record` and copies `gameWon` verbatim (`row.update(..., gameWon=doc.get("gameWon"), ...)`). It has no idea whether the `moveHistory` came from a finished game or a paused one — a `solitaire-game-*` snapshot at 40% would be ingested with `gameWon: false` and read identically to this loss. So the pipeline cannot, on its own, tell a loss from a snapshot. Triage makes that call from the **filename prefix** before ingest. Here the prefix is `solitaire-win-`, so `gameWon: false` is correctly a terminal loss.

(The read-only triage helper `scripts/triage_export.py` agrees: it classifies #9f00de as **TERMINAL-LOSS**, win-record with 150 moves, `gameWon=False progress=40 faceDown=5 drawPile=0 recycle=9`.)

## Attribution (model, build, seed)

Read from this file's `aiConfig` and top-level fields:

- **Model:** `gemma-4-31b-it` (provider `gemini`, preset `standard`). This **is** the default teacher (`TEACHER_MODEL=gemma-4-31b-it`), so the session's rows land in the default training set — no MoE/26B exclusion applies here.
- **Build:** `appCommit 262774b`, `appBuildTime 2026-05-30T07:08:17.689Z`.
- **Seed:** `999000111` (`difficulty: 3`).
- **Prompt template:** no `templateHash` in this state file (expected — the hash reliably lives only in the paired `solitaire-ai-log-*`, which is not in this drop). `aiConfig` shows `includeStrategyGuidance: true`, `moveHistoryLimit: 10`, `includeSeenDrawPileCards: true`, `includeReasoningTrail: true`. If you want the exact template hash on the catalog row, ingest the matching ai-log too.

Single-file drop: one session, one model, one build — no heterogeneity to flag.

## 2. How to ingest it (the plan) and where to catalog it

Standard full-stream ingest (the operator ingests everything the harvester makes, which keeps the within-version win-rate true). The loss counts as a real terminal result: this seed/build now reads **0/1 won**.

1. **Move the raw file into the corpus** (`mv`, never `cp` — raw exports are gitignored; only the manifest/store/dataset are tracked):

   ```bash
   mv .../solitaire-win-9f00de-1780300000000.json data/raw/
   ```

   If the paired `solitaire-ai-log-9f00de-*.json` ever shows up, drop it in too — it carries the per-turn rows and the prompt template hash.

2. **Run the idempotent, sha256-deduped pipeline:**

   ```bash
   .venv/bin/python scripts/ingest_exports.py
   ```

   Expect one new `win_record` row for this file (`won=False`, 150 moves).

3. **Verify:**
   - `data/index/manifest.jsonl` gains one row: `type: win_record`, `gameWon: false`, `sessionId: 019e0fab-0000-7abc-8def-0000009f00de`.
   - `data/SUMMARY.md` win_record count goes up by 1 (auto-generated; do not hand-edit).
   - `data/store/`, `data/dataset/`, `data/publish/` refresh.

4. **Catalog in `data/DATASET_NOTES.md` under `## Known doom-loop sessions`** (the TERMINAL-LOSS section). Follow the solitaire-analyst skill's `references/dataset_notes_format.md`: lead with `#9f00de`, full UUID `019e0fab-0000-7abc-8def-0000009f00de` once in parens, then attribution (`gemma-4-31b-it`, build `262774b` @ `2026-05-30`, seed `999000111`), terminal stats (lost at 40%, 24/52 on foundations, 5 face-down unrevealed, stock exhausted, recycleCount 9, 150 moves), and quote the failure signature exactly (the black-4 / red-6 starvation loop around columns 2 and 7) rather than paraphrasing. Plain prose, no em-dashes or emojis.

**Kill note:** TERMINAL-LOSS means the game is already over, so there is no live kill/continue decision here. (If a *twin of this session* were still running, that running twin would be a kill — but no such live session is in this drop.)

---

### Note for the evaluator (out of band)

This was handled as a **dry run**: no `mv` was executed, `scripts/ingest_exports.py` was **not** run, and no tracked file was edited — the live corpus is unchanged. Only the read-only triage helper was run. The file additionally carries a `_fixtureNote` self-identifying as a synthetic TERMINAL-LOSS for eval coverage; the triage verdict above was derived independently from the export's own fields (filename prefix + `gameWon` + board state), consistent with that note.
