# Harvester batch ingest triage: #5c25ad, #bcd6cf, #a11e74

Dry-run triage of a 5-file harvester drop (3 sessions). Read-only: no `mv`, no
`ingest_exports.py`, no tracked-file edits. Verdict first, then attribution, then
the ingest/catalog plan.

## 1. Triage verdict (one line per session)

- **#5c25ad -> TERMINAL-WIN.** Ingest; no kill decision (game is over); catalog under `## Won sessions`.
- **#bcd6cf -> TERMINAL-WIN.** Ingest; no kill decision (game is over); catalog under `## Won sessions`.
- **#a11e74 -> PENDING-SNAPSHOT (NOT terminal).** Ingest the file (full-stream), but record NO outcome from it (no win, no loss). It is a 76.9% mid-game `solitaire-game-*` snapshot; await the terminal `solitaire-win-a11e74-*` before cataloging an outcome.

Batch outcome line: **2/2 terminal records won.** The third session contributes
no terminal record and must not be counted in the denominator.

## 2. Per-session / per-file attribution

All three sessions are model `gemma-4-31b-it`, provider `gemini`, preset
`standard`, `seeHiddenCards: false` (imperfect information) -- so all three are in
the default training teacher (`TEACHER_MODEL=gemma-4-31b-it`); none are excluded.
Build and prompt-template provenance, however, are NOT uniform across the batch.

### #5c25ad  (full `019e7d96-4bd5-74eb-98f1-16991f5c25ad`)
- Files: `solitaire-win-5c25ad-1780260673575.json` + `solitaire-ai-log-5c25ad-1780260671117.json`
- Record class: **TERMINAL-WIN** (`gameWon: true`, progress 100, faceDown 0, drawPile 0, recycle 4; all 4 foundations complete to K)
- Model: `gemma-4-31b-it` (gemini, standard)
- Build: `f5c3870`, built `2026-05-31T10:27:59Z`; exported `2026-05-31T20:51:11Z`
- Prompt template: `hybrid-v1.3`, hash `7d9ecda4cb415ec2335b3e970421d297730773f541b523666332dd024e9772bb` (finalised 2026-05-28)
- Seed: `4221577640`
- Artifacts: ai-log 211 rows (154 success), win-record 205 moves; ai-log `session.outcome = won`, finalProgress 100, moveCount 205 (consistent with the win-file)

### #bcd6cf  (full `019e7cfc-3d20-73bf-be59-3d6786bcd6cf`)
- Files: `solitaire-win-bcd6cf-1780254407044.json` + `solitaire-ai-log-bcd6cf-1780254408610.json`
- Record class: **TERMINAL-WIN** (`gameWon: true`, progress 100, faceDown 0, drawPile 0, recycle 5; all 4 foundations complete to K)
- Model: `gemma-4-31b-it` (gemini, standard)
- Build: `262774b`, built `2026-05-30T07:08:17Z`; exported `2026-05-31T19:06:48Z`
- Prompt template: `hybrid-v1.3`, hash `7d9ecda4cb41...` (identical to #5c25ad; finalised 2026-05-28)
- Seed: `2044240526`
- Artifacts: ai-log 262 rows (200 success), win-record 257 moves; ai-log `session.outcome = won`, finalProgress 100, moveCount 257 (consistent)

### #a11e74  (full `019e7aaa-0dfa-722b-9fd9-03ff87a11e74`)
- Files: `solitaire-game-a11e74-1780215783804.json` only (NO win-file, NO ai-log in this drop)
- Record class: **PENDING-SNAPSHOT** -- a `solitaire-game-*` is a mid-play snapshot by construction, never a terminal record
- Model: `gemma-4-31b-it` (gemini, standard)
- Build: `262774b`, built `2026-05-30T07:08:17Z` (same build as #bcd6cf)
- Prompt template: not carried in a `solitaire-game-*` file (the template hash lives in the ai-log, which is absent here). Sibling build #bcd6cf ran `hybrid-v1.3`; do not assume, since no ai-log pins it for this session.
- Seed: `601852437`
- State at snapshot: `gameWon: false`, `completionProgress: 76.92` (40/52 foundation cards: H8 D11 C11 S10), faceDown 0, drawPile 1, discardPile 0, **recycleCount 26**, moveHistory 461 moves. Last logged move is a productive `tableau_to_foundation` (JC, col 7 -> clubs) at ts 1780215718321 with forward-looking reasoning -- i.e. it was still advancing at capture time, not visibly stalled.

### Heterogeneity call-out and what it implies
- **Builds are HETEROGENEOUS:** `f5c3870` (#5c25ad) vs `262774b` (#bcd6cf, #a11e74). Do not collapse the batch to one build; attribute per file (this is exactly the mixed-build case the skill warns about).
- **Prompt template is IDENTICAL where pinned:** both ai-logs carry the same full hash `7d9ecda4cb41...` (`hybrid-v1.3`). So the `262774b -> f5c3870` build delta is **harness-side, not a prompt change**. Consequence: any same-seed comparison spanning these two builds is testing app/harness behavior, not prompt behavior. (The triage helper flagged "HETEROGENEOUS prompt template" only because the `solitaire-game` snapshot carries no hash to compare; that is a missing-field artifact, not an actual second template.)
- **Model is uniform:** all `gemma-4-31b-it`. No `gemma-4-26b-a4b-it` MoE rows and no Gemini-proper rows in this drop, so no `TEACHER_MODEL` exclusions apply -- every ingested row is training-eligible.

## 3. Ingest plan and cataloging

### Why the pipeline cannot be trusted blind here
`scripts/ingest_exports.py::classify_file` (lines 203-208) returns `win_record`
for ANY doc containing `moveHistory`, and the manifest row copies `gameWon`
verbatim (lines 993-996). The `solitaire-game-a11e74` snapshot has a
`moveHistory` and `gameWon: false`, so the pipeline would write it to
`data/index/manifest.jsonl` as `type: win_record, gameWon: false` -- indistinguishable
from a terminal loss. That is the #a11e74 trap, and it is the reason this snapshot
must be cataloged by hand as pending, not read off the manifest as a loss.

### Ingest (full-stream -- ingest everything, including the snapshot)
On the live run (NOT performed here), all 5 files move into `data/raw/` and the
pipeline runs idempotently (sha256-deduped):
```
mv solitaire-win-5c25ad-*.json solitaire-ai-log-5c25ad-*.json \
   solitaire-win-bcd6cf-*.json solitaire-ai-log-bcd6cf-*.json \
   solitaire-game-a11e74-*.json   data/raw/
.venv/bin/python scripts/ingest_exports.py
```
Expected manifest delta: **+5 rows** -- 2 `ai_log` (5c25ad 211 rows, bcd6cf 262
rows) + 3 `win_record` (5c25ad win, bcd6cf win, and the a11e74 snapshot, which
the pipeline mislabels `win_record gameWon=false`). `data/SUMMARY.md` file counts
should move by +2 ai_log / +3 win_record. Ingesting the snapshot is correct
(full-stream keeps within-version win-rate unbiased); the only rule is do not let
its label settle an outcome.

### Cataloging in `data/DATASET_NOTES.md`
- **#5c25ad -> `## Won sessions`.** New entry led by `#5c25ad` (full UUID once in parens): model gemma-4-31b-it, build f5c3870 (2026-05-31), prompt hybrid-v1.3 `7d9ecda4`, seed 4221577640, 205 moves, recycle 4, progress 100.
- **#bcd6cf -> `## Won sessions`.** New entry led by `#bcd6cf`: same model, build **262774b (2026-05-30)**, prompt hybrid-v1.3 `7d9ecda4`, seed 2044240526, 257 moves, recycle 5, progress 100. Note in the entry that this win is on the older build but the identical prompt as #5c25ad.
- **#a11e74 -> NO terminal entry.** Do not write a win or loss. If noted at all, mark it explicitly as a mid-game `solitaire-game-*` snapshot (76.9%, 461 moves, recycle 26, seed 601852437, build 262774b) awaiting its terminal `solitaire-win-a11e74-*`, citing the #a11e74 precedent (a 77% gameWon=false snapshot of this same session id that later WON ~1h after capture; here the snapshot export is ~59 min before that known terminal-win export timestamp). When the terminal win-file arrives, it becomes canonical and supersedes the snapshot, which is then recorded as an archived predecessor with its move/row count.

### How batch win-rate should be counted
Count wins from terminal `solitaire-win-*` files only (`gameWon: true`), never
from a snapshot and never from a stall detector. For THIS batch:
- Terminal records: 2 (#5c25ad, #bcd6cf). Wins: 2. **Batch terminal win-rate = 2/2.**
- #a11e74 contributes **nothing** to numerator or denominator until its terminal
  export lands. Counting it as 2/3 (treating the snapshot as a loss) would be the
  exact #a11e74 error and would understate the true within-version win rate.
- Because builds differ, a strict within-version tally should bucket by build:
  `f5c3870` 1/1 (#5c25ad), `262774b` 1/1 (#bcd6cf), with #a11e74 (also `262774b`)
  pending. CIs on n=1-per-build are uninformative; the value here is full-stream
  corpus growth and keeping the snapshot from corrupting the denominator.
