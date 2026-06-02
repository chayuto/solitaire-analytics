# Ingest triage: harvester drop #62f09b

**Verdict: ALREADY INGESTED. No action needed.** Both dropped files are byte-for-byte
identical to copies already in `data/raw/` and already recorded in
`data/index/manifest.jsonl` (ingested 2026-05-31 21:10:28 UTC). Re-running the pipeline
would be a clean no-op (0 new files, 0 new interactions). This is a **re-drop**, not a new
session.

The underlying session is a legitimate, high-value **full-game terminal win** by the teacher
model on the current schema and current prompt build. If it had *not* already been in the
corpus, it would be a top-priority ingest. The rest of this report documents the triage so
the conclusion is auditable.

---

## 1. What the files are

| Drop file | Type | Size | sha256 (prefix) |
|---|---|---|---|
| `solitaire-win-62f09b-1780254328065.json` | **win_record** (terminal state + `moveHistory`) | 213 KB | `34e82622...` |
| `solitaire-ai-log-62f09b-1780254397973.json` | **ai_log** (232 per-decision interactions) | 3.5 MB | `022ca0b3...` |

Classification follows `scripts/ingest_exports.py::classify_file`: a doc with `moveHistory`
is a `win_record`; a doc with `interactions` is an `ai_log`. Both files belong to the **same
game session** `019e7d96-d8d9-7317-985a-e73b1c62f09b` (short id `#62f09b`), so they are a
matched pair (terminal snapshot + decision log) for one game.

## 2. Session key attributes

| Attribute | Value | Source |
|---|---|---|
| Short id | **#62f09b** | filename / `gameSessionId` tail |
| Full session id | `019e7d96-d8d9-7317-985a-e73b1c62f09b` | both files |
| Outcome | **WIN** (`gameWon: true`, `completionProgress: 100`, session block `outcome: "won"`) | win_record + ai_log |
| Model | **`gemma-4-31b-it`** (the teacher) -- uniform across all 232 interactions | `aiConfig.model`, per-interaction `model` |
| Provider | `gemini` (uniform) | `aiConfig.provider` |
| Seed | **`3590201206`** | win_record `seed` + every interaction |
| appCommit / build | **`f5c3870`** / `2026-05-31T10:27:59Z` (uniform) | both files |
| Prompt template | **`hybrid-v1.3`**, templateHash `7d9ecda4cb415ec2` | per-interaction (188 success rows carry it; 44 are error rows with null template) |
| Temperature | `0.3` | `inferenceParams` (matches known teacher production temp) |
| Difficulty / perceived | engine `3` / perceived `47` | win_record |
| recycleCount | `5` | win_record |
| moveHistory length | **259 moves** (all `aiMove: true`) | win_record |
| ai-log exportedAt | `2026-05-31T19:06:37Z` | ai_log top-level |

**ai-log interaction breakdown (232 total):**
- Outcomes: **200 success**, 32 error. The 32 errors are the known provider unavailable/timeout
  pattern (they carry no `decision` and are not training-eligible; no need to re-flag).
- Schema tier: **232/232 "current"** -- every row has `id` + `sessionId` + `turnIndex` +
  `appCommit`, so all satisfy `SCHEMA_CONTRACT_FIELDS`.
- Success-with-decision (publishable) rows: **156**.
- turnIndex 0..258, 195 distinct (turn coverage gaps are normal -- auto-played / filtered moves;
  the engine logged 259 moves total).
- All 232 `id`s are distinct (UUIDv7); no intra-file dupes.

**Win-record internal consistency (sanity check passed):** all four foundations are complete
A..K (13 each), drawPile and discardPile empty, tableau fully cleared (0 cards remaining). This
is a genuine 52-card clear, not a mislabeled stall.

## 3. Whether and how to ingest

**It is already ingested -- do nothing.** Evidence:

1. **Byte-identical copies already in `data/raw/`.** Both dropped files hash-match files
   already present:
   - `data/raw/solitaire-ai-log-62f09b-1780254397973.json` -> sha `022ca0b3...` (same as drop)
   - `data/raw/solitaire-win-62f09b-1780254328065.json` -> sha `34e82622...` (same as drop)
2. **Both shas are already in the manifest**, ingested 2026-05-31 21:10:28 UTC:
   - ai_log row: `rows: 232, usable: 200, new: 232, duplicate: 0, schemaTiers: {current: 232}`
   - win_record row: `moves: 259, gameWon: true, sessionId: 019e7d96-...-62f09b`
3. **All 232 interactions are already in the store** (`data/store/interactions.jsonl`):
   232/232 ids present, 0 missing, and 232 store rows are tagged with this `sessionId`.

The ingest pipeline is content-addressed and idempotent: it skips any file whose sha256 is
already in the manifest. So the **correct handling is to recognize the re-drop and stop** -- no
move, no script run, no catalog edit.

### If this were a NEW session (for reference), the handling would be:

1. Move both files into `data/raw/` (the win_record and its paired ai_log).
2. Run `python scripts/ingest_exports.py` (incremental; venv: `.venv/bin/python`).
   - The **ai_log** unions its 232 interactions into the store by UUIDv7 id. All 156
     success-with-decision rows are **training-eligible** here: model is the teacher
     `gemma-4-31b-it`, schema is "current", and a winning game has no stalled stretch to
     filter. They flow into `training.jsonl`, the clean-raw, and clean-lean publish configs.
   - The **win_record** contributes a manifest row only (its `moveHistory` + `gameWon`); it is
     **not** unioned into the interaction store (only ai_log interactions are id-keyed).
3. This drop would be especially valuable: the dataset card notes wins are under-represented
   and **end-game (foundation_cards > ~10) is particularly sparse** -- a full A..K clear directly
   fills that gap. It would also raise the teacher's win-rate numerator on seed `3590201206`.

## 4. Catalog entry I would write

Because the session is already ingested, **no new `DATASET_NOTES.md` entry is warranted**
(re-cataloging an existing session would create a duplicate). I would instead confirm an entry
for `#62f09b` already exists and leave the corpus untouched.

For reference, the entry this session *should* carry (a clean win, not a failure-mode case) is:

> **#62f09b** (`019e7d96-d8d9-7317-985a-e73b1c62f09b`) -- WIN. `gemma-4-31b-it` via gemini,
> build `f5c3870`, prompt `hybrid-v1.3` (hash `7d9ecda4`), seed `3590201206`, temp 0.3.
> Full 52-card clear (completion 100, 259 moves, 5 recycles, perceived difficulty 47).
> ai-log: 232 interactions, 200 success / 32 provider-error, 156 success-with-decision, all
> current-schema. Class: **clean win** (end-game-positive sample; no doom-loop, no stall).
> Paired files: `solitaire-win-62f09b-1780254328065.json` + `solitaire-ai-log-62f09b-1780254397973.json`.

---

## Bottom line

- **Files:** a matched win_record + ai_log pair for session **#62f09b**.
- **Session:** teacher `gemma-4-31b-it`, build `f5c3870`, prompt `hybrid-v1.3`, seed
  `3590201206` -- a verified full-game **WIN** (52-card clear, 259 moves), 156 training-eligible
  decisions.
- **Action:** **none.** Already in `data/raw/`, the manifest, and the store as of
  2026-05-31 21:10 UTC. Re-running ingest is a guaranteed no-op. Recognize the re-drop and
  skip; do not re-catalog.
