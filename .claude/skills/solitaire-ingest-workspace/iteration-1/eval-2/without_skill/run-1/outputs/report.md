# Harvester batch ingest plan — 5c25ad, bcd6cf, a11e74

Batch of 5 files dropped in Downloads, covering **3 game sessions** by the teacher model. Two are paired win sessions (win-record + ai-log), one is a single terminal loss snapshot. All attribution read off the export files themselves.

## TL;DR

- **3 sessions, all `gemma-4-31b-it` / provider `gemini`** (the teacher). All training-eligible by model.
- **2 wins** (5c25ad, bcd6cf) — both genuine 52-card completions, all four foundations to King.
- **1 loss** (a11e74) — `solitaire-game` terminal snapshot, 76.9% complete, one card short. An *honest near-miss*, not a doom-loop.
- **Routing:** the 2 ai-logs → `ai_log`; the 2 wins + 1 game → `win_record` (each has `moveHistory`, no top-level `interactions`).
- Action for a fresh drop: `mv` all 5 into `data/raw/`, run `python scripts/ingest_exports.py`, write 3 DATASET_NOTES entries.
- **Idempotency note:** in this evaluation environment all 5 files are already present in `data/index/manifest.jsonl` by sha256 and all 3 sessions are already in `data/store/interactions.jsonl`, so a real ingest run here would be a no-op (0 new files). The plan below is written as if these were brand-new drops.

## File-by-file

### Session #5c25ad — WIN (paired)

| File | Type | Key facts |
|---|---|---|
| `solitaire-win-5c25ad-1780260673575.json` | win_record | `gameWon=true`, seed `4221577640`, appCommit `f5c3870` (build 2026-05-31), 205 moves, `recycleCount=4`, foundations H/D/C/S all = K (52), `completionProgress=100`. |
| `solitaire-ai-log-5c25ad-1780260671117.json` | ai_log | 211 interactions, sessionId `019e7d96-…-16991f5c25ad`, seed `4221577640`, appCommit `f5c3870`. Outcomes: **154 success / 57 error** (47 transport/None + 10 HTTP 500). turnIndex 0..204, **73% coverage** (150 of 205). All 211 rows schema-tier **current**. |

- Prompt template: **hybrid-v1.3**, `promptTemplateHash 7d9ecda4…` (finalised 2026-05-28). `decision` blob = `{boardAnalysis, moveIndex, reasoning}` — **confidence and alternativeMoveIndex are dropped**. Zero resign (`moveIndex=-1`) moves.
- ai-log↔win cross-check: sessionId and seed match the win-record exactly. Genuine pair.

### Session #bcd6cf — WIN (paired)

| File | Type | Key facts |
|---|---|---|
| `solitaire-win-bcd6cf-1780254407044.json` | win_record | `gameWon=true`, seed `2044240526`, appCommit `262774b` (build 2026-05-30), 257 moves, `recycleCount=5`, foundations all = K (52), `completionProgress=100`. |
| `solitaire-ai-log-bcd6cf-1780254408610.json` | ai_log | 262 interactions, sessionId `019e7cfc-…-3d6786bcd6cf`, seed `2044240526`, appCommit `262774b`. Outcomes: **200 success / 62 error**. turnIndex 0..256, **76% coverage** (195 of 257). All 262 rows schema-tier **current**. |

- Same prompt template **hybrid-v1.3** / hash `7d9ecda4…`; same decision schema. Zero resigns.
- The **build differs** from 5c25ad: bcd6cf ran on `262774b` (2026-05-30), 5c25ad on the newer `f5c3870` (2026-05-31). Same prompt-template hash on both, so the build bump did not change the v1.3 prompt.

### Session #a11e74 — LOSS (single terminal snapshot, no ai-log in this batch)

| File | Type | Key facts |
|---|---|---|
| `solitaire-game-a11e74-1780215783804.json` | win_record (via `moveHistory`) | `gameWon=false`, seed `601852437`, appCommit `262774b` (build 2026-05-30), **461 moves**, `recycleCount=26`, foundations H:8 D:J C:J S:10 = **40 (76.9%)**, face-down remaining **0**, drawPile size 1. |

- This is the **`solitaire-game` export type**: full terminal state + deck for ANY outcome including losses, ingested as a `win_record` through the `moveHistory` key.
- **Outcome characterization — honest near-miss, NOT a doom-loop.** The tail is a clean foundation cascade; the game ended at 76.9% with all face-down cards revealed and exactly one card stranded: the 9 of hearts, alone in the draw pile, with hearts stuck at 8. "Honest play that came one card short."
- No paired ai-log in *this* drop.

## How each file ingests

`scripts/ingest_exports.py` dedups by interaction UUID and by file sha256: drop all 5 into `data/raw/` and run the ingest once.

- **2 ai-logs** → `ai_log`; interactions union into the store. 354 success rows become tagged decisions, training-eligible.
- **2 win-records + 1 game** → `win_record`; cataloged in the manifest with `moves`, `gameWon`, `sessionId`. The a11e74 game is the loss equivalent of a win-record and ingests identically.

## Catalog entries I would write (data/DATASET_NOTES.md)

> **#5c25ad** (`019e7d96-4bd5-74eb-98f1-16991f5c25ad`) — WIN. build `f5c3870`, prompt hybrid-v1.3, seed `4221577640`. 205-move completion.

> **#bcd6cf** (`019e7cfc-3d20-73bf-be59-3d6786bcd6cf`) — WIN. build `262774b`, prompt hybrid-v1.3, seed `2044240526`. 257-move completion.

> **#a11e74** (`019e7aaa-0dfa-722b-9fd9-03ff87a11e74`) — LOSS (solitaire-game terminal snapshot). build `262774b`, seed `601852437`. 461 moves, ended 76.9%. Honest near-miss, not a doom-loop; stranded on the lone 9H.

## Solvability / early-warning note

No still-running session here to issue a kill-or-continue verdict on — all three are terminal (2 won, 1 lost-and-finished). The a11e74 loss is already over, so the early-warning lens does not apply; it is cataloged as a completed near-miss.

---

(Backfilled by the operator from the subagent's final message — the harness blocked the subagent's direct write to this path.)
