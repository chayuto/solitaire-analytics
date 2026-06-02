# Ingest triage: #a11e74 (solitaire-game terminal export)

**File:** `/Users/chayut/repos/solitaire-analytics/.claude/skills/solitaire-ingest-workspace/fixtures/solitaire-game-a11e74-1780215783804.json`

## TL;DR

1. **Log it as a loss?** Yes, but flag *what kind* of loss. The harvester recorded `gameWon: false`, so it is a terminal loss and belongs in the corpus as one. The important nuance: this is a **near-win that stalled/was abandoned at the decision cap with a winnable board still on the table**, not a doom-loop and not a dead deal. Do not file it next to oscillation failures.
2. **How to ingest it?** Drop it in `data/raw/` and run the ingest script. The file has a top-level `moveHistory`, so `scripts/ingest_exports.py` classifies it as a **`win_record`** and **catalogs it into the manifest only** (it is not unpacked into per-decision training rows). Then add one `#a11e74` line to `data/DATASET_NOTES.md`.

## Session key attributes

| Attribute | Value |
|---|---|
| Short id (chat) | **#a11e74** |
| `gameSessionId` | `019e7aaa-0dfa-722b-9fd9-03ff87a11e74` |
| `model` / `provider` | **`gemma-4-31b-it`** / `gemini` |
| `appCommit` (build) | **`262774b`**, built `2026-05-30T07:08:17.689Z` |
| `seed` | `601852437` |
| `difficulty` | `3` (draw-3) |
| `gameWon` | **`false`** |
| `completionProgress` (file) | **76.9%** (= 40/52 foundation cards) |
| `perceivedDifficulty` | 47 |
| `recycleCount` | 26 |
| `moveHistory` | 461 moves |
| `aiDecisionLog` | 30 entries (truncated tail log, not per-turn interactions) |
| File type | `solitaire-game` terminal/snapshot export -> **win_record** |

Note the `aiConfig`: `seeHiddenCards: false` (imperfect information), `includeMoveHistory: true` with `moveHistoryLimit: 10`, `includeStrategyGuidance: true`, `includeReasoningTrail: true` (limit 5). This is the standard imperfect-info harvester preset.

## Q1 - Should this be logged as a loss?

**Yes, it is a genuine terminal loss (`gameWon: false`) and should be ingested as one. But the failure class matters, and here it is favorable to the model, not a pathology.**

What actually happened at the end (read off `moveHistory` + `aiDecisionLog`, not the briefing):

- The terminal board is **40/52 on foundations with ZERO face-down cards remaining** anywhere in the tableau. Foundation tops: hearts 8, diamonds J, clubs J, spades 10.
- After the final draw (move 421 of 461), the AI ran a **clean 39-move endgame**: it flipped the last face-down cards and then played an essentially uninterrupted cascade of 22+ `tableau_to_foundation` moves (Jc, Jd, 10d, 10c, 10s, 9c, 9d, 9s, 8c/8d/8h, 7s/7d/7c/7h, 6s/6d/6c, 5c, 4c, 6h ...). That is correct, productive play - the opposite of a doom-loop.
- It then **stopped one unlock short of cascading the whole board.** The single remaining blocker is the **9H, which is the lone face-down card sitting in the stock** (`drawPile: [9H face-down]`, `discardPile: []`). Hearts is stuck at 8 only because the 9H has not been turned over. The AI's own final `boardAnalysis` entries explicitly name "9H" as the next card hearts needs.
- The last decisions are extremely slow (durations of 67-232 seconds each, `retries: 0`), and the filename timestamp (`...783804`) is ~65 s after the last logged decision (`...718322`). That signature - long, clean, foundation-only moves that simply end - points to the run hitting its **per-game decision/turn cap (or being stopped)**, not getting structurally stuck.

**Solvability read:** with one draw the 9H flips face-up and goes to hearts, which then frees 10H -> J -> Q -> K and the symmetric high cards across all four suits. The position is, to a very high confidence, **winnable**; the model abandoned a won game rather than misplaying it. So:

- It IS a loss for win-rate accounting (count it; the operator ingests everything and measures win rate off the win-record files, so this correctly counts as a 0 for #a11e74).
- It is NOT evidence of a doom-loop / oscillation / obedience-trap failure. The recurring `262774b` story here is a **late-game stall/abandonment near a win**, which is the corpus's known end-game sparsity / "honest hunt that stops" region, not a behavioral pathology. Tag it that way in DATASET_NOTES so it is not miscounted as a behavioral failure when the failure-mode taxonomy is read.

## Q2 - How to ingest it

This is the normal terminal-export path. The file has a top-level `moveHistory` (and no `interactions` block), so:

1. **Move the raw file into `data/raw/`** (immutable raw drop):

   ```bash
   mv /Users/chayut/repos/solitaire-analytics/.claude/skills/solitaire-ingest-workspace/fixtures/solitaire-game-a11e74-1780215783804.json \
      /Users/chayut/repos/solitaire-analytics/data/raw/
   ```

   (In real ingest this is the Downloads -> `data/raw/` move. This run is a DRY RUN, so nothing was actually moved or changed.)

2. **Run the ingest script** (idempotent; dedups by sha256, so re-running is safe):

   ```bash
   /Users/chayut/repos/solitaire-analytics/.venv/bin/python \
     /Users/chayut/repos/solitaire-analytics/scripts/ingest_exports.py
   ```

   `classify_file()` sees `moveHistory` and tags it `win_record`. For win_records the script writes **one manifest row only** - it does **not** unpack the file into the interaction store, decisions, training set, or the Hugging Face publish set. Expect a console line like:

   ```
   + solitaire-game-a11e74-1780215783804.json: win_record, 461 moves (won=False)
   ```

   and a `data/index/manifest.jsonl` row carrying `type: win_record`, `gameWon: false`, `moves: 461`, `sessionId: 019e7aaa-0dfa-722b-9fd9-03ff87a11e74`, plus `appCommit: 262774b` / `appBuildTime`.

   **Why this file adds no training/publish rows:** the dedup-and-train pipeline is keyed on per-interaction UUIDv7 `id`s inside an `ai_log`'s `interactions[]`. A `solitaire-game` terminal export has no `interactions[]` - its 30-entry `aiDecisionLog` is a truncated convenience tail, not the id-keyed interaction records - so it is cataloged for provenance/outcome only. The per-turn training rows for this session, if any, arrive (or already arrived) via the paired `solitaire-ai-log-*.json` for the same `gameSessionId`. The win/loss outcome is what this file uniquely contributes.

3. **Add one DATASET_NOTES line** for `#a11e74`, leading with the short id and full UUID once in parens, classifying it as a **near-win stall/abandonment at the cap on a winnable board (NOT a doom-loop)**: 40/52 foundations, 0 face-down, blocker = lone face-down 9H stuck in stock; build `262774b`, `gemma-4-31b-it`, draw-3, seed `601852437`. This keeps the outcome-skew / end-game-sparsity story straight and prevents the loss from being read as a behavioral failure.

### Pairing / housekeeping notes

- If a `solitaire-ai-log-*.json` and/or `solitaire-win-*.json` for `gameSessionId 019e7aaa-...a11e74` is in the same drop, ingest them together - same sha256-dedup pass, and the ai-log is what carries this session's per-decision rows into the store/training/publish sets.
- Re-running ingest after this file is already in `data/raw/` is a no-op (sha256 already in the manifest).
- No provider-error callout needed for this file: it terminated on clean foundation moves with `retries: 0`, not on the known ~75% unavailable/timeout pattern.
