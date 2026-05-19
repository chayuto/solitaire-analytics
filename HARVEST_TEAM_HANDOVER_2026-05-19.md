# Harvesting Team Handover

Date: 2026-05-19
From: data pipeline review
Re: advisor data collection for the Gemma 4 E2B distillation project

## Purpose

Consolidated feedback from auditing the advisor export logs: the config-logging
ask, a prioritized action list, and pointers to the detailed documents.

## Files referenced (full paths)

- This handover:
  `/Users/chayut/repos/solitaire-analytics/HARVEST_TEAM_HANDOVER_2026-05-19.md`
- Schema and move audit:
  `/Users/chayut/repos/solitaire-analytics/GEMMA4_E2B_SCHEMA_AUDIT_2026-05-19.md`
- Progress metric proposal:
  `/Users/chayut/repos/solitaire-analytics/GAME_PROGRESS_METRIC_2026-05-19.md`
- Ingest pipeline:
  `/Users/chayut/repos/solitaire-analytics/scripts/ingest_exports.py`
- Raw export drop folder:
  `/Users/chayut/repos/solitaire-analytics/data/raw/`

## The harvest runs multiple configs: log them explicitly

The harvest is intentionally being run under different `aiConfig` settings, to
compare how the AI performs, for example whether perfect information plays
better than blind play. That is expected and useful. But for that comparison
to be possible, every game must be attributable to the exact config that
produced it, and right now the config is not in the advisor export at all
(`aiConfig` appears only in the win-record).

Some toggles can be inferred from the prompt structure:

| aiConfig field | Recoverable from the advisor export? |
|---|---|
| `seeHiddenCards` | yes, from `faceDown` arrays and `hiddenInfo.drawPileOrder` |
| `includeMoveHistory` | yes, `recentMoves` present |
| `includeReasoningTrail` | yes, `reasoningTrail` present |
| `includeSeenDrawPileCards` | yes, `seenDrawPileCards` present |
| `includeStrategyGuidance` | yes, the STRATEGY GUIDANCE block in the prompt |
| `model`, `provider` | yes, logged fields |
| `preset` name, `moveHistoryLimit`, `reasoningTrailLimit` | no, not in the export at all |

The last row is the real problem. `preset`, `moveHistoryLimit`, and
`reasoningTrailLimit` cannot be recovered by any means. Two runs with
different presets, or different history limits, are indistinguishable in the
data. Inference of the other toggles is also fragile: a schema change breaks
structural detection silently and mislabels the whole dataset.

Ask: log the full `aiConfig` object as a field on every advisor export, not
only on the win-record. At minimum, an explicit `infoMode` tag (`perfect` or
`imperfect`). An explicit tag from the harness is unambiguous and survives
schema changes; structural inference does not. The `seed` field is now
present, so the strongest experiment design is to replay the same seed across
configs for a direct head-to-head.

Until the harness logs it, the ingest pipeline derives `infoMode` from prompt
structure (`faceDown` and `hiddenInfo` presence) so current files can still be
segmented. The data collected so far is a mix of both modes; that is now
expected, but it must be tagged, not blended.

## Consolidated priority list

Re-verified against the latest build `ec38c03` (file
`/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-7381e0-1779187345109.json`)
and build `afa66cb`, session `9229e2cc3adc` (files
`/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-cc3adc-1779184594840.json`
and
`/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-cc3adc-1779186724024.json`).

### Verified resolved, no action needed

- Teacher model: runs `gemma-4-31b-it`.
- `thinkingText`, `thoughtTokens`, `recentMoves` (populated), and
  `reasoningTrail` are all present.
- Rate-limit bug fixed: zero HTTP 429 errors in the latest runs.
- Forced-position calls are gone. The advisor is no longer called when only
  one move is legal; such moves are auto-played.
- `perceivedDifficulty` is a static per-deal estimate, not a broken counter.
- `legalMoves`: the latest runs offer 6 to 7 moves; the earlier "caps at 4"
  concern was a constrained board, not truncation.
- Deck seed: build `ec38c03` logs a numeric `seed` on every interaction. With
  `sessionId` as the game key this closes the reproducibility gap. One open
  check on your side: confirm the seed actually reproduces the deal.
- `moveCount` removed from metrics; `turnIndex` is now the single move counter.

### P0, blocking for distillation

1. Tag auto-played moves. In the latest runs some moves are auto-played with
   no log entry of any kind (for example turns 229 to 231, 256, 258 in session
   `9229e2cc3adc`; 2 of 7 turns in session `3cfcbb7381e0`). Emit a log entry or
   a flag for each auto-played move so the move sequence is complete.
2. Log the full `aiConfig` on every advisor export, or at minimum an explicit
   `infoMode` tag. `preset`, `moveHistoryLimit`, and `reasoningTrailLimit`
   cannot be recovered from the export by any means, so the config-comparison
   runs cannot otherwise be told apart. See the config section above.

### P1, quality and cost

3. Add a stall auto-terminator. Session `9229e2cc3adc` ran to turn 273 with
   completion progress flat at 17 percent across two exports, roughly 437
   calls, all poison labels. The stall test is in the progress metric document.
4. Adopt the `progressScore` metric and export its raw components,
   `foundationCards` and `faceDownTotal`. The metrics block is currently only
   `completionProgress`, `perceivedDifficulty`, `difficulty`.
5. Service errors. The latest runs lose 60 to 80 percent of calls to
   "temporarily unavailable" and 210s timeouts. Rate-limiting is fixed;
   availability and timeout on the 31B model are now the yield drag.

### P2, cleanup

6. `confidence` is saturated (0.8 to 1.0) and carries no signal. Drop it or
   change the prompt to force genuine calibration.

## Pipeline status on our side

The ingest pipeline
(`/Users/chayut/repos/solitaire-analytics/scripts/ingest_exports.py`) is ready.
Drop new exports in `/Users/chayut/repos/solitaire-analytics/data/raw/` and run
it. It deduplicates by interaction id, builds the store, and derives the local
and publish datasets. It absorbs schema changes without rework as long as the
current identity fields stay.
