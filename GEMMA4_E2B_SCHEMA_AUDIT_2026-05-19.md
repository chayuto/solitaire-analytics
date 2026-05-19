# Schema and Move Audit: Latest Export (v4)

Date: 2026-05-19
Author: data pipeline review
Audience: data harvesting / collection team

## Status update (build `afa66cb`)

This document audited the v4 export (`95cf4da`). A newer build, `afa66cb`, has
since landed and resolved several items below: the teacher model is restored,
`recentMoves` is populated, `thinkingText` is back, forced-position calls are
gone, and the `perceivedDifficulty` and `legalMoves` concerns were not bugs.
This file is kept as the dated v4 record. For the current open list, see
`HARVEST_TEAM_HANDOVER_2026-05-19.md`.

## Scope

Deep audit of the latest export schema and the moves it records, to guide the
next harvesting iteration for the Gemma 4 E2B distillation project.

File audited: `solitaire-ai-log-4aa914-1779058383236.json` (the only file in
the v4 schema). App commit `95cf4da`, build 2026-05-17T21:27. It holds one game
(session `019e37ee-7fac-7f9e-bbb8-c8c19d4aa914`), 104 moves, model
`gemini-3.1-flash-lite`.

## Summary

The v4 schema is structurally sound and the model output is well formed. The
problems are in coverage and reproducibility: there is no deck seed, more than
half of the game's moves are not logged, and one in five logged calls is spent
on a position with no choice to make. The recorded game also played very
poorly, which points at a missing-context bug rather than a model-quality
issue.

## Schema audit

### What works

- Clean identifiers. File-level `appCommit` and `appBuildTime`; interaction
  level `sessionId`, `turnIndex`, and a UUIDv7 `id`. Deduplication and
  game-level grouping work off these.
- Output discipline is solid. All 45 success rows carry a `rawResponse` that
  parses to the expected three keys (`board_analysis`, `strategic_plan`,
  `final_decision`). No malformed JSON.
- Move legality is clean. Every `moveIndex` and `alternativeMoveIndex` is
  within the `legalMoves` range. No invented indices.
- Compact prompt. 1058 prompt tokens per call, much leaner than the earlier
  gemma logs.

### What is broken or missing

1. No deck seed and no `gameId`. Not present in the file, the interaction, or
   the board JSON. `sessionId` groups the turns of one game but cannot
   reproduce a deal, verify it, or regenerate ground truth. A deck seed was a
   P0 request for the re-collection and is still absent.
2. `recentMoves` is always empty. The field exists in the board JSON but is
   never populated. The teacher decides every move with no short-term history.
3. `perceivedDifficulty` is frozen at 52 for all 104 moves. Either a static
   deal metric mislabeled as dynamic, or a broken counter.
4. `turnIndex` and `metrics.moveCount` are identical on every row. One of them
   is redundant.
5. `thinkingText` and `thoughtTokens` were dropped between v3 and v4. The
   structured `boardAnalysis` and `reasoning` fields survive, so the
   distillation target is intact, but raw chain-of-thought is no longer
   captured.

## Move audit

- The recorded game played very poorly: 0 to 6 percent completion progress
  over 104 moves. This pattern is consistent with the model cycling, and the
  always-empty `recentMoves` is the most likely cause.
- The advisor was logged for only 45 of 104 moves. 59 moves were made with no
  logged decision. The dataset captures less than half of the gameplay.
- 9 of 45 logged calls (20 percent) were on forced positions with exactly one
  legal move, all `draw_card`. The model was asked to choose when there was no
  choice.
- `legalMoves` count never exceeds 4 (distribution: 1 move on 9 calls, 2 on 5,
  3 on 27, 4 on 4). Klondike mid-game usually offers more options. Either this
  deal was unusually constrained or the move generator is under-producing
  legal moves.
- `confidence` is saturated between 0.9 and 1.0, the same miscalibration seen
  in the gemma logs.

## Recommendations

### P0, blocking for distillation

1. Log the deck seed and a stable `gameId`. Without a seed the data is not
   reproducible, and games cannot be verified or deduplicated at the game
   level.
2. Log a decision for every move, or explicitly tag auto-played moves. Today 56
   percent of moves are invisible. Document the auto-play policy in the export.
3. Run the teacher model. v4 logged `gemini-3.1-flash-lite`, but the
   distillation target is `gemma-4-31b-it`. If gemini runs are intentional,
   tag them and keep them in a separate stream.

### P1, quality

4. Stop calling the advisor on forced positions where `legalMoves` has length
   1. Auto-play them. This removes about 20 percent of calls, and API quota is
   the binding constraint given the high error rate in earlier collection runs.
5. Populate `recentMoves`. A teacher with no move history will loop, which is
   consistent with 104 moves for 6 percent progress.
6. Verify the move generator. Confirm `legalMoves` is complete. If it
   truncates, every chosen-move label is selected from a partial option set and
   is wrong.
7. Decide on `confidence`. It currently carries no signal. Either drop it or
   rewrite the prompt to force genuine calibration.

### P2, cleanup

8. Drop either `turnIndex` or `moveCount`. Fix or remove `perceivedDifficulty`.
9. Add a game-outcome record to the export: won, lost, or abandoned, plus final
   progress, at the session level. There is currently no way to filter for
   quality games without the separate win-record file.

## Reference

Audit reproducible from `scripts/ingest_exports.py` output and the raw file in
`data/raw/`. See `data/DATASET_NOTES.md` for the pipeline layout and
`data/SUMMARY.md` for current dataset statistics.
