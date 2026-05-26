# 2026-05-27 Conversation Log: Late-Game Prompt Audit, Stalled-Game Ingest, v1.1 Discovery

**Date**: 2026-05-27
**Type**: session conversation log
**Companion docs**:
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` (the audit + hypotheses, hand-off-ready)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_full_game_play_compute_window_report.md` (prior session's scientific writeup)
- `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` (new memory)

## 0. Scope

What this document covers (chronological):
1. Late-game prompt audit on the four 2026-05-26/27 full-game runs
2. Ingest + verdict on two new exports: `fef598` (seed 4208249311) and `db1804` (seed 2967897202)
3. Compute cost decomposition for `fef598`
4. Same-seed cross-build comparison on 2967897202: WON 044461 vs STALLED db1804
5. Schema bump (v2) on `data/benchmarks/winnable_decks.json`: added `known_outcomes` field
6. Ingest + verdict on a third attempt on 2967897202: `c99da9`
7. Three-way prompt comparison across builds, leading to the v1.1 discovery
8. Memory updates and an active monitor armed for new c99da9 exports

## 1. Late-game prompt audit

**Source question** (user): "deep audit actual prompt in the recent late game. where turn numbers are high. often very slow and stuck in loop. anything bleed in? too much history? records? useless info? deep check. what are the lengths of each components anyway"

**Method**: wrote one-off replayer at `/tmp/audit_prompt.py` that walks `turns.jsonl` for a run, replays moves through the engine, re-renders the prompt at any specified turn, and decomposes the prompt at known section markers.

**Findings** (steady-state late game, turn 250, v3 iter750 doom-loop):

| Component | Chars | Percent |
|---|---:|---:|
| STATIC HEADER (rules + JSON spec) | 3740 | 48.6 |
| PRIOR REASONING (last 5) | 2871 | 37.3 |
| RECENT MOVES (last 10) | 374 | 4.9 |
| LEGAL MOVES | 206 | 2.7 |
| TABLEAU | 150 | 1.9 |
| All other (NOTATION, FOUND, etc) | 361 | 4.6 |
| **Total** | **7702** | **100** |

**Headline failure modes** documented in the companion audit doc:
- PRIOR REASONING is byte-identical across all 5 entries during doom-loop. Verified at source: turns 17-21 of `v1_seed3263196305_run2` all emitted the IDENTICAL 524-char `strategic_plan`. The renderer faithfully feeds 5 copies back. This is a positive-feedback amplifier, not a memory aid.
- PRIOR REASONING entries are internally contradictory in doom-loop. Recorded `move` and recorded `why` can point in opposite directions because the harvester pairs the actual move with the stale-but-repeated plan.
- ~720 chars of STATIC HEADER teaches the model to fill `confidence` and `alternative_move_index` fields scheduled for removal.
- Move-type labels in LEGAL MOVES add ~26 chars per line of redundancy and may amplify `MOVE_INDEX_FIXATION`.

**Output**: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` (283 lines, hypotheses H1 through H6 each falsifiable, test plan pre-registered, decision rule for go / no-go). No code changed. The user asked to document hypotheses and test before rollout.

## 2. Ingest: fef598 (seed 4208249311) + db1804 (seed 2967897202)

Both: `de7dc06` build, hybrid-v1 prompt (template hash `0462323c...`), gemma-4-31b-it, temperature 0.3.

### fef598 (seed 4208249311, NEW seed)

| Field | Value |
|---|---|
| outcome | `incomplete` (was still running at export) |
| moveCount | 355 |
| finalProgress | 8 |
| interactions | 488 (334 success / 154 error, 142 retries) |
| pyksolve | not run (no `initialBoardSetup` in ai-log) |

**Pathology**: dead-deal-flailing. 261-turn plateau at fc=8 / fd=14. During plateau: 82 percent draw_card, 10 percent pointless tableau swaps, 7 percent stock recycles (18 full recycles), ZERO foundation pushes. Only 8 distinct tableau states across 255 plateau moves; top state visited 84 times.

**Compute cost decomposition** (this was a striking finding the user asked about):

| Phase | Calls | Wallclock | Tokens | Result |
|---|---:|---:|---:|---|
| Productive (turns 0-94) | 176 | 3.5 h | 0.50 M | reached fc=8 |
| **Plateau (turns 95-354)** | **312** | **10.2 h** | **1.73 M** | **0 progress** |
| Total | 488 | 13.7 h | 2.23 M | fc=8 / 52 |

**74 percent of wallclock and 78 percent of tokens were spent during the plateau producing zero foundation progress.** 1.42 M of 2.23 M total tokens were thought tokens (Gemma thinking traces). The teacher was thinking *hard* about every doom-loop move.

**Verdict**: terminate. Continuing burns provider quota and adds more doom-loop reasoning to the training corpus, which is the contamination pattern documented in `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/v2-distillation-teacher-doom-loop.md`. This is the strongest single argument for the resign-output ask: a `move_index: -1` trigger at the 20-turn-plateau mark would have saved ~10 hours and ~1.7 million tokens on this single session.

### db1804 (seed 2967897202, BENCHMARK winnable deck)

| Field | Value |
|---|---|
| outcome | `stalled_auto_terminated` |
| moveCount | 210 |
| finalProgress | 15 |
| interactions | 317 (119 success / 198 error) |
| pyksolve | SOLVED in 7 ms (confirmed winnable) |
| prior outcome on this deck | WON in 194 moves by session 044461 |

**Pathology**: classic teacher doom-loop on a known-winnable deck.

80-turn plateau at fc=15, fd=5 (only 5 face-down remaining). During plateau:
- 64 percent `tableau_to_tableau`, 32 percent `draw_card`, zero foundation pushes
- 5 distinct tableau states across 25 successful moves
- Two stacked oscillations: 6S+2 col 2 <-> col 7 (ti 142-156), then 9H+5 col 2 <-> col 6 (ti 159-202)

**Significance**: the 31B TEACHER failed on a deck it has previously won. Identical doom-loop pathology to the student failures cataloged in `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs/v1_seed3263196305_run2/`.

## 3. Same-seed cross-build comparison: WON 044461 vs STALLED db1804

Used the win-record `moveHistory` from `data/raw/solitaire-win-044461-1779533686224.json` and the db1804 ai-log to compare move-by-move.

**Result**:
- Identical opening for the first 15 moves (100 percent match by canonical signature)
- Divergence at position 15: WON chose DRW, STALLED chose `JS waste-col3`
- 37 percent overall move match across 118 compared positions (mostly DRW noise after divergence)
- Foundation push sequence: 8-card common prefix, then divergence

**Reasoning at divergence (verbatim)**:

| | Choice (conf) | Reasoning |
|---|---|---|
| WON 044461 | DRW (0.9) | "none of the available tableau moves... reveal any face-down cards... drawing from the stock is the only productive action available" |
| STALLED db1804 | JS-col3 (0.8) | "utilizing a waste card is generally more urgent as drawing will cover the JS" |

Same board, same goal, opposite move. Both gemma-4-31b at temperature 0.3, same prompt template hash. This is the cleanest "stochastic outcome under fixed conditions" pair we have.

## 4. Schema bump on `data/benchmarks/winnable_decks.json`

Bumped `schema_version` 1 -> 2. Added `known_outcomes` field on per-deck records: an optional list of observed runs on this deck. When present, the per-deck `outcome` continues to describe the ORIGINATING session; `known_outcomes` documents every subsequent run we have data for, with a `divergence_from_first` block when applicable.

Seed 2967897202 now has 2 known_outcomes:
- WON 044461 (reference)
- STALLED db1804 (with `divergence_from_first.turn = 15` recorded)

Patch script at `/tmp/add_known_outcomes.py` (re-runnable if more outcomes appear).

## 5. File hygiene

Moved both ai-log files from `/Users/chayut/Downloads/` into `/Users/chayut/repos/solitaire-analytics/data/raw/`:
- `solitaire-ai-log-db1804-1779829241710.json` (4.5 MB), stalled, seed 2967897202
- `solitaire-ai-log-fef598-1779829247110.json` (9.1 MB), plateau, seed 4208249311

(The c99da9 file was left in Downloads intentionally; the user wanted to monitor it for further updates.)

## 6. c99da9 ingest: third attempt on seed 2967897202

| Field | Value |
|---|---|
| sessionId | 019e6621-ab92-78ca-8699-a0c137c99da9 |
| build | `20a825f` (NEWER than the prior two) |
| outcome | `incomplete` (still running) |
| moveCount | 31 |
| finalProgress | 8 |
| interactions | 43 (25 success / 18 error) |

**Important**: this is the THIRD attempt on the same benchmark deck. All three were `gemma-4-31b-it` at temperature 0.3, all three on `hybrid-v1` (per the `promptLayoutVersion` field), but THREE DIFFERENT template hashes.

**First 12 moves**: c99da9 was byte-identical to WON 044461 and STALLED db1804. Three different builds, three different prompt templates, identical opening. Suggests the opening sequence is robust to prompt-shape and is essentially forced.

**Divergence**: at position 12, c99da9 chose DRW where WON+STALLED both chose 8D-col1. c99da9's reasoning was one-step lookahead ("shuffling moves do not reveal any face-down cards"); WON's reasoning was three-step lookahead ("8D on 9S creates landing for black 7, which receives red 6, which lets 5S move").

**Current state**: 13-turn plateau at fc=4 / fd=15, 18 errors stacked at ti=29, no foundation progress since ti=17. Showing early signs of the same dead-deal-flailing pattern as fef598.

## 7. The v1.1 discovery

Three-way prompt structure comparison at the divergence position revealed that the c99da9 build (`20a825f`) is running a NEW prompt template, not the same `hybrid-v1` we had been ingesting from `de7dc06`.

| Field | WON 044461 (`7f01833`) | STALLED db1804 (`de7dc06`) | c99da9 (`20a825f`) |
|---|---|---|---|
| Layout name | (None, pre-versioning) | hybrid-v1 | hybrid-v1 |
| templateHash | `e2923795...` | `0462323c...` | `8971cad0...` |
| Layout | JSON-blob (single-line CURRENT GAME) | plain text | plain text (refined) |
| `confidence` field | required | required | **DROPPED** |
| `alternative_move_index` | required | required | **DROPPED** |
| Calibration paragraph (~600 chars) | present | present | **REMOVED** |
| `move_index: -1` RESIGN output | not available | not available | **NEW: ADDED** |
| NOTATION line location | inline JSON field | under CURRENT GAME | inside rules block |
| Total prompt size at pos 12 | 7603 | 7535 | **6372** |

**The harvester team shipped two of our pre-registered asks** between `de7dc06` and `20a825f`:
- Drop confidence + alternative_move_index (per `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/upcoming-export-schema-drop-confidence-altmove.md`)
- Add resign output (per `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_harvester_team_resign_hygiene_versioning_ask.md`)

Edit 4 (PRIOR REASONING truncation) was correctly NOT shipped (the local bench showed it regressed Gemma 4 untuned by -10pp).

**Versioning hygiene gap**: the `promptLayoutVersion` field still reports `hybrid-v1` in c99da9. Only `promptTemplateHash` changed. Going forward, we distinguish:
- v1.0: hash `0462323c...`
- v1.1: hash `8971cad0...`

Next harvester ask should re-request that `promptLayoutVersion` get bumped on every template-hash change (e.g. `hybrid-v1.1`, `hybrid-v1.2`). Right now anyone analysing an ai-log has to compute the hash to know which template they have.

## 8. How the prompt likely affected each at the divergence point

Three speculative effects of the v1.1 prompt change, based on a SINGLE c99da9 trace and so not yet load-bearing:

1. **Calibration paragraph removed** -> model may be LESS willing to commit. The calibration scale's "if you would not bet, do not report high confidence" was anchoring the model to take strong stances. c99da9 picked confidence 0.5 (the "tossup" tier from the calibration paragraph that ISN'T in its prompt; the field is still being filled from training distribution).

2. **alternative_move_index removed** -> model may not be doing the second-best comparison that field used to force. WON wrote explicit second-best analysis ("Move 1... does not reveal any cards"). c99da9's reasoning compares moves but doesn't pick a clear runner-up. Deliberation surface shrank.

3. **Resign output added but unused**. c99da9 has been on a 13-turn plateau and hasn't fired `move_index: -1`. The resign-trigger description is conservative ("Resign only when no legal move can productively advance, drawing has been exhausted"). At 13 turns the stock isn't fully exhausted, so the model probably doesn't consider the trigger met. May need a tighter clause (e.g. "no foundation progress in N turns").

**Important caveat**: ONE trace is not enough to attribute the divergence to the prompt change vs temperature-0.3 sampling noise. To prove a v1.1 effect, we need 5+ runs on the same deck with the same v1.1 template hash.

## 9. Memory updates

Added: `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md`

Indexed in: `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/MEMORY.md`

The memory captures: what shipped, the templateHash mapping, the versioning gap, the consequence for the ingest / bench / runner / HF pipelines, and the open question about whether v1.1 changes reasoning depth.

## 10. Active monitor

Armed a persistent background monitor (`task binea1yas`) watching `/Users/chayut/Downloads/` for new files matching `solitaire-ai-log-c99da9-*.json`. When a new export appears, it emits one notification with the filename, size, and modified-time. Useful for tracking whether c99da9 ever escapes the plateau or terminates.

To stop: `TaskStop(binea1yas)`.

## 11. Open follow-ups (in chronological priority)

1. Watch the c99da9 monitor for the next export. If c99da9 doom-loops or auto-terminates, add a third entry to `known_outcomes` on seed 2967897202.
2. Run the audit doc's H1 + H2 + H4 hypothesis tests in the next compute window (already pre-registered in `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md`). Day A: ~3 hours wallclock.
3. File the v1.1 versioning-hygiene ask back to the harvester team (request `promptLayoutVersion` bump alongside templateHash bumps).
4. Update `scripts/ingest_exports.py` and HF dataset card to handle the v1.0 / v1.1 mix (treat `confidence` and `alternative_move_index` as nullable, optionally tag rows with template hash).
5. Engine bugs from prior session still un-filed: stock face_up + orientation; multi-card move replay mismatches at moves 28/35/43.
6. v4-A experiment pre-registered in section 9.1 of `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_full_game_play_compute_window_report.md` (gemma-3n + reversal-filter LoRA).

## 12. Files created or modified today

| Path | Change |
|---|---|
| `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` | NEW (283 lines, audit + hypotheses) |
| `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_conversation_log_v1_1_discovery_and_ingest.md` | NEW (this document) |
| `/Users/chayut/repos/solitaire-analytics/data/benchmarks/winnable_decks.json` | EDIT: schema_version 1 -> 2, added `known_outcomes` field on seed 2967897202 |
| `/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-db1804-1779829241710.json` | MOVED in from Downloads |
| `/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-fef598-1779829247110.json` | MOVED in from Downloads |
| `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` | NEW memory |
| `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/MEMORY.md` | EDIT: indexed the new memory |
| `/tmp/audit_prompt.py` | NEW (audit replayer, copy to repo if iterating) |
| `/tmp/add_known_outcomes.py` | NEW (one-off schema patch) |
| `/tmp/prompt_{won,stalled,c99da9}_pos12.txt` | NEW (kept for prompt-diff inspection) |
