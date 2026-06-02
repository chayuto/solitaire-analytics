# Ingest triage: solitaire-win-9f00de (baseline, no skill)

## TL;DR
1. **Is this a loss?** Yes, a terminal loss. The deck is exhausted (stock+waste empty after 9 recycles), halted at 21/52 foundation cards, gameWon false. The `win-` prefix is a filename convention, not an outcome; read gameWon, not the filename.
2. **How to ingest?** Drop into data/raw/ and run python scripts/ingest_exports.py. classify_file tags it win_record (has moveHistory), manifest records gameWon=false. Losses are first-class; the corpus keeps everything.

## Key attributes
gameWon false; completionProgress 40%; foundations 21/52 (H6 D5 C4 S6); faceDown 5; stock/waste 0/0; recycleCount 9; model gemma-4-31b-it (gemini); build 262774b (2026-05-30); seed 999000111; difficulty 3; session 019e0fab-0000-7abc-8def-0000009f00de (#9f00de); moveHistory 150.

## Reasoning
gameWon false is authoritative on a terminal-state export. The game is genuinely stuck (stock exhausted, 9 recycles, 5 cards still buried), not paused. Honest progress that ran out of deck, not a doom-loop write-off. A kill/continue verdict only applies to a still-running session; this game is over.

## Ingest (loss does not change the procedure)
mv into data/raw/, run ingest_exports.py (idempotent, sha256). win_record = terminal state + move history; per-turn training rows come from the paired ai-log (not in this drop).

## Caveat
The file carries a _fixtureNote declaring it a synthetic terminal-loss for eval coverage; if it were ever about to enter the real corpus, confirm with the harvest team before ingesting fabricated board data.

(Backfilled from the subagent's final message; the harness blocked its direct write.)
