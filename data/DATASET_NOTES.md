# data/ layout and pipeline

Raw exports land in `data/raw/`. The ingest script deduplicates them into a
canonical store and derives two datasets from it. Run it after dropping new
exports in:

    python scripts/ingest_exports.py

## Directories

| Path | Contents | Tracked |
|---|---|---|
| `raw/` | Raw export files, immutable. Drop new exports here. | no (gitignored) |
| `raw/archive/` | Superseded raw exports (see below). Not read by the pipeline. | no (gitignored) |
| `store/interactions.jsonl` | Every interaction, deduplicated by id. | yes |
| `index/manifest.jsonl` | One row per ingested raw file, with provenance. | yes |
| `dataset/decisions.jsonl` | Every successful decision, tagged with training eligibility. | yes |
| `dataset/training.jsonl` | Local set: full records selected for the local fine-tune. | yes |
| `publish/` | Publishing set: Hugging Face dataset plus card. | yes |
| `SUMMARY.md` | Auto-generated statistics. Do not hand-edit. | yes |

## How deduplication works

Each interaction carries a unique UUIDv7 `id`. The store keys on that id, so
overlapping files and re-exports merge cleanly. No file is ever discarded, and
re-running the script is safe and idempotent.

## The two datasets

Both are selections over the same store.

- Local set (`dataset/training.jsonl`): teacher model `gemma-4-31b-it` on the
  current export schema, with stalled-game decisions filtered out. This is
  the input to `gemma4_finetune`.
- Publishing set (`data/publish/`): every successful decision across all models
  and schema versions, *including* stalled games as a research baseline.
  Published to Hugging Face under CC-BY-4.0.

Selection criteria live in one config block at the top of
`scripts/ingest_exports.py` (`TEACHER_MODEL`, `SCHEMA_CONTRACT_FIELDS`,
`STALL_TURNS`).

## Stall filter

A game is "stalled" when `foundationCards` (sum of foundation ranks) and
`faceDownTotal` (sum of `faceDownCount` across the 7 columns) have both been
unchanged for at least `STALL_TURNS` consecutive interactions. Stalled
decisions are kept in the store and the publish set, but are excluded from
the local set: every stalled decision in the harvest so far is a doom-loop
draw, and training on those teaches the model to loop. The threshold is
defined in `scripts/ingest_exports.py`. Each decision row also carries
`foundationCards`, `faceDownTotal`, `progressScore`, and `turnsSinceProgress`
for downstream analysis.

## Won sessions

Complete wins in the corpus. Kept as the positive baseline alongside the
doom-loop corpus.

- Session `…0ce0b2ce0fb4` (full: `019e3583-f286-7a29-8217-0ce0b2ce0fb4`),
  seed **unknown** (pre-logging), model `gemma-4-31b-it`, build **unknown**
  (no `appCommit` on per-interaction record at the time). Surfaced via
  cross-corpus audit on 2026-05-23. Two artefacts in `raw/`:
  `solitaire-ai-log-1779050738885.json` (interaction log, terminal turn 283,
  `completionProgress: 98`, board state shows H:KH, D:KD, C:QC, S:KS — 51
  of 52 cards on foundations, last card pending) and
  `solitaire-win-1779050713349.json` (the harvester's full-state win
  export, `gameWon: true`, `completionProgress: 100`, `moveHistory` of
  284 moves). **Confirmed second win in the corpus.** Discovered during
  the 2026-05-23 prompt-template audit (see
  `/Users/chayut/repos/solitaire-analytics/docs/reports/20260522_prompt_template_audit.md`).
  Prompt template at the time was the older 3001-char variant (hash
  `719b1734…d49703`) — predates the calibration-bands edit. **Build and
  seed are permanently lost for this session** because the harvester
  wasn't logging them yet; the only reason we know this is a win is
  the separate `solitaire-win-*.json` file. Material precedent for
  handover ask 1 (per-interaction `promptTemplateHash` +
  `promptTemplateFinalisedAt`) — without those, the same loss happens
  to the next escape too.

- Session `…1abf260154e1`, seed `3263196305`, model `gemma-4-31b-it`, app
  build `6dfc8a9`. Three exports cover the full trajectory and are all
  active in `raw/`: `solitaire-ai-log-0154e1-1779360419122.json` (178 rows,
  through turn 75), `solitaire-ai-log-0154e1-1779363194612.json` (191 rows,
  through turn 80), and the canonical `solitaire-ai-log-0154e1-1779380748971.json`
  (200 rows, terminal). Final stored state: 319 interactions, max successful
  turn `173`, `moveCount: 174`, `finalProgress: 100%`, `outcome: won`.
  The session **broke out of an emerging 8C col 6 ↔ col 7 oscillation** —
  caught at 16× in the 11:33Z snapshot — when the final stock pass
  surfaced the `10S` the model had correctly named as the bottleneck
  ("the only available black 10 is the 10C, currently buried in column 3
  under the 9D"). The unbury chain then cascaded; the terminal export
  shows 10 consecutive foundation plays in `recentMoves`. Prompt template
  was the newer 3527-char variant (hash `a39354fa…5dc551c` — the one with
  calibration-bands guidance added). **First win on the newer template**,
  and the first win with full build+seed attribution (`6dfc8a9`/
  `3263196305`) — locked as one of two candidate seeds for the pending
  same-seed cross-build experiment.

- Session `…688f5a044461`, seed `2967897202`, model `gemma-4-31b-it`, app
  build **`7f01833`** (new template, hash `e2923795…2b91b2`,
  `promptTemplateFinalisedAt` `2026-05-22T00:00:00Z`). Four exports
  cover the full trajectory: `solitaire-ai-log-044461-1779512216030.json`
  (3 rows, the early 3-error opening — would have looked like a
  "dead-on-arrival" if killed at this point),
  `solitaire-ai-log-044461-1779532129400.json` (335 rows, 150 success,
  at `finalProgress: 71%` mid-game),
  `solitaire-ai-log-044461-1779532834013.json` (342 rows, +7 new),
  and the canonical `solitaire-ai-log-044461-1779533681032.json` (362
  rows, terminal, `outcome: won`). Companion win record
  `solitaire-win-044461-1779533686224.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of **194 moves**) — and
  importantly, **the win record stamps `seed: 2967897202` and
  `appCommit: 7f01833` directly at the top level**, so attribution is
  intact even without joining against the interaction log. **First win
  on build `7f01833`**, **second win on the new calibration-bands
  template**, and the **first win on the harvester's post-Ask-1
  schema** (every successful turn now carries `promptTemplateHash` +
  `promptTemplateFinalisedAt`). Notable: this session was almost
  written off after the 3-row opening snapshot — operator instinct
  to wait for the second snapshot was the right call, and the
  framing in the doom-loop section below has been updated to
  reflect that. Material data points: (1) build `7f01833` does
  produce wins, retiring the morning's "all-losses-on-7f01833"
  framing; (2) `2967897202` is the second known-winnable seed in
  the corpus and a candidate alongside `3263196305` for same-seed
  cross-build experiments; (3) this is the first end-to-end win
  with full prompt-template + inference-config attribution on the
  per-interaction record, so it's the cleanest training-data win
  the corpus has.

- Session `…2c84bac05ad4`, seed `3263196305`, model `gemma-4-31b-it`, app
  build **`cef6291` (hybrid-v1.2)**. Two artefacts in `raw/`:
  `solitaire-ai-log-c05ad4-1779911237818.json` (364 rows, 173 success /
  191 errors, canonical interaction log) and the win record
  `solitaire-win-c05ad4-1779911235693.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of **296 moves**, seed and
  appCommit stamped at top level). Final stored state: `moveCount: 296`,
  `finalProgress: 100%`, outcome `won`. **Third end-to-end win in the
  corpus and the first win on v1.2.** Material context: this is the
  same seed as the prior `0154e1` win (build `6dfc8a9`, 174 moves), so
  it confirms v1.2 can still reach a win on a known-winnable deck — but
  it took **296 moves vs 174 on the same deck** (1.7× longer), and the
  trajectory shows a **94-turn flat plateau on `(foundationCards=14,
  faceDownTotal=8)` from turn 129 to turn 223** with active oscillation
  (`3H col 3 ↔ col 4` 85× and `4C col 3 ↔ col 4` 73× session-wide)
  before the breakout. After turn 223 the cascade was rapid: faceDown
  8 → 0 in ~26 turns then foundations 14 → 51 in ~70 turns. This is the
  **first corpus session to break out of a >50-turn doom-loop** — every
  prior session with a comparable plateau either stalled
  (`stalled_auto_terminated`) or was operator-killed. Two lessons: (1)
  v1.2 has not eliminated the doom-loop pathology; it has shown that
  the model can occasionally escape one given enough recycles, (2) the
  stall-filter `STALL_TURNS=25` would have excluded turns 154-222 of
  this win from `dataset/training.jsonl` — worth confirming whether the
  filter handles "stalled but eventually escaped" correctly, since
  those are the most interesting recovery decisions in the corpus.
  Doom-loop fingerprint (`3H/4C col 3 ↔ col 4` for 94 turns) is the
  same shape as the `…993e6cadf71b` adf71b loop catalogued below —
  evidence that the loop type recurs across builds and seeds even on
  decks that win.

- Session `…3ced34aca45a`, seed `2853966634`, model `gemma-4-31b-it`, app
  build **`cef6291` (hybrid-v1.2)**. Two artefacts in `raw/`:
  `solitaire-ai-log-aca45a-1779936098329.json` (494 rows, 253 success /
  241 errors, canonical interaction log) and the win record
  `solitaire-win-aca45a-1779936096358.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of **418 moves**, seed and
  appCommit stamped at top level). Two earlier snapshots archived to
  `raw/archive/` (the mc-69 emerging-loop snapshot and the mc-393
  mid-cascade snapshot). Final stored state: `moveCount: 418`,
  `finalProgress: 100%`, outcome `won`. **Fourth end-to-end win in the
  corpus, second win on v1.2, and the first v1.2 win on a fresh
  (non-anchor) seed.** **This is the canonical doom-loop-then-breakout
  example.** The session ran a `5H/4S/3H col 3 ↔ col 6` three-card
  oscillation that bounced roughly 290 times across the mid-game
  plateau (session-wide counts `3H` 106×, `4S` 97×, `5H` 84×) before
  breaking out, revealing all 21 face-down cards, and cascading to the
  win. **Material implications**: (1) v1.2 produces wins on seeds whose
  early game looks like a dead doom-loop, so kill-verdicts at first-
  plateau would have wrongly terminated a winnable session. (2) The
  v1.3 bench at
  `/Users/chayut/repos/solitaire-analytics/docs/reports/20260528_prompt_v1_3_candidate_spec.md`
  MUST run sessions to terminal state (win, auto-terminate, or operator
  kill) rather than scoring at first-plateau detection, otherwise the
  v1.2 baseline win rate is undercounted. (3) The `STALL_TURNS=25` filter
  excludes the ~290-turn oscillation decisions from `dataset/training.jsonl`
  but keeps the post-breakout cascade, which is the correct behavior for
  a training corpus (don't teach the loop, do teach the recovery). (4)
  Together with `c05ad4` (seed 3263196305, 94-turn plateau then win),
  this establishes that the v1.2 doom-loop is frequently RECOVERABLE
  rather than terminal. The pathology is "slow and wasteful," not
  "fatal," on at least some winnable decks. This reframes the v1.2
  failure-mode severity: the regression is a throughput problem (sessions
  burn hundreds of retries oscillating before winning) more than a
  win-rate problem.

- Session `…e12004d29c8c` (full: `019e6e6b-035d-794d-b373-e12004d29c8c`),
  seed `1388178981`, model `gemma-4-31b-it`, app build `7c946d4`, prompt
  **`hybrid-v1.3`** (templateHash `7d9ecda4…9772bb`). Two artefacts in `raw/`:
  `solitaire-ai-log-d29c8c-1780006961542.json` (149 rows, 140 success / 9
  errors, canonical interaction log) and the win record
  `solitaire-win-d29c8c-1780006959645.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 180 moves, seed and appCommit
  stamped at top level). Final stored state: `moveCount: 180`,
  `finalProgress: 100%`, outcome `won`. **First clean v1.3 win in the corpus,
  on a fresh (non-anchor) seed.** Clean is literal here: zero exact-reversal
  tableau oscillations across the whole game, a longest no-reveal shuffle
  burst of 1, and all 21 face-down cards revealed. The dataset retains 119
  success decisions from this session, all 119 `trainingEligible` with empty
  `excludeReasons`, so it is the first v1.3 session to contribute its entire
  trace to `dataset/training.jsonl` with nothing dropped by the stall filter.
  Its 6% error rate (9 of 149 rows) is far below background, so the trajectory
  is dense. Role in corpus: the positive control for v1.3, the counterpart to
  `9b1c4a` (same model, same template, same 2026-05-28 export batch) showing
  that v1.3 wins efficiently when the deck presents no oscillation trap.

- Session `…61e2fa9b1c4a`, seed `3263196305`, model `gemma-4-31b-it`, app
  build `8e65592`, prompt **`hybrid-v1.3`** (templateHash `7d9ecda4…9772bb`,
  byte-identical to `d29c8c` despite the earlier build commit). Two artefacts
  in `raw/`: `solitaire-ai-log-9b1c4a-1780006955362.json` (364 rows, 260
  success / 104 errors, canonical interaction log) and the win record
  `solitaire-win-9b1c4a-1780006954228.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 360 moves, seed and appCommit
  stamped at top level). Final stored state: `moveCount: 360`,
  `finalProgress: 100%`, outcome `won`. **First v1.3 win on the anchor seed
  `3263196305`, and the slowest win yet recorded on that deck.** Same seed as
  `…1abf260154e1` (build `6dfc8a9`, 174 moves) and `…2c84bac05ad4` (v1.2
  `cef6291`, 296 moves); the move count to win has now climbed 174, 296, 360
  across `6dfc8a9`, v1.2, and v1.3 on this identical deck, so v1.3 did not
  improve throughput here (whether the trend is real or `temp=0.3`
  stochasticity needs more runs). The pathology is the established
  slow-and-wasteful loop: the `7H` plus `6S` run (sometimes the full
  `7H 6S 5H 4S 3H` sequence) slid `col 6 ↔ col 7`, where column 6's top card
  was `8S` and column 7's was `8C`, two interchangeable black 8s. The move
  stream shows 26 `7H` tableau moves, 23 of them exact reversals (32 exact
  reversals across all cards). Every toggle carried the identical rationale
  ("the priority is to expose face-down cards"), but the run always rested
  directly on an already-face-up black 8, so it exposed nothing: only 17 of
  the 63 tableau-to-tableau moves actually flipped a face-down card, and the
  stall filter excluded 82 of the 260 decisions as `stalled-game` (every
  tableau shuffle, all 26 `7H` moves), so none of the oscillation reached
  `dataset/training.jsonl`. v1.3's anti-undo bullet ("Do not move a card to a
  tableau column it occupied in the last 5 moves shown in RECENT MOVES") was
  present and the model invoked it elsewhere ("I must avoid undoing recent
  moves, which eliminates move [0]"), but it failed to stop this loop for two
  reasons: the reveal-priority bullet is listed first and the model treats it
  as primary, overriding anti-undo whenever it believes a move exposes a
  face-down card; and the oscillation period exceeded the 5-move RECENT MOVES
  horizon, so each reversal looked fresh. Role in corpus: confirms the
  v1.2-era finding (`c05ad4`, `aca45a`) that the doom-loop is recoverable
  rather than fatal carries into v1.3, shows the v1.3 anti-undo predicate does
  not close the reveal-priority override hole, and pairs with `d29c8c` as the
  negative control in the same export batch.

- Session `…0698032fd837`, seed `4161700176`, model `gemma-4-31b-it`, app
  build `7c946d4`, prompt **`hybrid-v1.3`** (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-2fd837-1780089504045.json` (348 rows,
  267 success / 81 errors, ingested 2026-05-30) and the win record
  `solitaire-win-2fd837-1780089502228.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 335 moves, seed and appCommit
  stamped at top level). Final stored state: `moveCount: 335`,
  `finalProgress: 100%`, outcome `won`. **Third v1.3 win, second messy one
  (after `9b1c4a`).** A genuine win on a fresh seed, but stock-heavy and
  oscillation-laced: 135 of 248 success decisions were draws (plus 9 stock
  recycles), and a `9H 8C 7D` run sloshed `col 5 ↔ col 6` (session-wide window
  counts `9H` 85×, `8C` 76×, `7D` 68×; 16 exact reversals on the ordered move
  stream, led by `3C`, `TS`, `9H`). The stall filter excluded 86 of 248
  decisions as `stalled-game`, leaving 162 (65%) in clean-lean, a yield between
  `d29c8c` (100%, clean) and `9b1c4a` (68%). Reinforces the v1.3-on-31B
  pattern: it wins on winnable decks but burns hundreds of moves cycling the
  stock and toggling a mid-board run rather than committing, the recoverable
  slow-and-wasteful loop rather than a fatal one.

- Session `…8850c4825d4b` (full: `019e739e-05d9-7422-8214-8850c4825d4b`),
  seed `549440324`, model `gemma-4-31b-it`, app build `3136c81`
  (2026-05-29T09:45:21Z), prompt **`hybrid-v1.3`** (templateHash
  `7d9ecda4…9772bb`, byte-identical to `d29c8c`/`9b1c4a`/`2fd837` despite the
  different build commit). Two artefacts in `raw/`:
  `solitaire-ai-log-825d4b-1780091777318.json` (227 rows, 165 success / 62
  errors, ingested 2026-05-30) and the win record
  `solitaire-win-825d4b-1780091779917.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 226 moves, seed and appCommit
  stamped at top level). Final stored state: `moveCount: 226`,
  `finalProgress: 100%`, outcome `won`. **Fourth v1.3 win, and the cleanest of
  the messy ones (sits between clean `d29c8c` and messy `9b1c4a`/`2fd837`).** A
  fresh seed with a near-monotonic trajectory: faceDownTotal fell 21 to 0
  (reaching 0 at success-turn 137 of 165), foundations climbed 0 to 51, longest
  foundationCards plateau only 17 turns, all 21 face-down cards revealed (21
  `flip_card` moves). Draws were a minority (70 of 226 applied moves, 31%), so
  it was not the stock-heavy grind `2fd837` was.
  **Window-count vs exact-reversal correction (read before trusting the
  briefing).** The `load_export.py` briefing flags heavy session-wide
  oscillation (`8H col 5 ↔ col 7` 24×, `9D col 1 ↔ col 3` 22×, `3H col 3 ↔
  col 4` 19×), but those are `session_oscillation` window counts and are
  overlap-inflated. The true ordered move stream (overlapping recentMoves
  windows stitched, then inverses counted) holds only **18 exact reversals
  across the whole game, distributed with no card exceeding 3** (`8H` 3×, `QC`
  3×, then `7C`/`TS`/`9D`/`JH` at 2× each). That is fewer than `9b1c4a` (32,
  of which 23 on `7H` alone) and on par with `2fd837` (16), and the churn is
  scattered rather than concentrated in one sustained loop. This is why the
  stall filter excluded only **1 of 165** decisions as `stalled-game` (164
  entered `dataset/training.jsonl`, a 99.4% yield, against 65-68% for the two
  messy wins and 100% for clean `d29c8c`): low-amplitude distributed toggling
  never forms a 25-turn flat window. Role in corpus: a near-clean v1.3 positive
  example, and a standing caution that the briefing's window-count headline
  overstates loop severity, the ordered-stream exact-reversal count is the one
  to cite.

- Session `…79e63413180a` (full: `019e739f-50dc-7e88-876a-79e63413180a`),
  seed `2003817730`, model `gemma-4-31b-it`, app build `3136c81`
  (2026-05-29T09:45:21Z, same build as `825d4b`), prompt **`hybrid-v1.3`**
  (templateHash `7d9ecda4…9772bb`). Two artefacts in `raw/`:
  `solitaire-ai-log-13180a-1780100997223.json` (237 rows, 164 success / 73
  errors, ingested 2026-05-30) and the win record
  `solitaire-win-13180a-1780100999156.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 269 moves, seed and appCommit
  stamped at top level). Final stored state: `moveCount: 269`,
  `finalProgress: 100%`, outcome `won`. **Fifth v1.3 win, on a fresh seed.**
  faceDown fell 21 to 0, foundations climbed 0 to 51, and all 156 success
  decisions entered `dataset/training.jsonl` (100% clean-lean yield, like
  `825d4b`), so despite 269 moves no 25-turn flat plateau ever formed.
  **Reveal discipline (the metric now tracked, see memory
  `reveal-pass-up-kill-signal`): 0% pass-up, the model took all 19 of the 19
  offered `(reveals a hidden card)` moves.** That is the cleanest reveal
  discipline in the corpus and the strongest win-side point on the pass-up
  gradient (wins now 0/6/6/7/12%, the lone kill `a1d118` at 27%). As with
  `825d4b`, the briefing's window counts read heavy (`9H col 2 ↔ col 6` 62×,
  `6H col 1 ↔ col 5` 40×, `5C col 3 ↔ col 5` 36×) but overstate severity: the
  ordered-stream top card reversed only about 8 times (no single sustained
  loop), and 63% of the 52 tableau-to-tableau moves flipped nothing, the
  familiar diffuse no-reveal-branch churn that costs throughput without
  stalling. Role in corpus: a clean confirmation that v1.3 wins keep reveal
  pass-up low, and another caution that the window count alone cannot grade a
  v1.3 session.

- Session `#5e2558` (full: `019e7aa9-cf17-74f6-844b-c634a45e2558`), seed
  `839179948`, model `gemma-4-31b-it`, app build `262774b`. Two artefacts in
  `raw/`: `solitaire-ai-log-5e2558-1780209396614.json` (217 rows, 186 success
  / 31 errors, canonical interaction log) and the win record
  `solitaire-win-5e2558-1780209395682.json` (`gameWon: true`,
  `completionProgress: 100`, 244 moves). Ingested 2026-05-31 (+155 success
  decisions, +150 local-set rows). Final stored state: max successful turn
  `243`, `moveCount: 244`, `finalProgress: 100%`, outcome `won`.

  Near-monotonic win. foundationCards climbs 0 to 51 (the KD off the waste then
  completes 52) and faceDownTotal falls 21 to 0, first reaching 0 at turn 228.
  The only genuine no-progress plateau is a short `(foundationCards=3,
  faceDownTotal=16)` stretch around turns 23 to 48 that the teacher worked
  through; the later `faceDownTotal=2` hold (turns 136 to 224) is not a stall,
  since foundations climb 12 to 36 across it (active endgame).

  Canonical false-positive for the session-wide oscillation detector. The
  ingest briefing flagged `7D col 4 ↔ col 6` 42×, `8S col 4 ↔ col 6` 37×, and
  `9H col 4 ↔ col 6` 32×, but a de-inflated recount on the stitched ordered
  move stream (222 of the 244 moves, reconstructed from the rolling windows)
  finds zero exact two-move reversals for any card or column pair. The 65
  tableau-to-tableau moves are spread across cards (top: 7D 7×, TC 7×, 9C 6×,
  8S 6×) with no card ever moved straight back to the column it just left. The
  inflated count is rolling-window lingering (one real move is recounted once
  per turn it remains in the last-10 window) compounded by the
  `frozenset({4,6})` column-pair grouping, which merges col 4 to col 6 with the
  reverse direction even when the two are non-consecutive. The disambiguator is
  the trajectory: faceDownTotal falls monotonically to 0 and the game wins, so
  a sustained `col 4 ↔ col 6` doom-loop is impossible here. Contrast
  `…2c84bac05ad4` above, where a comparable session-wide count (`3H/4C col 3 ↔
  col 4` 85×/73×) WAS a real 94-turn plateau: the count alone cannot separate
  the two, only the foundationCards/faceDownTotal trajectory can.

- Session `#a11e74` (full: `019e7aaa-0dfa-722b-9fd9-03ff87a11e74`), seed
  `601852437`, model `gemma-4-31b-it`, app build `262774b`. WON on continuation.
  Canonical interaction log `solitaire-ai-log-a11e74-1780219324803.json` (487 rows,
  449 success) and win record `solitaire-win-a11e74-1780219323762.json`
  (`gameWon: true`, `completionProgress: 100`, 474 moves). An earlier 77% snapshot
  pair was ingested the same day (`solitaire-ai-log-a11e74-1780215777088.json` 474 rows +
  the new-format `solitaire-game-a11e74-1780215783804.json`, `gameWon: false`); this
  terminal export supersedes it. Final stored state: `moveCount: 474`,
  `finalProgress: 100%`, outcome `won`.

  Slow-breakout win. It sat at `(foundationCards=4, faceDownTotal=17)` for 119 successful turns (turn-index span ~16 to 336) before breaking out (faceDown 17 to 0, all revealed) and cascading to the win;
  de-inflated reversals over the full game are 0 (the briefing's `9S/8D col 4 ↔ col 5`
  window counts are inflation, not a loop). pyksolve on the 77% snapshot's all-face-up
  state had already confirmed 10/10 winnable, so the no-kill call held: the ~300-turn
  early plateau cost throughput but not the game. Notable as the session that proved the
  `solitaire-game-*` export is a mid-game SNAPSHOT, not a terminal record (its 77%
  `gameWon: false` snapshot preceded this win by ~1 hour).

- Session `#5c25ad` (full: `019e7d96-4bd5-74eb-98f1-16991f5c25ad`), seed
  `4221577640`, model `gemma-4-31b-it`, app build `f5c3870`
  (2026-05-31T10:27:59Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`).
  Two artefacts in `raw/`: `solitaire-ai-log-5c25ad-1780260671117.json` (211
  rows, 154 success / 57 errors, canonical interaction log) and the win record
  `solitaire-win-5c25ad-1780260673575.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 205 moves, seed and appCommit
  stamped at top level). Ingested 2026-05-31. Final stored state:
  `moveCount: 205`, `finalProgress: 100%`, outcome `won`, terminal faceDown 0,
  recycleCount 4. The ai-log's last logged board sits at `foundationCards=51`
  with only the KC pending in column 1 ("the only remaining card is the King
  of Clubs ... the final card needed to win"); the win record carries the
  closing move. The briefing's session-wide window counts read heavy (`6S col
  4 ↔ col 6` 21×, `4S col 4 ↔ col 6` 18×, `5H col 4 ↔ col 6` 16×) but are the
  familiar rolling-window inflation, not a sustained loop: faceDownTotal
  reaches 0 and the game wins, so no `col 4 ↔ col 6` doom-loop is possible
  here (see `#5e2558` for the canonical false-positive write-up).

- Session `#bcd6cf` (full: `019e7cfc-3d20-73bf-be59-3d6786bcd6cf`), seed
  `2044240526`, model `gemma-4-31b-it`, app build `262774b`
  (2026-05-30T07:08:17Z, the older of the two builds in this batch), prompt
  `hybrid-v1.3` (templateHash `7d9ecda4…`, identical to the two `f5c3870` wins
  ingested alongside it). Two artefacts in `raw/`:
  `solitaire-ai-log-bcd6cf-1780254408610.json` (262 rows, 200 success / 62
  errors, canonical interaction log) and the win record
  `solitaire-win-bcd6cf-1780254407044.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 257 moves). Ingested 2026-05-31.
  Final stored state: `moveCount: 257`, `finalProgress: 100%`, outcome `won`,
  terminal faceDown 0, recycleCount 5. Last logged board at `foundationCards=51`,
  KC pending in column 3; win record carries the close. Session-wide window
  counts (`2C col 5 ↔ col 6` 65×, `6D col 2 ↔ col 3` 44×, `7S col 2 ↔ col 3`
  42×) are rolling-window inflation over a long (257-move) but winning game,
  not a loop. Notable for attribution: this carries the same v1.3 prompt
  (`7d9ecda4…`) as the two `f5c3870` wins ingested with it, so the `262774b`
  to `f5c3870` build delta is harness-side, not a prompt change, and decks on
  both builds still win.

- Session `#62f09b` (full: `019e7d96-d8d9-7317-985a-e73b1c62f09b`), seed
  `3590201206`, model `gemma-4-31b-it`, app build `f5c3870`
  (2026-05-31T10:27:59Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-62f09b-1780254397973.json` (232 rows,
  200 success / 32 errors, canonical interaction log) and the win record
  `solitaire-win-62f09b-1780254328065.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 259 moves). Ingested 2026-05-31.
  Final stored state: `moveCount: 259`, `finalProgress: 100%`, outcome `won`,
  terminal faceDown 0, recycleCount 5. Last logged board at `foundationCards=51`,
  KH pending in column 1; win record carries the close. The lowest error count
  of the batch (32 / 232) and a clean endgame: 9 of the last 10 logged moves
  are foundation plays. Session-wide window counts (`4D col 2 ↔ col 3` 76×, `6D
  col 2 ↔ col 3` 75×, `5C col 2 ↔ col 3` 75×) are rolling-window inflation, not
  a loop (faceDownTotal reaches 0, game wins).

- Session `#c27334` (full `019e823f-e4fe-784f-89c5-f1251fc27334`), seed
  `350743738`, model `gemma-4-31b-it`, app build `df3a89b` (2026-05-31T12:10:49Z,
  the newest build in the corpus), prompt `hybrid-v1.3` (templateHash
  `7d9ecda4…`). Two artefacts in `raw/`:
  `solitaire-ai-log-c27334-1780340238184.json` (210 rows, 178 success / 32
  errors) and the win record `solitaire-win-c27334-1780340236876.json`
  (`gameWon: true`, `completionProgress: 100`, `moveHistory` of 205 moves).
  Ingested 2026-06-02. Final stored state: `moveCount: 205`, `finalProgress:
  100%`, outcome `won`, terminal faceDown 0, recycleCount 6. A clean win: the
  session-wide window counts are modest (`TD col 1 ↔ col 3` 37×, `JC col 1 ↔
  col 3` 17×, `7S col 1 ↔ col 6` 14×, well below the doom-loop range) and the
  latest window is a non-looping endgame (`KH`/`KS` to foundation). All 132
  success decisions entered `clean-lean`/`training.jsonl` (100% yield, no
  25-turn plateau to stall-filter). First win on build `df3a89b`.

- Session `#50aff7` (full `019e87fe-eaf7-72eb-b930-cb5bfe50aff7`), seed
  `405489085`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-50aff7-1780438039873.json` (263 rows,
  178 success / 85 errors, canonical interaction log) and the win record
  `solitaire-win-50aff7-1780438038906.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 248 moves, seed and appCommit
  stamped at top level). Ingested 2026-06-03. Final stored state: `moveCount:
  248`, `finalProgress: 100%`, outcome `won`, terminal faceDown 0, recycleCount
  4. Last logged board at `foundationCards=51` (H:KH D:KD C:KC S:QS), KS the
  last card pending; the win record carries the close. Second win on build
  `df3a89b` (after `#c27334`), same `hybrid-v1.3` template.

- Session `#bf6d85` (full `019e87ff-6b03-799c-8e05-543f48bf6d85`), seed
  `3841057237`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-bf6d85-1780433292446.json` (238 rows,
  162 success / 76 errors, canonical interaction log) and the win record
  `solitaire-win-bf6d85-1780433293821.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 210 moves, seed and appCommit
  stamped at top level). Ingested 2026-06-03. Final stored state: `moveCount:
  210`, `finalProgress: 100%`, outcome `won`, terminal faceDown 0, recycleCount
  3. Last logged board at `foundationCards=51` (H:KH D:QD C:KC S:KS), KD the
  last card pending; the win record carries the close. Third win on build
  `df3a89b`.

- Session `#a6acda` (full `019e8a63-79a8-7601-ad05-f79bb4a6acda`), seed
  `3123337720`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-a6acda-1780486179894.json` (300 rows,
  186 success / 114 errors, canonical interaction log) and the win record
  `solitaire-win-a6acda-1780486178906.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 274 moves, seed and appCommit
  stamped at top level). Ingested 2026-06-04. Final stored state: `moveCount:
  274`, `finalProgress: 100%`, outcome `won`, terminal faceDown 0, recycleCount
  4. Last logged board at `foundationCards=51`, KS the last card pending in
  column 3; the win record carries the close (final move KS col3 -> spades
  foundation). A messy win on a fresh seed: the briefing's session-wide window
  counts read like a loop (`4S col 2 ↔ col 4` 64×, `3H col 2 ↔ col 4` 60×, `9H
  col 5 ↔ col 7` 60×), but those are rolling-window inflation, not verified exact
  reversals -- faceDownTotal reached 0 and the game won, so the concentrated
  col2↔col4 churn on the 4S/3H pair is the same recoverable slow-and-wasteful
  pattern as `9b1c4a`/`2fd837`, not a fatal lock. Error rate is the highest of
  the four df3a89b wins (114 / 300, 38%) but does not change the outcome. Fourth
  win on build `df3a89b`.

- Session `#aa3e4d` (full `019e8d55-22e3-7d8b-831d-0f7ad8aa3e4d`), seed
  `1145639637`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…9772bb`).
  Two artefacts in `raw/`: `solitaire-ai-log-aa3e4d-1780556562059.json` (351
  rows, 202 success / 149 errors, canonical interaction log) and the win record
  `solitaire-win-aa3e4d-1780556562609.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 276 moves, seed and appCommit
  stamped at top level). The harvester re-exported this same won session about
  four minutes later (`solitaire-win-aa3e4d-1780556319392.json`, byte-identical
  sha256, plus `solitaire-ai-log-aa3e4d-1780556321230.json`, differing only in
  `exportedAt`); the store dedups interactions by UUIDv7 `id`, so a re-export
  adds no rows, and only the canonical pair was ingested (the redundant pair left
  in Downloads). Ingested 2026-06-04.
  Final stored state: max successful turn `275`, `moveCount: 276`,
  `finalProgress: 100%`, outcome `won`, terminal faceDown 0, recycleCount 7.

  A messy, stock-heavy win. faceDownTotal falls 21 to 0 (reaching 0 at move 237
  of 276) and foundations climb 0 to 52 (40 from the tableau, 12 from the waste).
  Draws are 95 of the 276 moves (34%) on top of 7 recycles. The mid-game stalls
  around nine hidden cards: a 39-move no-progress plateau at `(foundationCards=9,
  faceDownTotal=9)` over moves 131 to 170, part of a longer roughly 86-move churn
  pinned near faceDownTotal 9. The oscillation is real but diffuse, 36 exact
  adjacent reversals across the 96 tableau-to-tableau moves, with the top card
  `3C` reversed only 4 times (then `5C`/`3D`/`4D`/`7H`/`9D` at 3 each), spread
  across many cards rather than one dominant sustained loop. Per the corpus
  disambiguator, faceDownTotal reaches 0 and the game wins, so this is the
  recoverable slow-and-wasteful diffuse pattern (cf. `9b1c4a`/`2fd837`/`a6acda`),
  not a fatal lock. Error rate (149/351, 42%) is in line with the messier
  `df3a89b` wins. Fifth win on build `df3a89b`.

- Session `#ca9bbe` (full `019e8f38-5f6c-7d8d-93c3-481ea2ca9bbe`), seed
  `4197389931`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). Two
  artefacts in `raw/`: `solitaire-ai-log-ca9bbe-1780573235553.json` (339 rows,
  209 success / 130 errors, canonical interaction log) and the win record
  `solitaire-win-ca9bbe-1780573233672.json` (`gameWon: true`,
  `completionProgress: 100`, `moveHistory` of 298 moves, seed and appCommit
  stamped at top level). Ingested 2026-06-04. Final stored state: max successful
  turn `297`, `moveCount: 298`, `finalProgress: 100%`, outcome `won`, terminal
  faceDown 0, recycleCount 6.

  A messy but recovered win. faceDownTotal falls 21 to 0 (reaching 0 at move 249
  of 298) and foundations climb 0 to 52 (44 from the tableau, 8 from the waste).
  Draws are 96 of the 298 moves (32%) on top of 6 recycles. The one real stall is
  late and single-card: a 36-move plateau at `(foundationCards=14,
  faceDownTotal=1)` over moves 212 to 248, the last hidden card pinned until a
  reveal broke it and foundations cascaded 14 to 52 in the final ~49 moves.
  Oscillation is diffuse, 35 exact adjacent reversals across the 113
  tableau-to-tableau moves with the top cards `4D` and `2D` reversed 4 times each
  (then `9D`/`3D`/`3C` at 3), spread rather than one sustained loop. Reveal
  discipline is clean: 19 reveal-turns offered, 18 taken (5% pass-up, the winning
  band). Sixth win on build `df3a89b`, and the second of today's batch alongside
  `#aa3e4d`.

- Session `#3e91a0` (full `019e8f95-c361-7aaa-9ba8-8129aa3e91a0`), seed
  `3169322146`, model **`gemma-4-26b-a4b-it`** (26B MoE cohort; excluded from the
  default training set by the `TEACHER_MODEL=gemma-4-31b-it` filter but kept in
  the `client_v1_26b_*` comparison cohort), app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). **The
  first 26B win in the corpus.** Three artefacts in `raw/`: the terminal
  interaction log `solitaire-ai-log-3e91a0-1780654875837.json` (521 rows, 220
  success / 301 errors, canonical) and an earlier mid-game export
  `solitaire-ai-log-3e91a0-1780636587030.json` (434 rows, captured at
  `finalProgress: 58%`, outcome `incomplete`, exported ~5h before the finish);
  on ingest the terminal log added only 87 new interactions over the 434 already
  carried by the mid-game export (deduped by UUIDv7 `id`), so both are kept and
  unioned. Plus the win record `solitaire-win-3e91a0-1780654874570.json`
  (`gameWon: true`, `completionProgress: 100`, `moveHistory` of 370 moves, seed
  and appCommit stamped at top level). Ingested 2026-06-05. Final stored state:
  max successful turn `369`, `moveCount: 370`, `finalProgress: 100%`, outcome
  `won`, terminal faceDown 0, recycleCount 4.

  A doom-loop-then-breakout win, and far loopier than any 31B win. faceDownTotal
  falls 21 to 0 early (reaching 0 at move 185 of 370) and foundations climb 0 to
  52 (38 from the tableau, 14 from the waste), but the game then sat at
  `(foundationCards=28, faceDownTotal=0)` for a 124-move plateau (moves 191 to
  315) with every card already face-up, before the final cascade 28 to 52. The
  oscillation is a real sustained loop, not the diffuse churn of the 31B wins: 68
  exact adjacent reversals across 194 tableau-to-tableau moves, dominated by `9H`
  (23×) and `8S` (23×) sliding back and forth, the 26B "ignores-and-loops"
  signature (cf. the `cbced2` 26B example in the obedience-trap note) that this
  time resolved into a win rather than a stall. Reveal discipline is poor by win
  standards: 22 reveal-turns offered, 6 passed up (27% pass-up), which equals the
  `a1d118` kill-signal level yet still won, a caution that the pass-up threshold
  may not transfer from 31B to the 26B MoE. Error rate is high (301/521, 58%).
  Role in corpus: the first existence proof that the 26B MoE can finish a game,
  and the first won trace in the `client_v1_26b_*` cohort, which the pipeline's
  hardcoded "no wins" print and the HF dataset card no longer describe correctly.

- Session `#6eb393` (full `019e99f9-05a8-7eb3-9d21-76995f6eb393`), seed
  `4250754298`, model `gemma-4-31b-it`, app build **`6810750`**
  (2026-06-05T22:28:30Z), prompt **`hybrid-v1.5`** (promptTemplateHash
  `8a46ca22…`, `promptTemplateFinalisedAt` 2026-06-06). **The first
  `hybrid-v1.5` session in the corpus, and a win.** (The operator flagged it as
  "1.4"; the version fields and templateHash confirm it is v1.5, the shipped
  form of the `docs/reports/20260606_v1_5_harvester_ask.md` ask, NOT the
  `fa14fe3`/`818edeb2` v1.4.) The two v1.5 prompt changes are present in the
  rendered prompt: the STRATEGY GUIDANCE draw-directive bullet ("Drawing from the
  stock is the correct action when...") is DELETED, and the PROGRESS line carries
  the two new counts ("turns since foundation grew", "turns since a card was
  revealed"). Two artefacts in `raw/`: the interaction log
  `solitaire-ai-log-6eb393-1780743062479.json` (227 rows, 145 success / 82 errors,
  36% error rate) and the win record `solitaire-win-6eb393-1780743061393.json`
  (`gameWon: true`, `completionProgress: 100`, `moveHistory` 254 moves, seed and
  appCommit stamped). Ingested 2026-06-06 via the new Parquet-shard append (3 tail
  shards rewritten; integrity verified by id and field). Final stored state:
  `moveCount: 254`, `finalProgress: 100%`, outcome `won`, terminal faceDown 0,
  recycleCount 2; the game finished on the last card `KC` to the clubs foundation
  after a closing foundation cascade.

  A messy win, not a clean one: session-wide window counts show oscillation
  (`3C col 2 ↔ col 4` 65x, `4D col 2 ↔ col 4` 58x, `9C col 3 ↔ col 5` 54x), the
  usual rolling-window inflation over a long game rather than a measured
  reversal count (see the oscillation-window-count-inflates note; exact-reversal
  severity and the v1.5 reasoning-behaviour change are analysed separately). The
  point that matters for v1.5: deleting the draw-directive did NOT make the model
  under-draw and lose this board; it recycled the stock twice and completed.

The harvester's v1.1+ prompt offers `move_index: -1` as an explicit resign
("Resign only when no legal move can productively advance the game, drawing has
been exhausted, and you would not bet on any of the available moves to recover.
Resign is final and ends the session."). This section records sessions that
exercised it. A resign is correct only if the board is genuinely unwinnable from
the resign position; a resign on a winnable board would be a new false-resign
failure mode, so each resign needs the winnability adjudication recorded.

- Session `#30e5e5` (full `019e87ff-89eb-7cc2-b8e2-0acf9e30e5e5`), seed
  `770499954`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). One
  artefact in `raw/`: `solitaire-ai-log-30e5e5-1780438153952.json` (246 rows,
  149 success / 96 errors, 1 resigned). Ingested 2026-06-03. Final stored state:
  `moveCount: 186`, `finalProgress: 48%`, session-level outcome `incomplete`
  with a terminal `resigned` interaction at turnIndex 186. This is the **first
  time the `move_index: -1` resign action actually fired in the corpus, and it
  is a correct resign.** Prior notes record the resign output failing to fire on
  dead boards (v1.2 across 90 plateau turns, below) and the `#4a9fe1` "false
  resignation" where the 31B teacher gave up in its reasoning but never emitted
  the action; #30e5e5 is the first time the action itself was emitted. Resign
  board:
  foundations H:4H D:KD C:2C S:6S (25/52), col5 `?? ?? ?? ?? 7H 6C 5H 4C` (4
  face-down), col6 `KS QH JC TH 9C 8H 7S 6H`, col2 `KH QS JH TS 9H`, col3 `KC`,
  col4 `QC`, col1 and col7 empty, waste 5C 7C TC JS, stock empty with recycle
  available. The board is provably dead: by elimination across all 52 cards the
  4 face-down cards in col5 are exactly {3C, 8C, 8S, 9S}, and the col5 top card
  4C can never move (the only red 5s are 5H, trapped directly beneath it, and
  5D, already up in the complete diamond foundation) nor be covered (both red 3s
  are in the foundation), so col5 is frozen and 3C and 5H are buried forever
  (clubs locked at 2C, hearts at 4H). Exhaustive reachability over the repo
  engine from this position confirms it: only 36 distinct states are reachable,
  no win is reachable, and the foundation count cannot advance past 25/52 at all
  (the King-only-to-empty-column rule even keeps 6H from moving off 7S, so spades
  cannot reach 7S). The model's own rationale is sound: it named 4C stuck because
  "the 5H is located directly beneath the 4C" and "the 5D is already in the
  foundation," correctly inferred "the 3C must be one of the face-down cards
  beneath the 4C," and concluded "the game is mathematically unwinnable." This is
  the positive counterpart to the doom-loop: on a dead board the 31B model
  recognised the deadlock and resigned instead of oscillating. Tooling caveat:
  `check_winnability.py` reports this board as "40/40 winnable, failure is
  behavioural not structural," which is wrong; as it runs today the script does
  not solve the intended board (load_pysol + reset_game, plus omitted foundation
  cards; see Operating notes). The repo-engine reachability search above is the
  authoritative check for this entry.

## Known doom-loop sessions (kept; flagged by stall filter)

These sessions are ingested as-is. The stall filter (`STALL_TURNS=25`)
excludes their stalled decisions from `dataset/training.jsonl` while keeping
every interaction in the store and the publish set as a research record of
how the teacher fails.

- Session `#7b6318` (full `019e87ff-10ae-7387-a4e3-ec0a6b7b6318`), seed
  `3161115466`, model `gemma-4-31b-it`, app build `df3a89b`
  (2026-05-31T12:10:49Z), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). One
  artefact in `raw/`: `solitaire-ai-log-7b6318-1780518304454.json` (626 rows,
  308 success / 318 errors, 41 forced moves). Ingested 2026-06-04. No win-record
  was exported. Final stored state: `moveCount: 501`, `finalProgress: 13%`,
  outcome `incomplete`. Terminal board `foundationCards=7`, `faceDownTotal=11`,
  `plateauTurns=173`, drawPile 2, recycle unavailable.

  **Behavioural stall on a structurally dead board, with no resign.** The engine
  solver proves the terminal board STRUCTURALLY DEAD: 12/12 sampled worlds
  provably unwinnable, exhausting in a mean of 48 states each (node_cap 200k), a
  hard surface lock with the same fast-exhaust profile as `#f75866` / `#8a5d12`.
  So at move 501 the stall is correct (no win exists), but the model never
  recognised it: it emitted `move_index: -1` (resign) zero times all session and
  thrashed the tableau instead. Of the 501 applied moves, **273 are
  tableau-to-tableau against only 10 flips and 7 foundation plays in the entire
  game** (the rest: 196 draws, 15 waste-to-tableau). The thrash is concentrated,
  not diffuse: the same three cards shuttle the same two columns, `8S col 4 ↔
  col 7` 121×, `9D col 4 ↔ col 7` 111×, `TC col 4 ↔ col 7` 100× (window counts,
  overlap-inflated, but the concentration on three cards and one column pair is
  the real-loop tell, not the magnitude). faceDownTotal fell 21 to 11 early then
  froze for the 173-turn plateau; the last 11 face-down cards sit behind the lock
  and are unreachable. Initial-deal winnability is not separately proven (opening
  boards adjudicate too slowly under the cap), but the plateau moves are
  reversible tableau shuffles, so the position very likely sat in the same dead
  reachable component throughout the 173-turn plateau, not just at the end. Role
  in corpus: the no-resign counterpart to `#30e5e5` (correctly resigned a dead
  board) and `#4a9fe1` (correct dead-board diagnosis in reasoning, resign action
  never emitted). #7b6318 is the worst rung: dead board, no diagnosis, no resign,
  173 turns of a 3-card shuffle, and a 31B v1.3 "ignores-and-loops" instance (cf.
  the obedience-trap split where 31B usually obeys-and-freezes). Bears on the
  parked v1.4 dead-board-recognition / resign gap.

- Session `#f75866` (full: `019e765a-540d-7596-9be6-963081f75866`), seed
  `3925117923`, model `gemma-4-31b-it`, app build `3136c81` (v1.3-era). One
  artefact in `raw/`: `solitaire-ai-log-f75866-1780215619568.json` (572 rows,
  485 success / 87 errors, ingested 2026-05-31). Final stored state:
  `moveCount: 525`, `finalProgress: 6%`, outcome `incomplete` (operator KILL).

  Behavioural stall, not a doom-loop. foundationCards/faceDownTotal sat flat at
  `(3, 10)` for the last 338 successful turns; foundations reached only AH/AC/AS
  (diamonds still null, AD buried) and the 10 face-down stayed pinned in columns
  6 and 7 behind a frozen column-3 run (`KC-QD-JC-TD-9S-8H-7C-6D-5S-4D-3C`). The
  briefing's `3C/4D col 3 ↔ col 7` 49×/44× is window inflation; de-inflated
  reversals on the stitched 510-move stream are 0. pyksolve on 10 determinised
  worlds solved 10/10, so the deal is most likely winnable and the failure is
  behavioural (the model froze for 338 turns without finding the line), not a
  structural dead deal. Caveat: no game-state file for this session, so the true
  10 face-down identities are unverified; the 10/10 is over random worlds. KILL
  was on stall grounds regardless.
  [Correction 2026-06-03: the "most likely winnable, failure is behavioural"
  call rested on the broken `check_winnability.py` (pyksolve). The fixed engine
  solver proves this board STRUCTURALLY DEAD in 40/40 sampled worlds (exhausts
  in <=248 states). So it WAS a structural dead deal, not a behavioural freeze
  on a winnable board. The original KILL stands (it was on stall grounds), but
  the structural-vs-behavioural characterisation is corrected to structural.]

- Session `…d46eb2645d03`, seed `3689552861`, model `gemma-4-31b-it`, app
  build `ce6afe1`. Exported across three files in raw/, latest is
  `solitaire-ai-log-645d03-1779331841599.json` (200 rows, canonical;
  earlier `1779227803496` and `1779270371464` are kept as overlapping
  exports — neither is a strict subset). Final stored state: 294
  interactions, max successful turn `143`, `moveCount: 145`,
  `finalProgress: 12%`. **Session is now LOCKED as the baseline** for
  the same-seed validation experiment (see "Same-seed validation
  experiments" below). Cause is **bad AI decisions, not a bad deck**:
  the model itself wrote that black 7s, red 7s, and a red King would
  unblock the board, and every named card except `KH` is in the
  seen-draw pile or face-up on the tableau (`7H` sits face-up on
  column 5). The model failed to play those cards when they reached
  the waste top, then oscillated 5C/4D between columns 4 and 6
  indefinitely.

- Session `…5061b71279a3`, seed `2439067361`, model `gemma-4-31b-it`.
  Latest canonical export `solitaire-ai-log-1279a3-1779329889383.json`
  (80 rows; earlier 39-row and 80-row duplicate exports archived).
  Final stored state: 80 interactions, max successful turn `76`,
  `finalProgress: 4%`. Initial 34-turn snapshot was **NOT a doom-loop
  — honest early-game hunt for the missing Aces** (`AH`, `AS`, never
  surfaced in the draw pile across multiple recycles). After 41 more
  turns the verdict flipped: foundation_cards unchanged at 2, only one
  more face-down revealed, and a `7H`/`8S` two-card oscillation
  between columns 4 and 5 began appearing in `recentMoves` (74
  repetitions over the tail-30 window). Mixed move types in last 20
  turns: 12 draws + 7 tableau shuffles. The two unaccounted Aces are
  almost certainly in the 18-card face-down stack; drawing further
  cannot help and the AI failed to pivot to face-down-revealing
  tableau moves. Retains its role as the empirical
  honest-hunt-then-degrades counterexample for the *shuffle-fraction*
  refinement: a plain plateau-only rule would have terminated this
  session at turn 35 (wrongly, while honest), whereas the
  shuffle-fraction gate gives it grace through honest-hunt phase and
  fires once tableau oscillation ramps up (~turn 60–65).

- Session `…aa24ed222c73fd85`, seed `191155745`, model `gemma-4-31b-it`.
  Latest export `solitaire-ai-log-73fd85-1779310300860.json` (167 rows)
  is canonical; earlier export `solitaire-ai-log-73fd85-1779308972435.json`
  (160 rows, strict subset) archived to `raw/archive/`.
  Final outcome: incomplete, `finalProgress: 19%`, `moveCount: 128`. Made
  real early progress (10 foundation cards built, face-down reduced 21 →
  8) but then descended into a **`TS`/`9D` two-card oscillation between
  columns 4 and 7** — the same class of failure as `645d03`, caught
  earlier (15-turn plateau when exported). Critically, at the final-turn
  state the AI had 6 legal moves and **5 of them dominated the chosen
  loop move**: two `KC + 7 chain` moves to empty columns, a `5S` reveal
  move, a recycle (the only path to the missing `AD`), and a `TC` move.
  Saturated 0.8 confidence on the looped pick. Used as the
  empirical doom-loop case where the *shuffle-fraction* gate fires at
  ~70% during the plateau window (well above the proposed 0.6 threshold).

- Session `…fd700d1f2fd2`, seed `3642085723`, model `gemma-4-31b-it`, app
  build `afa8c24`. Canonical export
  `solitaire-ai-log-1f2fd2-1779380943828.json` (191 rows, 132 success / 59
  errors). Final stored state: `moveCount: 146`, `finalProgress: 13%`,
  outcome `incomplete` (session still running at export time —
  **recommend operator kill**). **Class: behavioural-doom-loop, record
  oscillation counts in the corpus**: session-wide sweep shows `8C col 6
  ↔ col 7` at **266×** and `7D col 6 ↔ col 7` at **82×** across the
  33-turn plateau on `(foundationCards=7, faceDownTotal=11)`. The latest
  10-move window is 9× pure `7D` ping-pong. Critically, the agent's own
  `boardAnalysis` on the final turn correctly names the productive move
  ("moving the 7D to the 8S is a legal move that will reveal a face-down
  card in Column 6") yet the actual move sequence shows it oscillating
  rather than committing — a self-aware-but-impotent reveal-then-undo
  cycle. Distinct from `645d03`'s and `73fd85`'s "agent ignores legal
  productive move" pattern: here the agent *identifies* the productive
  move every turn but never executes it stably. **Material data point:
  build `afa8c24` is a third distinct build alongside `71130ac` (failing)
  and `6dfc8a9` (won — see Won sessions). Suggests prompt/scoring deltas
  across builds matter, but a single seed per build isn't enough to
  attribute the lift — needs a same-seed validation arm.**

- Session `…8b03bd502768`, seed `821908579`, model `gemma-4-31b-it`, app
  build `71130ac`. Four exports cover non-overlapping turn ranges and
  are all kept active in `raw/`: `solitaire-ai-log-502768-1779331813666.json`
  (45 rows), `solitaire-ai-log-502768-1779342158182.json`,
  `solitaire-ai-log-502768-1779360684635.json` (200 rows), and the
  terminal `solitaire-ai-log-502768-1779361355983.json` (200 rows,
  contains the manual abort). Final stored state: `moveCount: 223`,
  `finalProgress: 23%`, session **manually aborted by operator at turn
  223** after the structural lock was confirmed. Made the corpus's
  best early progress for build `71130ac` (foundations to 12, face-down
  21 → 6) before degenerating into a **32-turn plateau on
  `(foundationCards=12, faceDownTotal=6)`** with the AI recycling the
  stock looking for cards that are face-down and structurally
  unreachable. Specifically: the missing `AD` is in the 6 face-down
  cards (never in `seenDrawPileCards` across multiple recycles), and
  no red-7 exists to peel col 4's `6C`. Final follow-up export
  contributed +1 foundation card (`5H → 6H`) over the last ~53
  retried turns before the operator killed it — a worst-case ratio
  that quantifies the cost of running without a stall auto-terminator.
  Final-turn reasoning explicitly identifies the deadlock ("none of
  the available tableau moves reveal hidden cards or advance the
  foundations") but the chosen remedy — recycle and keep drawing —
  cannot help. Same pathology class as `645d03` and `73fd85`, just
  delayed. **Class: dead-deal-flailing** (structural lock, not a
  winnable board with bad play) — distinct from the behavioural
  doom-loops on `645d03`/`73fd85`/`29a7f5`.

- Session `…8e7159391920`, seed `3841211007`, model `gemma-4-31b-it`, app
  build `afa8c24`. Two exports cover the session:
  `solitaire-ai-log-391920-1779398622316.json` (200 rows) and the latest
  `solitaire-ai-log-391920-1779409652731.json` (200 rows, 18 new).
  Final stored state: `moveCount: 356`, `finalProgress: 10%`, outcome
  `incomplete`. **Class: behavioural-doom-loop, late-game with terminal
  stock**. Plateau on `(foundationCards=5, faceDownTotal=8)` for 87 turns.
  Session-wide oscillation `3C col 3 ↔ col 5` 88× and `4H col 3 ↔ col 5`
  82× across the plateau. **Endgame is structurally bounded**: 4 stock
  cards remain, fully known via `seenDrawPileCards` (`3S, 6H, 8D, 8S`),
  `canRecycleStock=False`, and the agent's own reasoning enumerates the
  needed cards as "a red 6 for Col 5, a red 7 for Col 6, a red Jack for
  Col 7" — none of which are in the remaining stock. **Second `afa8c24`
  data point reinforcing the build's self-aware-but-impotent failure
  pattern** (cf. `1f2fd2`): the model correctly diagnoses deadlock in
  `boardAnalysis` but its move selection keeps drawing or oscillating
  anyway. Played one productive `2H waste → col 4` park in the final
  window — the one legitimate stock-fed opportunity, taken correctly.
  Operator kill recommended; will terminate naturally at stock exhaustion
  within ~5 turns regardless.

- Session `…fa36193d03e5`, seed `841422313`, model `gemma-4-31b-it`, app
  build **`7894202` (NEW)**. Three exports kept:
  `solitaire-ai-log-3d03e5-1779398629965.json` (231 rows),
  `solitaire-ai-log-3d03e5-1779406890160.json` (323 rows, +92), and the
  terminal `solitaire-ai-log-3d03e5-1779411082008.json` (336 rows, +13).
  An intermediate 1779409633914 snapshot had 0 new rows and was a session
  idle re-export; archived/dropped. Final stored state: `moveCount: 412`,
  `finalProgress: 12%`, outcome `incomplete`, **operator killed after
  follow-up snapshot confirmed the chain-relocation reversal**. **Class:
  behavioural-doom-loop with frozen session-wide oscillation**: counts
  for `9D col 3 ↔ col 7` (82×), `TC col 3 ↔ col 7` (73×), and `JD col 3
  ↔ col 7` (65×) are identical across all three snapshots, meaning the
  loop completed early and the model subsequently moved to non-oscillating
  but still non-productive chain shuffling on `(foundationCards=6,
  faceDownTotal=11)` for 76+ turns. **Key behavioural finding for the
  `7894202` debut**: at the 23:41Z snapshot the model relocated the
  full 10-card JD-2C chain from col 3 → col 4, plausibly staging an
  empty-col-3 + KD-from-waste park. In the next snapshot (00:51Z) the
  model **reversed that chain back to col 3 without ever recycling the
  stock** (`drawPileCount=0, canRecycleStock=True, discardTop=KD`
  remained un-played across the entire window). The promising move was
  not part of a multi-step plan — single-step planning only. **First
  `7894202` data point: not materially better than `afa8c24` on this
  seed.**

- Session `…f31fb63e77cc`, seed `831006668`, model `gemma-4-31b-it`, app
  build **`7894202`**. Two exports:
  `solitaire-ai-log-3e77cc-1779398638448.json` (297 rows) and the latest
  `solitaire-ai-log-3e77cc-1779406886677.json` (425 rows, +128). Final
  stored state: `moveCount: 370`, `finalProgress: 2%`, outcome
  `incomplete`. **Class: catastrophic stall — worst plateau in the
  corpus**. Plateau on `(foundationCards=1, faceDownTotal=14)` for
  **290 turns**. The model played exactly one foundation card (`AH`)
  across 370 game moves. Session-wide oscillation is only 40× (mild —
  the model isn't even looping much, it's just drawing repeatedly).
  Stock fully known: 16 cards remaining (`8S, TS, KC, 3C, 4S, 2D, 8C,
  6H, JC, 3S, 4H, KH, 7D, 2S, 7H, QS`), `canRecycleStock=False`. Likely
  but unconfirmed dead-deal (one foundation in 370 moves is suggestive
  of structural blockage from the open). Operator kill recommended;
  worth a post-mortem winnability check via Monte Carlo determinisation
  to attribute (dead-deal vs `7894202` early-game blind spot) for build
  evaluation. **Second `7894202` data point: pairs with `3d03e5` as
  evidence the debut build does not show planning or stock-recycle
  prioritisation improvements over `afa8c24`/`71130ac`.**

- Session `…e141c4c7fdb9`, seed `2600933760`, model `gemma-4-31b-it`, app
  build **`7894202`**. Two snapshots kept:
  `solitaire-ai-log-c7fdb9-1779425375340.json` (203 rows, 22 May 04:49Z,
  `moveCount: 125`, `finalProgress: 25%`, plateau 3 turns) and the
  terminal `solitaire-ai-log-c7fdb9-1779510019738.json` (476 rows, 23 May
  04:20Z, `moveCount: 202`, `finalProgress: 27%`, plateau **67 turns**).
  Between snapshots: +77 moves yielded **+1 foundation card** (13 → 14),
  faceDownTotal unchanged at 3. **Class: dead-deal-flailing on a
  near-finished board**. The 3 remaining face-down cards are all in
  column 6, pinned under a `7H/8S/9D/TC` face-up stack that requires
  cards buried in column 2 (`8C/9H/TS/JD`) to peel — a mutual lock.
  Stock fully known: `drawPileCount=6 + seenDrawPileCards=6 = 12`,
  `canRecycleStock=False`. Session-wide oscillation `7H col 2 ↔ col 6`
  ran **161×** across the 67-turn plateau before the model exhausted
  productive moves and degenerated to pure stock-cycling (latest 10-move
  window: 9 draws + 1 oscillation step). Final-turn `boardAnalysis`
  explicitly names the deadlock: *"The board is currently in a deadlock.
  The only hidden cards are in Column 6, but the cards on top of them
  ... cannot be moved without cards that are currently buried in Column
  2."* **Self-aware-but-impotent endgame**: textbook case for the
  resignation-output handover ask (`docs/reports/20260522_harvester_team_handover.md`,
  Ask 2) — model has the right verbal diagnosis with no structural
  primitive to act on it.

- Session `…ca1cefd5a63f`, seed `598648106`, model `gemma-4-31b-it`, app
  build **`7894202`**. Single canonical export
  `solitaire-ai-log-d5a63f-1779510016213.json` (284 rows, 76 success /
  208 errors). Final state: `moveCount: 157`, `finalProgress: 21%`,
  outcome `incomplete`. **Class: dead-deal-flailing, short plateau but
  provable lock**. Plateau on `(foundationCards=11, faceDownTotal=4)`
  for only 9 turns at export, but the structural reasoning is
  conclusive. Stock fully known: `drawPileCount=3 + seenDrawPileCards=3
  = 6`, `canRecycleStock=False`. Model's `boardAnalysis` lays out the
  lock cleanly: revealing col 7 needs the only red 7 (`7H`), which is
  buried under `6C/5H/4C` on col 1; revealing col 3 needs the black
  King `KS`, which is blocked by `QD` on col 4. RecentMoves show the
  model assembled a 10-card `JD/TC/9D/8C/7H/6C/5H/4C` chain onto col 1
  in the final window — that chain *is* the lock, since it consumed
  the only red 7 the col-7 cleanup needs. **Class diagnosis ahead of
  plateau threshold**: this is the second session in two days where
  the heuristics gave a confident kill verdict before the 25-turn
  stall filter would have triggered. Operator kill recommended.

- Session `…de3bbdb89064`, seed `4074802352`, model `gemma-4-31b-it`, app
  build **`7894202`**. Single export
  `solitaire-ai-log-b89064-1779509974784.json` (341 rows, 96 success /
  245 errors). Final state: `moveCount: 172`, `finalProgress: 17%`,
  **outcome `stalled_auto_terminated`** — the **first session in the
  corpus** to terminate via the harvester-side stall auto-terminator
  rather than operator kill, manual abort, or `incomplete`. **This
  closes one of the four standing harvester P0s** referenced in the
  `645d03` validation entry below (the stall auto-terminator is now
  live). **Class: behavioural-doom-loop with concurrent structural
  suspicion**. Plateau on `(foundationCards=9, faceDownTotal=8)` for
  24 turns at termination. Heavy multi-pair oscillation: session-wide
  `4C col 3 ↔ col 6` 85×, `5D col 3 ↔ col 6` 77×, `6S col 5 ↔ col 6`
  43×; latest 10-move window shows fresh `7H col 5 ↔ col 6` cycling
  (3× in the tail). Model's final reasoning correctly notes the chain
  shifts don't reveal face-down cards but picks one anyway — the
  classic self-aware-but-impotent pattern. The face-down distribution
  (col 6: 4, col 5: 2, others smaller) plus the consistent fixation on
  cols 5/6 hints this may be structural as well as behavioural, but
  the auto-termination fired before the lock could be proved.
  **Material harvester-side finding**: the stall threshold is firing
  cleanly, on the right kind of session, without an operator in the
  loop. This is the day's good news.

- Session `…07b5515ffb25`, seed `2284386365`, model `gemma-4-31b-it`, app
  build **`7f01833` (NEW — first appearance in the corpus, appBuildTime
  2026-05-22T05:21Z)**. Single export
  `solitaire-ai-log-5ffb25-1779525838374.json` (219 rows, 88 success /
  131 errors). Final state: `moveCount: 120`, `finalProgress: 27%`,
  outcome `incomplete`. **Prompt template is unchanged from the prior
  build line** — md5 `a39354fa5f16e03285e389dee5dc551c` (the
  3,527-char calibration-bands template that 7894202 / 6dfc8a9 /
  afa8c24 / 71130ac / ce6afe1 also use). So `7f01833` is yet another
  build advancement on the same prompt; the audit's open question on
  what build hashes mean keeps getting louder. **Class:
  behavioural-doom-loop, latest-window saturated**: 6× `TD col 2 ↔ col 4`
  oscillation in the last 10 moves (i.e. nearly the entire latest
  window is the same 2-card cycle), on top of session-wide `6S col 3 ↔
  col 7` 34×, `7D col 3 ↔ col 7` 32×, and `5C col 5 ↔ col 6` 29×
  across the plateau. **Self-aware-but-impotent**: the model's
  `boardAnalysis` correctly states *"Column 4 can be emptied in two
  moves: first, by moving the red Ten of Diamonds (TD) to the black
  Jack of Clubs (JC) in Column 2, and second, by moving the black Jack
  of Spades (JS) to..."* — and then executes the TD oscillation that
  defeats that plan. Operator kill recommended. **First behavioural
  signal on `7f01833`: not different from `7894202`.**

- Session `…688f5a044461` — **won; full entry moved to "Won sessions"
  section above** as the third corpus win and first win on build
  `7f01833`. Listed here historically because the mid-game `71%`
  snapshot was originally classified under doom-loop watching;
  reclassified once the terminal win export landed at 18:54Z.

- Session `…5d992198fe0e`, seed `4200745230`, model `gemma-4-31b-it`,
  app build `7f01833`. Three exports in `raw/`:
  `solitaire-ai-log-98fe0e-1779532140716.json` (127 rows, the original
  midgame snapshot, retained for the time series),
  `solitaire-ai-log-98fe0e-1779541526045.json` (211 rows; 95 success /
  116 errors), and the canonical latest
  `solitaire-ai-log-98fe0e-1779542126968.json` (215 rows; 95 success /
  120 errors). Final state: `moveCount: 186`, `finalProgress: 19%`,
  `foundationCards: 10`, `faceDownTotal: 6`, outcome `incomplete`.
  **Terminal pathology: provider-timeout-freeze on turn 186.** The two
  late snapshots are 10 min apart and show **zero new successful
  turns** with 4 new error rows, all on turnIndex 186 ("The model did
  not respond within 240s." × 3 then cancelled). Session-wide
  oscillation history: `8H col 1 ↔ col 2` (10×), `3C col 1 ↔ col 4`
  (7×), `5S col 1 ↔ col 4` (6×) — modest counts, the failure is not a
  pure behavioural doom-loop. Board at the freeze still has 8 legal
  moves and reveal paths (the model's last successful turn correctly
  planned a 10-card chain from col 4 to expose `KC` so `QD` could be
  parked on it). What changed across the game is the **thinking-token
  budget**: per-quartile mean `thoughtTokens` grew 1741 → 1546 → 1611
  → **4280**, and mean response duration grew 60s → 53s → 60s →
  **133s** with max 231s — the 240s timeout wall caught the Q4 thinking
  explosion. This session is the canonical example of "model-side
  late-game thinking blowup masquerading as a doom-loop". **Recommend
  kill**: harness-side fix is a thinking-token / duration ceiling, not
  prompt rewording.

- Session `…770556668fda`, seed `2856463832`, model `gemma-4-31b-it`,
  app build `7f01833`. Two exports: the prior
  `solitaire-ai-log-668fda-1779532136283.json` (289 rows) and the
  canonical latest `solitaire-ai-log-668fda-1779542123127.json` (401
  rows; 126 success / 275 errors, ingested 2026-05-23). Final state:
  `moveCount: 207`, `finalProgress: 27%`, `foundationCards: 14`,
  `faceDownTotal: 4`, outcome `incomplete`, plateau **17 turns** at
  export. **WATCH verdict resolved → KILL**: the productive `7D →
  diamonds` breakout seen in the prior snapshot did not extend; the
  session reverted to oscillating and added 42 more moves with **zero
  progress on foundation or face-down**. The `3C col 4 ↔ col 5`
  session-wide pair count stays at **159×** (it was already that high
  at the prior snapshot — the model has not added more to that pair
  but has not escaped it either). Late state confirms structural
  lock: `drawPileCount: 2`, `canRecycleStock: false` (stock burned
  for this cycle), and the model's own boardAnalysis correctly
  diagnoses the impasse: *"deadlock between Column 5 and Column 7…
  9D buried in Column 5 and 9H buried in Column 7… The only black 6
  (6C) is buried in Column 5."* Chicken-and-egg: every card needed
  to unblock a column is buried in that same column. This session is
  now the canonical example of *self-rescue-from-doom-loop fails* —
  the model can sometimes break out of a 159× oscillation for one
  productive move, but cannot string two together. Thinking-budget
  late-game blowup is present here too: per-quartile mean
  `thoughtTokens` grew 1800 → 2348 → 4374 → **5863** with mean
  duration 66s → 80s → 138s → **179s**. **Recommend kill**;
  reinforces the P0 ask for a stall auto-terminator on the harness
  side.

- Session `…ac8e98cf40af`, seed `251180270`, model `gemma-4-31b-it`,
  app build `6d92ddd`. Single export
  `solitaire-ai-log-cf40af-1779569910807.json` (196 rows; 70 success
  / 126 errors, ingested 2026-05-24). Final state: `moveCount: 86`,
  `finalProgress: 13%`, `foundationCards: 7`, `faceDownTotal: 17`,
  outcome `incomplete`, plateau **1 turn** at export (model had just
  played `3C waste -> clubs foundation` and chained `TC/9H/8C col 6 ->
  col 1` in the last window). Despite the fresh breakout, the
  session-wide oscillation signature is extreme: `4D col 4 ↔ col 5`
  recurs **65×** across recentMoves windows, `5S col 4 ↔ col 5` **59×**,
  and `8C col 3 ↔ col 6` **25×** — i.e. cols 4/5 were a tight 4D/5S
  pump for most of the session. Stock effectively exhausted at export:
  `drawPileCount: 2`, `canRecycleStock: false`, `seenDrawPileCards`
  only `6C, JC`. Last-turn reasoning self-diagnoses the structural
  problem accurately: *"None of the available tableau moves (Move 0,
  1, 4) or the waste move (Move 2) reveal any face[-down cards]"* —
  the model knows the breakout chain didn't expose anything new and
  is back to the same pinned-column set. Pattern matches the
  `…770556668fda` canonical *self-rescue-from-doom-loop fails* shape:
  one productive move out of a tight oscillation, no second productive
  move available. **Recommend kill**.

- Session `…55e0809f22c2`, seed `885853979`, model `gemma-4-31b-it`, app
  build `de7dc06`. Single export
  `solitaire-ai-log-9f22c2-1779746889045.json` (957 rows, 260 success /
  697 errors, ingested 2026-05-26). Final state: `moveCount: 297`,
  `finalProgress: 15%`, `foundationCards: 8`, `faceDownTotal: 9`,
  outcome `incomplete`, plateau **85 turns**. **Class: dead-deal-flailing
  (missing-ace lock)**. Both `AH` and `AS` are absent from
  `seenInWasteThisCycle`, both unseen face-up, and `canRecycleStock: false`
  with `drawPileCount: 2` (terminal stock pass), so both Aces are pinned
  among the 9 face-down cards. The only KC-rooted run (`KC QH JS TH 9S 8H
  7S 6H 5S 4H`) sits in col 5 with no face-down beneath it; relocating it
  to col 1 or col 3 (both empty) does not expose anything. col 4/6/7 each
  carry 3 face-down beneath `6C`/`5H`/`5C` respectively, and the cards
  needed to peel them (a red 7, a black 6, a red 6) are all face-down or
  buried. Session-wide top pair `draw 3H / draw QC` 78× = the model has
  been recycling looking for cards that are not in the stock. Final-turn
  `boardAnalysis` self-diagnoses: *"The board is currently in a stalemate.
  Hidden cards remain in columns 4, 6, and 7... no such cards are
  available on the tableau or in the waste."* Recommend kill.

- Session `…14ec18e11e3b`, seed `1296147461`, model `gemma-4-31b-it`, app
  build `de7dc06`. Single export
  `solitaire-ai-log-e11e3b-1779746885211.json` (626 rows, 78 success /
  548 errors, ingested 2026-05-26). Final state: `moveCount: 116`,
  `finalProgress: 15%`, `foundationCards: 8`, `faceDownTotal: 8`, outcome
  `incomplete`, plateau **15 turns**. **Class: behavioural-doom-loop on a
  winnable-looking board** — distinct from the three other `de7dc06`
  sessions ingested the same day, which are all dead-deal. Session-wide
  oscillation `move 5C col 3 -> col 6` paired with `move 4H col 3 -> col 6`
  recurs **25×** across the 78 successful turns; a secondary `4C/3H col 4
  -> col 7` swap also appears. Board has clear reveal paths: col 4 holds
  a clean 9-card KH-rooted alternating run (no face-down); col 7 holds 5
  face-down beneath a runnable 5-card `7D-6C-5H-4C-3H`; col 5 has only 2
  face-down beneath `4D`; foundations already at `AH/AD/3C/3S` so `2H/2D`
  are still findable. Stock just exhausted (`drawPileCount: 0`,
  `canRecycleStock: true`). Final-turn `boardAnalysis` correctly names
  the plan (*"The top run (4C, 3H) can be moved to the 5D in column 4...
  the 6S, 7H, and 8C will be available to receive the 5H, 6C..."*) but
  the model keeps swapping 5C/4H instead. **Material corpus value:
  cleanest behavioural-doom-loop on the new hybrid-v1 prompt build
  `de7dc06`, and the first such case where the model both verbalises the
  productive plan and stays in the loop.** Useful as a paired counterexample
  to the dead-deal `de7dc06` exports below — same build, same model, same
  hybrid prompt, opposite failure class.

- Session `…4db3cea0d9fe`, seed `1236111563`, model `gemma-4-31b-it`, app
  build `de7dc06`. Single export
  `solitaire-ai-log-a0d9fe-1779674937536.json` (531 rows, 233 success /
  298 errors, ingested 2026-05-25). Final state: `moveCount: 232`,
  `finalProgress: 15%`, `foundationCards: 8`, `faceDownTotal: 11`,
  outcome `incomplete`, plateau **153 turns**. **Class: dead-deal-flailing
  (missing-ace lock)**. `AH` is absent from `seenInWasteThisCycle`, all
  face-up tableau, and any other observable pool, leaving it face-down
  among the 11. Every face-down column is reveal-blocked: col 4 face-downs
  need a JD/JH base to relocate the 6-card `TC-9H-8C-7D-6S-5H` chain
  (JD buried in col 2 beneath the full `KD-QS-JD-TS-9D-8S-7H-6C-5D` run;
  JH unseen); col 5 face-downs need an empty column to relocate
  `KC-QH-JS-TD-9C-8H` (col 1 holds `KS QD`, col 3 holds `6D`, neither
  empty); col 6 face-down needs a red T for `9S` (TH/TD unseen); col 7
  face-downs are blocked by `4D-3S-2H` whose `4D` has no black 5 anywhere.
  Session-wide top pair `draw 6H / draw 3H` 132× = pure stock recycle.
  Final-turn `boardAnalysis`: *"the AH is not yet visible. The stock is
  empty, meaning the only way to access the cards in the waste pile is
  to recycle..."*. Recommend kill.

- Session `…8a10b6c98e59`, seed `3976147086`, model `gemma-4-31b-it`, app
  build `de7dc06`. Single export
  `solitaire-ai-log-c98e59-1779674930290.json` (561 rows, 247 success /
  314 errors, ingested 2026-05-25). Final state: `moveCount: 281`,
  `finalProgress: 10%`, `foundationCards: 5`, `faceDownTotal: 16`,
  outcome `incomplete`, plateau **208 turns** — **worst plateau in the
  `de7dc06` cohort and second-worst in the corpus overall** (after
  `…f31fb63e77cc`'s 290 turns on a 1-foundation board). **Class:
  dead-deal-flailing (double-ace lock)**. Both `AD` and `AC` are absent
  from every observable pool, leaving both among the 16 face-down. All
  five face-down columns are reveal-blocked: col 2's `9S` needs a red T
  (none); col 4's `QS-JD-TC-9D-8S` 5-card alternating chain needs a red K
  base or empty column (no empty, `KH`/`KD` unseen); col 5's `6C-5H` needs
  a black 6 placement (`6S` is exposed on col 6 but cannot be shifted off
  without a red 7); col 6's `6S` needs a red 7 (unseen); col 7's `JH`
  needs a black Q (`QS` buried in col 4, `QC` sits on waste with no
  tableau target). Session-wide top pair `move 9D col 7 -> col 4 / move
  8S col 7 -> col 4` 59× = early-game shuffle that has since exhausted
  itself; the model has subsequently shifted to pure stock-draw flailing
  in the final window. Final-turn `boardAnalysis`: *"AD and AC are
  missing. There are 16 hidden cards remaining in the tableau."* Recommend
  kill. **Three of the four `de7dc06` exports ingested 25-26 May (this
  session, `9f22c2`, and `a0d9fe`) are dead-deal; only `e11e3b` is
  behavioural. The new hybrid-v1 prompt does not appear to be the
  bottleneck on the dead-deal class — the underlying seeds are unsolvable
  and the missing-ace heuristic catches all three uniformly.**

- Session `…993e6cadf71b`, seed `114946100`, model `gemma-4-31b-it`, app
  build `de7dc06` (hybrid-v1 plain-text prompt). Canonical export
  `solitaire-ai-log-adf71b-1779779357575.json` (189 rows, 132 success /
  57 errors). Final stored state: `moveCount: 210`, `finalProgress: 25%`,
  outcome `incomplete` (session was killed on operator instruction after
  this verdict). **Class: behavioural-doom-loop on a structurally
  winnable board.** 40-turn plateau on `(foundationCards=13,
  faceDownTotal=3)` from turn 169 to 209, with `seenDrawPileCards`
  exhausted (stock=3, `canRecycleStock: no`). Oscillation is a 3-card
  chain `6D-5C-4D` migrating col 2 to col 6 and back; the latest
  10-move window at turn 209 contains the round-trip 3 times. Same
  oscillation class as `645d03` (2-card `5C/4D`) and `73fd85` (2-card
  `TS/9D`), one card wider. Final-turn `boardAnalysis` is the
  canonical self-aware-but-impotent failure mode: *"The board is
  currently in a deadlock. To reveal the hidden cards in column 6 (??,
  ??, 7C), a red 8 is needed. The available red 8s are 8D (buried in
  column 2) and 8H (previously seen in the waste)."* Then picks an
  oscillating move anyway. **Solver ground truth contradicts the AI's
  deadlock claim.** All 6 unknown cards are constrained:
  `{8H, 2D, 9C, 3S, 5S, TS}` fill the 3 face-down slots and the 3 stock
  slots. Enumerating all 720 permutations through pyksolve (draw-3,
  `max_closed_count=100000`) yields **720/720 solvable** in ~34 s
  total. The board is winnable for every conceivable assignment of the
  unknowns, so there is no dead-deal hypothesis to fall back on. This
  is a pure behavioural failure. **Material data point: this is the
  first catalogued doom-loop under build `de7dc06` / hybrid-v1**, on a
  build that wins seed `3263196305` (see Cross-version teacher
  benchmarks). Hybrid-v1 fixes neither the oscillation pathology nor
  the confidence saturation. The 3-card-chain variant should be added
  to the failure-mode taxonomy alongside the existing 2-card
  oscillations.

- Session `…d2fde06b491a`, seed `2967897202`, model `gemma-4-31b-it`, app
  build **`cef6291` (NEW)**, prompt **`hybrid-v1.2` (NEW — first appearance
  in the corpus, templateHash `645f77b252b95a1b...`)**. Single canonical
  export `solitaire-ai-log-6b491a-1779877547228.json` (230 rows, 114 success
  / 116 errors, ingested 2026-05-27). Final stored state: `moveCount: 199`,
  `finalProgress: 17%`, outcome `incomplete`. **Class: behavioural-doom-loop
  on a winnable board** (corrected from initial dead-deal verdict — see
  methodological note below). Plateau on `(foundationCards=9,
  faceDownTotal=13)` from turn 108 / turn 50 onward — zero foundation
  progress for 90 turns, zero face-down reveal for 148 turns at export.
  **Session-wide oscillation signature**: `8D col 1 ↔ col 3` 85× and `9S
  col 1 ↔ col 3` 80× across the plateau, i.e. the model repeatedly applies
  legal move [0] *"Move 9S plus 1 more from column 3 to column 1"* (moving
  the `9S-8D` pair onto `TD`) and then later moves it back to col 3,
  cycling ~165 times. Secondary `3D col 5 ↔ col 7` 59× indicates the
  4-card `6S-5H-4C-3D` sub-chain on col 7 also gets relocated back to col
  5 periodically. Latest 10-move window happens to be draws + a one-way
  col 5 → col 7 unload, hiding the oscillation from a tail-only inspector
  — must use session-wide pair counts (see methodological note). Solver
  verdict via `check_winnability.py` (pyksolve, 10 samples, seed 42):
  **10/10 solvable, mean 8 ms/sample**. Result is over-optimistic on
  v1.2 because the Monte Carlo does not constrain talon identity from the
  DRAW TIMELINE (see "Methodology updates" below), but 10/10 is strong
  enough evidence that the board has many winnable arrangements — the
  failure is behavioural, not structural. Final-turn `boardAnalysis`
  self-diagnoses but executes oscillation anyway: *"Move [0]... and Move
  [1]... only shift face-up cards and do not expose any hidden cards.
  Drawing the 6H (Move [2])... will immediately reveal a hidden card in
  column 6"*. The plan is also wrong on the merits (`6H` cannot land on
  tableau because both black 7s are face-down), but the proximal failure
  is the 165× col 1↔3 cycle, not the misjudged 6H plan. **Material
  harvester-team findings on v1.2 debut**: (1) prompt switches from a
  `CURRENT GAME (JSON):` block to a plain-text `CURRENT GAME:` board
  layout, which broke `load_export.py` and `check_winnability.py` until a
  text-format parser was added (this session was the trigger for that
  fix); (2) new **DRAW TIMELINE** block (`8C QC KC JC KD TS 6D JD {QD}
  6H`) makes the full 10-card talon observable to the analyst and the
  model — yet the model still picked Move [2] (draw) believing it would
  unblock col 6, ignoring that a strict read of the timeline plus the
  "6H needs a black 7 to land" check rules that out; (3) **first
  behavioural data point on v1.2: same doom-loop pathology as the
  hybrid-v1 cohort**. Pairs with `e11e3b` (hybrid-v1 on `de7dc06`) and
  `adf71b` (hybrid-v1 on `de7dc06`) as evidence that the prompt
  iteration from v1 → v1.2 does not address oscillation. The
  `move_index: -1` resign output did not fire across 90 turns of zero
  foundation progress despite the model's own reasoning enumerating the
  deadlock — first signal that v1.2's resign trigger needs a plateau-aware
  rule, not just a "no productive move" verbal cue. Operator killed.

  **Methodological note**: initial verdict (before the parser fix) was
  `dead-deal-flailing` based on hand-analysis of the board state. That
  verdict was *wrong*. The error path: the v1.2 text-format board broke
  `parse_board()`, which returned `None`, which made
  `session_oscillation()` return no signal — so the 165× col 1↔3 cycle
  was invisible. The hand-analysis then over-weighted the "AD face-down
  with circular reveal dependencies" structural argument and missed that
  even with a hard structural barrier, the proximal failure is still the
  oscillation. **Always run the briefing's session-wide oscillation count
  before declaring dead-deal**, even when the latest-10 window is clean.
  Logged in memory as a parser-broke-the-verdict case.

  **Session-history note**: an earlier snapshot
  `solitaire-ai-log-6b491a-1779861014358.json` (86 rows, turns 0-81) is a
  strict subset of the canonical 230-row export and is archived to
  `raw/archive/`.

- Session `…a0c137c99da9`, seed `2967897202`, model `gemma-4-31b-it`, app
  build **`20a825f` (hybrid-v1.1)**. Canonical
  `solitaire-ai-log-c99da9-1779851673647.json` (190 rows, 113 success / 77
  errors, ingested 2026-05-27). Two earlier snapshots
  (`solitaire-ai-log-c99da9-1779833581648.json` 43 rows turns 0-30, and
  `solitaire-ai-log-c99da9-1779834631435.json` 68 rows turns 0-48) are
  strict subsets, archived to `raw/archive/`. Final stored state:
  `moveCount: 197`, `finalProgress: 29%`, `foundationCards: 15`,
  `faceDownTotal: 5`, **outcome `stalled_auto_terminated`** (second corpus
  session to terminate via the harvester-side stall auto-terminator after
  `…de3bbdb89064` b89064 was first; confirms the terminator is live on
  build `20a825f`). **Class: behavioural-doom-loop, late-game**.
  Session-wide oscillation `4C col 2 ↔ col 7` 80×, `5H col 2 ↔ col 7` 72×,
  `6S col 2 ↔ col 7` 65× — the 6S-5H-4C three-card chain bouncing between
  cols 2 and 7 ~217 chain-moves across the plateau. Latest-window also
  shows the loop (4C 3× in last 10). Reached deep into game (foundation
  15, faceDown 5) before stalling. **Material for same-seed comparison**:
  this seed `2967897202` won under build `7f01833` (session `…688f5a044461`
  in 194 moves), stalled under `20a825f` v1.1 (this session) at 29% in 197
  moves, and stuck under `cef6291` v1.2 (session `…d2fde06b491a`) at 17%
  in 199 moves. **Same deck, three successive builds, monotonically worse
  outcome** (won → stalled-late → stuck-mid). See Same-seed validation
  experiments section.

- Session `…3ced34aca45a`, seed `2853966634` — **WON; full entry moved to
  "Won sessions" section above.** Originally classified under doom-loop
  watching when the mid-game snapshots (mc 69 then mc 393) showed a
  ~290-repetition `5H/4S/3H col 3 ↔ col 6` oscillation; reclassified once
  the terminal win export landed at mc 418 / fp 100%. Canonical doom-loop-
  then-breakout example on a fresh seed.

- Session `…6a3d6d49b05f`, seed `3263196305`, model **`gemma-4-26b-a4b-it`
  (NEW model in corpus)**, app build **`8934147` (NEW)**, prompt
  **`hybrid-v1.3` (NEW first appearance, templateHash `7d9ecda4cb...`)**.
  Three snapshots: early
  `solitaire-ai-log-49b05f-1779919722784.json` (22 rows, mc 16, fp 6%),
  `solitaire-ai-log-49b05f-1779933186983.json` (81 rows, 40 success / 41
  errors, ingested 2026-05-28), and the now-canonical
  `solitaire-ai-log-49b05f-1780055383974.json` (481 rows, 137 success / 344
  errors, ingested 2026-05-29; supersedes the 81-row export, 81 rows re-seen,
  400 new). Terminal state at the canonical export: `moveCount: 246`,
  `finalProgress: 29%`, `foundationCards: 15`, `faceDownTotal: 8`, outcome
  `incomplete`, plateau **20 turns** at export (the 2026-05-28 snapshot was
  mc 40 / fp 10% / fd 20). **Class: behavioural-doom-loop on a known-winnable
  seed, confirmed structural-vs-behavioural by the solver.** pyksolve returns
  **10/10 consistent worlds solvable** from the terminal state (mean 8 ms,
  verdict "failure is behavioural, not structural"), and `3263196305` is the
  anchor seed the 31B has won three times (`0154e1` 174 moves, `c05ad4` 296,
  `9b1c4a` 360). The board is wide open at abandonment: `col1` holds a complete
  face-up `JC TD 9S 8H 7C 6H` run, diamonds are up to `8D`, only 8 face-down
  remain (col5 1, col6 2, col7 5), and productive moves are available (`4S 3H`
  from col3 onto the `5H` in col6 empties col3 for one of three exposed Kings),
  but the model spends the tail toggling instead. The early QS oscillation
  (`QS col 5 ↔ col 7` 16× at the 2026-05-28 snapshot) persisted and broadened:
  session-wide `QS col 5 ↔ col 7` 60×, `6D col 2 ↔ col 4` 44×, `7C col 2 ↔
  col 4` 41×, and the final 10-move window is a pure three-card
  `5H/7H/6S col 6 ↔ col 7` slosh. None of the 97 new success decisions entered
  clean-lean (all filtered `stalled-game`), so the loop does not contaminate
  training. **Three
  triple-firsts in one session**: (a) first 26B-a4b in corpus, (b) first
  v1.3 prompt, (c) first time the harvester team has shipped a prompt
  revision AND a model swap simultaneously, which is a single-variable
  attribution problem for the v1.3 bench at
  `/Users/chayut/repos/solitaire-analytics/docs/reports/20260528_prompt_v1_3_candidate_spec.md`
  Section 4. That bench requires 31B-on-v1.3 traces too for clean arm
  comparison.

  **Material v1.3 finding (DESIGN HOLE)**: the anti-undo predicate from
  v1.3 change 2.2 (*"Do not move a card to a tableau column it occupied
  in the last 5 moves"*) IS being recognized by the model but is being
  treated as a soft heuristic the model can override. At turn 36 the
  model correctly cited the rule: *"I will avoid move [1] because it
  violates the 'last 5 moves' rule, which is designed to prevent
  looping."* At turn 37 the model explicitly overrode it: *"Although
  this move involves returning a card to a column it recently occupied,
  the long-term benefit of exposing a face-down card outweighs the
  short-term restriction."* The override creates the QS-col5↔col7
  oscillation. The model is using the "exposes a face-down card" reveal
  bullet as license to override the anti-undo bullet. **Root cause**:
  the v1.3 STRATEGY GUIDANCE bullets are not priority-ranked, so the
  model is free to interpret reveal-priority as superseding anti-undo.
  **v1.3.1 fix proposed**: convert the bullets to an explicit priority
  list per anti-pattern 5.3 at
  `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/references/anti_patterns.md`,
  with anti-undo ranked ABOVE reveal-priority so the model cannot
  rationalise the override.

  **Material v1.3 finding (intended-behavior confirmed)**: the rewritten
  reveal-priority bullet (v1.3 change 2.4) IS being internalized cleanly.
  The model verbally invokes the tie-break predicate ("prefer the column
  with the most face-down cards remaining") when applicable. The
  reasoning style across all 40 turns is now dominated by explicit
  citation of the strategy guidance rules ("According to the strategy
  guidance...", "Following the strategy heuristic..."). This is the
  predicted cost of moving from soft heuristics to hard predicates and
  is acceptable for the bench's purpose. Watch whether the cite-y
  reasoning style affects student-LoRA training in unexpected ways
  downstream.

- Session `…3336ada5a161`, seed `663543359`, model `gemma-4-31b-it`, app
  build **`cef6291` (hybrid-v1.2)**. Two exports for this session:
  predecessor `solitaire-ai-log-a5a161-1779933184182.json` (202 rows, a
  snapshot at `moveCount: 151` / `finalProgress: 12%`, ingested
  2026-05-28) and the now-canonical, later export
  `solitaire-ai-log-a5a161-1779968626089.json` (372 rows, 192 success /
  180 errors, 170 new rows over the predecessor's 202, ingested
  2026-05-28). Canonical final stored state: `moveCount: 286`,
  `finalProgress: 33%`, `foundationCards: 17`, `faceDownTotal: 3`,
  outcome `incomplete`. The session DID progress between snapshots
  (12% -> 33%, foundations 6 -> 17, face-down 12 -> 3) before entrenching.
  `promptTemplateVersion` field is `None` on the success rows
  (unexplained gap; build hash `cef6291` confirms it is the v1.2 build
  whose template should be `hybrid-v1.2`). **Class: behavioural-doom-loop
  on a CONFIRMED-WINNABLE board.** pyksolve solved 20/20 sampled worlds
  (8 ms each), and the deck is near-fully determined here: only 4 cards
  are unknown (`5H`, `6D`, `3C`, `8C`) across the 3 face-down slots in
  col 5 plus the 1 remaining stock card, so 20/20 is effectively
  ground-truth winnable. Session-wide oscillation counts on the canonical
  export: `7C col 4 ↔ col 5` 114×, `2C col 2 ↔ col 3` 69×, `2C col 3 ↔
  col 7` 64× (the loops migrated and intensified from the predecessor's
  `6C`/`7H col 5 ↔ col 6` patterns as the board changed). Root cause is a
  waste-relocation blind spot: the latest `boardAnalysis` self-reports
  "The game is currently in a deadlock" and correctly states that `9S`
  (needed to clear col 5) requires a red 10 "(TD or TH)", but treats the
  waste `TD` as inaccessible. It never searches the 2-ply unlock that was
  a legal move on the final turn: play `TD` from the waste onto `JS`
  (col 3), which manufactures the red-10 landing spot, then move `9S`+`8H`
  off col 5 onto it, revealing the three face-down cards (`5H`/`3C`/`8C`),
  all of which are immediately or near-immediately playable to
  foundations (hearts `4H` -> `5H`, clubs `2C` -> `3C`). An empty col 1
  gave ample maneuvering slack the whole time. This is the canonical
  example of the model dismissing a relocatable waste card and
  declaring deadlock on an open, winnable board. **Recommend operator
  kill** (plateau entrenched, 100+ session-wide oscillations on `7C`
  alone, and the agent now self-reports deadlock while a legal unlock
  sits in its move list).

- Session `…74a30422b87e`, seed `3631548599`, model **`gemma-4-26b-a4b-it`**,
  app build `7c946d4`, prompt **`hybrid-v1.3`** (templateHash `7d9ecda4…`).
  One export in `raw/`: `solitaire-ai-log-22b87e-1780056307410.json` (349
  rows, 115 success / 234 errors, ingested 2026-05-29). Terminal state:
  `moveCount: 345`, `finalProgress: 27%`, `foundationCards: 14`,
  `faceDownTotal: 1`, outcome `incomplete`, plateau **16 turns**. **Class:
  behavioural-doom-loop on a winnable board** (pyksolve 10/10 consistent
  worlds solvable, mean 8 ms, "failure is behavioural, not structural").
  **The canonical "refuses to draw, misreads the blocker, with only 1
  face-down left" case.** Foundations H:`4H` D:`3D` C:`AC` S:`6S`; the model
  has already turned 20 of 21 face-down cards and still stalls. The lone
  remaining face-down sits at the bottom of column 4, under the run
  `8H 7S 6D 5C 4D 3C`, and can only be exposed by moving that 6-card run onto
  a black 9. Both black 9s (`9S`, `9C`) are unaccounted (somewhere in the
  14-card stock or the single face-down), and `TH` is exposed on top of column
  3 to receive a drawn black 9, so the unlock is mechanical: draw the stock,
  play a black 9 onto `TH`, move the column-4 run onto it, reveal the last
  card, cascade. The model never draws: the stock sits at 14 with the waste
  empty across the whole tail while it oscillates the 10-card
  `KD QS JH TC 9H 8C 7D 6C 5H 4C` run between the empty column 1 and column 2
  (session-wide `9H col 1 ↔ col 2` 41×, `TC col 1 ↔ col 2` 36×, `JH col 1 ↔
  col 2` 33×). It also misdiagnoses the lock, writing "no legal move can
  expose the face-down in column 4 because there is no available Red 4 to
  receive the 3C", but the face-down is under the entire run rather than the
  `3C`, and the real unlock is a black 9 from the stock, not a red 4: the
  agent's board model is wrong and it ignores the 14-card stock. Second
  26B-a4b doom-loop on v1.3, pairs with `…6a3d6d49b05f` (also seed-winnable,
  also oscillates a long run rather than drawing). None of the 115 success
  decisions entered clean-lean (all filtered `stalled-game`).

- Session `…d52502a1d118`, seed `1792828001`, model `gemma-4-31b-it`, app
  build `7c946d4`, prompt **`hybrid-v1.3`** (templateHash `7d9ecda4…`). Two
  exports in `raw/`: `solitaire-ai-log-a1d118-1780056875803.json` (287 rows,
  196 success / 91 errors, ingested 2026-05-29) and the superseding canonical
  `solitaire-ai-log-a1d118-1780090870848.json` (424 rows, 265 success / 159
  errors, 137 new, ingested 2026-05-30). State at the first (mc 373) snapshot:
  `moveCount: 373`, `finalProgress: 21%`, `foundationCards: 11`,
  `faceDownTotal: 6`, outcome `incomplete`, plateau **3 turns**. **Class:
  behavioural-doom-loop, winnable (pyksolve 10/10), apparently mid-breakout at
  the snapshot.** **Heaviest session-wide oscillation in the corpus to date**:
  `5S col 2 ↔ col 3` 118×, `6H col 2 ↔ col 3` 109×, `7S col 2 ↔ col 3` 100×
  (a three-card run sloshed between the empty column 2 and column 3 around 100
  times each across 373 moves at only 21% progress). The tail breaks the
  pattern, which is why the verdict was WATCH rather than kill: the last window
  consolidated the `9S 8D 7S 6H 5S` run back onto column 5, emptied column 2,
  played `3D` to the diamonds foundation, and drew from the stock, resetting
  plateau to 3. Foundations H:`4H` D:`3D` C:`2C` S:`2S`; 6 face-down remain
  (col3 1, col4 2, col7 3), 11 cards in stock, and crucially the model is
  drawing. Unlike the same-day 26B-a4b loops (`49b05f`, `22b87e`) that refused
  to draw, this 31B both draws and makes foundation progress, so 72 of its 160
  success decisions entered clean-lean (the rest filtered `stalled-game`). On
  the `c05ad4` / `aca45a` recoverable-loop precedent the call is WATCH; kill
  only if a later snapshot shows foundations still at 11 with the
  `col 2 ↔ col 3` slosh resumed. Canonical "extreme oscillation count but the
  31B is drawing its way out" datapoint, contrast with the 26B refuse-to-draw
  loops above.

  **Update 2026-05-30 (tripwire fired, operator killed).** The later
  `…1780090870848.json` snapshot (`moveCount: 508`, `finalProgress: 23%`) shows
  the WATCH grace was not vindicated: 135 more moves bought only +1 foundation
  card (11 to 12) and ZERO new reveals (faceDown stuck at 6), while plateau
  exploded to **61 turns** and the `5S/6H/7S col 2 ↔ col 3` slosh resumed and
  intensified to the heaviest in the corpus (`5S` 151×, `6H` 138×, `7S` 125×).
  Damning: the model's own boardAnalysis names the unlock ("a King of Diamonds
  available in the waste pile and an empty column (col 5)"), `Move KD from the
  waste to column 5 (empty)` is in the legal-move list, and it played
  `5S col 3 -> col 2` instead. The earlier "breakout" (one `3D` to foundation
  plus a couple draws) was a one-card blip, not a recovery, and faceDown never
  fell. **Final class: confirmed entrenched behavioural-doom-loop;
  operator-killed 2026-05-30.** Only 16 of the 69 new decisions cleared the
  stall filter. Methodological lesson: do not credit a breakout on a foundation
  play plus draws alone; require `faceDownTotal` to actually fall (real reveals)
  as the recovery signal, as in `c05ad4` / `aca45a`.

- Session `#b2d946` (full `019e75b5-2b71-7de5-aa06-ea6394b2d946`), seed
  `1152037935`, model `gemma-4-31b-it`, app build `3136c81`, prompt
  `hybrid-v1.3` (templateHash `7d9ecda4…`). Single
  canonical export `solitaire-ai-log-b2d946-1780173641817.json` (518 rows,
  494 success / 24 error). Final stored state: 518 interactions, max
  successful turn `504`, `moveCount: 505`, `finalProgress: 19%`,
  `foundationCards: 10`, `faceDownTotal: 18`, plateau `416` turns.
  **Class: behavioural-doom-loop on a winnable board** (pyksolve 10/10 on
  consistent worlds). Session-wide oscillation `9H col 2 ↔ col 4` 115×,
  `TS col 2 ↔ col 4` 97×, `JD col 2 ↔ col 4` 86× (window-inflated counts) —
  the single `KD/QS-JD-TS-9H` stack hopping between the two red kings KD
  (col 2) and KH (col 4) across the plateau, one coherent loop rather than
  diffuse churn. **First-in-corpus look at the v1.3 anti-undo rule
  firing-but-insufficient:** on the final turn the model correctly *declines*
  the loop move ("This violates the strategy heuristic to avoid moving cards
  back to a column they recently occupied to prevent infinite loops") and
  draws instead — but the stock is down to its last card (`drawPileCount: 1`,
  `canRecycleStock: false`) so the draw is futile, and the only tableau move
  is the anti-undo-forbidden one. Damning: the model itself notes that move
  would make progress ("move [0] would allow the 10 of Clubs (TC) from the
  waste to be played onto the 9 of Hearts (9H)") yet rejects it because it
  "does not expose" a face-down. The v1.3 predicate cannot distinguish a loop
  iteration from the loop-shaped move that unlocks a waste play. Operator-killed
  2026-05-31.
  [Correction 2026-06-03: the "behavioural-doom-loop on a winnable board" basis
  was the broken `check_winnability.py` (pyksolve). The fixed engine solver
  proves this board STRUCTURALLY DEAD in 40/40 sampled worlds (exhausts in <=18
  states, so the lock is at the surface and independent of the 18 face-down
  cards). The TC-from-waste-onto-9H observation still holds as a v1.3-predicate
  illustration, but the board was not winnable; the obedience-trap framing
  shared with `#783780`/`#cbced2` should note that this arm was on a dead deal.]

- Session `#783780` (full `019e759f-87bc-7dd6-9dad-f354f9783780`), seed
  `1514988667`, model `gemma-4-31b-it`, app build `3136c81`, prompt
  `hybrid-v1.3`. Single canonical export
  `solitaire-ai-log-783780-1780173634325.json` (453 rows, 426 success / 27
  error). Final stored state: 453 interactions, max successful turn `504`,
  `moveCount: 505`, `finalProgress: 31%`, `foundationCards: 16`,
  `faceDownTotal: 12`, plateau `332` turns. **Class: behavioural-doom-loop on
  a winnable board** (pyksolve 10/10). Spades foundation still `None` — `AS`
  never surfaced. Session-wide oscillation `6S col 4 ↔ col 7` 181×, `7D col 4
  ↔ col 7` 179×, `8S col 4 ↔ col 7` 161× — the single `9D-8S-7D-6S` run
  bouncing between TS (col 7) and TC (col 4). Same v1.3 tail behaviour as
  `b2d946`: the model declines the loop move on the final turn ("This violates
  the strategy guidance against undoing recent moves... the only productive
  action is to draw") and draws, but `drawPileCount: 8` with
  `canRecycleStock: false` and it never breaks the run apart to reveal the 12
  buried cards. Operator-killed 2026-05-31.

- Session `#cbced2` (full `019e7193-6bee-7682-bd63-12c11dcbced2`), seed
  `764521981`, model `gemma-4-26b-a4b-it` (MoE, active-4B), app build
  `7c946d4`, prompt `hybrid-v1.3`. Canonical export
  `solitaire-ai-log-cbced2-1780174347096.json` (468 rows, 195 success / 273
  error — error-heavy; see the 26B-cohort note); the earlier
  `…1780124475823.json` (402 rows) is a strict subset, archived to
  `raw/archive/`. Final stored state: 468 interactions, max successful turn
  `~466`, `moveCount: 467`, `finalProgress: 13%`, `foundationCards: 7`,
  `faceDownTotal: 13`, plateau `74` turns. **Class: behavioural-doom-loop on
  a winnable board** (pyksolve 10/10). **Operator-killed 2026-05-31, and the
  85-move tail between the two exports confirms the KILL**: from `moveCount`
  382→467 the foundation never left 7 and `faceDownTotal` never fell below 13
  (no breakout — the `faceDownTotal`-must-fall recovery test never fired), the
  loop only intensified (`8H col 1 ↔ col 3` 48×→63×, `9S` 39×→52×, `TH`
  37×→50×) and was still active in the final-10 window (`TH col 1 ↔ col 3`
  2×; recentMoves show the textbook `8H/9S/TH col 1 ↔ col 3` reversal). All 27
  decisions added by the newer export were stall-filtered out of
  `training.jsonl`. **Contrast with the 31B v1.3 sessions:** the 26B model
  *ignores* the anti-undo rule, repeatedly playing the loop move (e.g. the
  earlier-export final turn chose `JC+3 col 1 -> col 3`, moveIndex 1, with a
  "most productive... organizes the tableau" rationale) — the v1.3
  predicate-override failure on the MoE base. Two empty columns (col 2, col 4)
  sat available and unused throughout.

- Session `#564fc9` (full `019e73e3-b53a-70a0-87ef-c677d1564fc9`), seed
  `1352343714`, model `gemma-4-26b-a4b-it` (26B MoE cohort, not the 31B
  teacher), app build `3136c81` (2026-05-29), prompt `hybrid-v1.3` (templateHash
  `7d9ecda4…`). One artefact in `raw/`:
  `solitaire-ai-log-564fc9-1780285177525.json` (550 rows, 225 success / 325
  errors), ingested 2026-06-01. Final stored state: `moveCount: 648`,
  `finalProgress: 12%`, outcome `incomplete`. Behavioural doom-loop on a
  confirmed-winnable board (pyksolve 10/10, mean 8 ms): the model shuttles a
  7-card alternating run `TH-9S-8H-7S-6H-5C-4H` between col 1 and col 5 and
  back, frozen at `foundationCards=6, faceDownTotal=14` for `plateauTurns=71`
  with no foundation gain and no reveal. The session-wide window count is
  dominated by a single card (`TH col 4 ↔ col 5` 445×, `TH col 1 ↔ col 5` 194×,
  `TH col 1 ↔ col 4` 30×), the one-dominant-card real-loop signature rather than
  diffuse churn; recentMoves show the full run going `col 5 -> col 1` then
  immediately `col 1 -> col 5`. The model's final rationale calls the run-move
  "highly beneficial ... consolidates a large sequence" while revealing none of
  the 14 face-down cards, and it names empty columns (col 2, col 4) it never
  uses to unbury a stack. A 26B-specific MoE failure: a severe single-card
  oscillation (445×) consistent with the predicted long-context routing
  instability, distinct from the 31B obedience-trap. Excluded from the default
  training set by the `TEACHER_MODEL=gemma-4-31b-it` filter (this ingest added
  +222 rows to the `full` publish config and +0 to `clean-lean`/`training.jsonl`).

- Session `#8a5d12` (full `019e765b-4a67-75d3-a6c3-3dbcc38a5d12`), seed
  `3208238335`, model `gemma-4-31b-it` (the 31B teacher), app build `3136c81`
  (2026-05-29), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in
  `raw/`: `solitaire-ai-log-8a5d12-1780285508580.json` (798 rows, 297 success /
  501 errors), ingested 2026-06-01. Final stored state: `moveCount: 568`,
  `finalProgress: 42%`, outcome `incomplete`. Behavioural failure on a
  confirmed-winnable board (pyksolve 10/10, mean 8 ms): the teacher is frozen
  for `plateauTurns=175` at `foundationCards=22, faceDownTotal=3`, unable to
  expose the last three face-down cards (2 in col 4, 1 in col 6) even though the
  deal is solvable. Reasoning is lucid but impotent: it names the exact goal,
  "only 3 face-down cards remaining ... The most critical objective is to expose
  the remaining face-down cards to unlock the rest of the deck", and fails at it
  for 175 turns. Not the canonical tight 2-card oscillation (the latest window
  is a non-looping 8-card reorganization `col 1 -> col 7`); it is diffuse
  late-game churn dominated by `7C`/`6D` shuffling in and out of col 6, the
  column holding the lone face-down (`7C col 2/col 3 ↔ col 6` 126×/104×, `6D
  col 2 ↔ col 6` 113×). Role in corpus: the end-game-sparsity weak spot in the
  TEACHER's own play, distinct from the 26B loops. Because it is the teacher
  model it still feeds training: this ingest added +280 success decisions, of
  which +100 entered `clean-lean`/`training.jsonl` (the productive pre-plateau
  play), the 175-turn stall stretch excluded by the stall filter. Contrast
  `#564fc9` (26B, +0 to clean-lean, model-filtered).
  [Correction 2026-06-03: the "winnable board" basis here was the broken
  `check_winnability.py` (pyksolve). The fixed engine solver proves this board
  STRUCTURALLY DEAD in 40/40 sampled worlds (exhausts in <=96 states, a
  surface-level lock). So this was a lucid-but-impotent stall on a DEAD board,
  not "on a winnable board"; the model's inability to progress was correct, and
  the only fault is that it never emitted the `move_index:-1` resign.]

- Session `#ffec5a` (full `019e76b2-53ae-7264-950d-0fac60ffec5a`), seed
  `3948741078`, model `gemma-4-26b-a4b-it` (26B MoE cohort), app build
  `819012b` (2026-05-30, a newer build than `#564fc9`'s `3136c81`), prompt
  `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in `raw/`:
  `solitaire-ai-log-ffec5a-1780301451712.json` (692 rows, 220 success / 472
  errors), ingested 2026-06-01. Final stored state: `moveCount: 502`,
  `finalProgress: 17%`, outcome `incomplete`. Behavioural doom-loop on a
  confirmed-winnable board (pyksolve 10/10, mean 8 ms): the 5-card run
  `8C-7D-6C-5H-4C` shuttles between col 1 and col 7 and straight back, frozen at
  `foundationCards=9, faceDownTotal=12` for `plateauTurns=92` with no reveal of
  the 12 face-down cards. Unlike the diffuse `#8a5d12` churn, the latest window
  is the textbook tight reversal: the full run goes `col 1 -> col 7`
  (`8C,7D,6C,5H,4C`) then immediately `col 7 -> col 1`; the session-wide count
  is dominated by the run's own cards (`4C col 1 ↔ col 7` 179×, `5H` 165×, `7D`
  150×). Reasoning self-deluded ("moving the run ... will significantly clear
  column 1" while it just moves back). Role in corpus: the third 26B MoE
  doom-loop, and confirmation the loop persists on a newer build (`819012b`)
  under the same v1.3 prompt. Excluded from the default training set
  (`TEACHER_MODEL=gemma-4-31b-it`): this ingest added +219 rows to the `full`
  publish config and +0 to `clean-lean`/`training.jsonl`, the same profile as
  `#564fc9`.

- Session `#4a9fe1` (full `019e7f6d-d6eb-7ea1-aec7-9fbd864a9fe1`), seed
  `811891845`, model `gemma-4-31b-it` (the 31B teacher), app build `df3a89b`
  (2026-05-31), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in
  `raw/`: `solitaire-ai-log-4a9fe1-1780381846668.json` (926 rows, 360 success /
  566 errors), ingested 2026-06-02. Final stored state: `moveCount: 472`,
  `finalProgress: 27%`, outcome `incomplete`. Behavioural failure with a
  signature distinct from the other batch sessions: FALSE RESIGNATION. The model
  self-diagnoses a structural lock that the solver contradicts (pyksolve 10/10
  over sampled face-down worlds; the Monte Carlo over-optimism caveat applies,
  the true arrangement is unverified). Its reasoning: "no legal moves exist that
  would expose any of these face-down cards: moving 7H (col 4), 5D (col 5), or
  9H (col 7) is not possible because the required destination cards (black 8,
  black 6, black 10) are either not present or are buried". It then flails:
  `plateauTurns=232` at `foundationCards=14, faceDownTotal=8`, `recentMoves`
  mostly draws cycling an exhausted stock (`drawPileCount=1`,
  `canRecycleStock=False`, 6 stock cards seen) plus a tiny `4S col 3 ↔ col 5`
  twitch. Contrast the deluded-looping of `#564fc9`/`#ffec5a` (the model thinks
  the run-shuffle is progress) and the lucid-but-trying `#8a5d12` (names the
  goal and keeps reorganizing): here the teacher gives UP on a likely-winnable
  board rather than emitting the `move_index:-1` resign. As the teacher model it
  still feeds training: +347 success decisions tagged, +131 into
  `clean-lean`/`training.jsonl` (the pre-plateau play), the 232-turn stall
  stretch excluded by the filter.
  [Correction 2026-06-03: the "false resignation on a likely-winnable board"
  verdict rested on the broken `check_winnability.py` (pyksolve 10/10). The
  fixed engine solver proves this board STRUCTURALLY DEAD in 40/40 sampled
  worlds (exhausts in <=112 states). So the model did NOT give up on a winnable
  board: it correctly judged a dead board. The accurate failure here is narrow:
  it recognised the deadlock in its reasoning but never emitted the
  `move_index:-1` resign action (cf. `#30e5e5`, which did). Reclassify from
  "false resignation" to "correct-diagnosis, resign-action-not-taken".]

- Session `#3fd319` (full `019e7f6e-b01c-7e2c-9798-4e56983fd319`), seed
  `1965004236`, model `gemma-4-31b-it` (the 31B teacher), app build `df3a89b`
  (2026-05-31), prompt `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in
  `raw/`: `solitaire-ai-log-3fd319-1780381874122.json` (869 rows, 367 success /
  502 errors), ingested 2026-06-02. Final stored state: `moveCount: 567`,
  `finalProgress: 17%`, outcome `incomplete`. Behavioural failure on a winnable
  board (pyksolve 10/10), caught earlier and milder than the other batch stalls:
  `plateauTurns=67` at `foundationCards=9, faceDownTotal=2` (2 face-down in
  cols 5/6). The `7H-6S-5H` run oscillates `col 6 ↔ col 7` (the latest window
  shows both directions) alongside a `col 1 -> col 2` run shuffle; session-wide
  counts are modest (`5H col 6 ↔ col 7` 46×, `2H col 1 ↔ col 5` 43×, `6S col 6 ↔
  col 7` 41×, far below the 26B loops). The teacher self-reports the lock ("No
  legal move immediately exposes these cards") yet leaves an out unused:
  `canRecycleStock=True` and its own reasoning names "7D in the stock/waste
  cycle" (which would take the waste `6C`), but it never recycles to reach it. A
  third distinct teacher signature this batch (after `#8a5d12` stuck-endgame and
  `#4a9fe1` false-resignation): a stall that ignores a known stock-recycle path.
  As the teacher it still feeds training: +315 success decisions, +132 into
  `clean-lean`/`training.jsonl` (pre-plateau), the 67-turn stall stretch
  excluded.

- Session `#8dcf34` (full `019e7dca-5f1e-7b6a-a413-fe663b8dcf34`), seed
  `2209184236`, model **`gemma-4-26b-a4b-it`** (26B MoE cohort; excluded from the
  default training set by the `TEACHER_MODEL=gemma-4-31b-it` filter), app build
  `df3a89b` (2026-05-31T12:10:49Z). One artefact in `raw/`:
  `solitaire-ai-log-8dcf34-1780398077766.json`, ingested 2026-06-03. The session
  block reports `moveCount: 404`, `finalProgress: 19%`, outcome `incomplete`: a
  clear stall (404 moves for only ~10/52 foundations). No behavioural signature
  can be quoted because the export's `interactions` array is empty (`count: 0`),
  so the decision log carries zero turns and contributes zero rows to the store.
  Kept for full-stream completeness and flagged as a data-quality gap: an empty
  26B decision log on build `df3a89b` (the session stats survive, the per-turn
  reasoning does not).

- Session `#404d11` (full `019e8a60-c4ad-7120-90cc-53857a404d11`), seed
  `3255629335`, model `gemma-4-31b-it`, app build `df3a89b` (2026-05-31), prompt
  `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in `raw/`:
  `solitaire-ai-log-404d11-1780568790074.json` (751 rows, 442 success / 309
  errors), ingested 2026-06-04. Final stored state: max successful turn `499`,
  `moveCount: 500`, `finalProgress: 17%`, outcome `incomplete`, cap-terminated at
  the ~500-turn budget (no `solitaire-win-*`/`solitaire-game-*` file emitted).
  Structurally dead at the cap: `check_winnability.py` (repo-engine best-first,
  recycle modelled, node_cap 500k) proves the latest board dead in 10/10
  consistent worlds (mean 52 states/world, exhaustive), and the small-search
  terminal checkpoints agree (turn 250 onward, 8/8 dead at ~950 states). The exact
  death-turn earlier is not pinnable here: with 11+ cards still hidden the Monte
  Carlo determinisation is noisy and non-monotonic (turn 200 sampled 6/6 dead, yet
  the real game then climbed foundations 5 to 9 by turn 225, so those were unlucky
  face-down assignments, not the true board), which is itself the caution that
  this method only gives a sound verdict once few cards remain hidden; a
  turn-resolved early verdict needs the seed replayed for the true deck. The board
  barely opened: foundationCards never exceeded 9 and faceDownTotal never fell below 11
  (only 10 of 21 hidden cards ever revealed). It reached its high-water mark
  `(foundationCards=9, faceDownTotal=11)` at turn 225, then spent the final ~275
  turns there: the `JS-TD-9S-8D` run oscillates `col 2 ↔ col 7` (session-wide
  `TD` 182×, `8D` 180×, `9S` 167×; latest window is the run going col 7 -> col 2
  then col 2 -> col 7 around a `draw 6D`). This is NOT the reveal-pass-up kill
  signature: reveals were almost never on offer (9 reveal-tagged legal moves all
  game, 2.3% of turns) and the teacher took 8 of the 9 (11% pass-up, in the
  winning-session band, far below `a1d118` 27%). It played its reveals correctly
  and simply had no foundation or reveal move in the terminal phase, so with no
  resign emitted it toggled a movable run to the cap. Companion to `#c36b7b` from
  the same 2026-06-04 batch (same class). Contrast `#3fd319` above (also 17%, same
  batch) which is behavioural on a winnable board; here the board is provably dead.

- Session `#c36b7b` (full `019e8a61-3fe7-7cb2-8fa9-e65984c36b7b`), seed
  `3602844246`, model `gemma-4-31b-it`, app build `df3a89b` (2026-05-31), prompt
  `hybrid-v1.3` (templateHash `7d9ecda4…`). One artefact in `raw/`:
  `solitaire-ai-log-c36b7b-1780532018201.json` (520 rows, 256 success / 264
  errors), ingested 2026-06-04. Final stored state: max successful turn `493`,
  `moveCount: 504`, `finalProgress: 13%`, outcome `incomplete`, cap-terminated,
  ai-log only. Structurally dead, not behavioural: `check_winnability.py`
  (repo-engine, recycle modelled) proves the latest board dead in 10/10 worlds
  (mean 656 states/world, exhaustive); the terminal-plateau checkpoints are
  consistently dead from turn 142 on (8/8 at both 142 and 161, last winnable
  checkpoint at turn 136, moderate search), so it died about when it plateaued.
  The board opened even less than `#404d11`:
  foundationCards never exceeded 7 and faceDownTotal never fell below 13 (only 8
  of 21 hidden cards ever revealed). It hit `(foundationCards=7, faceDownTotal=13)`
  at turn 142 and oscillated for the remaining ~358 turns: the 11-card
  `QC-JD-TS-9H-8C-7D-6C-5H-4C-3D-2C` alternating-color run slides `col 4 ↔ col 6`
  (session-wide `9H` 86×, `QC` 85×, `8C` 85×), the model believing moving it onto
  the `KD` in col 4 will expose col 6's five face-down cards. Reveal discipline is
  good, not the kill signature: only 10 reveal-tagged moves offered all game (3.5%
  of turns), 8 of 9 reveal-turns taken (11% pass-up). At the end `drawPileCount=1`
  and `canRecycleStock=False`, so the stock is spent. Same class and batch as
  `#404d11`: a structural lock the teacher could not escape and did not resign from.

- Session `#136236` (full `019e927b-e4bd-790b-9272-99c41e136236`), seed
  `2945049884`, model `gemma-4-31b-it`, app build **`fa14fe3`**
  (2026-06-04T10:16:45Z), prompt **`hybrid-v1.4`** (templateHash `818edeb2…`).
  **The first `hybrid-v1.4` session in the corpus.** One artefact in `raw/`:
  `solitaire-ai-log-136236-1780636605884.json` (504 rows, 400 success / 104
  errors), ingested 2026-06-05. Final stored state: max successful turn `527`,
  `moveCount: 528`, `finalProgress: 21%`, outcome `incomplete`, cap-terminated at
  the ~500-turn budget (ai-log only, no terminal state file). Structurally dead,
  not behavioural: `check_winnability.py` (repo-engine, recycle modelled) proves
  the latest board dead in 10/10 worlds (mean 34 states/world, exhaustive). Same
  structural-lock signature as the v1.3 dead-deals `#404d11`/`#c36b7b`: the board
  barely opened (foundationCards never exceeded 11, faceDownTotal never fell below
  15, only 6 of 21 hidden cards ever revealed), reaching `(foundationCards=11,
  faceDownTotal=15)` at turn 144 then oscillating a `6D-3C-4D` run `col 5 ↔ col 6`
  for the final ~384 turns (session-wide window counts 164×/140×/130×, the usual
  rolling-window inflation over a dead board). Reveal discipline is clean and
  beside the point here: only 3 reveal-turns offered all game, all 3 taken (0%
  pass-up), the reveal-starved structural lock rather than a pass-up failure. One
  v1.4 datapoint worth noting: the error rate is much lower than the df3a89b v1.3
  sessions (104/504, 21%, vs ~45 to 58%), but whether that is the v1.4 prompt or
  provider-side cannot be told from N=1, and the rendered v1.4 prompt text has not
  yet been diffed against v1.3. The dead deck makes this session uninformative
  about whether v1.4 changes the behavioural-loop rate; it shows only that v1.4
  does not (and cannot) rescue a structurally lost deal.

- Session `#523f19` (full `019e926f-c0a7-71fe-a50c-632cfc523f19`), seed
  `521496738`, model `gemma-4-31b-it`, app build **`fa14fe3`**
  (2026-06-04T10:16:45Z), prompt **`hybrid-v1.4`** (templateHash `818edeb2…`).
  One artefact in `raw/`: `solitaire-ai-log-523f19-1780697647886.json` (879 rows,
  695 success / 184 errors), ingested 2026-06-06. Final stored state:
  `moveCount: 700`, `finalProgress: 12%`, outcome `incomplete`, cap-terminated at
  the ~700-turn budget (ai-log only, no terminal state file). This is the longest
  plateau in the recent cap-stall set: foundation froze at `foundationCards=6`
  and `faceDownTotal=12` with `plateauTurns=542` out of 700, i.e. the board
  stopped opening around turn ~158 and the remaining ~540 turns were pure churn.
  The signature is a single dominant card: `4S` moved `col 5 ↔ col 7` **631×**
  session-wide; the next two loops (`4H col2↔col4` 38×, `5S col2↔col4` 35×) are an
  order of magnitude smaller. By the one-dominant-card test (see
  oscillation-window-count-inflates) this is a true tight loop, not diffuse
  rolling-window inflation. The teacher was self-aware and looped anyway: its
  latest boardAnalysis states the col-7 top card `4S` "can only move to 5D in
  Column 5, which does not reveal any hidden cards", yet that is the move it ran
  631 times. No resign emitted. Winnability is **indeterminate** here, unlike the
  sibling v1.4 dead-deal `#136236` (proved dead 10/10, mean 34 states): the
  repo-engine solver ran 5+ minutes pinned at the 500k-node cap without
  completing a single determinized world, the expected non-result on a
  high-unknown board (12 face-down + 10 stock = 22 unknown) where Monte-Carlo
  determinization is not sound (see winnability-montecarlo-false-dead). So this is
  classified on behaviour as a frozen-progress cap-stall with a dominant
  reveal-inert oscillation, not adjudicated dead. Like `#136236`, the v1.4 error
  rate is low (184/879, 21%, vs the df3a89b v1.3 sessions' ~45 to 58%). Companion
  v1.4 31B stall to `#136236`: same build, prompt, low error rate, and
  cap-terminated frozen-board outcome, differing only in that `#136236` was
  provably dead while `#523f19` could not be adjudicated. Batch note (the
  2026-06-06 kills): the same ingest carried three sessions that were live at
  ingest and the operator then killed on the kill recommendation (2026-06-06), so
  all three are now terminal losses in the win-rate denominator. `#4c3a11` (full
  `019e922b-d7c3-7576-ad53-09d0314c3a11`, seed `282557647`, 31B v1.4, killed near
  `moveCount 483`) is the standout: `check_winnability` proved it **WINNABLE
  10/10** (mean 46 states, sound at faceDown=2) while the model looped `4C/3S/5D`
  and asserted "7S is blocked", a behavioural-loop-on-winnable loss and the clean
  exemplar of the class the v1.5 ask targets. The other two are 26B behavioural
  run-loops, killed before they could be adjudicated (high-unknown): `#37a4cd`
  (seed 778149891, 26B v1.4, `5D-4C` run `col 4 ↔ col 5` 199x/169x) and `#ea00f0`
  (seed 4046349543, 26B v1.3, `4C-3D` run `col 5 ↔ col 7` 211x/190x). No terminal
  `solitaire-win`/`solitaire-game` file was emitted on the manual kills, so the
  last-logged ai-log boards above are the final record unless a terminal export
  arrives later (dedup-safe to re-ingest).

- Session `#15c62d` (full `019e99f5-12ed-7be5-ade5-7de5a615c62d`), seed
  `2129367171`, model `gemma-4-31b-it`, app build **`6810750`**
  (2026-06-05T22:28:30Z), prompt **`hybrid-v1.5`** (templateHash `8a46ca22…`,
  `promptTemplateFinalisedAt` 2026-06-06). **The first `hybrid-v1.5` doom-loop in
  the corpus** (the only other v1.5 session, `#6eb393`, is a win under Won
  sessions). One artefact in `raw/`: `solitaire-ai-log-15c62d-1780780633670.json`
  (427 rows, 277 success / 150 errors), ingested 2026-06-07. Final stored state:
  `moveCount: 351`, `finalProgress: 12%`, outcome `incomplete` (ai-log only, no
  terminal state file; KILL recommended on this verdict 2026-06-07, counted as a
  loss in the win-rate denominator). Terminal board `foundationCards=6` (H:4H,
  S:2S), `faceDownTotal=11`, drawPile 3, recycle unavailable.

  **Dead-deal flailing, proven by structure (identity-independent, no solver
  needed).** All four next-needed foundation cards are face-down: AD, AC (both
  Aces), 3S, 5H. No legal-move sequence reveals any hidden card, so the face-down
  identities are irrelevant and the proof holds for every arrangement. col 3 is
  pinned by `QC` (a queen needs a red K in an empty column); col 5 by `7H` (needs
  an exposed black 8, but 8S is covered by 7D and 8C is in the stock, a circular
  block); col 6 by `2C` over 4 hidden (needs a red 3, none face-up in the
  tableau); col 7 by `8H` (needs an exposed `9C`, itself buried under col 2's run,
  circular). The only free single card, `2D` in col 4, cannot move (needs AD, or a
  black 3, none available), so no empty column can ever be created and the two
  king-rooted runs (col 1 `KD..7D`, col 2 `KC..4D`) can never clear. The reachable
  move set is therefore closed: shuffle the two runs, plus draw/recycle a
  fully-known stock. Stronger than a Monte-Carlo verdict on the 14 unknowns (11
  face-down + 3 stock), where determinization would be unsound (see
  winnability-montecarlo-false-dead); here no determinization is needed.
  Foundations froze at turn ~59 with the export at turn 350, a ~290-turn dead
  plateau; an early-kill tripwire near turn 100 would have saved ~240 turns.

  **Dominant symptom is a draw-stall, not the tableau oscillation.** Of 256 parsed
  moves, **186 draws + 15 recycles = 201 (78%)**. The tableau oscillation is small
  and overstated by the rolling-window briefing (`4D/5C/6H col 2 ↔ col 7`
  115×/102×/91×). De-inflated stitched reversals total **27**: `7C col 2 ↔ col 7`
  15× (the 7C-6H-5C-4D run) and `9D col 1 ↔ col 3` 11× (the 9D-8S-7D run), plus one
  `TS`. The 4D/5C/6H figures were the same 7C-run move counted per-card per-window.

  **Root cause of the draw-stall: a confident illegal plan.** Late reasoning
  (turns 349-350) draws to "reach 8C ... 8C (black) can receive the 7H (red) from
  Col 5 ... moving 7H to the waste will reveal a hidden card". The model believes a
  drawn stock card, which lands on the WASTE, can serve as a tableau base for a
  tableau card. It cannot: the waste only gives its top card, it never receives
  one. The teacher chases this impossible play for hundreds of draws. This is a
  Gemma rules error, not a render bug; the prompt correctly offered only moves
  [0]/[1]/draw.

  **First dead board under `hybrid-v1.5`, and the new stall counts did not fire.**
  v1.5 deleted the draw-directive and added `turns since foundation grew` / `turns
  since a card was revealed` precisely so the model could perceive a stall and
  resign. On this 290-turn-frozen board the deletion did NOT stop compulsive
  drawing (186 draws), and the counts produced no resign: across 277 successful
  turns, `turns since...` is cited 3 times, resign/give-up language appears 0
  times, and `move_index: -1` was never emitted, even while the model used
  dead/locked language about columns on 62 turns. This is the counterexample to the
  `#6eb393` win and the first real test of the v1.5 stall-count lever: its intended
  perceive-stuck-then-resign payoff failed. Same no-resign-on-a-dead-board class as
  the v1.3 `#7b6318` and the v1.4 `#136236`/`#523f19`; the dead-board recognition /
  resign gap persists into v1.5.

- Session `#7a4b10` (full `019e99f6-f1f2-7381-9e33-9f3ff37a4b10`), seed
  `1428760046`, model **`gemma-4-26b-a4b-it`** (MoE comparison cohort, excluded
  from the 31B training set by the `TEACHER_MODEL` filter), app build `6810750`
  (2026-06-05T22:28:30Z), prompt `hybrid-v1.5` (templateHash `8a46ca22…`). One
  artefact in `raw/`: `solitaire-ai-log-7a4b10-1780781086425.json` (181 rows, 70
  success / 111 errors), ingested 2026-06-07. Final stored state: `moveCount:
  222`, `finalProgress: 10%`, outcome `incomplete` (ai-log only; KILL recommended
  2026-06-07, behavioural loss in the 26B-cohort denominator). Terminal board
  `foundationCards=5` (H:AH, D:2D, C:2C, S:null), `faceDownTotal=14`, drawPile 18,
  recycle unavailable.

  **Behavioural doom-loop on a NOT-dead board, driven by a fabricated reveal.**
  The `9H-8C-7D-6C` run shuttles `col 3 ↔ col 5` for **47 clean stitched
  reversals** (48 of 70 moves; a true tight loop, not the briefing's window-
  inflated 42×/27×/27×). The board is far from dead: 18 cards sit unseen in the
  stock (only 4 seen) and `draw` is legal move [1] every turn, yet the model drew
  on only 6 of 70 turns (8%). The loop engine is reveal-misattribution: col 5 is
  `[4 face-down] TS 9H 8C 7D 6C`, and the boardAnalysis repeatedly claims moving
  the `9H-8C-7D-6C` run "will reveal the face-down card beneath the TS" (reveal
  language on 49 of 70 turns). It will not: moving the run leaves the already-
  face-up `TS` on top and the 4 hidden cards stay pinned under TS (which needs a
  red J, none exposed). The prompt is NOT at fault: neither tableau move carries
  the "(reveals a hidden card)" tag (the two tag strings in the prompt are only the
  static STRATEGY GUIDANCE bullets), so the reveal is the 26B model's own
  hallucination, manufacturing the reveal-priority heuristic's justification for an
  untagged, reveal-inert move. No resign emitted (`move_index: -1` zero times). The
  `#3e91a0`-style 26B loop that escapes did not recur; caught early (53-turn
  plateau, move 222, well under the cap).

- Session `#4aa9f1` (full `019e99f7-1bb5-7a07-9d57-d8ea724aa9f1`), seed
  `2893821912`, model **`gemma-4-26b-a4b-it`** (MoE cohort, excluded from the 31B
  training set), app build `6810750`, prompt `hybrid-v1.5` (templateHash
  `8a46ca22…`). One artefact in `raw/`:
  `solitaire-ai-log-4aa9f1-1780781091449.json` (296 rows, 116 success / 180
  errors), ingested 2026-06-07. Final stored state: `moveCount: 306`,
  `finalProgress: 17%`, outcome `incomplete` (ai-log only; KILL recommended
  2026-06-07, behavioural loss in the 26B-cohort denominator). Terminal board
  `foundationCards=9` (H:5H, D:AD, C:null, S:3S), `faceDownTotal=8`, drawPile 7,
  recycle unavailable.

  **Same class as its batch-mate `#7a4b10`: a fabricated-reveal behavioural
  doom-loop, here with multiple concurrent oscillations.** Stitched reversals total
  **69**: the `6C-5D-4S-3D-2C` run `col 1 ↔ col 6` 41×, a `JD col 4 ↔ col 5`
  shuttle 19×, and a `2C col 2 ↔ col 5` shuttle 9× (the briefing's 102×/61×/33×
  are the inflated rolling-window versions). Drew on only 7 of 116 turns (6%)
  despite a legal draw [1] (7 stock cards) and a real foundation line it ignored
  (spades sit at `3S`, and `4S` is in the col-6 run, extractable). The reveal
  fabrication is identical: col 6 is `[4 face-down] 7H 6C 5D 4S 3D 2C`, and the
  model claims moving the `6C-5D-4S-3D-2C` run "will reveal the 7H ... and
  subsequently the hidden card beneath it" (reveal language on 68 of 116 turns),
  but `7H` is already face-up and only re-surfaces as the new top, revealing
  nothing (the 4 hidden stay under 7H, which needs an exposed black 8). Again no
  move carries the reveal tag and no resign was emitted. Board not proven dead (8
  face-down + 7 stock = 15 unknowns, plus the playable 4S line). Caught at a
  55-turn plateau. Batch context: ingested 2026-06-07 alongside the 31B v1.5 dead-
  deal `#15c62d`, and the cohort contrast under v1.5 is sharp: the 31B over-drew on
  a dead board (78% draws, despite the deleted draw-directive) while these two 26B
  sessions under-draw on playable boards (6-8%) and loop the tableau on
  hallucinated reveals. All three emitted zero resigns, so the v1.5 resign /
  stall-count lever fired in none of them.

- Session `#78c130` (full `019e99f5-ba6c-78d2-9c09-c323ce78c130`), seed
  `2358770568`, model `gemma-4-31b-it`, app build `6810750`
  (2026-06-05T22:28:30Z), prompt `hybrid-v1.5` (templateHash `8a46ca22…`). One
  artefact in `raw/`: `solitaire-ai-log-78c130-1780809593625.json` (466 rows, 163
  success / 303 errors), ingested 2026-06-07. Final stored state: `moveCount:
  325`, `finalProgress: 35%`, outcome `incomplete` (ai-log only; KILL recommended
  2026-06-07). Terminal board `foundationCards=18` (H:AH, D:8D, C:3C, S:6S),
  `faceDownTotal=4` (1 in col 4, 3 in col 6), drawPile 0 with a recyclable 3-card
  waste (JH, JC, TS), an empty col 7, and four kings available.

  **Played-into dead-end at an advanced foundation, then flailed (NOT a dealt-dead
  deal, NOT a winnable-board loop).** Despite reaching 18 foundation with only 4
  face-down, an empty column, and four kings, `check_winnability.py` (engine,
  recycle modelled) proves the board STRUCTURALLY DEAD: 10/10 sampled worlds
  exhausted as unwinnable in a mean of 64 states (sound at 4 unknowns; this is the
  small-search regime, not a high-unknown false-DEAD). The model maneuvered the two
  big king-runs (col 1 `KC..4H`, col 3 `KD..3H`) into a closed reshuffle component
  where the 4 hidden cards cannot be revealed and no foundation can advance, then,
  over 163 success turns, drew 63 times (39%), recycled the 3-card waste 5 times,
  and oscillated a `7S`-led run `col 1 ↔ col 6` (13 exact stitched reversals; 27
  total, `7S` 13x / `5C` 4x / `8H` 4x; the briefing's 78x/46x/40x is window
  inflation). It used stuck/dead language on 45 turns and emitted `move_index: -1`
  zero times, although the solver confirms a resign here would be CORRECT. The
  reached-position deadness is proven; whether the DEAL was winnable (a thrown-away
  win) is not separately adjudicated and would need a seed replay.

- Session `#8bbec1` (full `019e99f8-e8ee-701a-8820-817acc8bbec1`), seed
  `1350517526`, model `gemma-4-31b-it`, app build `6810750`, prompt `hybrid-v1.5`
  (templateHash `8a46ca22…`). One artefact in `raw/`:
  `solitaire-ai-log-8bbec1-1780809583950.json` (436 rows, 163 success / 273
  errors), ingested 2026-06-07. Final stored state: `moveCount: 420`,
  `finalProgress: 25%`, outcome `incomplete` (ai-log only; KILL recommended
  2026-06-07). Terminal board `foundationCards=13` (H:5H, D:2D, C:6C, S:null),
  `faceDownTotal=5` (3 in col 6, 2 in col 7), stock FULLY exhausted (drawPile 0,
  waste empty, recycle unavailable), an empty col 1. Spades never started: `AS` is
  among the 5 face-down and unreachable.

  **Same class as `#78c130`: a played-into dead-end at an advanced foundation,
  here a hard terminal lock.** `check_winnability.py` proves STRUCTURALLY DEAD
  10/10 in a mean of just 8 states (sound at 5 unknowns; an extremely tight lock).
  The variant restricts empty columns to kings (every empty-column legal move is a
  King-led run), and with the stock spent and no exposed black-9 or red-6 base, the
  `8H-7S` over col 6 and the `5S-4D-3S` over col 7 cannot be cleared, so the 5
  hidden (incl. `AS`) are permanently locked. Over 163 success turns the model drew
  43 times (26%) and thrashed: 49 exact stitched reversals but DIFFUSE (`TC col 2 ↔
  col 4` 11x, `6S col 2 ↔ col 6` 10x, `9H col 2 ↔ col 4` 7x, `7S col 1 ↔ col 6`
  7x; top-card dominance ~0.22, i.e. diffuse dead-board churn rather than one tight
  loop, per oscillation-window-count-inflates). Stuck/dead language on 54 turns, 0
  resigns. Batch context: ingested 2026-06-07 with `#78c130`; both are 31B v1.5
  played-into dead-ends at advanced foundations (18 and 13), distinct from the same
  day's dealt-dead `#15c62d` (foundation 6, dead from early) and the winnable-board
  26B loops `#7a4b10`/`#4aa9f1`. All five 2026-06-07 sessions emitted 0 resigns;
  on both #78c130 and #8bbec1 the engine solver confirms a resign would be correct,
  so the no-resign-on-a-dead-board gap now spans dealt-dead AND played-into-dead at
  advanced foundations under v1.5.

## Student full-game play

Full-session runs where the deployed LoRA student plays one of the
benchmark winnable decks end to end via
`gemma4_finetune/play_deck_with_student.py`. These are model artefacts
(not corpus); the published HF dataset stays teacher-only.

- **v1.1 LoRA (gemma-3n + adapters_t5_at750), seed 3263196305, run #2
  (2026-05-26).** 58 turns played to `final_foundation_cards=3`, then
  aborted via the runner's illegal-move safety cap. **Classification:
  MIDGAME STALL with behavioural-doom-loop signature**, same pathology
  class as the 31B teacher's adf71b / 645d03 / 73fd85 doom-loops.
  Turns 0-35 were competent play (reduced face-down from 21 to 12,
  played AC and AH to foundations, built sensible KH+QC tableau
  sequences). Turns 36-54 oscillated `JD col 4 <-> col 7` 18 times in
  19 turns, confidence saturated at 0.95-1.0 throughout. The illegal
  picks at turns 55-57 are a discrete additional failure mode:
  **move_index fixation**, where the student picked `move_index=4`
  three consecutive turns even after the legal-moves list dropped to
  4 entries. Run artefacts at
  `gemma4_finetune/play_runs/v1_seed3263196305_run2/`. **Material
  finding**: the deployed student inherits the teacher's doom-loop
  pathology on a solver-confirmed winnable deck. The resign and
  state-repetition annotation asks in the 2026-05-26 harvester ask
  are doubly justified by this; the student would benefit from them
  as much as the teacher does.

## Same-seed validation experiments

When the harvest team re-runs a known-failing seed under a different
build/prompt, the original session ID becomes the locked baseline and
the new session is the comparison arm.

- Seed `3689552861` — baseline session `…d46eb2645d03` (build `ce6afe1`,
  documented above; 75-turn doom-loop on 5C/4D). Comparison arm:
  session `…4a46c829a7f5`, model `gemma-4-31b-it`, ingested 2026-05-21
  via `solitaire-ai-log-29a7f5-1779361593611.json` (200 rows, 86 success
  / 114 errors). **Result: prompt v1 did not address the pathology.**
  Final state `moveCount: 285`, `finalProgress: 12%`, killed by operator
  after stall confirmed. The comparison arm reproduced the **exact same
  5C/4D oscillation between cols 3 and 4** as the baseline, with a
  **longer plateau** (85 turns, 198 → 284) than the baseline's 75 turns,
  on 99% of parsed turns. RecentMoves tail at terminal state: 16x
  `move 4D col 3 -> col 4`, 16x `move 4D col 4 -> col 3`, 14x
  `move 5C col 3 -> col 4`, 13x `move 5C col 4 -> col 3`. Final-turn
  reasoning shows the same self-aware-but-impotent pattern as the
  baseline (correctly identifies "neutral shuffles that do not reveal
  any face-down cards or advance the foundations" then picks an
  unproductive action anyway). Saturated 0.91 mean / 0.95 max
  confidence throughout the plateau. **Conclusions:** (1) the same-seed
  validation method works as designed — it delivered a clean negative
  result on the controlled experiment; (2) prompt v1 is insufficient
  for the 5C/4D oscillation class; (3) the stall auto-terminator (see
  `docs/internal/HARVEST_TEAM_NEXT_CORRECTION_2026-05-20.md`) is now
  unambiguously P0: even explicit prompt fixes targeted at this
  pathology fail to interrupt the loop, so harness-side termination is
  the only reliable line of defence.

## Cross-version teacher benchmarks

Locked seeds that the harvest team should re-run on every new prompt
build, to track teacher win-rate and per-turn behaviour across versions.
Each seed has been validated solvable by `pyksolve` against the
ground-truth initial state from the matching `solitaire-win-*` record.

### Seed `3263196305` (draw-3): cross-version comparison ready

pyksolve solves the ground-truth initial state (from
`solitaire-win-010e01`'s `initialBoardSetup`) in 9 ms draw-1 and 49 ms
draw-3 (`SolvedMayNotBeMinimal`). Two teacher runs on file, both reached
the win state:

- Baseline arm: session `…a1fa1abf260154e1`, build `6dfc8a9`, json-format
  prompt (`promptLayoutVersion: None`), ingested via
  `solitaire-ai-log-1779376068820.json` (319 rows, 138 success, 181 errors
  = 56.7% error rate). Reached foundationCards 51/52 by turn 173 with KS
  as the next chosen move; no win-record was exported but the state is
  effectively won.
- Comparison arm: session `…b3cf-ca549d010e01`, build `de7dc06`, hybrid-v1
  plain-text prompt (`promptLayoutVersion: hybrid-v1`,
  `promptTemplateHash: 0462323c…ddd0cdb9c`), ingested 2026-05-26 via
  `solitaire-ai-log-010e01-1779766254801.json` (139 rows, 123 success,
  16 errors = 11.5% error rate) + `solitaire-win-010e01-1779766255660.json`
  (170-move win-record, `gameWon: true`).

**Strategic-path comparison.** Both runs played the **first 25 moves
identically** (same legal-move choices in the same order), then diverged
on tableau-organisation philosophy. LCS over the remaining sequence is
~46%. Trajectory milestones:

| foundation cards | baseline turn | comparison turn |
|---:|---:|---:|
| 5  |  51 |  60 |
| 15 | 113 | 129 |
| 30 | 140 | 147 |
| 40 | 161 | **157** |
| 50 | 172 | **168** |
| 51 | 173 | **169** |

The baseline arm pumped foundations earlier (greedy strategy: 47 draws,
28 tableau-to-tableau moves, 2 recycle_stocks); the comparison arm built
tableau sequences longer before opening foundations (patient strategy:
37 draws, 52 t2t, 0 recycles). Net: comparison wins in 4 fewer turns
despite being behind through fc=35.

**Hardest-turn signature differs.** Baseline's top-5 slowest decisions
(200-208s, 9K tokens) all came at turns 75-119 with 8-9 legal moves,
endgame foundation sequencing with face-up board. Comparison's top-5
slowest (227-238s, 10K tokens) all came at turns 41-125 with 23-28 legal
moves, mid-game branching with hidden cards still on board. Same teacher,
same deck, different *kind* of hard turn depending on which strategic
path it commits to.

**Behavioural carryover.** Confidence saturated identically (mean 0.93
vs 0.94, never below 0.8) in both arms; the overconfidence pathology is
independent of prompt format. Reasoning length per call comparable
(~470-490 char boardAnalysis, ~440-480 char reasoning).

- v1.1 comparison arm: session `…4de0cfe2cc8c`, build `20a825f`,
  hybrid-v1 (templateHash `8971cad0…b524b902ff51eb` = v1.1, the build
  that dropped confidence + alternative_move_index and added the
  resign output), ingested 2026-05-27 via
  `solitaire-ai-log-e2cc8c-1779860448889.json` (216 rows, 166 success,
  50 errors = 23.1% error rate) +
  `solitaire-win-e2cc8c-1779860450489.json` (233-move win-record,
  `gameWon: true`, `completionProgress: 100`). **Result: v1.1 wins
  this deck too**, taking 233 moves vs the v1.0 baseline's 170 moves
  (37% more). First confirmed v1.1 win in the corpus, and first
  evidence the v1.1 prompt is not categorically broken; it can win on
  a winnable deck. Per-turn cost is slightly cheaper than v1.0 (1966
  vs 2140 prompt-tok mean, 3544 vs 3635 thought-tok mean, 5800 vs
  6086 total-tok mean), but total wallclock is longer (5.13 h vs
  3.77 h) because the run needed more turns. The v1.1 adaptive
  thinking pattern shows up here too: light through ti 0-29 (1469
  thought tokens), heavier through ti 30-119 (3659-4659), spike at ti
  150-179 (5942) before easing in the finisher (2238 at ti 220-259).
  Read alongside the v1.1 STALL on seed 2967897202 (session c99da9):
  same prompt template, same model, opposite outcomes; the win path
  exists for v1.1, the deck-specific structural lock on seed
  2967897202 is what kept c99da9 stuck. **Conclusions:** (1) v1.1 is
  outcome-capable on this deck, (2) the move-count regression (170
  to 233) suggests v1.1 makes more shuffling moves to reach the same
  win, possibly because removing the calibration paragraph reduced
  decision-commitment pressure, (3) the per-turn cost reduction
  trades against move-count growth; net wallclock is worse.

### Seed `2967897202` (draw-3): single-version baseline, awaiting hybrid-v1 re-run

One teacher run on file:

- Session `…688f5a044461`, build `7f01833`, json-format prompt
  (`promptLayoutVersion: None`,
  `promptTemplateHash: e2923795…d397f045e7e6c2b91b2`), ingested via
  `solitaire-ai-log-044461-1779533681032.json` paired with
  `solitaire-win-044461-1779533686224.json` (194-move win-record,
  `gameWon: true`, `completionProgress: 100`).
- ai-log capture is complete: 362 rows covering turnIndex 0..193 with
  ~30 small interleaved gaps (retry-cosmetic, not missing decisions).
  168 success / 194 errors = 53.6% error rate, consistent with the
  pre-`de7dc06` reliability band; error mix `unavailable: 184,
  timeout: 7, invalid_key: 3`.
- Difficulty 3 (draw count 3). pyksolve solves the ground-truth initial
  state in 8 ms draw-1 and 50 ms draw-3 (`SolvedMayNotBeMinimal`).
- **Why register it now**: a SECOND cross-version benchmark seed. With
  only `3263196305` on file, any "did hybrid-v1 help" claim has n=1.
  Re-running `2967897202` under hybrid-v1 (build `de7dc06` or later)
  gives n=2 same-seed wins per prompt version, enough to start
  separating prompt effect from per-seed variance.
- Planned next step: harvest team re-runs seed `2967897202` under the
  current hybrid-v1 build; ingestion will mirror the 3263196305 pair.
- Replay URL: `solitaire.chayuto.com/?seed=2967897202`.

Same-seed validation experiments:

- Seed `2967897202`, comparison arm 1 (prompt v1.0): session
  `…9ec0a5db1804`, build `de7dc06`, `promptLayoutVersion: hybrid-v1`,
  `promptTemplateHash: 0462323c…d0cdb9c`, ingested 2026-05-27 via
  `solitaire-ai-log-db1804-1779829241710.json` (317 rows, 119 success
  / 198 errors). **Result: hybrid-v1.0 did NOT win this deck.** Final
  state `moveCount: 210`, `finalProgress: 29%`,
  `outcome: stalled_auto_terminated` at fc=15 / fd=5. Opening was
  byte-identical to the baseline 044461 for the first 15 moves
  (`AS-Sf, 2S-Sf, 9S col4-col1, AC-Cf, DRW, KH-col2, QS col3-col2,
  DRW, 8S-col5, 7H col6-col5, 2C-Cf, DRW, 8D-col1, DRW, DRW`), then
  diverged at position 15 when the baseline drew and db1804 committed
  `JS waste -> col3`. After divergence the 37% move-match was mostly
  DRW noise; the foundation push sequences share an 8-card prefix
  (`AS 2S AC 2C 3S AH 2H 4S`) and diverge after. Late game collapsed
  into a `9H + 5 more col 2 ↔ col 6` 6-card-unit oscillation (≥7
  swap-pairs in the final 20 moves). Reasoning at ti=209 was
  self-aware: *"The board is currently in a deadlock regarding the
  reveal of Column 7. The only way to break this is to obtain the 8C
  or 9D from the stock... drawing from the stock is the only
  productive action."* **Conclusions:** (1) the same model + prompt
  template can produce opposite outcomes on identical state under
  temperature 0.3; the divergence is at a single close call;
  (2) hybrid-v1.0 did not address the post-recycle "draw to discover"
  pathology; (3) the canonical 6S/9H oscillation pattern is a
  recurring failure mode at fc=15 on this specific deck.

- Seed `2967897202`, comparison arm 2 (prompt v1.1): session
  `…a0c137c99da9`, build `20a825f`, `promptLayoutVersion: hybrid-v1`
  (string unchanged, see versioning gap note),
  `promptTemplateHash: 8971cad0…b524b902ff51eb`, ingested 2026-05-27
  via `solitaire-ai-log-c99da9-1779851673647.json` (190 rows, 113
  success / 77 errors; 5.94 h wallclock). **Result: hybrid-v1.1 also
  did NOT win this deck.** Final state `moveCount: 197`,
  `finalProgress: 29%`, `outcome: stalled_auto_terminated` at fc=15
  / fd=5 — the EXACT same wall as db1804 reached, by a different
  intermediate path. Opening 12 moves byte-identical to BOTH 044461
  and db1804; diverged at position 12 (chose `DRW` where 044461 and
  db1804 both chose `8D waste -> col1`). Foundation push prefix
  matches 044461's first 7 (`AS 2S AC 2C 3S AH 2H`); reached fc=11
  earlier than db1804 via KS-to-empty-col4 followed by 6S+3 unit
  moves. Late-game collapsed into the canonical `6S + 2 more col 2
  ↔ col 7` oscillation (10× in the last 20 moves) — same structural
  pattern as db1804's 9H oscillation, just a different cycling unit.
  Final-turn reasoning correctly identifies the lock: *"The available
  black 8s are 8S (currently in column 2, but will be buried under
  the 6S-5H-4C run) and 8C (not yet seen, likely in the stock)."*
  The `SEEN IN WASTE THIS CYCLE` list at the final turn was `KD TS
  6D QD 6H` (5 cards = current stock); 8C is NOT in the stock,
  meaning 8C is either still in the waste under the JC top or face-
  down in col7's hidden 5. **The resign output (shipped in v1.1)
  was NEVER fired** despite 53+ turns of obvious oscillation; the
  conservative trigger language ("drawing has been exhausted, you
  would not bet on any of the available moves") never matched
  because stock + recycle remained available. **Conclusions:** (1)
  v1.1 prompt changes (drop confidence + alternative_move_index,
  add resign output) did not address the post-recycle discovery
  pathology; (2) three runs on this seed converge on the same fc=15
  / fd=5 / col7-locked wall via three different paths, indicating
  the wall is reachable from many openings and the model lacks the
  information to recognise it as unreachable from current state;
  (3) the resign trigger needs concrete failure-mode predicates
  (e.g. "no foundation progress AND no hidden-card reveal in N
  turns") to fire on this pathology, or — more in line with our
  design principle — we accept that resign should not fire on a
  winnable deck and the fix lies elsewhere (information gaps, not
  decision rules); (4) this seed is now the canonical "WON via one
  path, STALLED via two different paths" reference, perfect for
  future v1.2 (DRAW TIMELINE) and v1.3+ A/B testing.

### Known data-quality caveat: session `…0ce0b2ce0fb4` ai-log truncation

The third on-file win (`solitaire-win-1779050713349.json`, 284 moves,
`gameWon: true`) is **not usable for cross-version benchmarking** because:

- No `seed` and no `appCommit` in the export (predates seed-and-commit
  logging); the deck is not replayable in-browser and cannot be solved
  by `pyksolve` against ground truth.
- ai-log capture is incomplete: `solitaire-ai-log-1779050738885.json`
  covers only turnIndex 140..283 (200 rows, 103 distinct turns). The
  first 140 turns have no prompt/decision record. The harvester's
  in-memory log buffer at that build likely had a cap, and the user
  exported only after it had already rotated out the opening. The
  win-record's `moveHistory` retains `aiReasoning` text for ~92 of the
  first 140 moves but no prompts and no API-side telemetry.
- The other contemporaneous export
  (`solitaire-ai-log-1779050730424.json`, 122 rows) is for a different
  session (`019e3584-bc46-…ca5f7e`), not this one. Searched
  `raw/` and `raw/archive/`; no other file covers turnIndex 0..139 of
  this session.
- Treat as a historical artefact: its win still counts toward the
  teacher's lifetime win count, but it cannot anchor a same-seed
  comparison or yield clean prompt-version trajectory data.

## Same-seed baseline pair

Seed `4153653383` was harvested twice on build `ec38c03`, once with
`seeHiddenCards` on (perfect information, session `…3cfcbb7381e0`) and once
with it off (imperfect information, session `…78e0b5481557`). Both ended in
the same total deadlock (foundations stuck at 2, 18 face-down). Both files
are kept in `raw/` as a perfect-vs-imperfect comparison baseline; the stall
filter prevents them from polluting the local training set.

## Archiving superseded exports

The pipeline globs `raw/*.json` non-recursively, so `raw/archive/` is ignored.
Move a raw file there only when it adds nothing to the store:

- An empty export (`count: 0`, no interactions).
- A true duplicate: another active file in `raw/` carries the exact same set of
  interaction ids.

A file is not superseded just because another file shares its ids. Check
coverage against the active `raw/` set only. If two files are mutually
duplicate, archiving both loses their interactions -- keep one copy in `raw/`.

Current archive contents:

- `solitaire-ai-log-1779050756211.json` -- same interaction set as the active
  `solitaire-ai-log-1779050730424.json` (122 interactions, re-exported twice).
- `solitaire-ai-log-eadb0a-1779058770667.json` -- empty export (`count: 0`,
  build `afa66cb`).
- `solitaire-ai-log-73fd85-1779308972435.json` -- session `…22c73fd85`,
  160 interactions, strict subset of the active
  `solitaire-ai-log-73fd85-1779310300860.json` (167 interactions = 160 known
  + 7 new continuation turns).
- `solitaire-ai-log-1279a3-1779292309912.json` -- session `…71279a3`, 39
  interactions, strict subset of the active
  `solitaire-ai-log-1279a3-1779329889383.json` (80 interactions, includes
  the original 39 plus 41 more turns of the same session).
- `solitaire-ai-log-1279a3-1779326207510.json` -- session `…71279a3`, 80
  interactions, exact duplicate of the active
  `solitaire-ai-log-1279a3-1779329889383.json` (same id set, re-exported).

## Operating notes

- Incremental by default. Files already recorded in the manifest, matched by
  sha256, are skipped. Use `--rebuild` to reprocess everything.
- Raw exports are gitignored because they are large and reproducible from the
  collection harness. The derived store and datasets stay in git.
- Reference audit (2026-05-29): cross-checked every session id and raw
  filename cited in this file against the store and disk. No phantom games,
  every cited session resolves to real data in the store. Two stale filename
  citations remain where the named export is no longer on disk but the
  session's data is intact under other filenames: the 22-row
  `solitaire-ai-log-49b05f-1779919722784.json` (a subset of the 481-row
  `…1780055383974.json`), and the seedless
  `solitaire-ai-log-1779376068820.json` baseline-arm name for session
  `…1abf260154e1` (superseded on disk by `solitaire-ai-log-0154e1-1779380748971.json`).
  Cause: gid-renaming of older exports left the old/seedless names behind in
  prose. Note also that a couple of entries write a 16-char session tail (e.g.
  `…aa24ed222c73fd85`) instead of the usual last-12; harmless, but it breaks
  naive suffix matching.
- **`check_winnability.py` defect, found 2026-06-03 (Monte-Carlo path
  unreliable; `pyksolve 10/10` verdicts in this file are suspect).** While
  adjudicating the `#30e5e5` resign, the script reported its board "40/40
  winnable, failure is behavioural not structural," but the board is provably
  unwinnable: exhaustive reachability over the repo engine
  (`solitaire_analytics.engine.generate_moves`/`apply_move`) finds only 36
  reachable states, no win, and the foundation count cannot pass 25/52 (col5's
  4C is permanently immovable, both red 5s gone, so 3C and 5H are buried
  forever). Root cause, confirmed by direct measurement: as it runs today
  `_solve_one_pyksolve` calls `sol.load_pysol(...)` then `sol.reset_game()`, and
  the board pyksolve actually solves is not the input board (the loaded talon
  never matches the input, even at turn 0; `reset_game()` after `load_pysol`
  discards the position). Compounding bugs: `gamestate_to_pysol` omits the
  foundation cards (pyksolve does not infer them, loads `H-0 C-0 D-0 S-0`), and
  `known_cards_from_board` counts only `discardTop` as a known waste card while
  ignoring the rest of the known stock/waste (`seenDrawPileCards`), so the
  face-down sampling pool is polluted and genuinely-buried cards can be sampled
  into the stock. Net effect: the Monte-Carlo path returns "winnable" largely
  independent of the input, so any `pyksolve N/N consistent worlds solvable ->
  failure is behavioural, not structural` verdict in this file rests on it and
  needs re-checking against the repo engine. Affected (Monte-Carlo over sampled
  face-down worlds on mid-game boards): the "behavioural-doom-loop on a winnable
  board" entries and the `#4a9fe1` "false resignation on a likely-winnable
  board" call (the solver was used there to overrule the model's own correct-
  looking lock diagnosis). NOT yet verified either way: the "ground-truth
  initial state" validations of full 52-card decks from win-records
  (same-seed/cross-version sections), which may use a different code path
  (`load_solitaire` / full deck) and should be checked separately before being
  trusted or discarded. Authoritative substitute until fixed: exhaustive
  reachability or the repo `ParallelSolver` on the actual `GameState` (handles
  non-empty foundations), as used for `#30e5e5`.
- **`check_winnability.py` FIXED 2026-06-03.** New sound backend
  `scripts/winnability_solver.py`: best-first search over the repo engine
  (`generate_moves`/`apply_move`) with safe-autoplay, a transposition table,
  and stock recycle modelled (the engine omits recycle; the session layer owns
  it). Verdicts are sound by construction: SOLVED is a constructive win,
  UNSOLVABLE means the full reachable space was exhausted under the node cap,
  UNKNOWN means the cap was hit (never read as winnable or dead). `--solver
  engine` is now the default; `pyksolve`/`beam` are retained but deprecated
  (pyksolve's `load_pysol` is non-functional in 0.0.15, solving a default deck
  regardless of input; `load_solitaire` round-trips but only for turn-0 full
  decks). The sampling-pool bug is also fixed (`seenDrawPileCards` now counted
  as known). Validated: known-won deals `#50aff7`/`#bf6d85` SOLVED (315 / 3669
  states), the `#30e5e5` resign board UNSOLVABLE (36 states), draw-1 confirmed
  from the won-deal move histories. **Re-adjudication of the ten flagged
  entries (engine solver, sampled worlds):** four "winnable board" claims are
  WRONG, the boards are structurally dead in 40/40 worlds (small surface-level
  locks): `#8a5d12` (fc22/fd3), `#4a9fe1` (fc14/fd8, the "false resignation" ->
  the model correctly judged a dead board), `#b2d946` (fc10/fd18), `#f75866`
  (fc3/fd10). Confirmed winnable: `#3fd319` (6/6). Winnable in most sampled
  worlds (claim supported): `#783780` (4/6), `#ffec5a` (5/6). Genuinely mixed:
  `#564fc9` (2 win / 1 dead / 3 cap). Inconclusive at the fast cap: `#cbced2`
  (fc7/fd13, all hit cap; raise the node cap to settle). `#a11e74` actually
  won, so winnability was never in question there. Each flipped entry carries a
  dated correction note inline.
