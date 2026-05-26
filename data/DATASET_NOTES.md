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

## Known doom-loop sessions (kept; flagged by stall filter)

These sessions are ingested as-is. The stall filter (`STALL_TURNS=25`)
excludes their stalled decisions from `dataset/training.jsonl` while keeping
every interaction in the store and the publish set as a research record of
how the teacher fails.

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
