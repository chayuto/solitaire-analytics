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

- Session `…8b03bd502768`, seed `821908579`, model `gemma-4-31b-it`, app
  build `71130ac`. Export `solitaire-ai-log-502768-1779331813666.json`
  (45 rows). Final stored state: 45 interactions, `moveCount: 31`,
  `finalProgress: 4%`. New session at time of writing — early game,
  insufficient data to verdict. Listed here for completeness.

## Same-seed validation experiments

When the harvest team re-runs a known-failing seed under a different
build/prompt, the original session ID becomes the locked baseline and
the new session is the comparison arm.

- Seed `3689552861` — baseline session `…d46eb2645d03` (build `ce6afe1`,
  documented above; 75-turn doom-loop on 5C/4D). New comparison arm:
  session `…29a7f5`, **in progress** under a new prompt (build TBD).
  Not yet ingested. Expected to be the first end-to-end signal of
  whether the prompt-side changes reduce the doom-loop pathology on a
  deck where the baseline pathology is known and structurally
  documented.

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
