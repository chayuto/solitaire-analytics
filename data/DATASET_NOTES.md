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
defined in `scripts/ingest_exports.py`; the proposal and rationale are in
`GAME_PROGRESS_METRIC_2026-05-19.md`, and the worked example that motivated
this filter is `DEAD_DEAL_ANALYSIS_2026-05-20.md`. Each decision row also
carries `foundationCards`, `faceDownTotal`, `progressScore`, and
`turnsSinceProgress` for downstream analysis.

## Known doom-loop sessions (kept; flagged by stall filter)

These sessions are ingested as-is. The stall filter (`STALL_TURNS=25`)
excludes their stalled decisions from `dataset/training.jsonl` while keeping
every interaction in the store and the publish set as a research record of
how the teacher fails.

- Session `…d46eb2645d03`, seed `3689552861`, model `gemma-4-31b-it`, app
  build `ce6afe1`. Exported across `solitaire-ai-log-645d03-1779227803496.json`
  (146 rows) and `solitaire-ai-log-645d03-1779270371464.json` (200 rows, 142
  new). Final outcome: incomplete, `finalProgress: 12%`, `moveCount: 137`.
  Foundation stuck at 6 cards / face-down stuck at 17 from turn 60 through
  turn 135 (75-turn plateau). Cause is **bad AI decisions, not a bad deck**:
  the model itself wrote that black 7s, red 7s, and a red King would unblock
  the board, and every named card except `KH` is in the seen-draw pile or
  face-up on the tableau (`7H` sits face-up on column 5). The model failed
  to play those cards when they reached the waste top, then oscillated 5C/4D
  between columns 4 and 6 indefinitely. Same class of failure as the open
  P1 in `HARVEST_TEAM_HANDOVER_2026-05-19.md` (no stall auto-terminator).

## Same-seed baseline pair

Seed `4153653383` was harvested twice on build `ec38c03`, once with
`seeHiddenCards` on (perfect information, session `…3cfcbb7381e0`) and once
with it off (imperfect information, session `…78e0b5481557`). Both ended in
the same total deadlock (foundations stuck at 2, 18 face-down). Both files
are kept in `raw/` as a perfect-vs-imperfect comparison baseline; the stall
filter prevents them from polluting the local training set. See
`DEAD_DEAL_ANALYSIS_2026-05-20.md` for the reachability proof and the
behavioural analysis.

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

## Operating notes

- Incremental by default. Files already recorded in the manifest, matched by
  sha256, are skipped. Use `--rebuild` to reprocess everything.
- Raw exports are gitignored because they are large and reproducible from the
  collection harness. The derived store and datasets stay in git.
