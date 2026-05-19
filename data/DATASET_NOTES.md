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
  current export schema only. This is the input to `gemma4_finetune`.
- Publishing set (`data/publish/`): every successful decision across all models
  and schema versions. Published to Hugging Face under CC-BY-4.0.

Selection criteria live in one config block at the top of
`scripts/ingest_exports.py` (`TEACHER_MODEL`, `SCHEMA_CONTRACT_FIELDS`).

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
