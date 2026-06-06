# HF upload efficiency: research and proposal

**Date:** 2026-06-06 | **Scope:** the `data/publish/` generation in `scripts/ingest_exports.py` and the manual `HfApi().upload_folder` step. No change to what is published, only how it is packed and uploaded.

## Problem

The dataset only grows (12337 full rows on 2026-06-04, 15213 on 2026-06-06, and climbing). The current pipeline re-materialises the whole corpus as monolithic JSONL on every push and uploads the lot. The 2026-06-06 push moved **974MB** to add ~1,300 new rows, and that number rises every time.

## Diagnosis (measured)

Current `data/publish/` payload, all re-written and re-uploaded each push:

| File | Size | Note |
|---|---:|---|
| `client_v1_full_corpus_raw.jsonl` | 340.0 MB | the corpus |
| `solitaire_advisor_decisions.jsonl` | 340.0 MB | byte-identical duplicate (sha256 `b077575a...`), an undeclared orphan config |
| `client_v1_teacher_clean_raw.jsonl` | 137.1 MB | |
| `client_v1_teacher_clean_lean.jsonl` | 82.7 MB | |
| `client_v1_26b_raw.jsonl` | 41.5 MB | |
| `client_v1_26b_lean.jsonl` | 32.7 MB | |
| **Total** | **974.0 MB** | |

Three root causes:

1. **A 340MB byte-identical duplicate.** `ingest_exports.py:1122` writes the full corpus a second time under the old filename. It is not referenced by any declared config in the card YAML, so it is an orphan, 35% of the payload for nothing.
2. **A richness re-sort that scrambles row order.** The canonical store is written in append-stable `(timestamp, id)` order (`ingest_exports.py:1072`), but `_front_load_rich` (`:1112-1114`, key `(-richness, timestamp)`) re-sorts the raw configs so schema-rich rows lead. That moves new rows into the middle of the file by richness band, shifting bytes throughout. It defeats any content-level dedup, and it exists only to satisfy Arrow first-batch type inference, which an explicit schema makes unnecessary.
3. **Uncompressed JSONL, rewritten in full** (`write_jsonl`, `:149-151`). Our rows are mostly the repeated prompt template, so they compress about 6.6x; shipping them raw and whole every push wastes both bandwidth and HF storage.

A wire check matters here: **`hf_xet` 1.5.0 is installed**, so chunk-level dedup is available, yet that push still reported "New Data Upload" at the full file sizes. The richness re-sort and full rewrite are why dedup found little to skip.

## Compression measured (full corpus, 340MB JSONL, 15213 rows)

| Format | Size | Ratio |
|---|---:|---|
| `jsonl.gz` (gzip-6) | 51.9 MB | 6.6x |
| Parquet + zstd | 52.9 MB | 6.4x |
| Parquet + snappy | 93.5 MB | 3.6x |

For our prompt-heavy rows, plain gzipped JSONL matches Parquet+zstd on size. The deciding factor between them is not compression but schema robustness across shards (next section).

## The shard-schema constraint (measured)

Sharding splits rows across files, and `datasets` infers Arrow types per shard from JSON. The corpus has **three sparse nested columns** that a 2000-row shard could contain all-empty:

| Nested column | Non-empty fraction |
|---|---:|
| `decision` | 100.0% |
| `config` | 92.1% |
| `movesApplied` | 87.8% |
| `inferenceParams` | 78.2% |

An all-empty shard infers `list<null>` / `struct<>` for one of these and then fails to concat with a shard where it is populated. This is exactly the failure `_front_load_rich` papers over in today's single file. With JSONL shards we would have to ship an explicit `features` block to pin the types. **Parquet shards carry their own schema**, so an all-empty `inferenceParams` shard still writes `struct<...>` and concat is safe with no features block and no front-loading.

## Options considered

- **A. Keep monolithic JSONL, just gzip it.** Rejected. A single growing `corpus.jsonl.gz` recompressed each push re-uploads in full, because gzip output changes completely on any input change. Compression only helps incrementally when each compressed file is frozen.
- **B. Gzipped-JSONL shards + explicit `features` in the card.** Workable, same compression, but the features block for 28 columns with nested structs is verbose and must be regenerated on every schema change.
- **C. Immutable Parquet (zstd) shards, keyed by interaction id, with an open tail.** Chosen. Same 6.4x compression, self-describing schema per shard (no features block, no front-load-rich), HF-native viewer support, pyarrow already installed. Old shards are frozen so they are never re-hashed or re-uploaded; new ids land in the open tail shard or a new one. Upload scales with new data, not total data.

## Decision

**Immutable Parquet+zstd shards, keyed by id, with an open tail.** Each config becomes a directory of `part-NNNNN.parquet` files written against one declared schema. New rows (ids not already published) go to the open tail shard until it reaches a row cap, then it freezes and a new tail begins. The card's `data_files` globs the parts, so new shards register automatically. The declared schema removes the need for both `_normalise_schema` and `_front_load_rich` on the published path.

Two disciplines this design enforces, both load-bearing:

1. **Freeze old shards.** Only the small open tail (and any newly added full shards) change per push. This is what makes compression help incrementally.
2. **Define "new" by interaction id, not by a timestamp cursor.** Re-exports can deliver new ids with old timestamps (a mid-game export recovering early turns, per the harvester re-export dedup behaviour). "New shard = rows whose id is not in any published shard" is the only correct rule; a time watermark would duplicate or drop rows.

An explicit `--rebuild` path re-materialises all shards from scratch for the rare cases that need it: a schema-superset change (a new field appears), or refreshing rows that got richer on re-export (the immutable model otherwise freezes the first-published copy, which is fine for decisions that are stable once logged).

## Payoff (estimated from the measured numbers)

| | Now | After |
|---|---:|---:|
| Full config on disk / Hub | 340 MB | ~53 MB |
| Total publish payload | 974 MB | ~100 MB (one-time), then frozen |
| Bytes changed on a typical push (~1,300 new rows) | ~974 MB | **single-digit MB** (the open tail shard) |
| Scaling | grows with corpus | flat, grows with new data only |

The first migration push uploads the re-sharded corpus once (about 100MB of zstd Parquet across all five configs, down from 974MB because the 340MB duplicate alias is gone and everything is compressed). Every push after that uploads only the changed tail shard plus any new full shards.

## Caveats

- **Schema drift across shards.** Between schema changes every shard shares the superset; when a new field appears, run `--rebuild` once to re-pack all shards. The pipeline detects a superset change and prints the instruction.
- **Frozen first-seen rows.** A row that gets richer on a later re-export keeps its first-published copy until a `--rebuild`. Acceptable for stable decision rows.
- **File proliferation.** The open-tail design caps part count at roughly `ceil(rows / shard_rows)` per config (single digits today), not one-file-per-push.
- **The orphan alias.** `solitaire_advisor_decisions.jsonl` is retired. It is not a declared config, so only a caller hardcoding that exact filename in `data_files=` would be affected; named-config loads are unchanged. It is re-added as a zero-cost config alias pointing at the full shards.

## Sources

- [HF dataset upload guide (format, incremental)](https://huggingface.co/docs/hub/datasets-upload-guide-llm)
- [huggingface_hub upload guide (upload_large_folder, CommitScheduler)](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Using Xet Storage](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
- [Xet chunk-level deduplication spec](https://huggingface.co/docs/xet/en/deduplication)
- [From Chunks to Blocks](https://huggingface.co/blog/from-chunks-to-blocks)
- [HF storage backends](https://huggingface.co/docs/hub/storage-backends)
