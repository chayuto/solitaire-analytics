# HF sharding: implementation plan

**Date:** 2026-06-06 | **Companion to:** `20260606_hf_upload_efficiency_research.md` | **Target:** `scripts/ingest_exports.py` (+ dataset card YAML, + a small state file).

Goal: replace the five monolithic publish JSONL writes (and the duplicate alias) with immutable Parquet+zstd shards keyed by interaction id, so a push uploads only the changed tail shard. Parquet over gzipped JSONL because three nested columns are sparse (`inferenceParams` 78%, `movesApplied` 88%, `config` 92%) and a JSONL shard that is all-empty for one infers the wrong Arrow type; Parquet shards self-describe their schema.

## 1. On-Hub layout (under `data/publish/`, which uploads to repo root)

```
client_v1_full_corpus_raw/part-00000.parquet, part-00001.parquet, ...
client_v1_teacher_clean_raw/part-00000.parquet, ...
client_v1_teacher_clean_lean/part-00000.parquet, ...
client_v1_26b_raw/part-00000.parquet, ...
client_v1_26b_lean/part-00000.parquet, ...
README.md
```

The old flat files (`client_v1_full_corpus_raw.jsonl`, the four others, and `solitaire_advisor_decisions.jsonl`) are deleted from the Hub in the migration commit.

## 2. Pipeline state (git-tracked, NOT uploaded)

`data/index/publish_shards.json`:

```json
{
  "shardRows": 2000,
  "configs": {
    "client_v1_full_corpus_raw": {
      "parts": [
        {"file": "client_v1_full_corpus_raw/part-00000.parquet", "rows": 2000, "ids": ["..."]},
        {"file": "client_v1_full_corpus_raw/part-00001.parquet", "rows": 213,  "ids": ["..."]}
      ]
    }
  }
}
```

Under `data/index/` (already git-tracked, already outside the `data/publish/` upload). The Arrow schema lives inside the Parquet files, not here. Per-part `ids` keep frozen parts stable so git deltas them well; a config's published-id set is the union across its parts.

## 3. Append algorithm (per config, per run)

Source row lists are unchanged: `publish_full_raw`, `publish_clean_raw`, `publish_clean_lean`, `publish_26b_raw`, `publish_26b_lean`, each already derived from the `(timestamp, id)`-ordered store. No `_normalise_schema` or `_front_load_rich` on this path; the declared Parquet schema handles missing keys (null) and typed empties.

```
def publish_sharded(config_name, rows, state, rebuild):
    import pyarrow as pa, pyarrow.parquet as pq
    cfg = state["configs"].setdefault(config_name, {"parts": []})
    cfg_dir = PUBLISH_DIR / config_name
    cap = state["shardRows"]

    if rebuild or not cfg["parts"]:
        for p in cfg_dir.glob("*.parquet"): p.unlink(missing_ok=True)   # full re-pack
        cfg["parts"] = []
        to_pack = list(rows)
        schema = pa.Table.from_pylist(to_pack).schema if to_pack else None  # infer across ALL rows
    else:
        published = {i for part in cfg["parts"] for i in part["ids"]}
        new = [r for r in rows if r.get("id") not in published]   # id-keyed, NOT timestamp
        if not new:
            return 0                                              # idempotent no-op
        schema = pq.read_schema(PUBLISH_DIR / cfg["parts"][0]["file"])
        buffer = []
        tail = cfg["parts"][-1]
        if tail["rows"] < cap:                                    # reopen under-full tail
            buffer = pq.read_table(PUBLISH_DIR / tail["file"]).to_pylist()
            (PUBLISH_DIR / tail["file"]).unlink()
            cfg["parts"].pop()
        to_pack = buffer + new

    cfg_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(to_pack), cap):
        batch = to_pack[start:start + cap]
        idx = len(cfg["parts"])
        rel = f"{config_name}/part-{idx:05d}.parquet"
        try:
            tbl = pa.Table.from_pylist(batch, schema=schema)     # schema None on first infer
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            raise SystemExit(f"{config_name}: row does not fit the frozen schema "
                             f"({e}); a field changed -- re-run with --rebuild")
        pq.write_table(tbl, PUBLISH_DIR / rel, compression="zstd")
        cfg["parts"].append({"file": rel, "rows": len(batch),
                             "ids": [r.get("id") for r in batch]})
        written += 1
    return written
```

Notes:
- Schema is inferred once across all rows on rebuild (pyarrow scans every row, so sparse nested types resolve correctly, which is why front-load-rich is unnecessary). On append it is read from `part-00000` so new shards match the frozen ones exactly.
- A new field on a later export makes `from_pylist(batch, schema=...)` raise; we stop with a clear "re-run with --rebuild" message rather than write a mismatched shard.
- The open tail keeps part count at about `ceil(rows / cap)` per config instead of one file per push.

## 4. Wire into `main()`

Load state early (to snapshot prior counts), replace the publish block (`:1116-1122`, the five `write_jsonl(PUBLISH_*)` calls and the `PUBLISH_LEGACY_ALIAS` write) with five `publish_sharded(...)` calls, then `save_shard_state`. `--rebuild` already exists (reprocess every raw file); extend it to also re-pack all shards. When `publish_shards.json` is absent (first run), every config rebuilds.

Update the `prior` snapshot (`:1062-1069`) and the end-of-run print to read per-config row totals from the shard state (sum of `parts[].rows`) instead of `count_lines` on the now-deleted flat files. Keep `_normalise_schema` / `_front_load_rich` defined (other callers may use them) but unused by the published path.

## 5. Dataset card YAML (`render_dataset_card`, `:689-701`)

```yaml
configs:
- config_name: client_v1_full_corpus_raw
  data_files: client_v1_full_corpus_raw/*.parquet
  default: true
- config_name: client_v1_teacher_clean_raw
  data_files: client_v1_teacher_clean_raw/*.parquet
- config_name: client_v1_teacher_clean_lean
  data_files: client_v1_teacher_clean_lean/*.parquet
- config_name: client_v1_26b_raw
  data_files: client_v1_26b_raw/*.parquet
- config_name: client_v1_26b_lean
  data_files: client_v1_26b_lean/*.parquet
- config_name: solitaire_advisor_decisions   # back-compat alias, zero extra bytes
  data_files: client_v1_full_corpus_raw/*.parquet
```

Add one body line noting the data is zstd Parquet shards read transparently by `datasets`.

## 6. One-time migration

```
# 1. fresh shards from the current corpus (store is already current; no raw reprocessing needed)
.venv/bin/python scripts/ingest_exports.py --rebuild

# 2. local load test BEFORE any push
.venv/bin/python -c "from datasets import load_dataset; \
  d = load_dataset('data/publish', 'client_v1_full_corpus_raw', split='train'); \
  print(len(d)); print(sorted(d[0].keys())[:6])"
# expect 15213 rows and a well-formed row (prompt + decision present)

# 3. idempotency: re-run plain, expect 0 new shards
.venv/bin/python scripts/ingest_exports.py

# 4. migration push: add shards AND delete old flat files in one atomic commit
#    api.upload_folder(folder_path='data/publish', repo_id='chayuto/klondike-llm-decisions',
#                      repo_type='dataset', delete_patterns=['*.jsonl'],
#                      commit_message='Migrate to zstd Parquet sharded layout')

# 5. verify from the Hub
.venv/bin/python -c "from datasets import load_dataset; \
  print(load_dataset('chayuto/klondike-llm-decisions', 'client_v1_full_corpus_raw', split='train'))"
```

`delete_patterns=['*.jsonl']` removes the five monoliths and the alias from the Hub in the same commit that adds the shards, so there is no broken window.

## 7. Test plan

- Counts: after `--rebuild`, sum of `parts[].rows` per config equals the prior flat row counts (full 15213, clean 6636/6636, 26b 1243/1243).
- Idempotency: a second plain run writes 0 parts.
- Incremental: on the next real drop, confirm only the tail part (and any new full part) changes.
- Load: `datasets.load_dataset` on the local dir and the Hub returns the right counts and a clean row, including rows where `inferenceParams` is absent.
- Frozen-shard stability: the append path never rewrites a full frozen part (only the under-full tail), so unchanged shards never re-upload; this is structural, not dependent on byte-reproducible Parquet.

## 8. Rollback

The flat-file generator stays in git history. If anything is wrong post-push, regenerate the flat files from the current store (short revert of the publish block) and `upload_folder` them back. The store, decisions, training, and manifest outputs are untouched by this change, and the raw exports (the source of truth) are unchanged, so no data is at risk.

## 9. Out of scope (noted, not done)

- `CommitScheduler` / automated background pushes.
- `hf-transfer` (Xet supersedes it).
- Columnar selective-read tooling for consumers (Parquet enables it; not wired up here).
