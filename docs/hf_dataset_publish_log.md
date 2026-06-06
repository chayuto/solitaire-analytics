# Hugging Face Dataset Publishing Log

Permanent, version-controlled record of the public dataset
`chayuto/klondike-llm-decisions` and its push history. This file is the durable
successor to the `hf-dataset-published` working memory: update the "Push
history" table on every significant corpus bump or card change so the
public-facing record stays current.

- Dataset: https://huggingface.co/datasets/chayuto/klondike-llm-decisions
- License: CC-BY-4.0
- Companion model adapter: `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`

## What it is

Per-decision traces from LLMs acting as advisors in Klondike Solitaire,
collected to support distillation research and the study of LLM failure modes
in sequential decision tasks. Every row records one advisor call against a
reproducible game state.

## Current configs (as of the 2026-06-06 Parquet migration)

Stored as zstd Parquet shards (see "Storage format" below). Counts:

| Config | Rows | What it is |
|---|---:|---|
| `client_v1_full_corpus_raw` (default) | 15213 | every success decision including failure modes, full interaction record; multi-model (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemini-3.1-flash-lite`; the current split is auto-emitted in the card) |
| `client_v1_teacher_clean_raw` | 6636 | full interaction, current-schema, non-stalled, single teacher (`gemma-4-31b-it`) only |
| `client_v1_teacher_clean_lean` | 6636 | per-decision flat schema, single teacher only, easy analytics |
| `client_v1_26b_raw` | 1243 | comparison cohort: the `gemma-4-26b-a4b-it` MoE alone, current-schema, stalled decisions KEPT (754 of 1243), 1 winning session (`#3e91a0`) |
| `client_v1_26b_lean` | 1243 | same cohort, per-decision flat schema, for analytics against the teacher on matched states |
| `solitaire_advisor_decisions` | 15213 | back-compat alias config pointing at the same shards as the default |

Notes:

- Both `teacher_clean_*` configs are filtered to `gemma-4-31b-it`, so the 26B
  doom-loop cohort never enters the training-recommended subsets.
- The 26B cohort now has 1 winning session (`#3e91a0`); it remains primarily a
  behavioural-contrast set for studying how the MoE fails on the same game
  states, not a training subset. It deliberately keeps the stalled/loop
  decisions that the `teacher_clean_*` configs drop.
- The full-corpus model-breakdown line is auto-emitted in the card by
  `render_dataset_card()`, so it updates itself on each push.

## How to refresh

Source of truth is the auto-generated `data/publish/` directory plus its card,
both written by `scripts/ingest_exports.py`. To refresh on Hugging Face:

1. Ingest any new exports and regenerate the publish shards:
   `.venv/bin/python scripts/ingest_exports.py`
   (appends new ids to each config's tail shard; idempotent if nothing is new)
2. Validate integrity before pushing:
   `.venv/bin/python scripts/validate_shards.py --ref local --src local`
3. Upload the folder (unchanged shards are skipped, so only new/changed shards go up):
   `HfApi().upload_folder("data/publish", "chayuto/klondike-llm-decisions", repo_type="dataset", ignore_patterns=["*.jsonl"])`
   (uses the cached token at `~/.cache/huggingface/token`)
4. Optionally verify the Hub round-trip:
   `.venv/bin/python scripts/validate_shards.py --ref local --src hub`

The README `load_dataset` examples, the per-row schema docs, and the config
list are emitted by `render_dataset_card()`. Edit there, not the published
README; the changes flow into the next push.

## Storage format (Parquet shards, since 2026-06-06)

Each config is a directory of immutable zstd Parquet shards
(`<config>/part-NNNNN.parquet`, up to `SHARD_ROWS` = 2000 rows each), written by
`publish_sharded()`. Properties:

- **Append-only by id.** A run adds only interaction ids not already in a shard,
  reopening the under-full tail shard and adding new shards. Frozen shards are
  never rewritten, so a push uploads single-digit MB, not the whole corpus, and
  stays flat as it grows. (Replaced the old monolithic JSONL, which re-uploaded
  ~974MB including a 340MB duplicate every push; rationale in
  `docs/reports/20260606_hf_upload_efficiency_research.md`.)
- **One schema per config.** Rows are typed with `pyarrow.json.read_json` (the
  loader `datasets` uses), which unions heterogeneous rows so older rows that
  lack newer fields do not drop columns. ISO-8601 strings are pinned to string
  (no timestamp auto-parse) so the published schema matches the source JSON.
  This supersedes the old `_normalise_schema` / `_front_load_rich` JSONL
  workarounds, which were removed.
- **Schema change** (a new field) needs a one-time `--rebuild` to re-pack all
  shards under the new schema; `publish_sharded()` prints a note when it sees a
  field absent from the frozen schema.
- **Integrity gate.** `scripts/validate_shards.py` compares any source (local
  shards or the Hub) against a reference by id and by every field. The migration
  was verified lossless against the prior JSONL on all five configs.

## Push history

Newest first. Counts are the live config row counts at each push.

| Date | Commit | Configs and counts | Notes |
|---|---|---|---|
| 2026-06-06 | [`1fdb2d4a`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/1fdb2d4a30382d16ea9ca785b1c0b74498f96087) | full 15213, clean-raw 6636, clean-lean 6636, 26b-raw 1243, 26b-lean 1243 (unchanged; format change) | Storage migration: monolithic JSONL to immutable zstd Parquet shards per config (`<config>/part-NNNNN.parquet`), append-only by interaction id. Deleted the old flat `*.jsonl` (incl. the 340MB duplicate alias). Payload 974MB to ~100MB; future pushes upload only changed tail shards. Verified lossless by id and by field against the prior JSONL on all five configs (`scripts/validate_shards.py`). Repo code commit `f804b5f`. |
| 2026-06-06 | [`2b496262`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/2b496262c71140fda62f343cc5104bb94d2709a8) | full 15213, clean-raw 6636, clean-lean 6636, 26b-raw 1243, 26b-lean 1243 | First 26B win (`#3e91a0`) folded in; corrected the card's 26B win count from "no wins" to 1 winning session. Added the 2026-06-04/05 v1.3/v1.4 harvest (incl. `#523f19` 700-turn cap-stall and the `#4c3a11` winnable-loop). Refresh of +2876 full, +845 on each clean tier, +386 on each 26b tier versus the prior push. |
| 2026-06-04 | [`077dfdb1`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/077dfdb1d4e4e0781f53a585d989251537487fa1) | full 12337, clean-raw 5791, clean-lean 5791, 26b-raw 857, 26b-lean 857 | Added the `client_v1_26b_*` comparison configs (`gemma-4-26b-a4b-it` MoE, no wins, 519 of 857 stalled kept). Corpus refresh of +5094 full and +2369 on each clean tier versus the prior push. |
| 2026-05-30 | `2f9351dc` | full 7243, clean-raw 3422, clean-lean 3422 | First `hybrid-v1.3` sessions and the first `gemma-4-26b-a4b-it` cohort folded into `full` (then full breakdown 31b ~6925, 26b 252, gemini 66). Three configs. |
| 2026-05-26 | (card update) | unchanged | Added the "Build on top of this" section to the card (Track C invitation). |
| 2026-05-25 | (initial) | three configs | Initial publish alongside the LoRA adapter `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`. |

## Build on top of this (Track C)

The data card carries a "Build on top of this" section inviting others to extend
the corpus. It points at:

- the `solitaire.chayuto.com/?seed=<seed>` replay URL pattern for reproducing
  any board,
- the source-repo skill at `.claude/skills/solitaire-analyst/` for running
  kill-or-continue analyses (winnability via `check_winnability.py`),
- the 20-state Phase 1.5 bench under
  `experiments/a4_phase1.5_2026_05_24/prompts/C0/` for like-for-like model
  comparisons,
- citation guidance and a contact note.

Re-push on each significant corpus bump so the public record stays current, and
add a row to the Push history table above.

## Related

- Corpus catalog and per-session verdicts: `data/DATASET_NOTES.md`
- Publishing flow and card generator: `scripts/ingest_exports.py`, `data/publish/`
- Distillation track context: the `gemma4-to-3n-pivot-mlxlm-blocker` and
  `26b-harvester-cohort-cataloging` working memories
