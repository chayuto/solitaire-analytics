# Hugging Face Dataset Publishing Log

Permanent, version-controlled record of the public dataset
`chayuto/klondike-llm-decisions` and its push history. This file is the durable
successor to the `hf-dataset-published` working memory: update the "Push
history" table on every significant corpus bump or card change so the
public-facing record stays current.

- Dataset: https://huggingface.co/datasets/chayuto/klondike-llm-decisions
- License: CC-BY-4.0
- Companion model adapters: [`chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`](https://huggingface.co/chayuto/gemma-3n-e2b-it-solitaire-advisor-lora) (v1 baseline) and [`chayuto/gemma-4-e2b-it-solitaire-advisor-lora`](https://huggingface.co/chayuto/gemma-4-e2b-it-solitaire-advisor-lora) (lead student, 2026-06-17); see [Companion models](#companion-models)

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
| 2026-06-13 | [`a83873ca`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/a83873ca7109b9d1337e86d66b2513e40e150a85) | full 20807, clean-raw 9134, clean-lean 9134, 26b-raw 1843, 26b-lean 1843 | +3455 full / +1526 each clean tier / +321 each 26b tier vs `908824dd`. Folds in the 2026-06-09..13 harvest across four ingests: four 31B `hybrid-v1.6` wins (`#6b7bf8`, `#2c5ac3` first on build `a410495`, `#d87d2e` first on build `ffc1cb4`, `#ec8ce7` the 6-recycle record), the SECOND-ever resign `#a29c9a` (correct, exact-proven dead), and the proven-dead / winnable-killed loss set from the 2026-06-13 operator kill batch (`#92762f` exact-dead incl. its re-export extension to move 422, `#989ea1`/`#61b7f2` exact-dead, `#6d7d80` killed-while-winnable, plus 26B `#783eb5`/`#a4b5fa` and gemini `#fd6e56`/`#248b67` in `full` only). Append-only; integrity gate `--ref local --src hub` OK on all five configs by id and field. |
| 2026-06-08 | [`908824dd`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/908824ddd63cd9343565d0885b584c66fdc03198) | full 17352, clean-raw 7608, clean-lean 7608, 26b-raw 1522, 26b-lean 1522 | +527 full / +146 each clean tier / +103 each 26b tier vs `b619768e`. Adds 6 v1.6-era sessions: the first gemini-3.1-flash-lite WIN `#b594d7` (clean, `full` only), the v1.6 31B dead board `#5094d7` (highest structural foundation lock 21/52, solver-proven, the no-resign action gap), the gemini doom-loop `#595e0c` (solver-proven STRUCTURALLY DEAD, a dead-flail not behavioural), and the v1.6 cross-model stall batch (`#00a2eb`/`#5c1ebc`/`#ba33bd`). Append-only: 6.7MB of new tail shards uploaded. |
| 2026-06-08 | [`b619768e`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/b619768eae8429149e3e6b03bc2eedf00cbffdaa) | full 16825, clean-raw 7462, clean-lean 7462, 26b-raw 1419, 26b-lean 1419 | +1612 full / +826 each clean tier / +176 each 26b tier vs `1fdb2d4a`. Adds the 2026-06-07 v1.5 doom-loops (incl. 26B `#7a4b10`/`#4aa9f1`), the three `hybrid-v1.6` wins (`#4c73b8`/`#fdc52f`/`#109f85`), the v1.5 dead-board flail `#c67807` (solver-proven STRUCTURALLY DEAD, the no-resign gap), and gemini-3.1-flash-lite `#eace21` (in `full` only; internal experiment, not a featured cohort, no dedicated config, operator-confirmed OK in full). Append-only: ~22.5MB of new tail shards uploaded. |
| 2026-06-06 | [`1fdb2d4a`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/1fdb2d4a30382d16ea9ca785b1c0b74498f96087) | full 15213, clean-raw 6636, clean-lean 6636, 26b-raw 1243, 26b-lean 1243 (unchanged; format change) | Storage migration: monolithic JSONL to immutable zstd Parquet shards per config (`<config>/part-NNNNN.parquet`), append-only by interaction id. Deleted the old flat `*.jsonl` (incl. the 340MB duplicate alias). Payload 974MB to ~100MB; future pushes upload only changed tail shards. Verified lossless by id and by field against the prior JSONL on all five configs (`scripts/validate_shards.py`). Repo code commit `f804b5f`. |
| 2026-06-06 | [`2b496262`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/2b496262c71140fda62f343cc5104bb94d2709a8) | full 15213, clean-raw 6636, clean-lean 6636, 26b-raw 1243, 26b-lean 1243 | First 26B win (`#3e91a0`) folded in; corrected the card's 26B win count from "no wins" to 1 winning session. Added the 2026-06-04/05 v1.3/v1.4 harvest (incl. `#523f19` 700-turn cap-stall and the `#4c3a11` winnable-loop). Refresh of +2876 full, +845 on each clean tier, +386 on each 26b tier versus the prior push. |
| 2026-06-04 | [`077dfdb1`](https://huggingface.co/datasets/chayuto/klondike-llm-decisions/commit/077dfdb1d4e4e0781f53a585d989251537487fa1) | full 12337, clean-raw 5791, clean-lean 5791, 26b-raw 857, 26b-lean 857 | Added the `client_v1_26b_*` comparison configs (`gemma-4-26b-a4b-it` MoE, no wins, 519 of 857 stalled kept). Corpus refresh of +5094 full and +2369 on each clean tier versus the prior push. |
| 2026-05-30 | `2f9351dc` | full 7243, clean-raw 3422, clean-lean 3422 | First `hybrid-v1.3` sessions and the first `gemma-4-26b-a4b-it` cohort folded into `full` (then full breakdown 31b ~6925, 26b 252, gemini 66). Three configs. |
| 2026-05-26 | (card update) | unchanged | Added the "Build on top of this" section to the card (Track C invitation). |
| 2026-05-25 | (initial) | three configs | Initial publish alongside the LoRA adapter `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`. |

## Companion models

Two LoRA advisor adapters are published alongside this dataset, each distilling
the 31B `gemma-4-31b-it` teacher into a locally-runnable small model. Both are
research artifacts; the model cards carry the full eval and caveats.

| Model | Base | Role | Eval headline |
|---|---|---|---|
| `gemma-3n-e2b-it-solitaire-advisor-lora` | gemma-3n E2B (4-bit DWQ) | v1 baseline | 20-state single-turn tier bench; gap to teacher -1.32 to -0.27 |
| `gemma-4-e2b-it-solitaire-advisor-lora` | Gemma 4 E2B int4 (mlx-community) | lead student (2026-06-17) | full-game: 5 wins vs base 1 on 13 held-out; +12.9 paired fc and 5 vs 1 on 12 fresh decks (generalizes) |

### gemma-4-e2b-it-solitaire-advisor-lora (published 2026-06-17)

The Gemma 4 E2B successor promised on the 3n card, now live at
[`chayuto/gemma-4-e2b-it-solitaire-advisor-lora`](https://huggingface.co/chayuto/gemma-4-e2b-it-solitaire-advisor-lora).
It is the project's first Gemma 4 E2B student and its current lead.

- Recipe (the "volume" arm). The full non-eval success pool: 6,859 decisions
  across 77 games (36% won), with the 13 eval seeds held out; game-level split
  5,663 train / 531 val / 665 test. LoRA rank 16, scale 2.0, dropout 0.05, keys
  `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj`, top 16 layers, lr
  2e-4, iters 1,000, batch 1, grad-checkpoint, max_seq 2,048. Base
  `mlx-community/Gemma4-E2B-IT-Text-int4`. Shipped weights = iter-1,000,
  selected over the 250/500 checkpoints (same 5 wins, roughly 3x cleaner JSON).
- Eval. Full-game, faithful v1.6 harness, cap 200, greedy, with exact engine
  replay plus a sound solver:
  - In-distribution (13 held-out winnable decks): 5 wins, meanFC 27.7, versus
    untuned base 1 win / 14.2.
  - Generalization (12 fresh, solver-winnable, zero corpus overlap): 5 wins,
    meanFC 28.5, +12.9 mean paired fc, better than base on 9 of 12, versus base
    1 / 15.6. Verdict: generalizes (gen-run report
    `docs/reports/20260614_generalization_run_plan.md` section 8).
- Card metadata (community convention): `base_model: mlx-community/Gemma4-E2B-IT-Text-int4`,
  `base_model_relation: adapter`, `library_name: mlx`, `license: gemma`,
  `datasets: chayuto/klondike-llm-decisions`.
- Known caveats (in-card): a JSON-discipline regression (fix is constrained
  decoding at inference), the ~31% teacher imitation ceiling, and the base needs
  a small local `sanitize()` patch (`gemma4_finetune/gemma4_text_patch.py`) to
  load on current `mlx-lm`.

### Publishing a model

Model staging dirs are gitignored (`gemma4_finetune/publish_hf*/`), derived from
`gemma4_finetune/adapters_*/`. The Gemma 4 folder
`gemma4_finetune/publish_hf_gemma4_volume/` holds `README.md` +
`adapter_config.json` + `adapters.safetensors` (iter-1,000) +
`checkpoints/{250,500,750}`. To publish or refresh:

    api = HfApi(token=...)
    api.create_repo("chayuto/<repo>", repo_type="model", exist_ok=True)
    api.upload_folder(folder_path="gemma4_finetune/publish_hf_gemma4_volume",
                      repo_id="chayuto/<repo>", repo_type="model")

For a card-only change, re-upload `README.md` alone with `api.upload_file(...)`.

### Model push history

| Date | Repo | What |
|---|---|---|
| 2026-06-17 | `gemma-4-e2b-it-solitaire-advisor-lora` | Initial publish: volume arm, iter-1,000. First Gemma 4 E2B student; beats base in-distribution (5 vs 1) and generalizes to fresh decks (+12.9 paired fc). The 3n card was updated to link it. |
| 2026-05-25 | `gemma-3n-e2b-it-solitaire-advisor-lora` | Initial publish (v1 baseline), alongside the dataset. |

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
