# Next session pickup pointer

**Last touched**: 2026-05-26
**Compute state at end**: idle, no active runs
**Branch**: `main`, one commit ahead of `origin/main` (carried from prior session), no new commits this session

## Read this first

This is the single read-on-resume document. After reading this, the order to read other docs (only if you need the depth) is:

1. `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_session_close_v2_gemma4_text_HELD.md` — what got done and why v2 HELD
2. `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md` — the pre-registered v3 experiment, one-shot recipe
3. `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v2_gemma4_distillation_lab_log.md` — full scientist lab log if you want the depth on v2

## State of the world in one paragraph

v1.1 LoRA on Gemma 3n shipped earlier this week at `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora` (canonical, unchanged). This session investigated whether the architecturally-correct Gemma 4 E2B base could replace it. A 15-line mlx-lm loader patch unblocked Gemma 4 entirely (the original "mlx-lm can't load Gemma 4" framing was wrong; it's a `sanitize()` oversight that drops cleanly). Untuned Gemma 4 E2B beats untuned Gemma 3n on the 20-state bench at half the memory. But the v2 LoRA trained on the same v1.1 recipe regressed: every checkpoint scored worse than v2-untuned, with all 5 regressions on `oscillation-*` states because the training corpus contains the 31B teacher's own doom-loop responses. v2 LoRA is HELD; v3 experiment (with a per-turn shuffle filter) is scaffolded and ready to fire when compute opens.

## What's queued for the next compute window

### Option A: v3 experiment (recommended, ~105 min compute)

Pre-registered hypothesis at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md`. One-shot execution:

```bash
cd /Users/chayut/repos/solitaire-analytics

# 1. Apply per-turn shuffle filter (already dry-run validated: drops 114 of 1635 rows, 7%)
.venv/bin/python gemma4_finetune/filter_shuffles.py \
    --in  data/dataset/training.jsonl \
    --out data/dataset/training_shuffle_filtered.jsonl

# 2. Game-level split
cd gemma4_finetune
venv/bin/python prepare_dataset.py \
    --log ../data/dataset/training_shuffle_filtered.jsonl \
    --out dataset_v3

# 3. Train in background (~85 min)
venv/bin/python train_v2.py --config lora_config_v3.yaml > /tmp/v3_train.log 2>&1 &

# 4. After training, run the sweep + score (auto-prints PROMOTE / PARTIAL / INCONCLUSIVE / HOLD)
bash sweep_v3_checkpoints.sh 2>&1 | tee /tmp/v3_sweep.log
```

The scorer reads against pre-committed thresholds in the experiment design doc. Whatever it prints is the answer; no judgement calls required at that step.

### Option B: ship v2 untuned as companion (no compute, ~30 min editing)

Rework `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/publish_hf_v2/README.md` to frame the artifact as untuned-only (apache-2.0, 3.35 GB peak, oscillation-resistant), drop the LoRA training section, point users at v1.1 for the high-foundation-recovery path. Push to `chayuto/gemma-4-e2b-it-solitaire-advisor-untuned` or similar slot.

Decision point: do this *before* v3 runs (since v3 might supersede it) or *after* v3 runs (to know whether v3 supersedes it)? Default to after, unless you want to publish something quickly.

### Option C: mlx-lm upstream PR (no compute, one afternoon)

Draft staged at `/Users/chayut/repos/solitaire-analytics/docs/internal/mlx_lm_gemma4_text_pr_draft.md`. Sequence:

1. `git clone https://github.com/ml-explore/mlx-lm /tmp/mlx-lm-upstream`
2. Check `mlx_lm/models/gemma4_text.py:610-640` for whether the fix is already in main
3. If not, fork on GitHub, branch, apply the patch, open the PR using the drafted text

Independent of any v2/v3 outcome.

## Uncommitted work

Working tree contains 25 new files and 4 modified files from this session. Nothing has been staged or committed. Per user preferences, no auto-commit was attempted. The list is in the HELD session-close doc.

If you want to commit before the next compute window, the natural unit is one commit per logical artifact, in this order:

1. `.gitignore` (just `venv_vlm/` added)
2. `gemma4_finetune/gemma4_text_patch.py` (the patch itself, smallest atomic unit)
3. `gemma4_finetune/baseline_n20_gemma4*_runner.py` + `score_n20_gemma4*.py` (the eval scripts)
4. `gemma4_finetune/train_v2.py` + `lora_config_v2*.yaml` + `posttune_n20_gemma4_text_runner.py` + `sweep_v2_checkpoints.sh` + `score_v2_learning_curve.py` (v2 training pipeline)
5. `gemma4_finetune/filter_shuffles.py` + `lora_config_v3.yaml` + `sweep_v3_checkpoints.sh` + `score_v3_learning_curve.py` (v3 scaffold)
6. `gemma4_finetune/publish_hf_v2/` (HF staging, needs rework before push)
7. `docs/reports/2026052{5,6}*.md` + `docs/internal/mlx_lm_gemma4_text_pr_draft.md` (all session docs)

The four modified files (`data/SUMMARY.md`, `data/index/manifest.jsonl`, `data/publish/README.md`, `scripts/ingest_exports.py`) are inherited from the prior session and represent the published HF dataset state; they are likely already in the commit ahead of `origin/main`. Verify with `git log -1 --stat` before committing.

## Confounds and caveats to remember when you resume

- All v2/v3 measurements are single-run, deterministic (temp 0.0), no multi-seed variance. N=20 bench, oscillation effects rest on 5-7 states. The Phase 1.5 noise floor is plus or minus 0.40 mean tier per single-run measurement. The observed deltas are above noise but not multiply-confirmed.
- The text-only Gemma 4 quant runs at max_tokens=2048 vs multimodal's 512 because thinking-mode chains are longer on the same prompts. Quantisation scheme difference (int4 vs 4bit) accounts for ~0.20 mean tier; the rest of the v2-vs-v2-multimodal gap is unknown.
- The teacher's effective inference temperature in production is not known; if it differs from our 0.0, that's a confound for "untuned student beats teacher" claims.
- v2 LoRA training collapsed inference from 15.7 s (thinking) to 5.0 s (direct JSON). Loss of the deliberation chain may itself be a contributor to the regression, separate from corpus content. v3 isolates corpus content alone; a deliberation-format experiment would be a later v4.

## Environment state

- `gemma4_finetune/venv/` (mlx-lm 0.31.3, the proven v1 + v2 stack) — keep
- `gemma4_finetune/venv_vlm/` (mlx-vlm 0.5.0, exploration scratch) — now gitignored, safe to delete to reclaim ~2 GB if you want
- `.venv/` (root project venv for ingest scripts and scoring) — keep
- Adapters at `gemma4_finetune/adapters_v2/` (~190 MB across 4 checkpoints + final) — gitignored, keep for any future inspection
- Smoke adapter at `gemma4_finetune/adapters_v2_smoke/` (~45 MB) — gitignored, deletable

## One-line summary if you only have time for one line

v2 HELD on doom-loop regression; v3 scaffold ready to fire (~105 min compute); v1.1 still canonical.
