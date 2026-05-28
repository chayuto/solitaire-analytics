# Session close: v2 Gemma 4 E2B distillation HELD; v3 scaffolded for next compute window

**Date**: 2026-05-26
**Status**: **HOLD** on shipping v2 LoRA. v3 experiment scaffold complete, not yet executed.
**Branch**: `main` (one local commit ahead of `origin/main` from prior session; nothing new committed this session)
**Compute used**: ~3 hours M5 time (mostly the 83-min v2 training run plus four 20-state bench evals)

## What this session delivered

1. **Diagnosed and patched the mlx-lm 0.31.3 Gemma 4 loader bug.** The `gemma4_text.py` model class correctly skips allocating `k_proj`/`v_proj`/`k_norm`/`v_norm` modules for KV-shared layers (layer_idx >= num_hidden_layers - num_kv_shared_layers; layers 15-34 for Gemma 4 E2B), but the published quants ship those weights anyway. `sanitize()` did not strip them, producing the `Received 140 parameters not in model` blocker that pushed v1 onto Gemma 3n. Six-line patch lives at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/gemma4_text_patch.py`; v1 toolchain works unchanged with it. PR draft for upstream submission staged at `/Users/chayut/repos/solitaire-analytics/docs/internal/mlx_lm_gemma4_text_pr_draft.md`.

2. **Selected `mlx-community/Gemma4-E2B-IT-Text-int4` as the v2 base.** Apache-2.0 licensed (vs Gemma TOS for the multimodal variant), audio Conformer and vision encoder weights physically absent, 60% smaller download. Three-way untuned comparison on the 20-state Phase 1.5 bench:

   | metric | v1 base (gemma-3n) | gemma-4 multimodal | **v2 base (gemma-4 text, patched)** |
   |---|---:|---:|---:|
   | mean tier | 2.10 | 2.75 | 2.55 |
   | gap vs teacher (3.42) | -1.32 | -0.67 | -0.87 |
   | foundation recovery (of 7) | 2 | 4 | 4 |
   | peak inference memory | 6.26 GB | 5.30 GB | **3.35 GB** |
   | illegal moves | 1/20 | 0/20 | **0/20** |

3. **v2 LoRA distillation: trained cleanly, regressed at every checkpoint.** 1000-iter QLoRA, same hyperparameters as v1.1's iter-750-winning recipe, training ran ~83 min wall, val loss 3.16 -> 0.36, peak 8.41 GB. But all four trained checkpoints (iter 250 / 500 / 750 / 1000) underperformed untuned baseline on mean tier:

   | config | json | illegal | agree | mean tier | gap | foundations |
   |---|---:|---:|---:|---:|---:|---:|
   | v2 untuned | 20/20 | 0/20 | 12/20 | **2.55** | -0.87 | 4/7 |
   | v2 iter 250 | 20/20 | 0/20 | 10/20 | 2.30 | -1.12 | 3/7 |
   | v2 iter 500 | 20/20 | 2/20 | 10/20 | 2.45 | -0.97 | 4/7 |
   | v2 iter 750 | 20/20 | 2/20 | 9/20 | 2.20 | -1.22 | 4/7 |
   | v2 iter 1000 | 20/20 | 2/20 | 8/20 | 2.05 | -1.37 | 3/7 |
   | v1.1 iter 750 (3n, shipped) | 20/20 | 2/20 | 11/20 | **3.15** | -0.27 | 6/7 |

4. **Localised the regression to a specific failure mode.** Per-state diff (v2-untuned vs v2-iter-1000): 12 unchanged, 1 improved, 2 shifted-but-still-wrong, and **5 regressed**. All 5 regressions are `oscillation-*` states. Response inspection on `oscillation-21cc5243e1d8` shows the trained model *confabulating* a non-existent move ("Move [1] (6S from waste to Col 3)") to justify picking a King swap at confidence 0.9, where untuned had correctly chosen `draw_card`. This is the exact doom-loop confabulation pattern the project memory `[[flag-unsolvable-boards-early]]` had already flagged in the 31B teacher. The corpus contains those responses; LoRA training transferred them.

5. **Surfaced a surprising scientific finding.** A 2.3B-active student locally outperforms a 31B teacher on a specific failure mode (oscillation recognition). Naive distillation of the teacher's average behaviour erodes that strength. v1.1 (Gemma 3n) didn't show this because Gemma 3n was *already weak* at oscillation recognition. Gemma 4 had a genuine strength to lose, and the corpus contained signal that actively eroded it.

6. **Scaffolded v3 experiment** to test whether per-turn corpus filtering fixes the regression. Pre-registered hypothesis, seven numeric predictions, and one-shot execution recipe staged at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md`. Filter dry-run already validates prediction P1 (114 of 1635 rows would be dropped, 7%, inside the predicted 5-15% band).

## Decision: HOLD

Pre-registered ship gate (best v2 mean tier > 3.15) not met. The v2 LoRA is *not* published. v1.1 remains the canonical recommended model.

Two follow-up tracks remain open:

- **v3 experiment (compute-required)**: re-train on the shuffle-filtered corpus to test the corpus-content hypothesis. ~105 min wall (filter + 85 min train + 20 min sweep). Pre-registered to print one of {PROMOTE, PARTIAL, INCONCLUSIVE, HOLD} when scoring completes.
- **v2 untuned as apache-2.0 companion (no compute)**: publish the untuned text-only Gemma 4 base alongside v1.1 with a use-case-based README. Different value proposition than v1.1 (lower foundation recovery but oscillation-resistant, half the memory, Apache-2.0). Staging at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/publish_hf_v2/` was built assuming a LoRA ship and needs rework if you take this path.

## Memory updates this session

- `gemma4-to-3n-pivot-mlxlm-blocker.md` updated: blocker is `mlx-lm`-specific and resolvable via a 6-line patch. `mlx-vlm` is a working alternative but the patched mlx-lm path dominates.
- New memory `v2-distillation-teacher-doom-loop.md` records the HOLD decision, the diagnosis, and the rule "do not re-train on the current corpus expecting different results; corpus shape is the bottleneck."

## Files produced this session (none committed)

**Documentation** (all under `/Users/chayut/repos/solitaire-analytics/docs/reports/`):
- `20260525_gemma4_e2b_v2_exploration_plan.md` (rung-by-rung plan + results)
- `20260526_v2_gemma4_distillation_lab_log.md` (scientist's lab log, full experiment record)
- `20260526_v3_experiment_design.md` (pre-registered v3 hypothesis and predictions)
- `20260526_session_close_v2_gemma4_text_HELD.md` (this doc)
- `20260526_NEXT_SESSION_PICKUP.md` (one-read resume pointer)
- `/Users/chayut/repos/solitaire-analytics/docs/internal/mlx_lm_gemma4_text_pr_draft.md` (upstream PR text)

**Code** (all under `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/`):
- `gemma4_text_patch.py` — the 15-line mlx-lm loader patch
- `train_v2.py` — wrapper that applies the patch before calling mlx_lm.lora.main()
- `lora_config_v2.yaml`, `lora_config_v2_smoke.yaml`, `lora_config_v3.yaml`
- `baseline_n20_gemma4_runner.py`, `baseline_n20_gemma4_text_runner.py`, `posttune_n20_gemma4_text_runner.py`
- `score_n20_gemma4.py`, `score_n20_gemma4_text.py`, `score_v2_learning_curve.py`, `score_v3_learning_curve.py`
- `filter_shuffles.py` — v3 per-turn corpus filter (dry-run confirmed)
- `sweep_v2_checkpoints.sh`, `sweep_v3_checkpoints.sh`
- `publish_hf_v2/` — HF staging tree (drafted before HOLD; needs rework before any push)

**Data and adapter artifacts** (will be gitignored, expected):
- `dataset_v2/` (1168 / 126 / 189 train/valid/test split)
- `adapters_v2/`, `adapters_v2_smoke/` (LoRA checkpoints)
- `baseline_n20_gemma4/`, `baseline_n20_gemma4_text/` (all eval responses + scored JSONs + learning_curve.json)
- `venv_vlm/` (now added to .gitignore)

**Source modifications**:
- `.gitignore` — added `venv_vlm/`

## Resource budget consumed

| phase | wall | peak MLX RAM |
|---|---:|---:|
| Rung 1 (load test, mlx-vlm multimodal) | ~5 min | 3.65 GB |
| Rung 2 (untuned bench, multimodal) | ~5 min | 5.30 GB |
| Rung 2b (untuned bench, patched text-only) | ~10 min | 3.35 GB |
| Rung 3 (30-iter smoke) | 2.5 min | 8.36 GB |
| Rung 4 (1000-iter real training) | 83 min | 8.41 GB |
| Checkpoint sweep (4 evals at max_tokens=2048) | ~22 min | 3.41 GB |
| **Total** | **~128 min** | **8.41 GB** |

Memory envelope (16 GB unified) held throughout with substantial headroom; v1.1's T5 had peaked at 11.49 GB, so v2 was actually more comfortable.

## Open items carried into next session

1. **v3 training run** (highest leverage, ~105 min compute).
2. **v2 untuned companion publish** (zero compute, README rewrite + HF push).
3. **mlx-lm upstream PR** (zero compute, just clone main + verify + submit).
4. **Track A harvester escalation note** (carried over from v1 session; still undrafted).
5. **TEMP HF token revoke** at https://huggingface.co/settings/tokens (carried over).
6. **Decide whether to commit this session's work**: 25 new files, 4 modified, all reflecting genuine artifacts of the v2 investigation. No automatic commit attempted; awaiting user decision.

## Pointer to the lab log and pickup doc

- Full scientist-style experiment record: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v2_gemma4_distillation_lab_log.md`
- Single-read resume pointer for the next session: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_NEXT_SESSION_PICKUP.md`
