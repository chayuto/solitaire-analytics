# Session close: v1.1 published, iter-750 promoted, harvest phase resumed

**Date**: 2026-05-25
**HF repo**: https://huggingface.co/chayuto/gemma-3n-e2b-it-solitaire-advisor-lora
**Local commits** (ahead of origin/main): two on `data-ingestion-pipeline`
**Status**: compute window closed; back to passive harvest mode

## What this session delivered

1. **Track B complete**, all six tiers (T0 inventory through T5 real training) cleared end-to-end on Apple M5 16 GB. Pipeline validated from `prepare_dataset.py` -> `mlx_lm.lora` -> `posttune_n20_runner.py` -> tier scoring with no manual intervention beyond config edits.

2. **v1 LoRA published to HuggingFace** as `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`. Public, Gemma TOS license. Includes the iter-750 adapter as canonical, all four training checkpoints under `checkpoints/`, full training scripts under `training/`, and a 20-state eval bench with scored pre/post results under `eval/`.

3. **Intermediate-checkpoint sweep flipped the shipping decision.** The originally-shipped weights were iter 1000 (the final checkpoint). After evaluating all four saved checkpoints against the 20-state bench, iter 750 turned out to be the strict winner on both mean tier (3.15 vs 2.75) and foundation recovery (6/7 vs 4/7). v1.1 was re-published with iter 750 as the canonical adapter. Iter 1000 remains available under `checkpoints/` for callers that prioritise format strictness over strategy.

4. **Dataset removed from git tracking.** Seven bulk JSONLs totalling ~370 MB are now in `.gitignore` and untracked. Files remain on disk; the metadata (SUMMARY.md, DATASET_NOTES.md, manifest, demos) stays tracked.

5. **Documentation pass to plain-text typography.** Em-dashes, unicode minus signs, emoji, and right-arrow glyphs stripped from all session-written docs. Memory updated with the writing-style preference so this doesn't repeat.

## Final eval numbers (iter-750 shipped weights)

| metric | untuned base | shipped (iter 750) | iter 1000 (back-pocket) | 31B teacher |
|---|---:|---:|---:|---:|
| JSON validity | 20 / 20 | 20 / 20 | 20 / 20 | n/a |
| Illegal moves chosen | 1 / 20 | 2 / 20 | 0 / 20 | n/a |
| Teacher-pick agreement | 11 / 20 | 11 / 20 | 11 / 20 | n/a |
| Mean tier (all 20) | 2.10 | **3.15** | 2.75 | 3.42 |
| Gap to teacher | -1.32 | **-0.27** | -0.67 | n/a |
| Foundation recovery (of 7) | 2 | **6** | 4 | n/a |

Iter 750 nearly closes the teacher gap (-0.27 vs untuned's -1.32, about 80% recovery) and recovers all but one of the seven teacher-foundation states. The single remaining miss (`midgame-4ab5735a4f20`) was also unrecovered at iter 1000, suggesting that state needs targeted data augmentation rather than more training time.

## Resource budget consumed

| phase | wall | peak MLX RAM | peak system RAM |
|---|---:|---:|---:|
| T1 proxy pipeline (0.5B model) | ~1 min | 1.18 GB | n/a |
| T2 E2B baseline (3 prompts) | ~6 min | 6.26 GB | ~9 GB |
| N=20 untuned baseline | 4 min 41 s | 6.26 GB | ~10 GB |
| T3 mini-smoke (10 iters) | 44 s | 8.88 GB | 13.92 GB |
| T4 full smoke (50 iters synth) | 3 min 5 s | 8.99 GB | 14.06 GB |
| T5 real training (1000 iters) | ~95 min | 11.49 GB | 15.25 GB |
| Intermediate checkpoint sweep (3 x N=20 evals) | ~15 min | 6.37 GB | n/a |
| **Total compute** | **~125 min** | **11.49 GB** | **15.25 GB** |

Peak memory stayed within the 16 GB envelope throughout. The only mild swap pressure was during T5; everything else fit in physical RAM with room.

## What's waiting in harvest phase

Watch `/Users/chayut/Downloads/` for new `solitaire-ai-log-*.json`. The triggers that would re-open a compute window:

| trigger | unlocks |
|---|---|
| >= 10 post-cutover sessions accumulated | A1 pre/post observational analysis (real signal on the cutover) |
| >= 1000 post-cutover training rows | T5 v2 on post-cutover-only slice (removes template-shift confound) |
| `mlx-lm > 0.31.3` ships Gemma 4 architecture support | Re-train on Gemma 4 E2B, publish at `chayuto/gemma-4-e2b-it-solitaire-advisor-lora` |
| Any harvester-side prompt change | Re-eval v1 adapter against the new template |

Current state: 2 post-cutover sessions (both stalled), 351 post-cutover training rows. Need ~8 more sessions plus more rows-per-session to clear the first two triggers.

## Open items not done this session

1. **Track A (harvester escalation note) never drafted.** Still the highest-leverage outstanding item: covers solvability pre-screening, the `PRIOR REASONING` anti-pattern (which is just renamed `reasoningTrail`), and the three open harvester P0s (no deck seed, untagged auto-plays, missing aiConfig). Roughly 30 minutes of writing when ready.

2. **TEMP HF token still active.** Token `local_machine_mac` (write scope, created 2026-05-24T10:54Z) is at `~/.cache/huggingface/token`. Revoke at https://huggingface.co/settings/tokens when fully done.

3. **`git push origin main`** not done. Two local commits ahead. Push when ready.

4. **Targeted foundation-move augmentation** for `midgame-4ab5735a4f20` (the one stubborn missed-foundation state). Lower priority than getting more clean post-cutover data.

## Reproducibility pointers

- Full model card and learning curve: https://huggingface.co/chayuto/gemma-3n-e2b-it-solitaire-advisor-lora
- Methodology doc: `training/METHODOLOGY.md` on the HF repo
- Untuned baseline report: `docs/reports/20260524_gemma3n_e2b_untuned_n20_baseline.md`
- First T5 run notes: `docs/reports/20260525_t5_first_distillation_run.md`
- Phase 1.5 prompt-format study (companion bench): `experiments/a4_phase1.5_2026_05_24/`
- Local training artifacts (gitignored): `gemma4_finetune/adapters_t5/`, `gemma4_finetune/baseline_n20/`
