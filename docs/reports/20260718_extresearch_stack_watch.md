# External research 5 of 5: stack watch (2026-07-18)

Strand 5 of the sequential external research pass requested 2026-07-17.
Method: direct registry checks run in-session (gh CLI against GitHub, HF
model API via curl) on 2026-07-18, no web-search agent needed; every fact
below was read off the primary registry response. Companions: strands 1 to
4 (same date) and `20260718_external_strategy_stresstest.md`. Bears on: the
pinned gemma4_text sanitize patch, the QAT re-baseline candidate, and the
constrained-decoding build path.

## 1. mlx-lm: no release since 0.31.3, patch stays

- Latest GitHub release: v0.31.3, published 2026-04-22. Prior cadence:
  v0.31.2 on 2026-04-07, v0.31.0 on 2026-03-07. The release train has been
  quiet for ~3 months. PyPI agrees (0.31.3 latest, checked 2026-07-17).
- PR #1240 "Fix Gemma 4 sanitize() not stripping KV projections for shared
  layers": state MERGED, mergedAt 2026-05-04T22:26:19Z, i.e. AFTER the
  v0.31.3 release. The upstream fix therefore remains UNRELEASED.
- Action: `gemma4_finetune/gemma4_text_patch.py` stays, with its version
  guard (built-for 0.31.3) unchanged. The guard will warn on the next
  release; verify the release contains #1240, then delete the patch and its
  imports per the plan in `20260704_mlx_gemma4_ecosystem_update.md` (M3).
- Cross-reference from strand 3 (`20260718_extresearch_inference_layer.md`):
  mlx-lm PR #845 (native structured outputs via Outlines) was CLOSED
  UNMERGED 2026-02-08 with the maintainer declining the dependency, so no
  native constrained-decoding API is coming; the user-side Outlines
  logits-processor route is the sanctioned path.

## 2. Gemma 4 E2B QAT: official checkpoints live, one MLX conversion exists

- `google/gemma-4-E2B-it-qat-w4a16-ct`: created 2026-06-04, lastModified
  2026-07-15, 575,375 downloads. Base model chain shows it is quantized
  from `google/gemma-4-E2B-it-qat-q4_0-unquantized`, the variant Google
  publishes specifically for downstream conversion. This confirms the
  standing memory note (QAT E2B released ~2026-06-05, next-cycle
  re-baseline candidate).
- MLX conversion found: `mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit`
  (created 2026-06-12, lastModified 2026-07-14, 2,360 downloads). Caveats
  read off its tags: mixed-precision 4bit/8bit "OptiQ" scheme (not the
  plain int4 our LoRA stack uses today) and image-text-to-text, i.e. the
  FULL multimodal E2B, not a Text-only slice.
- No Text-only QAT MLX conversion exists in mlx-community: their full
  gemma-4-e2b listing (bf16, 4bit, 5bit, 6bit, 8bit, mxfp4, mxfp8, nvfp4,
  both -it and base) was all created 2026-04-02, before the QAT drop.
- Re-baseline paths, when that experiment is scheduled (next cycle, not
  blocking): (a) try the OptiQ-4bit full model (memory cost and
  mixed-precision compatibility with the LoRA trainer unverified), or (b)
  self-convert `gemma-4-E2B-it-qat-q4_0-unquantized` to a Text-only MLX
  int4 with mlx-lm convert, mirroring the current base's recipe. Either
  path still needs the sanitize patch until the next mlx-lm release.

## 3. Current pinned base: upstream unmoved

- `mlx-community/Gemma4-E2B-IT-Text-int4` repo HEAD sha is
  61d85e83c959ac93109dc5be7104c8de9942ef66, lastModified 2026-05-19, which
  IS the pinned revision (61d85e83) in use. No upstream drift since the
  pin; nothing to re-verify.

## 4. Verdict

Stack: STABLE, no action forced before the round-2 gate. Watch items, both
cheap: (1) next mlx-lm release (verify #1240 inclusion, then delete the
patch), (2) appearance of a Text-only QAT MLX conversion, which would make
the re-baseline experiment nearly free to set up.
