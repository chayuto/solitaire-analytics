# MLX + Gemma 4 ecosystem update (2026-07-04)

Deep research pass over the MLX ecosystem and the Gemma 4 model family,
focused on roughly mid-May through 2026-07-04, run while the GRPO run-2 eval
occupied the GPU. Two web-research strands (~36 searches/fetches, all sources
retrieved this session) plus local verification against our pinned stack.
Companion docs: `20260704_next_move_deep_research.md` (method research this
updates in places) and `20260703_long_term_project_strategy.md`.

## 0. TL;DR

- **No MLX upgrade exists to take.** mlx 0.31.2 and mlx-lm 0.31.3 (both
  2026-04-22) are STILL the latest releases on GitHub and PyPI; June traffic
  on main was ~6 commits. Our pins are the current ecosystem state.
- **Our gemma4_text sanitize monkey-patch is fixed upstream, but only on
  unreleased main** (mlx-lm PR #1240, merged 2026-05-04). Keep the patch;
  pre-stage its retirement for the next release.
- **WWDC 2026 (June) made MLX a first-party citizen**: Metal 4, GPU Neural
  Accelerators (M5), distributed inference AND training over Thunderbolt
  RDMA, and a Foundation Models Python SDK + `fm serve` (macOS 27) exposing a
  free local Apple model with guided (schema-constrained) generation.
- **Gemma 4: no 4.1, no silent checkpoint refresh.** The family news is
  official QAT checkpoints for every size (2026-06-05) and a new 12B Unified
  (2026-06-03). The QAT E2B is a candidate next-cycle student re-baseline;
  do not swap mid-program.
- **Teacher risk low**: no Gemma 4 deprecation or serving change on the
  Gemini API (deprecations page checked as of 2026-07-02).
- **Verified locally**: our student base is `Gemma4-E2B-IT-Text-int4`
  revision `61d85e83`, pulled 2026-05-24, comfortably post-dating the April
  PLE-quantization bug window (which hit the multimodal e2b conversions).
- **One plan-relevant convergence**: mlx-lm-lora v2.1.0 ships a maintained
  GRPO/GSPO/Dr.GRPO/DAPO trainer family sized for our hardware; worth a
  compat smoke-test as a cross-check for our custom TR trainer, though none
  of its variants carries our one-sided frozen-reference trust region.

## 1. Local grounding (verified on this machine, 2026-07-04)

- Pins: `mlx 0.31.2`, `mlx-lm 0.31.3` (the newest releases that exist).
- `mlx_lm.generate_step` exposes `logits_processors` natively (verified by
  introspection 2026-07-04; the constrained-decoding plan needs no fork).
- Student base snapshot: `mlx-community/Gemma4-E2B-IT-Text-int4` revision
  `61d85e83c959ac93109dc5be7104c8de9942ef66`, cached 2026-05-24. Its chat
  template ships as a separate `chat_template.jinja` (the F13 packaging
  pattern); mlx-lm's `apply_chat_template` handles this correctly.
- `gemma4_finetune/gemma4_text_patch.py` (our sanitize workaround) is still
  required on 0.31.3 and still active in every trainer/harness entry point.

## 2. MLX ecosystem findings

**M1. Releases stalled at our pins.** mlx v0.31.2 and mlx-lm v0.31.3, both
2026-04-22, remain the latest on GitHub releases and PyPI; no release in over
ten weeks. Sources: github.com/ml-explore/mlx/releases,
github.com/ml-explore/mlx-lm/releases, pypi.org/project/mlx-lm/.
Applicability: "should we upgrade" is moot; stay pinned.

**M2. The 0.31.x line recap that matters to us.** 0.31.0 was YANKED from
PyPI (batched KV-cache cross-contamination); 0.31.2 added Gemma 4 support +
the batch-generator refactor with the logits-processor contract; 0.31.3
added the thread-local generation stream and partial Gemma 4 KV-shared
fixes. No breaking changes to generate/samplers/tuner across 0.30-0.31.
Applicability: never roll back below 0.31.2; our planned Outlines processor
rides an already-stable contract.

**M3. Sanitize bug fixed upstream, unreleased.** PR #1240 (merged
2026-05-04, commit df1d3f3): `sanitize()` now strips checkpoint-only
k_proj/v_proj/k_norm keys for KV-shared layers on quantized Gemma 4, the
exact failure our local patch works around. A regression-test PR (#1302)
is still open. Applicability: keep the patch while pinned; ACTION: diff our
patch against df1d3f3 for behavioral equivalence, and add a startup warning
if `mlx_lm.__version__` exceeds 0.31.3 while the patch is active, so it is
retired (not double-applied) the day the next release ships.

**M4. main-branch additions worth waiting for (not worth tracking main).**
`batch_generate` gained `return_logprobs` / `return_token_ids` (2026-06-12),
directly useful for batched GRPO logprob collection; a Gemma 4 MTP
speculative-decoding drafter is an open PR (#1276, since 2026-05-14) that
would speed our on-policy rollout harvesting if merged. Applicability: both
are reasons to take the NEXT release promptly, not to run unreleased main
mid-experiment.

**M5. WWDC 2026: MLX first-party.** Metal 4 support, GPU Neural
Accelerators (M5 hardware), and distributed inference + TRAINING across
Macs via RDMA over Thunderbolt (sessions 232/233). Applicability: neutral
on our single 16GB machine today; a supported two-Mac training topology
exists if the project ever needs more than one night per experiment.

**M6. Foundation Models framework, `fm` CLI, and a Python SDK (macOS 27).**
`fm serve` runs a local OpenAI-compatible server backed by the always-on
Apple model; the official Python SDK supports streaming, guided
(schema-constrained) generation, and tool calling. Third-party claims of an
`MLXLanguageModel` backend loading any mlx-community model into the
framework are UNVERIFIED. Applicability: a free local baseline advisor with
built-in structured output, and independent confirmation that
schema-constrained decoding is the platform-level answer to JSON
discipline; requires macOS 27, so a watch item, not a dependency.

**M7. mlx-lm-lora v2.1.0 (2026-04-22): 12 trainers including the GRPO
family** (GRPO, GSPO token/sequence, Dr.GRPO, DAPO dual-epsilon), QLoRA at
4/6/8-bit, file-based custom reward functions, memory guidance matching our
hardware class. Minimum mlx-lm version UNVERIFIED (same-day release as
0.31.3 suggests alignment). Applicability: the maintained alternative to
our custom trainer; none of its variants has our one-sided
frozen-reference TR, so it is a cross-check and a source of vetted
implementations (Dr.GRPO mean-only advantage, DAPO filtering) for the
round-2 upgrade list, not a drop-in replacement.

**M8. KV-cache quantization guidance.** June 2026 benchmarks recommend Q8
KV (group 64) as the quality-safe throughput win; available in our pinned
mlx-lm via kv-bits. Applicability: headroom lever if RFT rollouts pressure
16GB; not urgent at E2B context sizes.

## 3. Gemma 4 family findings

**G1. No Gemma 4.1; three family events total.** Launch 2026-03-31 (E2B,
E4B, 31B, 26B-A4B), Gemma 4 MTP drafter models for all sizes (2026-04-16),
Gemma 4 12B Unified (2026-06-03). No revised IT checkpoints, no
long-context bump (ai.google.dev/gemma/docs/releases). Applicability: our
held family facts are current; nothing to re-pull.

**G2. Official QAT checkpoints for the whole family (2026-06-05).** Q4_0
GGUF, W4A16 (group 32), unquantized-QAT, and a mobile wNa8o8 format for
E2B/E4B; E2B QAT ~1GB on disk, ~3GB to run, claimed ~72% memory cut at
near-BF16 quality. Applicability: the QAT E2B is a plausibly BETTER student
base at identical memory cost than our community post-training int4. It is
also a comparability hazard: every measured number in this project is on
the current base. Verdict: queue as a next-cycle re-baseline experiment,
gated on MLX generation-quality validation (G5's QAT metadata pitfall);
do NOT swap under the current program.

**G3. 12B Unified (2026-06-03).** Encoder-free multimodal, Apache 2.0, runs
in 16GB unified memory, benchmarks near the 26B. Applicability: not a
student candidate (advisor must stay small); a plausible LOCAL fallback
teacher/adjudicator if the 31B API teacher ever degrades. MLX support for
the unified arch lagged at first (gemma4_unified remap landed on mlx-lm
main 2026-06-05).

**G4. The April PLE-quantization bug is history but defines the safe
lineage.** Early 4-bit conversions of E2B quantized the Per-Layer
Embeddings and produced garbage (report 2026-04-04 on the multimodal
e2b-4bit repo); fixed via mlx-vlm PR #893 and PLE-safe re-quants; the
current recommended community lineage is OptiQ-4bit. Applicability:
verified locally that our Text-int4 snapshot (`61d85e83`, cached
2026-05-24) post-dates the fix window and is a different (text-only)
conversion from the affected repo; our measured win rates independently
confirm a sane base. Recorded here so any future re-pull targets a pinned,
post-fix revision.

**G5. QAT-on-MLX has a metadata pitfall.** QAT checkpoints mix 4-bit
attention with 8-bit MLP but declare one global quant type, so some loaders
mis-read and emit garbage; separately, quantized 26B-A4B produces garbage
specifically on low-GPU-core M4 machines (mlx issue #3393). Applicability:
gates the G2 re-baseline experiment (validate generation quality first);
the 26B item is irrelevant unless we ever run that cohort locally.

**G6. Fine-tuning landmines in OTHER stacks (recorded for portability).**
transformers issue #45242: `use_cache=False` corrupts Gemma 4 training
forwards, and gradient checkpointing forces it, so standard
transformers/unsloth LoRA recipes hit it (fix PR merge status unconfirmed).
Unsloth #5386: SFT merge sets the wrong eos token (`<eos>` instead of the
turn delimiter), so merged models never stop under vLLM. Applicability: we
train via MLX and are unaffected; these become critical only if the RFT
program ever ports to a transformers-based stack, and the eos check applies
if we ever export merged weights.

**G7. Native function-call tokens as an alternative JSON fix.** Gemma 4
ships native function-call special tokens; May 2026 benchmarks report lower
output-structure variance and 15-20% fewer output tokens vs prompt-based
JSON, with fine-tuned E4B edging Qwen3-4B on tool calling. A claim that E2B
emits valid JSON while in-string reasoning quality drops is UNVERIFIED.
Applicability: a format-level attack on our JSON-discipline regression, BUT
switching output format breaks compatibility with the entire trained corpus
and every baseline; constrained decoding (already planned, preserves the
trained format) stays first. Function-call format is a candidate only for a
future retrain generation.

**G8. Teacher serving: no lifecycle risk found.** gemma-4-31b-it and
gemma-4-26b-a4b-it live on the Gemini API since 2026-04-02; the deprecations
page (updated 2026-07-02) lists no Gemma 4 entries; June changelog items are
all Gemini-model features. Applicability: our observed ~75% provider-error
pattern remains an operational capacity issue, not a published lifecycle
event. Keep the deprecations page on the periodic check list.

**G9. Competitive picture.** The credible on-device tool-calling bases are
Qwen3-4B-Instruct-2507, Gemma 4 E4B, Phi-4-Mini; E2B holds the
lowest-latency niche. Nothing dethrones E2B for THIS project: same-family
teacher/student (shared tokenizer, shared failure modes) is load-bearing
for our distillation and eval machinery. If capability-per-GB ever reopens,
QAT-E4B and Qwen3-4B are the challengers.

## 4. What it means for this project

1. **The current plan is ecosystem-aligned.** The constrained-decoding
   build rides a stable, current API (M2); the RFT trainer's round-2
   upgrade list can borrow vetted implementations from mlx-lm-lora (M7);
   nothing released in the window invalidates any measured number.
2. **The patch has an expiry path** (M3): diff vs upstream df1d3f3, add the
   version-guard warning, delete at the next mlx-lm release.
3. **Two queued experiments enter the Later list** (strategy doc roadmap):
   QAT-E2B student re-baseline (G2, gated on G5 validation, next cycle
   only) and, further out, function-call-token output format (G7, requires
   a retrain generation).
4. **Teacher continuity is safe** (G8); harvest steady-state continues
   unchanged.
5. **New free baseline available when convenient** (M6): the Apple
   Foundation Models local model with guided generation, after a macOS 27
   upgrade; worth one bench run as an external reference point for the
   write-up.

## 5. Actions

| When | Action | Basis |
|---|---|---|
| Now (CPU, minutes) | Diff `gemma4_text_patch.py` vs upstream df1d3f3; add version-guard warning to the patch | M3 |
| Now (doc-only) | Pin the base revision `61d85e83` in writing (done, this doc) | G4 |
| Next window | Constrained decoding build proceeds as planned (unchanged) | M2 |
| Round 2 (if gate passes) | Cross-check advantage/filtering code against mlx-lm-lora's Dr.GRPO/DAPO before extending the custom trainer | M7 |
| Watch | mlx-lm next release (take promptly: patch retirement + return_logprobs + possibly Gemma 4 MTP drafter) | M3, M4 |
| Watch | Gemini API deprecations page, monthly | G8 |
| Later cycle | QAT-E2B re-baseline experiment (generation-quality gate first) | G2, G5 |
| Later cycle | Function-call-token output format (retrain generation only) | G7 |
| Optional | macOS 27 + `fm serve` as an external baseline advisor | M6 |

## 6. Reliability notes

Verified locally this session: pins, logits_processors signature, base
snapshot revision + cache date. Verified against primary sources: all
release dates/versions (GitHub releases API, PyPI, ai.google.dev), PR #1240
merge, WWDC session pages, QAT/12B launch blogs, Gemini deprecations page.
UNVERIFIED (flagged inline): MLXLanguageModel Foundation-Models backend
(Medium only), Foundation Models open-sourcing claim, mlx-lm-lora minimum
mlx-lm pin, PR #1276 contents (search summary only), E2B JSON-vs-reasoning
quality claim, transformers #45242 fix-merge status, third-party KV-bench
version citations, the 26B "+1 shared expert / ~3.8B active" detail (held
from our earlier reference doc, nothing contradicts it).
