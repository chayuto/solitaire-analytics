# Gemma 4 Model Family: Architecture Reference

**Date**: 2026-05-28
**Status**: REFERENCE DOC. Compiled from external sources, not from internal training knowledge. Verify against the official model cards before using any single fact in a production decision.
**Why this exists**: the solitaire harvester pipeline now produces traces from both `gemma-4-31b-it` (dense) and `gemma-4-26b-a4b-it` (MoE). Same family, different architecture, different inference characteristics. The model assistant's own training-time knowledge of Gemma 4 variants is unreliable enough that this reference compiles externally-sourced facts for future cross-checking.
**Audience**: future-me looking up "is gemma-4-26b-a4b-it dense or MoE, and what does that imply for the corpus?"

## 0. Provenance

All facts below sourced from web searches conducted 2026-05-28. Cross-checked against multiple sources where possible; remaining discrepancies are called out explicitly in Section 3. Where Google's official `ai.google.dev/gemma/docs/core` page is one of the sources, that is treated as authoritative.

## 1. Release facts

- **Release date**: 2026-04-02 (April 2, 2026)
- **License**: Apache 2.0 (first time for the Gemma line; prior generations had a Gemma-specific license)
- **Family size**: four variants on launch

## 2. The four variants

| Variant | Total params | Active per token | Type | Context | Notes |
|---|---|---|---|---|---|
| `gemma-4-e2b-it` | ~5.1B | ~2.3B "effective" | Dense + PLE | 128K | Edge model. Runs on Raspberry Pi 5 at ~7.6 tok/s. Multimodal: text, image, audio. |
| `gemma-4-e4b-it` | ~8B | ~4.5B "effective" | Dense + PLE | 128K | Edge model. Runs on Android 12+ with 4 GB RAM via Google AICore. Multimodal: text, image, audio. |
| `gemma-4-26b-a4b-it` | 26B | ~3.8B active | MoE | 256K | Server-class but MoE-efficient. Multimodal: text, image, video. |
| `gemma-4-31b-it` | 30.7B | 30.7B (all active) | Dense | 256K | Flagship. Full multimodal: text, image, audio, video. |

### 2.1 What "effective" vs "active" mean (the naming is confusing)

- **Effective** (E2B / E4B): a marketing term capturing "the model performs comparably to a dense model of this size on benchmarks." The actual parameter count is higher (5.1B for E2B, ~8B for E4B) but inference cost is dominated by the "effective" portion because of Per-Layer Embeddings (PLE).
- **Active** (A4B in `26b-a4b`): a technically precise MoE term meaning "parameters used in any single forward pass per token." 26B-a4b loads 26B params into memory but only ~3.8B participate in any given token's computation, because the MoE router selects a small subset of experts.

These are NOT the same concept. Effective is about "behaves like" (a benchmark-comparison term). Active is about "computes with" (a hardware-cost term). Conflating them is a common error in third-party Gemma 4 writeups.

## 3. Per-architecture details

### 3.1 `gemma-4-31b-it` (dense flagship)

- **Parameter count**: 30.7B, all active per token
- **Attention**: hybrid local + global. Pattern is "5 local sliding-window layers per 1 global full-context layer." Global layers use unified K/V (shared keys/values across heads) and Proportional RoPE (p-RoPE) for long-context efficiency.
- **Context window**: 256K tokens
- **Multimodal inputs**: text, image, audio, video
- **VRAM (informational, not authoritative)**: ~20 GB at FP16, ~12 GB at INT4 quantization
- **Reported benchmarks**: MMLU-Pro ~72%, HumanEval 85%+
- **No PLE**: PLE is an edge-model efficiency technique, not used in the 31B dense flagship.

### 3.2 `gemma-4-26b-a4b-it` (MoE)

- **Parameter count**: 26B total loaded, ~3.8B active per token
- **MoE structure**: 128 small experts per MoE layer plus one shared expert
- **Routing**: this is where sources disagree. See Section 4 below.
- **Context window**: 256K tokens (same as 31B)
- **Multimodal inputs**: text, image, video (no explicit audio support in the surveyed sources, but worth verifying)
- **Performance positioning**: "knowledge of a 26B model with the speed of a ~4B model" (paraphrasing from multiple sources)
- **No PLE**

### 3.3 `gemma-4-e2b-it` and `gemma-4-e4b-it` (edge variants with PLE)

These are NOT used by the harvester. Included for completeness because the solitaire-analytics project's student LoRA work at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/` uses E2B as the base model and the Gemma 3n + adapters_t5_at750 run also references E2B-class.

- **Per-Layer Embeddings (PLE)**: a parameter-efficiency technique. Instead of one embedding table consumed at input, PLE adds a parallel, lower-dimensional conditioning pathway that feeds a small dedicated vector into every decoder layer.
  - For each token: produces a per-layer vector by combining (a) a token-identity component from a second embedding lookup table and (b) a context-aware component from a learned projection of the main embeddings.
  - E2B has 35 decoder layers; PLE produces a packed vector per token sliced into 35 pieces, one per layer.
  - Compute cost is a lookup plus an addition per layer (no matrix multiply); cheap.
- **E2B**: ~2.3B effective / ~5.1B total parameters
- **E4B**: ~4.5B effective / ~8B total parameters
- **Audio input**: yes (the only Gemma 4 variants with explicit on-device audio)
- **Context window**: 128K tokens (half the 26B/31B context)
- **Reported edge deployment**: E2B on Raspberry Pi 5 at 7.6 tok/s; E4B on Android 12+ with 4 GB RAM

## 4. Architectural ambiguity called out

### 4.1 MoE routing: top-2 vs top-8

Two surveyed sources disagree on the 26B-a4b MoE router behavior:

| Source class | Claim |
|---|---|
| Some dev-community writeups | "router selects 2 of 128 experts per token" |
| Other dev-community writeups + llama.cpp gist | "activates 8 of 128 experts per token plus 1 shared expert" |

The "top-8 plus 1 shared" figure is more commonly cited and is internally consistent with the "~3.8B active per token" number (8 experts of ~0.3B each plus a shared expert plus router overhead lands roughly at 3.8B). The "top-2" figure would give ~1B active, which contradicts the reported 3.8B.

**Likely truth**: top-8 + 1 shared, consistent with 3.8B active.

**Action if it matters for your purpose**: read the actual model card at `https://huggingface.co/google/gemma-4-26B-A4B-IT` (or wherever Google publishes the canonical card) to confirm. Don't trust this doc on this specific point.

### 4.2 "27B" vs "26B" naming

One source consistently refers to the MoE variant as "Gemma 4 27B" rather than 26B. This is likely a rounding choice (the true parameter count is probably around 26.4B, which can round either way) plus inconsistency in the early-week dev-community write-ups. The official Google identifier is `gemma-4-26b-a4b-it`.

## 5. Common-to-all-variants architecture

These hold across E2B, E4B, 26B-a4b, and 31B unless noted otherwise:

- **Attention scheme**: interleaved local sliding-window attention + global full-context attention. The 5-local-to-1-global ratio appears explicitly cited for 31B; the smaller models likely use a similar pattern but the surveyed sources do not specify.
- **Global layer optimizations**: unified K/V (shared keys/values across heads) and Proportional RoPE (p-RoPE) on global attention layers for long-context efficiency.
- **Multimodal inputs**: text and image are universal across the family. Audio is on E2B and E4B. Video is on 26B-a4b and 31B. (Video via frame-extraction; not native frame-stream input.)
- **Language coverage**: 140+ languages
- **Agentic features**: multi-step reasoning, function calling, "thinking modes" referenced by sources but spec-level details not surveyed here.
- **Tokenizer**: same across the family (this is the consistent Gemma 4 convention and matches the family pattern from prior Gemma generations; not explicitly verified per-variant in the surveyed sources).

## 6. Implications for the solitaire harvester pipeline

Translating Section 1-5 facts into what matters for this project:

### 6.1 31B vs 26B-a4b: same family, different cost profile

- Both have the SAME training data lineage and the SAME prompt-formatting requirements (Gemma 4 chat template). This means [[harvester-prompt-v1-2-shipped]] applies to both without modification.
- Both have 256K context, so long-session prompts (200+ turn games) fit identically.
- 26B-a4b inference is much faster per token (active 3.8B vs dense 30.7B, so roughly 8x cheaper compute per token). Harvesting throughput should be substantially higher.
- 26B-a4b VRAM footprint is HIGHER than 31B at quantization parity, because all 26B params must be loaded even though only 3.8B compute per token. MoE is fast-but-fat; dense is slow-but-thin (relatively).

### 6.2 Failure mode expectations

Family lineage strongly suggests:
- The Gemma-specific failure modes catalogued in `/Users/chayut/repos/solitaire-analytics/data/DATASET_NOTES.md` (oscillation, strategy-bullet parroting, reveal-narrative fabrication) should reproduce on 26B-a4b. The rule from memory `feedback_harvester_fidelity_model` (don't substitute stronger Gemini models because failures are Gemma-specific) continues to hold within the Gemma 4 family.
- BUT MoE routing introduces a NEW potential failure mode: expert-routing instability under long context. Long-running solitaire sessions are exactly the kind of long-context workload where MoE routers have been documented to drift. If 26B-a4b shows decode-incoherence patterns (malformed JSON, wrong move_index strings, schema-key hallucination) that 31B does NOT show, suspect routing instability before suspecting prompt-version effects.

### 6.3 The 31B-specific session analytics may not transfer cleanly

The session-wide oscillation pair-count thresholds in `/Users/chayut/repos/solitaire-analytics/.claude/skills/solitaire-analyst/scripts/load_export.py` (default minimum 4 to flag a pair) were calibrated against 31B behavior. MoE models can produce different oscillation absolute counts because of routing-driven variance in move selection at sampling temperature greater than 0. First few 26B-a4b sessions should be manually compared against the heuristic output and the threshold reconsidered.

### 6.4 Distillation track interaction

The student LoRA work uses Gemma 4 E2B (NOT E4B or 31B) as the base. E2B is dense + PLE. Training data is currently 31B teacher traces. If 26B-a4b traces eventually enter the training mix (currently they do not, per [[harvester-26b-cohort-cataloging]]):
- Both teacher families share Gemma 4 lineage, tokenizer, and chat template, so the corpus is internally consistent at the format level.
- Both teachers produce traces with the same JSON schema (`boardAnalysis`, `reasoning`, `moveIndex`), so the student's loss landscape doesn't change structurally.
- But the teachers may have systematically different decision profiles (MoE vs dense), in which case training on a mix gives the student a noisier signal than training on either alone.
- The standing decision (memory `harvester-26b-cohort-cataloging`): leave the `TEACHER_MODEL` filter at `gemma-4-31b-it` for now; explore mix options after the analysis pass establishes how 26B-a4b actually behaves.

## 7. What this doc deliberately does NOT cover

- Per-variant published benchmark numbers (MMLU, HumanEval, etc.). These shift fast and the dev-community writeups are inconsistent on specific scores. Refer to the official Google model cards for current numbers.
- Pricing and API access. Multiple providers (Together AI, Modular, NVIDIA NIM, Ollama) host Gemma 4; specifics change.
- Detailed comparison vs other model families (Llama, Qwen, etc.). Out of scope for this reference; relevant if the user later decides to consider non-Gemma alternatives.
- The exact published training data composition and the safety/RL-fine-tuning details. Not in the surveyed sources; would need the Gemma 4 technical report.

## 8. Open items to verify if any decision depends on them

1. Exact MoE routing: top-2 vs top-8 + 1 shared (Section 4.1). Verify against the official model card.
2. 26B-a4b audio input support: not explicitly stated in the surveyed sources for the 26B variant. Verify if multimodal audio is needed.
3. Whether the 26B-a4b sliding-window attention uses the SAME 5-local-to-1-global pattern as 31B, or a different ratio. Not specified in the surveyed sources for the MoE variant.
4. Whether the E2B 35-layer count is exact or approximate; relevant for the student LoRA training infrastructure at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/`.

## Sources

Cross-checked across multiple write-ups for each fact where possible. Primary sources where claims trace back to original material:

- Google AI for Developers - Gemma 4 model overview (official): https://ai.google.dev/gemma/docs/core
- Maarten Grootendorst - A Visual Guide to Gemma 4: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- Botmonster - Gemma 4 Architecture Explained: Per-Layer Embeddings, Shared KV Cache, and Dual RoPE: https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/
- Kaitchup - Gemma 4 31B and 26B A4B: Architecture and Memory Consumption: https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture
- MindStudio - What Is the Gemma 4 Mixture of Experts Architecture? How 26B Parameters Run Like 4B: https://www.mindstudio.ai/blog/gemma-4-mixture-of-experts-architecture
- BetterStack - Google Gemma 4: Per-Layer Embeddings, Multimodality, and On-Device Performance: https://betterstack.com/community/guides/ai/gemma-4/
- Hugging Face - google/gemma-4-31B model card: https://huggingface.co/google/gemma-4-31B
- Hugging Face - google/gemma-4-E2B model card: https://huggingface.co/google/gemma-4-E2B
- llama.cpp issue tracker - Gemma 4 E2B/E4B PLE implementation status: https://github.com/ggml-org/llama.cpp/issues/22243
- Modular - Gemma 4 31B Inference, Google's Dense Vision Model: https://www.modular.com/models/gemma-4-31b-it
- Reeboot - Gemma 4 31B-IT: 256K context, vision, video, function calling, Apache 2.0: https://reeboot.fr/en/blog/gemma-4-31b
- Datature Blog - Gemma 4: What Computer Vision Engineers Actually Need to Know: https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know
- DEV Community (shreya111111) - How Gemma 4's Per-Layer Embeddings Actually Work: https://dev.to/shreya111111/how-gemma-4s-per-layer-embeddings-actually-work-and-why-e2b-punches-above-2b-4l68
- AImadetools - Gemma 4: All Models Compared: https://www.aimadetools.com/blog/gemma-4-family-guide/
