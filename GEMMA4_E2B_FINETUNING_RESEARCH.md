# Gemma 4 E2B Fine-Tuning on Apple M5 16GB — Research Findings

> Status: **Research complete — pending implementation-plan discussion**
> Date compiled: 2026-05-17
> Target: fine-tune a Gemma 4 "~3B-class" model locally on a MacBook M5 (16GB unified memory)

---

## 0. Correction Log

An earlier assessment in this session dismissed "Gemma 4" and the tool `mlx-tune`
as fabricated. **That assessment was wrong.** Gemma 4 was released **April 2, 2026**,
after the assistant's January 2026 knowledge cutoff. Web research confirms Gemma 4,
`mlx-tune`, and `gemma-tuner-multimodal` are all real. Findings below supersede the
earlier critique.

The originally shared research report (the "Gemini doc") is now assessed as
**mostly accurate**, with these minor inaccuracies:
- `gemma-tuner-multimodal` uses **PyTorch + MPS**, not MLX (the doc said MLX).
- 26B-A4B active params: HF blog says **~4B active**, doc said 3.8B.
- E2B 4-bit footprint: HF says **~3.2 GB**, doc said 2.6 GB.
- The doc contained generation artifacts (leaked `[span_NN]` citation markup inside
  a code block; a broken `{"messages":}` JSON example).
- Specific benchmark numbers in the doc are stated with false precision — treat as
  indicative only.

---

## 1. Gemma 4 — Release & Architecture

- **Released:** 2026-04-02, **Apache 2.0** license (commercial use OK), built from
  Gemini 3 research.
- **Natively multimodal:** text + image + video across all sizes; **audio** on E2B/E4B.

### Model lineup

| Model | Parameters | Context | Notes |
|---|---|---|---|
| **E2B** | 2.3B effective / 5.1B total | 128K | text+image+audio, on-device target |
| **E4B** | 4.5B effective / 8B total | 128K | text+image+audio |
| **26B-A4B** | ~4B active / 26B total (MoE, 128 experts, top-8) | 256K | needs >40GB to fine-tune |
| **31B** | 31B dense | 256K | flagship |

### Architecture features (confirmed real)

- **Per-Layer Embeddings (PLE):** a distributed embedding matrix feeds a token-specific
  residual into every decoder layer. For E2B, PLE tables are a large share of the 5.1B
  total params but are O(1) lookups — so RAM footprint is high while FLOPs stay at the
  2.3B "effective" level.
- **Shared KV cache:** final N layers reuse earlier layers' K/V states — lowers
  activation memory in the backward pass (helps on 16GB).
- **Dual RoPE:** standard RoPE for local sliding-window attention (512–1024 tokens),
  pruned RoPE for global attention layers.
- **Vision encoder:** ~150M params, configurable token budgets (70/140/280/560/1120).
- **Audio encoder:** ~300M-param USM-style 12-layer Conformer, native 16kHz, no
  separate Whisper-style transcription model needed.

---

## 2. Fine-Tuning Frameworks — What Actually Runs on a Mac

| Framework | Backend | Mac/Apple Silicon? | Notes |
|---|---|---|---|
| Unsloth (official) | CUDA/Triton | ❌ "MLX support upcoming, not yet available" | Needs an NVIDIA GPU today |
| transformers + TRL | PyTorch | ⚠️ via MPS, fragile near memory limit | Official Google day-0 path |
| Vertex AI / NeMo-AutoModel | cloud / NVIDIA | ❌ local | H100-class hardware |
| **mlx-lm** | MLX | ✅ | Apple's own lib; `mlx_lm.lora` CLI; **most stable text-LoRA path** |
| **mlx-vlm** | MLX | ✅ | Multimodal (image/audio) sibling of mlx-lm |
| **mlx-tune** (`ARahim3/mlx-tune`, ex `unsloth-mlx`) | MLX | ✅ | Unsloth-compatible API (`FastLanguageModel`, `SFTTrainer`); same script can later run on CUDA |
| **gemma-tuner-multimodal** (`mattmireles`) | PyTorch + MPS | ✅ | Multimodal text/image/audio toolkit |

### Key fact

**Unsloth proper does not run on Apple Silicon yet.** Code samples in the wild that
use `FastLanguageModel` / `SFTTrainer` on a Mac are running **`mlx-tune`**, the
third-party MLX reimplementation of the Unsloth API — not Unsloth itself.

### Memory math (fine-tuning, not inference)

- E2B 4-bit weights ≈ **3.2 GB**.
- E4B 4-bit weights ≈ **4.2 GB**.
- Fine-tuning adds: BF16 LoRA adapters, AdamW optimizer states (32-bit),
  forward-pass activations (scale **quadratically** with sequence length), KV cache.
- **QLoRA is mandatory on 16GB.** Load 4-bit base, train BF16 LoRA adapters
  (+2–4 GB). Realistic E2B QLoRA peak: ~7–10 GB → fits the ~12GB usable envelope.

### Known sharp edge — GGUF export

Direct **GGUF export from a 4-bit-loaded model is broken** (upstream mlx-lm
limitation; affects mlx-tune too). Workarounds:
1. **Mac-native:** `save_pretrained_merged()` → ready-to-serve HF-format dir for
   `mlx_lm.server`. Recommended if deploying only on Apple hardware.
2. **GGUF target:** dequantize to FP16 on export, then re-quantize externally with
   `llama.cpp` (`llama-quantize ... Q4_K_M`).

---

## 3. Hardware Findings — This Machine

Probed 2026-05-17:

| Property | Value | Assessment |
|---|---|---|
| Chip | Apple **M5**, 10-core GPU, 10 CPU cores (4P + 6E), Metal 4 | ✅ |
| Unified memory | **16 GB** | ⚠️ tight but workable for E2B QLoRA |
| macOS | 26.3 | ✅ |
| Disk free | 359 GB | ✅ |
| Python (system) | **3.14.3** (Homebrew, `/opt/homebrew/bin/python3`) | ⚠️ too new — see below |
| MLX / mlx-lm | **not installed** | ☐ to install |
| Swap at probe time | 5.1 GB swapfile, **4.6 GB in use** | ⚠️ machine under memory pressure now |

### Verdict

- **E2B QLoRA: viable** with a comfortable safety margin.
- **E4B QLoRA: possible but tight** — batch 1, short sequences, expect some swap.
- **26B-A4B / 31B: not feasible** on 16GB (need >40GB / multi-GPU).

### Two blockers to clear before training

1. **Python 3.14 is too new.** MLX/mlx-lm wheels are reliable for Python 3.11–3.13;
   3.14 (released Oct 2025) frequently lacks prebuilt wheels. → Use a dedicated
   **Python 3.12 venv**, not the system 3.14 interpreter.
2. **4.6 GB already in swap.** The machine is under memory pressure at rest. Close
   browser/other apps before any run; ML working envelope on 16GB is ~12 GB.

---

## 4. Recommended Configuration (16GB M5, E2B QLoRA)

| Setting | Value | Reason |
|---|---|---|
| Base model | `mlx-community/gemma-4-E2B-it-4bit` (pre-quantized) | Never quantize FP16 locally — causes init OOM spike |
| Quantization | 4-bit QLoRA | Mandatory for 16GB |
| LoRA rank `r` | 16 | Enough expressivity for 3B-class without memory blowup |
| LoRA alpha | 32 (`2×r`) | Stable update scaling: `ΔW = (α/r)·AB` |
| Target modules | all linear (`q,k,v,o,gate,up,down`) | Omitting MLP/o_proj degrades reasoning & format adherence |
| `--lora-layers` | 16 | Subset of layers keeps memory bounded |
| Batch size | 1 (physical) | Prevents activation overflow |
| Grad accumulation | 4–8 | Simulates effective batch 4–8 |
| Max sequence length | 512 (baseline) – 1024 (ceiling) | Activations scale quadratically; >1024 risks OOM |
| Learning rate | 2e-4 | QLoRA sweet spot for 3B class |
| Epochs | 1–3 | Watch validation loss for overfit past epoch 2 |
| Warmup ratio | 0.05–0.1 | Early-training stability |
| Gradient checkpointing | on | Trades compute for activation memory |

### Memory-safety techniques

- Stream a **pre-quantized** model from disk; do not load FP16 then quantize.
- Cap the MLX allocator (`mx.set_wired_limit`, `mx.set_cache_limit`,
  `mx.set_memory_limit`) — note current MLX exposes these top-level; the older
  `mx.metal.*` namespace is deprecated.
- `train_on_responses_only` — mask prompt tokens from the loss to cut backward-pass
  memory.
- Periodic `mx.clear_cache()` + `gc.collect()` inside the loop.

---

## 5. Pre-Implementation Checklist

1. ☐ `brew install python@3.12`; create isolated venv
2. ☐ `pip install mlx mlx-lm` (+ `mlx-vlm` and/or `mlx-tune` per task choice)
3. ☐ HuggingFace account → accept Gemma 4 license → `huggingface-cli login`
4. ☐ Confirm pre-quantized repo availability (`mlx-community/gemma-4-E2B-it-4bit`)
5. ☐ Prepare dataset: JSONL chat format, train/valid/test split
6. ☐ Free RAM before runs (close apps; verify swap pressure dropped)
7. ☐ Decide export target: Mac-native merge vs GGUF dequantize path

---

## 6. Open Decisions (blocking the implementation plan)

1. **Task type** — text-only SFT (→ `mlx-lm`) vs audio STT/Q&A vs vision
   (→ `mlx-vlm` / `mlx-tune`). Changes the whole pipeline.
2. **Model variant** — E2B (recommended, safe fit) vs E4B (tight).
3. **Framework** — `mlx-lm` CLI (most stable) vs `mlx-tune` (Unsloth-compatible API,
   portable to CUDA later).
4. **Dataset** — what data, what task, how many examples? (Mac is good for <1k
   examples / overnight runs; large datasets favor cloud.)

---

## 7. Sources

- [Gemma 4 — Google DeepMind](https://deepmind.google/models/gemma/gemma-4/)
- [Welcome Gemma 4 — Hugging Face blog](https://huggingface.co/blog/gemma4)
- [Gemma 4: Byte for byte… — Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 Fine-tuning Guide — Unsloth docs](https://unsloth.ai/docs/models/gemma-4/train)
- [Gemma 4 — How to Run Locally — Unsloth docs](https://unsloth.ai/docs/models/gemma-4)
- [ARahim3/mlx-tune — GitHub](https://github.com/ARahim3/mlx-tune)
- [mattmireles/gemma-tuner-multimodal — GitHub](https://github.com/mattmireles/gemma-tuner-multimodal)
- [Fine-Tuning Gemma 4 on a Mac — Apple Silicon + MLX — Antigravity Lab](https://antigravitylab.net/en/articles/antigravity/gemma-4-finetuning-apple-silicon-mlx-guide)
- [Gemma model fine-tuning — Google AI for Developers](https://ai.google.dev/gemma/docs/tune)
- [From OOM Errors to Working Model: Fine-Tuning Gemma 4 E2B with Unsloth — Medium](https://medium.com/@gabi.preda/from-oom-errors-to-working-model-fine-tuning-gemma-4-e2b-step-by-step-using-unsloth-ef7873e59efd)
