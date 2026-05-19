# Gemma 4 E2B Distillation — Implementation Plan & Design

> Companion to `GEMMA4_E2B_FINETUNING_RESEARCH.md` (research) and
> `GEMMA4_E2B_DATA_EVALUATION.md` (dataset audit).
> This document is the **build plan**. The runway (scripts) lives in `gemma4_finetune/`.
> Date: 2026-05-17

---

## 1. Objective & Scope

Distill the `gemma-4-31b-it` Klondike Solitaire advisor into **Gemma 4 E2B**,
fine-tuned locally on an **Apple M5 / 16 GB** Mac with **MLX QLoRA**, then serve it to
the repo's MCP advisor path.

- **In scope:** text-only SFT; QLoRA on a pre-quantized 4-bit E2B; full ~1764-token
  rules prompt kept verbatim in every example.
- **Success criterion:** E2B **matches the 31B teacher's `move_index`** on held-out
  boards (top-1 agreement) — see §9.
- **Out of scope:** audio/vision quality (acceptable to degrade), the MoE/31B models.

---

## 2. Status & Dependency Gate

| Track | State |
|---|---|
| Pipeline runway (scripts, env, smoke test) | **buildable now — data-independent** |
| Data preparation finalization | **blocked** — pilot schema not final (P0 fixes in progress) |
| Training run | **blocked** on a pilot that clears the §10 acceptance gate in the data eval |

This plan builds everything that does **not** depend on the final dataset, and proves
the M5 can physically run the job (the §6 smoke test) before real data exists.

---

## 3. Pipeline Architecture

```
 collection harness          gemma4_finetune/ runway              deployment
 ┌──────────────┐   log.json   ┌───────────────────┐   adapters   ┌──────────────┐
 │ 31B teacher  │ ───────────► │ prepare_dataset.py │ ──────────►  │ mlx_lm.server│
 │ plays games  │              │   ↓ train/valid/   │              │  + LoRA      │
 └──────────────┘              │     test.jsonl     │              └──────┬───────┘
                               │ mlx_lm.lora (QLoRA)│                     │
                               │   ↓ adapters/      │              strategies/llm.py
                               │ evaluate.py        │              → MCP server
                               └───────────────────┘
```

Eight stages, executed in order. Stages 1–3 + 5(scaffold) + 6(core) are built now;
stage 4 finalize + the live training run wait on data.

---

## 4. Directory Layout (`gemma4_finetune/`)

```
gemma4_finetune/
├── README.md            # how to run the runway
├── requirements.txt     # mlx, mlx-lm, pyyaml, huggingface-hub
├── setup.sh             # create Python 3.12 venv + install
├── prepare_dataset.py   # collection log JSON → train/valid/test.jsonl (game-level split)
├── make_smoke_data.py   # synthetic 2048-token dataset for the memory smoke test
├── lora_config.yaml     # mlx-lm LoRA/QLoRA config (the documented hyperparameters)
├── smoke_test.sh        # run ~50 iters on synthetic data + sample peak memory
├── evaluate.py          # JSON-validity, move-legality, teacher-agreement scorer
├── .gitignore           # ignore venv/, dataset/, adapters/, *.jsonl, model cache
├── dataset/             # (generated) train.jsonl, valid.jsonl, test.jsonl
└── adapters/            # (generated) LoRA adapter weights
```

---

## 5. Stage 1–2 — Environment & Model

**Environment.** The system Python is 3.14 (too new for reliable MLX wheels). `setup.sh`
creates an isolated **Python 3.12** venv and installs `mlx`, `mlx-lm`, `huggingface-hub`,
`pyyaml`. Verifies `import mlx.core` reports a Metal device.

**Model.** Target repo: `mlx-community/gemma-4-E2B-it-4bit` (pre-quantized — never load
FP16 and quantize locally, that spikes memory at init). Requires accepting the Gemma 4
license on Hugging Face + `huggingface-cli login`. ~3 GB download.

---

## 6. Stage 3 — Memory Smoke Test (the critical de-risk)

**The single biggest unknown:** does QLoRA on E2B at **seq-len 2048** fit in 16 GB?
The research doc's generic "≤1024 or OOM" guidance is too conservative, but unverified
on *this* machine. We answer it before real data exists.

`make_smoke_data.py` generates a synthetic dataset whose examples match the real shape
(~1764-token prompt + ~289-token completion ≈ 2050 tokens). `smoke_test.sh` runs
`mlx_lm.lora` for ~50 iterations with the **exact** production config and samples peak
memory.

**Decision tree on the result:**

| Smoke-test outcome | Action |
|---|---|
| Peak < ~11 GB, no swap storm | ✅ proceed; config is final |
| Peak 11–13 GB, mild swap | ⚠️ acceptable; consider `num_layers` 12, monitor |
| OOM / kernel-panic risk | ❌ reduce `max_seq_length` to 1536, `num_layers` to 8; if still failing, drop grad accumulation and re-test |

Run with other apps closed (baseline machine already sits ~4.6 GB into swap).

---

## 7. Stage 4 — Data Preparation (spec; finalized post-pilot)

`prepare_dataset.py` transforms a collection log into mlx-lm training files:

1. Keep only `outcome == "success"` rows.
2. Validate `rawResponse` parses as JSON with exactly `board_analysis`,
   `strategic_plan`, `final_decision`; drop rows that fail.
3. Emit one JSON line per example in **completions format**:
   `{"prompt": <full prompt>, "completion": <rawResponse JSON string>}`.
   mlx-lm masks the prompt from the loss automatically — loss is computed only on the
   ~289-token answer (memory-cheap, prevents prompt memorization).
4. **Split by game, not by row.** If `gameId` is present (P0 fix), all turns of one
   game go entirely to train **or** valid **or** test — never split across — to prevent
   leakage between near-identical consecutive turns. Without `gameId`, the script warns
   loudly and falls back to a row-level split.
5. Default split ≈ 80 / 10 / 10 by game.

The script is written now and runs against the **current** schema; the only
pilot-dependent part is the `gameId` grouping, which it already handles defensively.

---

## 8. Stage 5 — Training

`mlx_lm.lora --config gemma4_finetune/lora_config.yaml`. Config (from the research doc,
tuned for 16 GB):

| Param | Value | Note |
|---|---|---|
| `model` | `mlx-community/gemma-4-E2B-it-4bit` | pre-quantized |
| `max_seq_length` | 2048 | covers the ~2050-token examples |
| `batch_size` | 1 | mandatory on 16 GB |
| `num_layers` | 16 | LoRA-adapted layer count |
| `lora rank / scale / dropout` | 16 / 2.0 / 0.05 | scale = α/r = 32/16 |
| `learning_rate` | 2e-4 | QLoRA sweet spot |
| `iters` | set from dataset size | ≈ 2–3 epochs; **few epochs — n is small** |
| `grad_checkpoint` | true | trades compute for activation memory |
| `steps_per_eval` / `save_every` | 50 / 100 | watch valid loss for overfit |

**Caveat to verify at first run:** gradient accumulation is not exposed by all mlx-lm
versions. If absent, batch stays 1; compensate with `iters` and LR, not larger batch.
LoRA target `keys` must be confirmed against E2B's actual module names on first load.

---

## 9. Stage 6 — Evaluation

Success = "matches the teacher", so the metric is **top-1 `move_index` agreement** on
the held-out test set. `evaluate.py` reports:

- **JSON validity rate** — output parses with the 3 required keys.
- **Legal-move rate** — chosen `move_index` exists in the prompt's `legalMoves`.
- **Teacher agreement (primary)** — E2B's `move_index` == teacher's `move_index`.
- Breakdown: agree / legal-but-disagree / illegal / malformed.

Target gate before deployment: validity ≥ 98%, legal ≥ 98%, agreement materially above
the trivial baseline (the share of the most common move in the test set).

---

## 10. Stage 7–8 — Export & Integration

**Export.** `mlx_lm.fuse` merges the LoRA adapters into the base weights →
Mac-native HF-format directory. (Direct GGUF export from a 4-bit base is broken
upstream; the merge path is the supported one — see research doc §2.)

**Integration.** Serve the fused model with `mlx_lm.server` (OpenAI-compatible
endpoint). Point `solitaire_analytics/strategies/llm.py` at `http://localhost:<port>/v1`
instead of the cloud provider. Validate end-to-end by playing games through
`solitaire_analytics/mcp_server.py` and comparing win rate against the `weighted` /
`lookahead` strategies.

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM at seq-len 2048 | medium | §6 smoke test before real data; fallback config in the decision tree |
| mlx-lm wheels unavailable for Python 3.12 | low | venv pinned to 3.12; setup.sh fails fast with a clear message |
| Tiny dataset → overfitting | high | few epochs, valid-loss monitoring, game-level split |
| Distilling a low-quality teacher | medium | data-eval acceptance gate must pass first; agreement metric, not absolute play strength |
| LoRA `keys` mismatch for E2B | medium | verified on first model load; config note flags it |

---

## 12. Runway Build Status

| Artifact | Built | Runs without pilot data |
|---|---|---|
| `setup.sh`, `requirements.txt` | ✅ | ✅ |
| `make_smoke_data.py`, `smoke_test.sh` | ✅ | ✅ |
| `lora_config.yaml` | ✅ | ✅ (smoke test) |
| `prepare_dataset.py` | ✅ | runs on current sample; finalize on pilot |
| `evaluate.py` | ✅ | needs a trained adapter |
| `README.md` | ✅ | ✅ |

**Next executable step:** run `gemma4_finetune/setup.sh`, then `smoke_test.sh` — proves
the M5 can do the job. Everything after that waits on the pilot dataset.
