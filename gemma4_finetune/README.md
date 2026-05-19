# gemma4_finetune — Runway

Pipeline to distill the `gemma-4-31b-it` Klondike advisor into **Gemma 4 E2B**, via
**MLX QLoRA** on an Apple M5 / 16 GB Mac.

Design & rationale: `../GEMMA4_E2B_IMPLEMENTATION_PLAN.md`
Research: `../GEMMA4_E2B_FINETUNING_RESEARCH.md` · Data audit: `../GEMMA4_E2B_DATA_EVALUATION.md`

## Status

The runway is built. The **training run is blocked** on a re-collected pilot dataset
(the current sample has a fatal label defect — see the data audit). What runs **now**,
without pilot data, is environment setup and the memory smoke test.

## Steps

```bash
# 1. Environment — creates a Python 3.12 venv (system Python is 3.14, too new for MLX)
./setup.sh

# 2. Hugging Face — accept the Gemma 4 license at huggingface.co/google/gemma-4-E2B-it,
#    then authenticate:
source venv/bin/activate && huggingface-cli login

# 3. Memory smoke test — proves the M5 can run QLoRA at seq-len 2048 (synthetic data).
#    Close other apps first. See implementation plan §6 for the result decision tree.
./smoke_test.sh

# --- below here needs the pilot dataset ---

# 4. Prepare data — collection log JSON -> dataset/{train,valid,test}.jsonl
python prepare_dataset.py --log ../data/<pilot-log>.json

# 5. Train — QLoRA; edit `iters` in lora_config.yaml to ~2-3 epochs for the real n
mlx_lm.lora --config lora_config.yaml

# 6. Evaluate — JSON validity, move legality, teacher agreement (primary metric)
python evaluate.py --test dataset/test.jsonl --adapter-path adapters

# 7. Export — merge LoRA into the base for Mac-native serving
mlx_lm.fuse --model mlx-community/gemma-4-E2B-it-4bit \
            --adapter-path adapters --save-path fused_model

# 8. Serve — OpenAI-compatible endpoint; point strategies/llm.py at it
mlx_lm.server --model fused_model --port 8080
```

## Files

| File | Purpose | Needs pilot data |
|---|---|---|
| `setup.sh` | Python 3.12 venv + MLX install | no |
| `make_smoke_data.py` | synthetic 2048-token dataset | no |
| `smoke_test.sh` | run QLoRA briefly, sample peak memory | no |
| `lora_config.yaml` | mlx-lm LoRA/QLoRA hyperparameters | no |
| `prepare_dataset.py` | collection log → train/valid/test JSONL | runs now; finalize on pilot |
| `evaluate.py` | score E2B vs the teacher | needs a trained adapter |

## Notes

- All generated artifacts (`venv/`, `dataset/`, `adapters/`, `fused_model/`,
  `smoke_data/`, `*.jsonl`) are git-ignored.
- `iters: 600` in `lora_config.yaml` is a placeholder — set it from the real dataset
  size once the pilot lands.
- Two things to verify on the first real model load: the LoRA `keys` match E2B's
  module names, and whether your mlx-lm version exposes gradient accumulation.
