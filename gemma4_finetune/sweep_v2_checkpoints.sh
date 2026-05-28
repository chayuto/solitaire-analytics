#!/usr/bin/env bash
# Post-training evaluation sweep for v2: runs posttune_n20_gemma4_text_runner.py
# against each saved adapter checkpoint (250, 500, 750, 1000 + final), then
# scores them all and emits a learning-curve summary.
#
# Assumes the working directory is gemma4_finetune/.

set -euo pipefail

VENV_PY=venv/bin/python
ADAPTERS=adapters_v2
OUT_DIR=baseline_n20_gemma4_text
PROJECT_VENV=../.venv/bin/python

echo "=== v2 checkpoint sweep ==="
ls $ADAPTERS/
echo

declare -a CHECKPOINTS=(
  "0000250_adapters.safetensors:posttune_at250.json:posttune_at250_responses"
  "0000500_adapters.safetensors:posttune_at500.json:posttune_at500_responses"
  "0000750_adapters.safetensors:posttune_at750.json:posttune_at750_responses"
  "0001000_adapters.safetensors:posttune_at1000.json:posttune_at1000_responses"
)

for entry in "${CHECKPOINTS[@]}"; do
  IFS=':' read -r ckpt out resp <<< "$entry"
  if [ ! -f "$ADAPTERS/$ckpt" ]; then
    echo "[skip] $ckpt missing"
    continue
  fi
  # Stage as adapters.safetensors so load() picks it up; mlx-lm load looks for
  # a generic adapters.safetensors, not the dated checkpoints.
  mkdir -p "$ADAPTERS/_eval_stage"
  cp "$ADAPTERS/$ckpt" "$ADAPTERS/_eval_stage/adapters.safetensors"
  cp "$ADAPTERS/adapter_config.json" "$ADAPTERS/_eval_stage/adapter_config.json"
  echo "=== eval $ckpt ==="
  $VENV_PY posttune_n20_gemma4_text_runner.py \
      --adapter-path "$ADAPTERS/_eval_stage" \
      --out-name "$out" \
      --responses-subdir "$resp" \
      --max-tokens 2048 2>&1 | tail -8
  echo
done

echo "=== aggregate + score ==="
$PROJECT_VENV score_v2_learning_curve.py 2>&1
