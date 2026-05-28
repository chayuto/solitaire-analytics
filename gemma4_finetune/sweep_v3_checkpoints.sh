#!/usr/bin/env bash
# v3 post-training checkpoint sweep. Mirrors sweep_v2_checkpoints.sh except
# the adapter source is adapters_v3/ and outputs go under
# baseline_n20_gemma4_text/posttune_v3_*.

set -euo pipefail

VENV_PY=venv/bin/python
PROJECT_VENV=../.venv/bin/python
ADAPTERS=adapters_v3

echo "=== v3 checkpoint sweep ==="
ls $ADAPTERS/
echo

declare -a CHECKPOINTS=(
  "0000250_adapters.safetensors:posttune_v3_at250.json:posttune_v3_at250_responses"
  "0000500_adapters.safetensors:posttune_v3_at500.json:posttune_v3_at500_responses"
  "0000750_adapters.safetensors:posttune_v3_at750.json:posttune_v3_at750_responses"
  "0001000_adapters.safetensors:posttune_v3_at1000.json:posttune_v3_at1000_responses"
)

for entry in "${CHECKPOINTS[@]}"; do
  IFS=':' read -r ckpt out resp <<< "$entry"
  if [ ! -f "$ADAPTERS/$ckpt" ]; then
    echo "[skip] $ckpt missing"
    continue
  fi
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

echo "=== aggregate + score (v2 + v3) ==="
$PROJECT_VENV score_v3_learning_curve.py 2>&1
