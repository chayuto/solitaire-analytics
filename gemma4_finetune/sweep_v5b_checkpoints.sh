#!/usr/bin/env bash
# v5 post-training checkpoint sweep. Clone of sweep_v3_checkpoints.sh: v5 shares
# the Gemma 4 TEXT base, so it reuses posttune_n20_gemma4_text_runner.py. Only the
# adapter source (adapters_v5b) and output names (posttune_v5b_*) differ.
#
# Run AFTER training produces adapters_v5b/000{0250,0500,0750,1000}_adapters.safetensors
# (train_v2.py --config lora_config_v5b.yaml). Pre-registration:
# docs/reports/20260530_v5_wononly_preregistration.md.

set -euo pipefail

VENV_PY=venv/bin/python
PROJECT_VENV=../.venv/bin/python
ADAPTERS=adapters_v5b

echo "=== v5b checkpoint sweep ==="
if [ ! -d "$ADAPTERS" ]; then
  echo "[error] $ADAPTERS/ does not exist. Train first:"
  echo "        $VENV_PY train_v2.py --config lora_config_v5b.yaml"
  exit 1
fi
ls "$ADAPTERS/"
echo

declare -a CHECKPOINTS=(
  "0000250_adapters.safetensors:posttune_v5b_at250.json:posttune_v5b_at250_responses"
  "0000500_adapters.safetensors:posttune_v5b_at500.json:posttune_v5b_at500_responses"
  "0000750_adapters.safetensors:posttune_v5b_at750.json:posttune_v5b_at750_responses"
  "0001000_adapters.safetensors:posttune_v5b_at1000.json:posttune_v5b_at1000_responses"
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

echo "=== aggregate + score (v5 vs v2) ==="
$PROJECT_VENV score_v5b_learning_curve.py 2>&1
