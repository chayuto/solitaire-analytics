#!/usr/bin/env bash
# v4-A post-training checkpoint sweep. Mirrors sweep_v3_checkpoints.sh, except:
#   - adapter source is adapters_v4a/ (gemma-3n base, NOT Gemma 4)
#   - the runner is posttune_n20_v4a_runner.py (gemma-3n, no gemma4_text_patch)
#   - outputs go under baseline_n20_v4a/posttune_v4a_*
#   - max-tokens is 512 to match the v1.1 bench protocol (the pre-registered
#     baseline tier 3.15 was measured at 512; keep apples-to-apples vs v1.1).
#
# Run AFTER training produces adapters_v4a/000{0250,0500,0750,1000}_adapters.safetensors
# (train_v2.py --config lora_config_v4a.yaml). Pre-registration:
# docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md section 3.

set -euo pipefail

VENV_PY=venv/bin/python
PROJECT_VENV=../.venv/bin/python
ADAPTERS=adapters_v4a

echo "=== v4-A checkpoint sweep ==="
if [ ! -d "$ADAPTERS" ]; then
  echo "[error] $ADAPTERS/ does not exist. Train first:"
  echo "        $VENV_PY train_v2.py --config lora_config_v4a.yaml"
  exit 1
fi
ls "$ADAPTERS/"
echo

declare -a CHECKPOINTS=(
  "0000250_adapters.safetensors:posttune_v4a_at250.json:posttune_v4a_at250_responses"
  "0000500_adapters.safetensors:posttune_v4a_at500.json:posttune_v4a_at500_responses"
  "0000750_adapters.safetensors:posttune_v4a_at750.json:posttune_v4a_at750_responses"
  "0001000_adapters.safetensors:posttune_v4a_at1000.json:posttune_v4a_at1000_responses"
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
  $VENV_PY posttune_n20_v4a_runner.py \
      --adapter-path "$ADAPTERS/_eval_stage" \
      --out-name "$out" \
      --responses-subdir "$resp" \
      --max-tokens 512 2>&1 | tail -8
  echo
done

echo "=== aggregate + score (v4-A vs v1.1) ==="
$PROJECT_VENV score_v4a_learning_curve.py 2>&1
