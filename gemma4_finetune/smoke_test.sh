#!/usr/bin/env bash
# Memory smoke test: run mlx-lm QLoRA for a short burst on SYNTHETIC data with the
# exact production config, and sample peak memory. Proves the M5 16GB can run the
# real job at seq-len 2048 BEFORE the pilot dataset exists.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d venv ]]; then echo "ERROR: run ./setup.sh first." >&2; exit 1; fi
# shellcheck disable=SC1091
source venv/bin/activate

ITERS="${ITERS:-50}"
echo "=== Generating synthetic 2048-token dataset ==="
python make_smoke_data.py --out smoke_data

echo
echo "=== Sampling memory in the background ==="
: > smoke_memory.log
( while true; do
    # Free + used pages -> rough RAM picture; plus swap.
    printf '%s ' "$(date +%H:%M:%S)" >> smoke_memory.log
    vm_stat | awk '/Pages active/{a=$3} /Pages wired/{w=$4} END{gsub(/\./,"",a);gsub(/\./,"",w);printf "active+wired=%.2fGB ",(a+w)*16384/1e9}' >> smoke_memory.log
    sysctl -n vm.swapusage >> smoke_memory.log
    sleep 3
  done ) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

echo
echo "=== Running mlx-lm LoRA: $ITERS iters on synthetic data ==="
echo "(close other apps — baseline machine already sits ~4.6GB into swap)"
set +e
mlx_lm.lora --config lora_config.yaml \
  --data smoke_data \
  --adapter-path smoke_data/adapters \
  --iters "$ITERS" \
  --save-every "$ITERS"
RC=$?
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
echo
echo "=== Smoke test exit code: $RC ==="
echo "Peak memory samples (smoke_memory.log):"
sort -t= -k2 -n smoke_memory.log | tail -3
echo
if [[ $RC -eq 0 ]]; then
  echo "PASS — training ran. Check peak 'active+wired' vs the §6 decision tree:"
  echo "  < ~11GB  -> config final"
  echo "  11-13GB  -> acceptable, consider num_layers=12"
  echo "  OOM      -> would have crashed; lower max_seq_length/num_layers"
else
  echo "FAIL — see output above. Reduce max_seq_length to 1536 and num_layers to 8,"
  echo "then re-run:  ITERS=$ITERS ./smoke_test.sh"
fi
