#!/usr/bin/env bash
# Create an isolated Python 3.12 venv and install the MLX fine-tuning stack.
# The system interpreter is Python 3.14 (too new for reliable MLX wheels) — this
# script deliberately uses 3.12.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for cand in python3.12 /opt/homebrew/bin/python3.12 /opt/homebrew/opt/python@3.12/bin/python3.12; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$cand"; break; fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Python 3.12 not found. Install it with:  brew install python@3.12" >&2
  echo "Then re-run, or pass PYTHON_BIN=/path/to/python3.12 $0" >&2
  exit 1
fi
echo "Using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

"$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Verifying MLX sees the Metal GPU..."
python - <<'PY'
import mlx.core as mx
d = mx.default_device()
print(f"  mlx default device: {d}")
assert "gpu" in str(d).lower(), "MLX is not using the GPU — check the install"
print("  OK — MLX + Metal ready.")
PY

echo
echo "Setup complete. Next:"
echo "  1. huggingface-cli login   (accept the Gemma 4 license on huggingface.co first)"
echo "  2. ./smoke_test.sh         (prove the M5 can run QLoRA at seq-len 2048)"
