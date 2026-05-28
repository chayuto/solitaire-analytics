#!/usr/bin/env python3
"""v2 training entry point: applies gemma4_text_patch then dispatches into
mlx_lm.lora.main() with the configured argv.

Why this wrapper exists: `mlx_lm.lora` is normally invoked as a CLI module
(`python -m mlx_lm.lora --config ...`), but that path calls `mlx_lm.load`
before our monkey-patch has a chance to apply. Importing the patch here
*first*, then forwarding into the same main() function, gives us the v1
mlx_lm.lora CLI behaviour with the gemma4_text loader bug worked around.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import gemma4_text_patch  # noqa: F401  -- patches mlx_lm.models.gemma4_text on import

import mlx_lm.lora as lora_mod


if __name__ == "__main__":
    # Default config if no args passed; otherwise forward whatever the caller
    # provided.
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--config", str(THIS_DIR / "lora_config_v2.yaml")]
    lora_mod.main()
