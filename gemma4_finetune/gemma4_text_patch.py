"""Monkey-patch for mlx_lm.models.gemma4_text loader bug.

Background: mlx-lm 0.31.3 ships a `gemma4_text.py` model class that
correctly skips allocating k_proj / v_proj / k_norm / v_norm modules for
the KV-shared layers (layer_idx >= num_hidden_layers - num_kv_shared_layers,
e.g. layers 15-34 for Gemma 4 E2B). At runtime those layers reuse the KV
projections of earlier layers via the model's `previous_kvs` mapping.

But every published Gemma 4 quant on mlx-community currently ships those
matrices for every layer anyway, and the model's existing `sanitize()` does
not strip them. The result is a hard load failure:

    ValueError: Received 140 parameters not in model.

This module extends `sanitize()` to drop the redundant weights. Import this
module before calling `mlx_lm.load(...)` against any Gemma 4 quant.

    import gemma4_text_patch  # noqa: F401 -- patches on import
    from mlx_lm import load
    model, tok = load("mlx-community/Gemma4-E2B-IT-Text-int4")

The patch is a no-op for non-Gemma-4 models (it only modifies the gemma4_text
Model class).

When mlx-lm upstream merges a fix for this, delete this file and remove the
import. The upstream bug is tracked at https://github.com/ml-explore/mlx-lm
(file an issue if none exists yet -- the fix is the 6 lines below).
"""
from __future__ import annotations

import re

import mlx_lm.models.gemma4_text as _g4t


_KV_SHARED_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)(\.|$)"
)
_orig_sanitize = _g4t.Model.sanitize


def _sanitize_with_kv_shared_strip(self, weights):
    sanitized = _orig_sanitize(self, weights)
    n_shared = self.args.num_kv_shared_layers
    if n_shared <= 0:
        return sanitized
    first_shared = self.args.num_hidden_layers - n_shared
    return {
        k: v
        for k, v in sanitized.items()
        if not (
            (m := _KV_SHARED_PATTERN.match(k)) is not None
            and int(m.group(1)) >= first_shared
        )
    }


_g4t.Model.sanitize = _sanitize_with_kv_shared_strip
