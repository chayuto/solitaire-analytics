#!/usr/bin/env python3
"""ORPO (odds-ratio preference optimization) loss for the mlx-lm LoRA trainer.

mlx_lm 0.31.3 ships no DPO/ORPO loss (``tuner.losses`` has only js_div / kl_div),
so this module supplies one that plugs into the custom loop in train_orpo.py. The
loop calls ``loss(model, *batch) -> (value, ntoks)`` exactly like ``default_loss``.

ORPO (Hong, Lee, Thorne 2024), matching TRL's ORPOTrainer:

    L_ORPO = L_SFT + beta * L_OR
    L_SFT  = mean token NLL on the CHOSEN response (= -Lp_chosen)
    L_OR   = -log sigmoid( logodds(chosen) - logodds(rejected) )
    Lp     = (1/|y|) * sum_t log P(y_t | x, y_<t)        # length-normalised
    logodds(y) = Lp(y) - log(1 - exp(Lp(y)))

Reference-model-free (no second frozen model -> fits 16 GB) and single-stage.

MEMORY (the 2026-06-02 smoke OOM-froze the laptop; a full-seq version lagged the
UI at ~10 GB peak): the loss only needs logits at the RESPONSE positions, but the
prompt is ~1500 of ~2000 tokens and is masked out. So we run the transformer body,
slice the hidden states to the response span, and only THEN apply the ~256k-wide
output head. That shrinks the dominant tensor ~8x. The result is numerically
identical because the head projection and Gemma's logit-softcap are per-position.

Why this is the right tool here: the won-game corpus is ~12% no-progress
shuffles, and imitation (SFT) has no gradient that says "do not loop". A pair
{chosen = the available progress move, rejected = the shuffle actually taken}
manufactures exactly that missing gradient.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma4_text import logit_softcap

# Lp is a length-normalised mean log-prob, so it is <= 0 and in practice lands in
# roughly [-2, -0.1] for these multi-hundred-token JSON responses. We still clamp
# it just below 0 so log(1 - exp(Lp)) can never hit log(0) = -inf on a pathological
# step (a near-perfectly-predicted response). The clamp never binds in normal play.
_LP_MAX = -1e-4


def _log1mexp(a: mx.array) -> mx.array:
    """Numerically stable log(1 - exp(a)) for a <= 0 (Machler 2012)."""
    return mx.where(
        a > -0.6931471805599453,
        mx.log(-mx.expm1(a)),
        mx.log1p(-mx.exp(a)),
    )


def _head(model, h):
    """Apply the output projection (tied embedding or lm_head) + Gemma softcap.

    Mirrors mlx_lm.models.gemma4_text.Model.__call__ exactly, but on whatever
    hidden states we pass -- which lets us project ONLY the response positions.
    """
    if getattr(model, "tie_word_embeddings", False):
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    cap = getattr(model, "final_logit_softcapping", None)
    if cap is not None:
        logits = logit_softcap(cap, logits)
    return logits


def _seq_logp(model, ids, offset, total):
    """Sum of log-probs over the response tokens of one B=1 sequence.

    ``offset`` / ``total`` are python ints (prompt length, last token index), so the
    response slice is a static slice -- enabling the head to run on ~250 positions
    instead of ~2000. Token-position p of the inputs predicts targets[p]=ids[p+1];
    a target is a response token when p in [offset-1, total-1].
    """
    inputs = ids[:, :-1]
    targets = ids[:, 1:]
    start = max(offset - 1, 0)
    end = max(total, start + 1)

    h = model.model(inputs)                 # [1, L, H] transformer body
    h = h[:, start:end, :]                  # response positions only -> cheap head
    logits = _head(model, h)                # [1, Lr, V]
    tgt = targets[:, start:end]

    token_logp = (-nn.losses.cross_entropy(logits, tgt)).astype(mx.float32)
    sum_logp = token_logp.sum(axis=1)       # [1]
    ntoks = mx.array([tgt.shape[1]], dtype=mx.float32)
    return sum_logp, ntoks


def orpo_terms(model, c_ids, c_off, c_total, j_ids, j_off, j_total, beta=0.1):
    """ORPO loss and components (B=1). Offsets/totals are python ints."""
    slp_c, n_c = _seq_logp(model, c_ids, c_off, c_total)
    slp_r, n_r = _seq_logp(model, j_ids, j_off, j_total)

    lp_c = mx.minimum(slp_c / mx.maximum(n_c, 1), _LP_MAX)   # length-norm mean logp
    lp_r = mx.minimum(slp_r / mx.maximum(n_r, 1), _LP_MAX)

    l_sft = -lp_c                                            # mean NLL on chosen
    log_odds = (lp_c - _log1mexp(lp_c)) - (lp_r - _log1mexp(lp_r))
    l_or = nn.softplus(-log_odds)                            # = -log sigmoid(log_odds)

    loss = (l_sft + beta * l_or).mean()
    return {
        "loss": loss,
        "l_sft": l_sft.mean(),
        "l_or": l_or.mean(),
        "log_odds": log_odds.mean(),
        "acc": (log_odds > 0).astype(mx.float32).mean(),   # preferred chosen over rejected
        "ntoks": n_c.sum(),
    }


# --------------------------------------------------------------------------- #
# Self-test: tiny model mirroring the real structure (.model body + lm_head),
# NO weights loaded, no crash risk.
# --------------------------------------------------------------------------- #
def _selftest():
    import math

    mx.random.seed(0)
    V, H = 16, 8

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(V, H)

        def __call__(self, x):
            return self.embed_tokens(x)          # [B, L, H]

    class Stub(nn.Module):
        """Mirrors gemma4_text.Model: .model returns hidden states, head projects."""

        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = nn.Linear(H, V, bias=False)
            self.tie_word_embeddings = False
            self.final_logit_softcapping = None

        def __call__(self, x):
            return _head(self, self.model(x))

    model = Stub()

    # 5-token sequence: 2-token prompt (offset=2), response = tokens 2..4, total=4.
    chosen = mx.array([[1, 2, 3, 4, 5]])
    rejected = mx.array([[1, 2, 9, 8, 7]])

    # (A) identical chosen/rejected -> log_odds == 0 -> L_OR == log 2.
    t_id = orpo_terms(model, chosen, 2, 4, chosen, 2, 4, beta=0.1)
    mx.eval(t_id["l_or"])
    assert abs(float(t_id["l_or"]) - math.log(2)) < 1e-5, float(t_id["l_or"])

    # (B) different rejected -> finite loss + finite gradient + correct ntoks.
    def loss_fn(m, *b):
        t = orpo_terms(m, *b, beta=0.1)
        return t["loss"], t["ntoks"]

    (val, ntoks), grad = nn.value_and_grad(model, loss_fn)(
        model, chosen, 2, 4, rejected, 2, 4
    )
    mx.eval(val, grad)
    leaves = [v for _, v in __import__("mlx").utils.tree_flatten(grad)]
    assert math.isfinite(float(val)), float(val)
    assert all(bool(mx.all(mx.isfinite(g))) for g in leaves), "non-finite gradient"
    assert int(ntoks) == 3, int(ntoks)

    # (C) log1mexp matches a direct computation in the safe range.
    a = mx.array([-0.1, -0.5, -1.0, -5.0])
    assert bool(mx.all(mx.abs(_log1mexp(a) - mx.log(1 - mx.exp(a))) < 1e-5))

    print("orpo_loss self-test PASSED")
    print(f"  (A) L_OR at log_odds=0 : {float(t_id['l_or']):.6f}  (expect {math.log(2):.6f})")
    print(f"  (B) loss={float(val):.4f}  ntoks={int(ntoks)}  grad finite=True")
    print(f"  (C) log1mexp stable")


if __name__ == "__main__":
    _selftest()
