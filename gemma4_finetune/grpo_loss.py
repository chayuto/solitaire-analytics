#!/usr/bin/env python3
"""GRPO / RLOO policy-gradient loss for the mlx-lm LoRA trainer (16 GB-safe).

The on-policy pivot's SFT-on-self-wins line closed at PARITY with volume across
every config (STaR x4/x1/diverse all <= volume, see the 2026-07-01 window). SFT
only ever REINFORCES winning trajectories; it has no gradient that says "do LESS
of what the losing rollouts did". This module supplies that missing gradient: an
advantage-weighted REINFORCE loss (the GRPO objective minus the PPO clip/KL,
which only matter when rollouts are reused across epochs -- we do a single epoch
on freshly-sampled, on-policy rollouts, so plain REINFORCE with a group baseline
is exact and unbiased).

The reward is the game OUTCOME (final foundation cards / 52); the baseline is the
per-deck group mean over the N rollouts of that deck; the advantage
A_i = (R_i - mean)/std is SHARED by every decision-turn of rollout i (standard
outcome-supervised credit assignment -- the whole trajectory gets one advantage,
exactly as DeepSeek-R1 GRPO scores a whole generation once). A losing rollout's
turns carry NEGATIVE advantage, so the update pushes the policy away from them.

Because the training examples are full-game decision turns, they span the entire
action space (draws, foundations, tableau moves), which structurally avoids the
ORPO draw-starvation failure ([[orpo-loop-penalty-next-step]]: preference pairs
that never contained draws collapsed the draw rate to 0.5%).

Per example (one decision turn, B=1):

    logp    = (1/|y|) * sum_t log P(y_t | x, y_<t)     # length-normalised mean
    L       = -advantage * logp                        # policy gradient

Minimising L for advantage>0 raises logp (reinforce); for advantage<0 it lowers
logp (suppress). train_grpo accumulates the gradient over a minibatch of turns
and steps once, so the update approximates the batch policy gradient rather than
a noisy per-turn SGD step.

MEMORY: reuses orpo_loss's response-only slice-trick verbatim (_seq_logp runs the
transformer body, slices hidden states to the response span, and only THEN
applies the ~256k-wide output head), which is what keeps the peak ~8x below a
full-sequence head projection and fits the 16 GB laptop.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Reuse the exact, already-tested slice-trick log-prob machinery from ORPO so the
# response masking / head projection / Gemma softcap are byte-for-byte identical
# to the SFT and ORPO paths (no risk of a second, subtly different implementation).
from orpo_loss import _seq_logp, _head  # noqa: F401  (_head re-exported for tests)

# logp is a length-normalised mean log-prob (<= 0). Clamp just below 0 for parity
# with orpo_loss (a near-perfectly-predicted response would otherwise give exactly
# 0 and, in a KL variant, log(0)); the clamp never binds in normal play.
_LP_MAX = -1e-4


def mean_logp(model, ids, offset, total) -> mx.array:
    """Length-normalised mean log-prob over the response tokens of one B=1
    sequence. offset/total are python ints (prompt length, last target index)."""
    slp, n = _seq_logp(model, ids, offset, total)
    return mx.minimum(slp / mx.maximum(n, 1), _LP_MAX)


def grpo_terms(model, ids, offset, total, advantage, ref_logp=None, tr_beta=0.0):
    """Advantage-weighted REINFORCE loss + components for one decision turn (B=1).

    advantage is a python float (the rollout's group-relative advantage, shared
    by every turn of that rollout). Offsets/totals are python ints so the loss
    statically slices the response span before the output head.

    Trust region (ref_logp + tr_beta): iteration-0 with NO trust region collapsed
    (logp -0.5 -> -23, all parse_failure) because the loss -A*logp is unbounded
    below for A<0 -- negative-advantage turns drive their completion logp toward
    -inf with nothing to stop them, and the shared JSON/reasoning tokens collapse
    with them. Run-1's grad norm stayed small (~0.3), so this is a slow DRIFT, not
    a spike -- gradient clipping does not bind. The fix is a ONE-SIDED downward
    penalty: whenever the current logp drops BELOW the reference (volume) logp for
    this turn, add tr_beta * (ref_logp - logp)^2. It opposes exactly the collapse
    direction while leaving positive learning (logp rising above ref) unpenalised.
    ref_logp is the length-normalised mean logp of this turn's completion under
    the volume policy, cached once before training (the policy starts AT volume).
    """
    slp, n = _seq_logp(model, ids, offset, total)          # ONE forward (was two)
    lp = mx.minimum(slp / mx.maximum(n, 1), _LP_MAX)       # length-norm mean logp
    loss = -advantage * lp                                 # [1], policy gradient
    if ref_logp is not None and tr_beta > 0:
        deficit = nn.relu(mx.array(float(ref_logp)) - lp)  # >0 only below reference
        loss = loss + tr_beta * deficit * deficit
    return {
        "loss": loss.mean(),
        "logp": lp.mean(),
        "advantage": mx.array(float(advantage)),
        "ntoks": n.sum(),
    }


# --------------------------------------------------------------------------- #
# Self-test: a tiny stub model mirroring the real structure (.model body +
# lm_head), NO weights loaded. Proves the gradient SIGN end-to-end: after a few
# AdamW steps, a +advantage example's completion logprob RISES and a -advantage
# example's FALLS. This is the one correctness gate that matters for an RL loss.
# --------------------------------------------------------------------------- #
def _selftest():
    import math
    import mlx.optimizers as optim

    mx.random.seed(0)
    V, H = 32, 8

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(V, H)

        def __call__(self, x):
            return self.embed_tokens(x)

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = nn.Linear(H, V, bias=False)
            self.tie_word_embeddings = False
            self.final_logit_softcapping = None

        def __call__(self, x):
            return _head(self, self.model(x))

    model = Stub()

    # Shared 2-token prompt (offset=2). DISJOINT completions so reinforcing one
    # and suppressing the other do not fight over the same token logits:
    #   good : prompt [1,2] + completion [3,4,5]      advantage = +1.0
    #   bad  : prompt [1,2] + completion [6,7,8]      advantage = -1.0
    good = mx.array([[1, 2, 3, 4, 5]])
    bad = mx.array([[1, 2, 6, 7, 8]])
    G = (good, 2, 4, +1.0)   # (ids, offset, total, advantage)
    B = (bad, 2, 4, -1.0)

    def lp_of(sample):
        ids, off, tot, _ = sample
        v = mean_logp(model, ids, off, tot)
        mx.eval(v)
        return float(v)

    lp_good_0, lp_bad_0 = lp_of(G), lp_of(B)

    def loss_fn(m, ids, off, tot, adv):
        t = grpo_terms(m, ids, off, tot, adv)
        return t["loss"], t["ntoks"]

    lvg = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=0.05)

    from mlx.utils import tree_map
    for _ in range(40):
        # Accumulate the gradient over the 2-example minibatch, then step once
        # (batch policy gradient, not per-example SGD).
        grads = None
        for (ids, off, tot, adv) in (G, B):
            (val, _nt), g = lvg(model, ids, off, tot, adv)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
        grads = tree_map(lambda a: a / 2.0, grads)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)

    lp_good_1, lp_bad_1 = lp_of(G), lp_of(B)

    assert math.isfinite(lp_good_1) and math.isfinite(lp_bad_1)
    assert lp_good_1 > lp_good_0 + 1e-3, f"+adv logp did not rise: {lp_good_0}->{lp_good_1}"
    assert lp_bad_1 < lp_bad_0 - 1e-3, f"-adv logp did not fall: {lp_bad_0}->{lp_bad_1}"

    # advantage 0 -> zero loss and zero gradient (a no-signal group contributes
    # nothing, e.g. the 5/5-win deck 9000237).
    (val0, _n0), g0 = lvg(model, good, 2, 4, 0.0)
    mx.eval(val0)
    leaves = [v for _, v in __import__("mlx").utils.tree_flatten(g0)]
    assert abs(float(val0)) < 1e-9, float(val0)
    assert all(float(mx.abs(v).max()) < 1e-9 for v in leaves), "adv=0 gave nonzero grad"

    # Trust region: repeat the -advantage suppression from a FRESH model, once
    # WITHOUT and once WITH a downward trust region anchored at the start logp.
    # Without it, logp runs away (as above, to ~ -1e3). With a strong tr_beta it
    # must NOT collapse far below the reference -- that is the whole anti-collapse
    # guarantee the real run relies on.
    def suppress_bad(tr_beta):
        mx.random.seed(0)
        m = Stub()
        ref = float(mean_logp(m, bad, 2, 4))  # reference = start logp under fresh m
        def lf(mm, ids, off, tot, adv):
            t = grpo_terms(mm, ids, off, tot, adv, ref_logp=ref, tr_beta=tr_beta)
            return t["loss"], t["ntoks"]
        g = nn.value_and_grad(m, lf)
        o = optim.AdamW(learning_rate=0.05)
        for _ in range(40):
            (v, _n), gr = g(m, bad, 2, 4, -1.0)
            o.update(m, gr)
            mx.eval(m.trainable_parameters(), o.state)
        end = float(mean_logp(m, bad, 2, 4))
        return ref, end
    ref_no, end_no = suppress_bad(0.0)      # no trust region -> runaway
    ref_tr, end_tr = suppress_bad(5.0)      # strong trust region -> bounded
    assert end_no < ref_no - 50, f"no-TR should run away, got {ref_no}->{end_no}"
    assert end_tr > ref_tr - 2.0, f"TR should bound drift, got {ref_tr}->{end_tr}"

    print("grpo_loss self-test PASSED")
    print(f"  +adv completion logp: {lp_good_0:.4f} -> {lp_good_1:.4f}  (rose)")
    print(f"  -adv completion logp: {lp_bad_0:.4f} -> {lp_bad_1:.4f}  (fell)")
    print(f"  adv=0 loss={float(val0):.2e}, grad=0  (no-signal group is a no-op)")
    print(f"  -adv WITHOUT trust region: {ref_no:.3f} -> {end_no:.1f}  (runs away)")
    print(f"  -adv WITH  trust region:   {ref_tr:.3f} -> {end_tr:.3f}  (bounded near ref)")


if __name__ == "__main__":
    _selftest()
