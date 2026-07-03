# Deep research: the next best move (2026-07-04)

Three parallel research strands (2024-2026 literature + practitioner practice,
~100 searches/fetches total) pressure-testing the strategy doc's plan before
the next compute window: (A) RL stability for small-model RFT, (B) constrained
decoding on MLX, (C) reward design, curriculum, and inference-time compute as
an alternative lever. Sources were retrieved this session; preprints and
vendor claims are flagged. Companion docs:
`20260703_long_term_project_strategy.md` (the plan this revises) and
`20260703_experiment_retrospective.md` (the internal evidence).

## 0. TL;DR: what changes, what holds

**Holds: the run-2 eval is still the next move.** Nothing found overturns the
pre-registered gate; the research sharpens what happens after it, in both
branches.

**The big validations:**
- Run 1's collapse is a KNOWN, named failure mode ("squeezing effect";
  negative-advantage collapse with format destruction is documented in 2026
  agent-RL work). Our diagnosis (unbounded negative branch, slow drift, not
  spikes) matches the literature's mechanism.
- Our one-sided trust region is a sound novel combination of three published
  ideas (DisCO's one-sided squared-hinge trust region + GSPO's sequence-level
  ratio + asymmetric negative handling). Keep it; add nothing heavier on top.
- int4 + LoRA rank-16 + single-process MLX is squarely known-good practice
  (QLoRA-style GRPO is mainstream; quantization noise can even HELP
  exploration; the one dangerous pitfall class, rollout/trainer numerics
  mismatch, is structurally impossible in our single-process loop).

**The big revisions:**
1. **Round-2 trainer upgrades** (cheap, evidence-backed): drop the std
   division from the advantage (Dr.GRPO), skip zero-variance groups (DAPO),
   normalize per-trajectory turn count (long losing games currently out-vote
   short wins ~2.5x), N=5 -> 8 rollouts/deck, success-band deck curriculum
   (pass@5 in ~20-80%), plus entropy/TR-bind instrumentation and a
   negative-downweight ablation arm.
2. **Dense per-step reward shaping is DROPPED, permanently.** Under gamma=1
   trajectory-level advantage, foundation-delta shaping telescopes to exactly
   our current reward (a mathematical no-op), and any rectified progress
   variant would subsidize the doom-loop (farmable). The legitimate successor
   if RFT plateaus is GiGPO-style anchor-state credit, not shaping.
3. **The fork-independent CPU track expands** from "constrained decoding" to
   a three-part inference-layer program with confirmed feasibility and strong
   Klondike-specific precedent: (a) constrained JSON decoding (Outlines
   JSONLogitsProcessor into mlx-lm's native `logits_processors` hook,
   verified in our installed 0.31.3), (b) an endgame exact-solver oracle that
   deletes the false-resign loss mode outright, (c) per-move solver
   veto/forcing over determinized worlds + small per-move best-of-N. The 2009
   Bjarnason et al. result lifts Klondike win-rate 13% -> 23-37% with exactly
   this class of search, and 2025 work (PokeChamp) shows an 8B LLM + shallow
   search beating frontier-model agents without any training.
4. **Publication finding:** no published LLM-plays-Klondike work exists (both
   major game benchmarks skip card games). The project is in genuinely
   unpublished territory, and our collapse-and-fix story sits inside a hot
   2025-26 literature thread.

## 1. Strand A: RL stability (the trainer is on solid ground, with upgrades)

**A1. The collapse is known.** The "squeezing effect" (Ren & Sutherland,
arXiv 2407.10490, ICLR 2025): negative gradients on low-probability responses
squeeze probability mass onto the argmax and monotonically decay everything
else, degrading output validity before reward moves; signature = slow drift
with small grad norms. A 2026 search-agent paper (arXiv 2604.18235) documents
GRPO training collapse driven by accumulated negative-advantage sequences with
exactly our phenomenology (format destruction first); its fix (CalibAdv)
downscales excessive negative advantages. NTHR (arXiv 2505.18830) shows GRPO's
uniform per-token penalty on bad rollouts suppresses correct-response
likelihood ("lazy likelihood displacement"), worst when one scalar advantage
is smeared across a whole trajectory, which is precisely our construction.

**A2. Why "KL-free GRPO works" did not save run 1.** Public KL-free recipes
(TRL default beta=0, Dr.GRPO, DAPO) all RETAIN the PPO clipped surrogate
against pi_old, itself a per-update trust region; pure -A*logp with no KL, no
ratio, no clip is unbounded below for A<0 (arXiv 2509.24203 analyzes
group-relative REINFORCE as secretly off-policy, needing exactly one update
regularizer). Reading: our one-sided TR plays the role the clip plays
elsewhere. Keep exactly ONE such mechanism.

**A3. Our TR has published cousins.** DisCO (arXiv 2505.12366) replaces
clipping with a one-sided squared-hinge trust-region penalty (+7% over GRPO at
1.5B); GSPO (arXiv 2507.18071, used for Qwen3) defines importance at the
sequence-mean-logp level, the exact quantity our TR penalizes; TOPR/AsymRE
(arXiv 2503.14286, 2506.20520) validate treating negative advantages
asymmetrically. No paper does literally our construction (one-sided quadratic
on sequence-mean logp vs a FROZEN reference with cached logps): a sound novel
variant, worth naming in any write-up. One caveat (ProRL, arXiv 2505.24864):
frozen anchors eventually dominate on LONG runs; the fix is periodic
reference reset-to-snapshot. At 200-2000 steps we are fine.

**A4. Known GRPO biases that DO apply to us.**
- Std-normalization bias (Dr.GRPO, arXiv 2503.20783): dividing by group
  reward std inflates noise into full-scale advantages on low-variance
  groups. With N=5 and clustered fc rewards this is live. Fix: mean-only
  baseline (RLOO-style) or a std floor.
- Trajectory-length coupling: each turn is a separate loss term, so a
  250-turn loss contributes ~2.5x the gradient mass of a 100-turn win at
  equal advantage. Fix: divide each trajectory's turn losses by its turn
  count. (GiGPO, arXiv 2505.10978, formalizes the step-credit problem and is
  the later escalation: anchor-state regrouping, +12% over GRPO on ALFWorld.)
- Zero/near-zero-variance groups: DAPO (arXiv 2503.14476) filters all-same
  groups as zero-signal; our extractor already drops exact-zero std (deck
  9000237); extend to a variance floor.

**A5. Quantization and LoRA: keep as is.** Unsloth ships 4-bit GRPO as
mainstream; QeRL (arXiv 2510.11696) finds 4-bit noise RAISES policy entropy
and can help exploration; a quantized-RL study (arXiv 2511.15694) recommends
training in the serving precision (we do). LoRA-for-RL works at rank 1
(Thinking Machines "LoRA Without Regret"), so rank 16 is generous. The one
documented quantized-RL killer (sampler/trainer numerics mismatch across
engines) cannot occur in our single-process MLX loop: an argument FOR the
stack.

**A6. Hyperparameter anchors** (published practice, 1-8B LoRA GRPO):
group size 8-16 (we run 5; raise to 8); one optimizer pass per rollout batch;
LoRA RL learning rates cluster ~5e-6 to 1e-5 (full-FT 1e-6 x the 10x LoRA
rule). We ran 8e-5 and stayed healthy UNDER THE TR, but the literature band
suggests our LR is hot; if round 2 shows TR-bind saturation or entropy
collapse, halving toward 2e-5 is the first knob. Expect slow reward movement
(hundreds of steps) as normal.

**A7. Small prompt sets are viable.** One-shot RLVR (arXiv 2504.20571)
trains on a SINGLE prompt and doubles MATH500 at 1.5B; OpenAI RFT guidance
says "several dozen to a few hundred examples"; LIMR shows selection beats
scale. 12 decks is low-end-viable; the known risk profile (in-dist up, OOD
down) is EXACTLY our project's measured pattern, so the fresh-deck holdout
governance already in the strategy doc is the right defense.

## 2. Strand B: constrained decoding (feasible, ~0.5-1 day, do it)

**B1. The hook exists in our exact stack (verified locally).** Installed
mlx_lm 0.31.3 `generate_step` accepts
`logits_processors: List[Callable[[tokens, logits] -> logits]]`, forwarded
from `generate(**kwargs)`. No fork or engine change needed.

**B2. The production pattern to copy is LM Studio's mlx-engine.** It appends
Outlines' `JSONLogitsProcessor` (from `outlines.processors.structured`) into
mlx-lm's logits_processors, with a tiny tokenizer-hash shim. Outlines also
ships an official mlxlm backend (`outlines.from_mlxlm`), but the
processor-only route integrates with our existing harness loop unchanged.
Fallback ladder if Outlines fights us: outlines-core's `Guide` (~40 lines of
our own mx masking), xgrammar (fastest, torch bitmask glue), lm-format-
enforcer (pure Python). Hand-rolling a token FSM is REJECTED: BPE tokens span
structural boundaries and escape-aware string states, so a correct
implementation converges to rebuilding outlines-core against a 256k vocab.

**B3. Quality evidence favors constraining OUR case.** The "Let Me Speak
Freely" degradation claim was rebutted with fixed methodology (dottxt "Say
What You Mean": structured >= unstructured on all three tasks when the schema
is in the prompt and prose/reasoning fields come FIRST). Our LoRA was trained
on this exact 3-key layout with board_analysis first, so masks bind only at
failure points. Honest nuance: a grammar cannot prevent an early unescaped
quote; it converts that failure into valid-JSON-with-truncated-prose instead
of a dead game, which is the desired trade. 2026 caution papers (constraint
tax, latency 3.6-8.2x on some engines) say: measure tokens/sec, constrain
only what must be constrained, log mask-bind events (a free LoRA-health
metric).

**B4. Free wins to layer regardless:** prefill the assistant response with
the `{"board_analysis": "` scaffold (kills text-before-object), brace-depth
stop tracking (kills text-after-object), and `json_repair` as the outer
fallback so the processor can never make things worse than today.

**B5. Build recommendation (adopted):** Outlines JSONLogitsProcessor via the
mlx-engine pattern; schema constrains structure + integer move_index, leaves
prose strings free; pin the Outlines version; warm the compiled index at
startup; A/B on the iteration bench (volume+constrained vs volume) before
adopting into any eval that feeds a claim. Effort ~0.5-1 day, CPU-side except
the validation window.

## 3. Strand C: reward design, curriculum, and the inference-time lever

**C1. Dense shaping is a no-op or a hazard here (drop it).** Potential-based
shaping (Ng/Harada/Russell 1999) is the only optimality-preserving
densification, and with gamma=1 it telescopes to Phi(terminal)-Phi(initial):
foundation-count potential reproduces our existing trajectory reward EXACTLY
under same-deck group advantage. Non-potential progress rewards get farmed
(CoastRunners; Motif's NetHack agent eating hallucinogens to fake the goal,
arXiv 2310.00166); in Klondike the farm is foundation-tableau cycling, i.e.
our documented doom-loop phenotype, subsidized. If step-level credit is ever
needed: GiGPO anchor-state grouping (deterministic engine + same-deck groups
means identical board states genuinely recur across rollouts) or ProAct-style
MC values from cheap ENGINE playouts (arXiv 2602.05327, preprint), not
reward shaping.

**C2. Curriculum: the success-band is the highest-leverage free change.**
AdaRFT (arXiv 2504.05520): sampling problems near ~50% success rate gives up
to 2x fewer steps; DAPO's dynamic sampling exists because all-pass/all-fail
groups carry no gradient; RAGEN/StarPO (arXiv 2504.20073) names the
multi-turn instability ("Echo Trap": reward-variance collapse + entropy drop)
and prescribes variability-based trajectory filtering, diverse initial
states. Our deck analogue: build the rollout pool from decks with observed
pass@5 in ~20-80% (the corrective-harvest near-misses are exactly this),
resample as win rates move, keep fresh holdouts untouched. Caveat: evidence
is single-turn math RLVR; unproven at 200-decision horizons (flagged).

**C3. Inference-time compute has Klondike-specific, quantified precedent.**
Bjarnason/Fern/Tadepalli (ICAPS 2009, draw-3): greedy heuristic 13.0% ->
HOP (solve determinized worlds per move, argmax average) 23.0-27.2% -> UCT
34.4% -> Ensemble-UCT 37.0%. Named failure mode: "averaging over
clairvoyance" overvalues moves whose success requires knowing hidden cards;
the fix is using determinized solves as VETO/FORCE filters (a move losing in
all sampled worlds is pruned; winning in all worlds is forced), with the LLM
breaking ties: exactly an LLM-proposes/solver-adjudicates division of labor.
PokeChamp (arXiv 2503.04094, ICML 2025): Llama-3.1-8B inside minimax beats
the prior GPT-4o agent with NO training. Test-time-compute scaling (Snell
arXiv 2408.03314; Jones 2104.03113) quantifies train-vs-inference compute as
a smooth exchangeable frontier. Our unique asset: an EXACT solver and a free
legal-move engine; latency budget (advisor, seconds/move) permits 2-5 samples
+ CPU solver calls per move.
Highest-certainty sub-items: (i) endgame oracle: at low/zero faceDown hand
the position to the exact solver, which deletes the false-resign-at-fd0 loss
mode outright; (ii) per-move veto/force over K determinized worlds; (iii)
per-move best-of-N with engine checks (reveals, strict fd reduction).

**C4. What RFT can realistically buy.** Yue et al. (arXiv 2504.13837): RLVR
mostly converts pass@N coverage into pass@1; the base policy's sampling
envelope is the practical ceiling at hobby compute (the ProRL counter-example
needed ~16k H100-hours). Our numbers: pass@1 ~33%, pass@5 recovered 5/6
near-miss decks. The RFT prize is a fraction of that gap (even half lands
~40-45%, the lower edge of the theoretical ceiling); the inference-time stack
captures part of the same gap WITH CERTAINTY, immediately, and the two
compose. Our STaR-parity result is the literature's mechanism observed
independently: SFT reweights coverage; policy gradient sharpens it.

**C5. Prior art gap.** No published LLM-plays-Klondike work found (BALROG,
lmgame-Bench skip card games). Klondike AI prior art is solver/planning-based
(Solvitaire: 81.9% perfect-information winnability; ~37% draw-3 search lower
bound; ~43% expert human draw-3; draw-1 strong-bot claims ~52% unverified).
Our 33%-of-winnable at 2B on-device is publishable novelty; the external
baselines to cite are the planner numbers.

## 4. Synthesis: revisions to the strategy doc

| Strategy item | Revision | Basis |
|---|---|---|
| Run-2 eval gate | UNCHANGED, still next | nothing overturns it |
| Round-2 trainer | ADD: mean-only advantage (drop std div), variance-floor group filter, per-trajectory turn normalization, N=8, success-band deck pool, entropy + TR-bind logging, negative-downweight ablation arm; LR watch (8e-5 is hot vs the 5e-6..1e-5 band; TR held it, but halve first if instability) | A4, A6, C2 |
| Per-token KL escalation | KEEP as designed; add ProRL-style reference resets only for long runs | A2, A3 |
| Dense reward shaping | DELETE from the option space (no-op/hazard) | C1 |
| Step-level credit | GiGPO anchor-state = the plateau escalation (v2) | A4, C1 |
| Pillar 2 | EXPAND to inference-layer program: constrained decoding (build rec adopted, hook verified) + endgame solver oracle + per-move veto/force + small best-of-N; A/B each on the iteration bench | B1-B5, C3 |
| Distill-back | If the verified advisor out-plays the raw policy, harvest its games for the NEXT RFT round (ExIt pattern); never static SFT (matches our form-3 result) | C3, C4 |
| Pillar 4 | Publication case STRENGTHENED: unpublished territory + our collapse/fix aligns with the squeezing-effect literature thread | A1, A3, C5 |

## 5. The revised next-move stack (ordered)

1. **Run-2 eval** (next GPU window, ~2-3h): unchanged gate, unchanged
   branches. Before it, 30 CPU minutes: add entropy + TR-bind logging to
   train_grpo so round 2 is instrumented either way.
2. **Inference-layer track (CPU, no window needed): build in this order:**
   endgame solver oracle (hours, biggest certain win), constrained decoding
   (0.5-1 day, Outlines processor), per-move veto/force + best-of-N advisor
   mode (1-2 days). One GPU validation window A/Bs all three vs plain volume.
3. **If gate passes: round 2 with the upgraded trainer** (config table
   above) on a success-band deck pool, fresh rollouts from the new policy.
4. **If gate fails: the inference-layer track becomes the ship path** (it is
   the only lever with near-guaranteed win-rate gain), and the write-up
   proceeds with the closure story.
5. **Either way, the write-up now has an external frame**: unpublished
   territory, a named failure mode we hit and fixed with a novel TR variant,
   and planner baselines to cite.

## 6. Reliability notes

Verified locally: mlx_lm 0.31.3 logits_processors signature. Peer-reviewed
or widely replicated: squeezing effect (ICLR 2025), Dr.GRPO, DAPO, GSPO,
DisCO, VinePPO, GiGPO (NeurIPS 2025), Bjarnason (ICAPS 2009), PokeChamp
(ICML 2025), Yue et al., Ng shaping theorem, One-shot RLVR. Preprints
(flagged): CalibAdv 2604.18235, NTHR 2505.18830, ProAct 2602.05327, MCPO
2604.16972, off-policy analysis 2509.24203. Vendor/blog (flagged): Unsloth
guides, Thinking Machines LoRA post, dottxt rebuttal, OpenPipe ART numbers,
draw-1 ~52% bot claim (unverified app data). Not found (absence unverified):
any LLM-Klondike publication; any GRPO study at 100-250-decision horizons
with ~12 prompt groups; a published method identical to our one-sided
frozen-reference TR.
