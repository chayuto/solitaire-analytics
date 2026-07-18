# External research 4 of 5: round-3 recipe inputs (2026-07-18)

Strand 4 of the sequential external research pass requested 2026-07-17.
Method: a single verification agent fetched each lead from the interrupted
2026-07-17 deep-research sweep and confirmed or corrected the
characterizations against primary sources (9 fetches, 3 searches, no
usage-limit errors). Companions: strands 1 to 3 and 5 (same date) and
`20260718_external_strategy_stresstest.md`. Bears on: the round-3 design
that only exists if the pre-registered round-2 gate PASSES
(`20260705_grpo_round2_preregistration.md`); also produces one mechanistic
hypothesis for why round moves have been parity-sized (signal dilution).

Report body below is the verification agent's output, verbatim except one
editorial addition: the parenthetical note in Round-3 implications item 5
cross-referencing the internal gamma=1 shaping no-op result was added by the
operator-side session, not the agent.

---

## Verdict summary

The mid-2026 literature partially supports and partially challenges the
round-2 recipe. SUPPORTED: group size N=8 (default in the Demystifying
recipe and in small-GPU GRPO practice), zero-variance group skip (ProRL v2
"dynamic sampling" discards uniform-reward groups), and reference-anchored
stabilization without per-token KL (ProRL v2 keeps a KL-to-reference but
must reset the reference every 200-500 steps to keep learning; our
per-round re-freeze at ~200 steps/round already implements that cadence, and
the multi-turn agent recipe paper drops KL entirely). CHALLENGED: smearing
one scalar advantage across 100-250 mostly-routine decisions is now a named
failure mode: "Drowning in Routine" derives that trajectory-level
estimators' signal-to-noise degrades as decision density rho falls (SNR
proportional to rho^(-1/2)), and our episodes are extreme low-rho; three
separate 2026 lines (Turn-PPO, GAGPO, GTPO) all move credit below
trajectory level for exactly this reason. The round-3 sketch of per-token
KL is weakly supported: the one multi-turn recipe paper that tests it
removes KL and widens clipping instead. The single highest-value round-3
change is critic-free step-level credit via a GAGPO-style state-matched
grouped value proxy (validated at 1.5B, terminal-only reward, exact state
matching is natural in our deterministic engine), with a cheap fallback of
masking or down-weighting routine forced-draw turns to raise effective
decision density; second is replacing fixed uniform N=8 with STEP-style
success-rate-aware rollout allocation, which subsumes the sketched
re-banded deck pools.

## Source-by-source findings

1. Turn-PPO (arXiv 2512.17008). VERIFIED. "Turn-PPO: Turn-Level Advantage
Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs", Li, Zhou,
Meng, Vadera, Li, Li; v1 18 Dec 2025, v2 23 Jan 2026; arXiv listing says
EACL 2026 (the lead's "Findings of EACL" anthology page was not fetched, so
main-vs-Findings is unconfirmed; consistent either way). SHOWS (abstract
level): proposes PPO on a turn-level MDP instead of token-level; evaluated
on WebShop and Sokoban, with and without long reasoning. CLAIMS: "directly
applying the widely used Group Relative Policy Optimization (GRPO) algorithm
to multi-turn tasks exposes notable limitations, particularly in scenarios
requiring long-horizon reasoning." Model sizes, quantitative results, and
critic memory cost are NOT in the abstract; full text NOT-FETCHED (budget).
Bearing on round 3: confirms peer-reviewed pressure against pure
group-relative trajectory advantages in multi-turn settings, but its fix
needs a learned critic, the costliest path at laptop scale.

2. GAGPO (arXiv 2605.13217). VERIFIED, preprint (submitted 13 May 2026,
cs.CL/cs.AI/cs.LG, no venue). SHOWS (full text fetched): critic-free;
builds a "non-parametric grouped value proxy from sampled rollouts" by
exact state matching across the rollout group: "For each state s, the
corresponding step group G(s)={(i,t) | s_t^(i)=s} gathers all occurrences
of s across the rollout group", with V-bar(s) = mean discounted return over
occurrences; then TD residual delta_t = r_t + gamma V-bar(s_t+1) -
V-bar(s_t) and GAE recursion A_t = delta_t + gamma lambda A_t+1,
gamma=0.95, lambda=0.8. Terminal-only rewards. Qwen2.5-1.5B and 7B.
Numbers: WebShop at 1.5B: 88.6 score / 78.1% success vs GRPO 80.5 / 66.4
(+11.7 success points); ALFWorld at 7B: 95.6% vs GiGPO 88.8%. Honest
drop-in assessment: needs nothing beyond rollout rewards plus a canonical
state key, which our deterministic engine provides exactly (same-deck N=8
rollouts share opening states, and Klondike transpositions and recycle
loops create genuine revisits); on unshared (singleton) states the TD error
collapses toward zero, so credit concentrates at branch points, which is
the desired behavior; the caveats are unknown shared-state density over
100-250 decisions and that gamma=0.95 was tuned for ~30-50 step horizons
(0.95^150 is about 4.5e-4, so we would need gamma near 0.99-1.0). Bearing
on round 3: the most plausible principled drop-in, at the closest scale
match (1.5B) in the whole set.

3. Demystifying RL for Long-Horizon Tool-Using Agents (arXiv 2603.21972).
VERIFIED, preprint (23 Mar 2026, code "Agent-STAR"). SHOWS (full text
fetched): empirical study on TravelPlanner across reward shaping, scale,
data, algorithm, environment stability. KL: "We remove the KL penalty and
increase the clipping bound epsilon_high to encourage broader exploration"
(KL-free plus clip-higher). Algorithms: "at the 7B scale, GRPO achieves the
highest success rate of 62.8%, outperforming both DAPO and ARPO";
exploration-heavy variants help below 7B. Group size G=8 default. Scale:
"smaller models benefit from staged rewards and enhanced exploration,
whereas larger models converge efficiently with simpler dense rewards";
smaller means 1.5B-3B, with dense-to-sparse staged reward curriculum. Data:
~1K training samples of balanced difficulty optimal; overly dense rewards
hurt OOD; environment failure rate of 10% noticeably degrades training.
Bearing on round 3: argues against adding per-token KL, for keeping G=8,
for a staged (denser early) reward at 2B scale, and flags our 8-deck prompt
pool as far below the ~1K balanced-difficulty guidance.

4. Drowning in Routine (arXiv 2606.22164). VERIFIED, but note evidence
grade: 2-author workshop paper (FAGEN Workshop, ICML 2026), submitted 20
Jun 2026. SHOWS: frames long-horizon credit failure as decision density rho
(fraction of consequential actions), not raw length; derives SNR of
trajectory-level estimators degrading as rho^(-1/2) when critic error is
controlled, and recovers the predicted scaling in controlled experiments
with R^2 = 0.999; model sizes unspecified in the abstract (likely
synthetic/controlled, not an LLM benchmark win). Quote: "Routine turns
create signal dilution: they add gradient variance to trajectory-level
estimators such as GRPO without adding expected signal." It is diagnostic,
not prescriptive: trajectory methods suffice at high rho, turn-level credit
becomes necessary at low rho. Bearing on round 3: this is not merely
title-relevant, it names our exact weakness (100-250 decisions dominated by
routine draws = very low rho), but it is a workshop-grade analytic result,
not a demonstrated method.

5. Credit-assignment survey (arXiv 2604.09459). VERIFIED as a survey,
single author (Chenchen Zhang), v1 10 Apr 2026, arXiv-only. SHOWS (abstract
level): covers 47 credit-assignment methods (2024-early 2026), taxonomized
by granularity (token, segment, step, turn, multi-agent) crossed with
methodology (Monte Carlo, TD, model-based, game-theoretic,
information-theoretic), and splits regimes into reasoning RL vs agentic RL
(100+ turns, 100K-1M tokens). Quote: "Reasoning CA is maturing around
process reward models and critic-free group comparison, while agentic CA is
driving genuinely new approaches." The abstract does NOT name
MT-GRPO/Tree-GRPO and says nothing about small-scale or few-prompt regimes;
full text NOT-FETCHED. CORRECTION to the lead: the survey lead's ID
"Tree-GRPO arXiv 2511.14846" is wrong; 2511.14846 is GTPO (see below).
Bearing on round 3: useful taxonomy confirming the field's direction
(critic-free group comparison for reasoning, new turn/step-level machinery
for agentic), but weak as an authority (non-peer-reviewed, single author).

6. STEP (aclanthology.org/2026.findings-acl.1532). VERIFIED: "STEP:
Success-Rate-Aware Trajectory-Efficient Policy Optimization", Chen, Liu,
Zhang, Gao, Luan, Liu; Findings of ACL 2026, pages 30681-30692. SHOWS:
instead of uniform rollouts per task, STEP "dynamically allocates sampling
based on per-task success rates" using a smoothed success-rate record to
prioritize harder tasks, plus success-rate-weighted advantages and
step-level augmentation, on OSWorld and AndroidWorld. CLAIMS faster
convergence and better generalization at the same sampling budget vs
trajectory-level and step-level GRPO variants; specific numbers were not in
the fetched metadata. Notable quote: "Step-level variants (e.g., GIGPO)
mitigate some interaction-cost constraints by decomposing trajectories, yet
they retain GRPO's sampling imbalance." Bearing on round 3: a direct
upgrade path for our fixed N=8 on 8 success-band decks; a smoothed per-deck
success rate driving rollout allocation is the continuous, self-tuning
version of the sketched re-banded pools.

7. NVIDIA ProRL v2 blog (developer.nvidia.com/blog/
scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/).
VERIFIED, vendor-grade. SHOWS: DeepSeek-R1-Distill-Qwen-1.5B trained 3,000
RL steps; KL-regularized objective L_PPO - beta KL(pi_theta || pi_ref) with
asymmetric clipping (eps_low 0.20, eps_high 0.28), scheduled cosine length
penalty, and dynamic sampling (discard prompts with uniform reward groups).
Reset rule per the blog: "Every 200-500 RL steps (or upon KL spikes/stalled
validation), the reference policy pi_ref is reset to the current policy,
optimizer state is not cleared." Rationale: "By regularly resetting the
reference policy, the model avoids being constrained by outdated guidance,
allowing it to continue learning effectively." Minor discrepancy flag: the
ProRL v1 paper and some secondary summaries describe hard resets of
reference AND optimizer; the blog text as fetched says optimizer state
persists in v2. Single-turn RLVR, not multi-turn agentic. Bearing on round
3: the lead's claimed 200-500 step schedule is CONFIRMED; at ~200
steps/round with a re-frozen reference each round, our recipe already
matches this cadence; formalize the KL-spike/stall trigger rather than
adding new regularizers.

Also answered (a): MT-GRPO is arXiv 2505.11821, "Reinforcing Multi-Turn
Reasoning in LLM Agents via Turn-Level Credit Assignment" (v2 retitled
"... via Turn-Level Reward Design", Zeng et al.): augments GRPO with
turn-level (tool-verifier) rewards blended with outcome rewards so
advantages differ by turn; no peer-review venue surfaced in search (arXiv;
code SiliangZeng/Multi-Turn-RL-Agent). Tree-GRPO is arXiv 2509.21240 "Tree
Search for LLM Agent Reinforcement Learning" (AMAP-ML), accepted ICLR 2026
per its official repo: replaces independent chain rollouts with tree-search
rollouts sharing prefixes, deriving process-level signal by comparing
sibling branches' outcomes (intra-tree plus inter-tree group-relative
advantages). The lead's 2511.14846 is actually GTPO, "Empowering Multi-Turn
Tool-Integrated Agentic Reasoning with Group Turn Policy Optimization":
turn-level reward assignment, return-based (normalized discounted return)
advantages, self-supervised reward shaping; venue not confirmed.

## Round-3 recipe implications

Ranked for a 2B LoRA, MLX laptop, ~200-step windows:

1. State-matched step-level advantages (GAGPO-style; GiGPO is the prior art
   both STEP and GAGPO benchmark). Critic-free: yes. Evidence: preprint,
   but with the best scale match (1.5B, +11.7 WebShop success points over
   GRPO, terminal-only reward). Cost: near-zero memory (a hash map from
   canonical state key to mean discounted return across the N=8 same-deck
   rollouts), moderate code change in the extract step (per-turn advantages
   instead of one smeared scalar; the trainer already consumes per-turn
   rows). Required adaptations: gamma near 0.99-1.0 for 100-250 step
   episodes, and measure shared-state density first; if the 8 rollouts
   rarely intersect after the opening, credit still concentrates correctly
   at early branch points but adds little late-game signal.
2. Cheap dilution countermeasure if (1) is deferred: raise effective
   decision density by masking or down-weighting routine turns in the loss
   (forced moves, single-legal-action draws), keeping the trajectory scalar
   only on decision turns. Critic-free: yes. Evidence: workshop-grade but
   analytic (SNR proportional to rho^(-1/2)); this is an intervention the
   paper implies rather than tests. Cost: trivial (a per-turn weight in
   extraction); lowest-risk change with a mechanistic justification for why
   round 2 moves are parity-sized.
3. STEP-style success-rate-aware rollout allocation replacing fixed N=8
   uniform. Critic-free: yes (pure scheduling). Evidence: Findings of ACL
   2026, but gains reported on GUI agents without numbers in the fetched
   metadata. Cost: trivial (the resumable rollout runner allocates next
   rollouts by smoothed per-deck success rate); subsumes and improves the
   sketched re-banded deck pools, and naturally stops burning rollouts on
   always-win and never-win decks (complementing the zero-variance skip,
   which currently discovers dead groups only after paying for them).
4. KL/reference handling: do NOT add per-token KL by default. The
   multi-turn agent recipe (source 3) removes KL and widens clipping; ProRL
   v2 (single-turn, 1.5B) keeps a KL but only stays healthy via 200-500
   step reference resets, which our per-round re-freeze already provides.
   Round-3 version: keep the one-sided TR, add ProRL v2's trigger
   discipline (reset or stop on KL spike / stalled validation), and treat
   per-token KL as a fallback only if the TR proves too loose at turn
   granularity. Critic-free: yes. Evidence: vendor-grade plus preprint,
   pointing the same direction.
5. Staged reward curriculum for small models (dense earlier, sparse later),
   per source 3's 1.5B-3B finding: e.g. early-round per-foundation shaping
   decaying toward win-only. Critic-free: yes. Cost: reward function only.
   Caution from the same paper: overly dense rewards hurt OOD
   generalization, and our fresh-deck generalization is the metric that has
   repeatedly falsified in-dist gains. (Note: the 20260704 internal pass
   proved gamma=1 potential shaping is a telescoping no-op in our setup;
   any staged variant must avoid the farmable non-potential forms already
   ruled out there.)
6. Tree-GRPO-style tree rollouts (ICLR 2026): shared-prefix branching would
   both cut rollout compute and produce branch-point credit (in a
   deterministic engine, shared prefix equals shared state, converging with
   option 1). Cost: a rollout-harness rewrite, the largest engineering
   item; defer unless round 3 graduates.
7. Turn-PPO learned critic (EACL 2026): strongest venue for the "trajectory
   GRPO is not enough" claim, but a critic/value head plus GAE is the most
   memory- and tuning-expensive option, with critic cost unquantified in
   the abstract; last resort at laptop scale.

## What round 2 already gets right/wrong per this evidence

- Right: N=8 group size (matches source 3's default and small-GPU practice,
  e.g. RobotxR1's G=8 at 3B).
- Right: zero-variance group skip (ProRL v2 dynamic sampling does the same
  for uniform-reward prompts).
- Right: reference-anchored stabilizer without per-token KL, re-frozen each
  ~200-step round (matches ProRL v2's 200-500 step reset cadence; and the
  one multi-turn recipe tested dropped KL entirely).
- Right (conditionally): critic-free group comparison is the field's mature
  default for the reasoning regime per the survey; nothing fetched says
  mean-only (no std division) is wrong.
- Wrong or under pressure: one scalar advantage smeared across 100-250
  mostly-routine decisions is the low-decision-density regime where
  trajectory-level estimators provably lose SNR (rho^(-1/2)); three
  independent 2026 method lines (Turn-PPO, GAGPO, GTPO) exist specifically
  to fix this.
- Wrong or under pressure: fixed uniform N per deck retains "GRPO's
  sampling imbalance" (STEP); rollout budget should follow per-deck success
  rate.
- Under pressure: an 8-deck prompt pool is one to two orders of magnitude
  below the ~1K balanced-difficulty task guidance (source 3); expect
  in-dist overfitting, consistent with this project's repeated
  in-dist-lift-no-transfer pattern.
- Unresolved: per-trajectory length normalization (equal gradient mass per
  trajectory) is not adjudicated by any fetched source; note the recipe
  borrows Dr.GRPO's mean-only advantage while keeping a length
  normalization Dr.GRPO itself removes, so this axis is a self-inconsistency
  to test, not a literature-settled choice.
- Unresolved: terminal-only foundation-count shaping vs staged/denser
  shaping at 2B (source 3 favors staged for 1.5B-3B but warns dense rewards
  hurt OOD).

## Searched-not-found

- GRPO/RLVR with LoRA specifically at 1-3B on consumer hardware with
  prescriptive step counts / group sizes (mid-2026 paper): NOT FOUND as
  such. Query: "GRPO LoRA reinforcement learning 1.5B 3B small model
  consumer GPU laptop MLX 2026 group size steps RLVR". Adjacent data points
  found: RobotxR1 (arXiv 2505.03238), Qwen2.5-3B GRPO via LoRA/Unsloth on
  RTX 4070 Ti/3090, ~11 GB VRAM, 750 steps, ~24 h, G=8; Unsloth docs,
  Qwen3-1.7B FP8 GRPO on 5 GB VRAM; Julien Simon (Medium, Apr 2026) states
  MLX still lacks GRPO ("if you need GRPO... you need CUDA"), confirming
  the project's homegrown MLX trainer has no off-the-shelf reference.
- MT-GRPO (2505.11821) peer-review venue: none surfaced in search; treating
  as arXiv preprint.
- NOT-FETCHED (budget, not absence): Turn-PPO full text (quantitative
  results, critic memory cost) and its aclanthology page (Findings vs main
  EACL); survey full text (whether it actually covers MT-GRPO/GTPO/
  Tree-GRPO by name and any small-scale guidance); STEP full text (numeric
  gains); GTPO full text.

One lead correction worth propagating to any round-3 doc: the prior pass's
"Tree-GRPO arXiv 2511.14846" conflated two papers; 2511.14846 is GTPO
(Group Turn Policy Optimization), and Tree-GRPO is arXiv 2509.21240 (ICLR
2026).
