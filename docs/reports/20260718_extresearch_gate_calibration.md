# External research 1 of 5: RFT round-2 gate calibration (2026-07-18)

Strand 1 of the sequential external research pass requested 2026-07-17
("support our strategy or otherwise"). Method: a single verification agent
fetched each lead surfaced by the interrupted 2026-07-17 deep-research sweep,
confirmed or corrected the search-phase characterization against the primary
source, and ran 2 targeted gap searches. 10 fetches, 2 searches, no
usage-limit errors. Companions: strands 2 to 5 (same date) and the synthesis
doc `20260718_external_strategy_stresstest.md`. Prior internal pass:
`20260704_next_move_deep_research.md`. The strategy under test:
`20260705_grpo_round2_preregistration.md` (7+/12 strict-win gate, hard stop).

Report body below is the verification agent's output, verbatim.

---

## Verdict summary

The round-2 gate bet is UNRESOLVED by external evidence, with an important
split: the DIRECTION of the bet is supported (multi-turn agentic RL is
precisely the regime where late-2025 to mid-2026 evidence says RLVR genuinely
expands capability rather than merely resharpening it, and where matched-data
SFT regresses, mirroring this project's own SFT-closed versus RFT-moving
history), but the RATE is unsupported (no fetched source measures
pass@k-to-greedy conversion inside a ~200-step, 8-prompt budget; every
quantified conversion precedent runs 400 to 2,000+ steps on hundreds to 136K
prompts). The single most decision-relevant finding is arXiv 2604.14877: in
compositional multi-turn agent tasks, "the RL agent's pass-curve pulls above
the base model's and the gap widens at large k rather than converging," and
"supervised fine-tuning regresses the boundary on the same compositional
tasks, isolating self-directed exploration as the causal factor." My prior
snippet for that paper was wrong (it claimed a base-overtakes-RL crossover;
the paper finds the opposite for agentic tasks), and the correction moves the
evidence toward RFT being the right lever while leaving the 200-step gate a
genuinely hard, roughly long-odds bet.

## Source-by-source findings

**1. arXiv 2510.04028, "The Debate on RLVR Reasoning Capability Boundary:
Shrinkage, Expansion, or Both? A Two-Stage Dynamic View" (Yao, Yu, Hu, Teng,
Cui, Zhou, Liu). Status: preprint, v1 submitted 2025-10-05; the arXiv page
shows NO venue; the "ICLR 2026 OpenReview" claim is NOT-FETCHED and stays
unverified.**
The snippet's core characterization is confirmed: both camps are "partially
valid" via a two-stage probability-mass dynamic. Stage 1 (exploitation): the
model reshuffles mass among already-sampled high- and low-reward tokens while
"the optimal token's probability remains largely unchanged";
"over-exploitation during the exploitation stage may lead to capability
boundary shrinkage." Stage 2 (exploration): entered only when high-reward
token probabilities saturate, at which point occasionally-sampled optimal
tokens can rise; "prolonged training into the exploration stage can promote
an expansion of the reasoning capability boundary." The abstract gives NO
quantitative stage markers (no step counts, no entropy thresholds); the
transition is defined by saturation, not steps.
Bearing on our gate: a 200-step LoRA round sits firmly in this framework's
exploitation stage, whose predicted signature is pass@1 up and pass@k at
risk. Round 1 showed the opposite (pass@5 doubled, greedy flat), so this
framework does not cleanly describe our run; it mainly warns that
expansion-type payoffs are a prolonged-training phenomenon.

**2. arXiv 2604.14877, "Does RL Expand the Capability Boundary of LLM Agents?
A PASS@(k,T) Analysis" (Zhai, Yan, Shao, Wang). Status: preprint, v1
submitted 2026-04-16, no venue listed.**
MY SNIPPET WAS WRONG and the correction matters. There is no crossover where
base overtakes RL at large budgets; the paper's headline is the reverse: for
agentic tool use with T interaction rounds, "tool-use RL genuinely enlarges
the capability boundary: the RL agent's pass-curve pulls above the base
model's and the gap widens at large k rather than converging," while "on
simpler tasks RL behaves as prior work predicts" (static-reasoning
convergence). It further claims that under matched training data SFT
regresses the compositional boundary, and that the mechanism is RL
reweighting the base strategy distribution toward strategies whose downstream
reasoning succeeds. These are abstract-level claims from a v1 preprint; no k
values, environments, or greedy (k=1) numbers were extractable from the page.
Bearing on our gate: Klondike at 100 to 250 decisions is a compositional
sequential task, the exact class where this paper says RL expands and SFT
regresses; it is the strongest external echo of this project's internal
record (on-policy SFT closed at parity, RFT showing movement), but it says
nothing about achieving that within 200 steps.

**3. arXiv 2506.14245, "Reinforcement Learning with Verifiable Rewards
Implicitly Incentivizes Correct Reasoning in Base LLMs" (Wen et al.,
Microsoft Research group). Status: preprint, v2 revised 2025-10-02 ("Update
with more experiments"), no venue listed.**
Confirmed as a direct response to the shrinkage claim: plain pass@k at large
k credits answers reached without correct reasoning, so base-model large-k
curves are inflated; their CoT-Pass@K requires "both the final answer and
intermediate reasoning steps" to be correct, and under it RLVR "can extend
the reasoning boundary for both mathematical and coding tasks." Notably for
step-scale: "it incentivizes correct reasoning early in the process." No
numbers were extractable from the abstract page.
Bearing on our gate: the lucky-guess critique mostly does not apply against
us, it applies FOR us. A Klondike win is a verified 100-to-250-decision
construction that cannot be guessed the way a boxed math answer can, so our
measured pass@5 doubling (18 to 40 percent) is closer in spirit to their
reasoning-verified metric and likely reflects real acquired capability rather
than metric artifact.

**4. arXiv 2512.16912, "Exploration vs Exploitation: Rethinking RLVR through
Clipping, Entropy, and Spurious Reward" (Chen, Li, Li, Yin, Chen, Lin).
Status: VERIFIED "Accepted by ICLR 2026" (arXiv comments field), v3
2026-01-26. The snippet's venue claim checks out; its content claim was
partly off.**
The paper is a mechanism study of why spurious rewards and entropy
minimization both seem to help: it shows "clipping bias under spurious
rewards reduces policy entropy, leading to more confident and deterministic
outputs, while entropy minimization alone is insufficient for improvement,"
plus a reward-misalignment model. The specific "pass@1 climbs while pass@k
stagnates at step X" account my snippet attributed to it is not in the
abstract; no step-scale numbers surfaced.
Bearing on our gate: its transferable lesson is that pure sharpening
(confidence up, entropy down) does not by itself produce real gains; gains
need an aligned reward. Ours is an exact win/loss verifier, so the
aligned-reward condition is met, and the relevant risk is entropy management,
which the one-sided trust region already addresses from the downside.

**5. arXiv 2604.17928, "HEALing Entropy Collapse: Enhancing Exploration in
Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment" (Liu et al.).
Status: VERIFIED "Accepted by ACL 2026 Main Conference," v1 submitted
2026-04-20.**
Confirmed in direction: "In low-resource scenarios, RLVR is prone to more
severe entropy collapse, which substantially limits exploration and degrades
reasoning performance." Their few-shot regime is 32 target-domain samples
(not single digits), against 1K full-shot; the fix is mixing in "high-value
general-domain data" plus a trajectory-level entropy-dynamics-alignment
reward, and with it "using only 32 target-domain samples, HEAL matches or
even surpasses full-shot RLVR trained with 1K target-domain samples." No
pass@1/pass@k failure-signature numbers or step counts were extractable from
the page.
Bearing on our gate: 8 decks is 4x fewer prompts than their already
collapse-prone few-shot setting, so elevated entropy-collapse risk in round 2
is externally corroborated; but the paper also proves few-prompt RLVR is not
intrinsically doomed, and its cheap mitigation maps to prompt diversity (more
decks) and entropy monitoring, both laptop-feasible. Round 1's zero parse
failures and rising pass@5 suggest collapse had not occurred yet.

**6. arXiv 2508.10751, "Pass@k Training for Adaptively Balancing Exploration
and Exploitation of Large Reasoning Models" (Chen et al.). Status: technical
report / preprint, v1 2025-08-14; arXiv page shows no venue (an OpenReview
forum exists, NOT-FETCHED).**
The snippet's step claim is CONFIRMED from the paper body: "Pass@k Training
leads to an increase in entropy, starting from step 200 of the RLVR
procedure," while "Pass@1 Training induces entropy to converge to a low
value." Also shown: "Pass@k Training does not compromise the model's Pass@1
performance and even results in Pass@1 performance gains," and "the
enhancement of LLM exploration ability is helpful to improve their
exploitation through continual training." Their runs span roughly 400 to
500+ steps; primary k=8 (tested k in {4,8,16}); models Qwen2.5-7B/32B-Instruct
and Seed1.5-VL-Small; and per their observation, standard pass@1-reward RLVR
improves pass@1 early then stagnates in later stages (local optimum).
Bearing on our gate: two concrete anchors. First, meaningful dynamics play
out on a hundreds-of-steps scale, so 200 steps is inside the
observable-movement window, not below it. Second, pass@k-as-reward at k=8
(matching our N=8 groups) is a verified objective that raises coverage
without sacrificing greedy; it would have been the natural round-3 objective,
though the pre-registered gate forecloses that on a FAIL.

## Conversion-lag question

Searched, not found as a directly measured quantity. I found no paper, in any
setting, that measures "N steps or rounds after pass@k rises, greedy pass@1
rises" as an explicit lag, and nothing at all in game or long-horizon agent
settings. The closest quantified anchors: (a) ProRL, arXiv 2505.24864
(preprint, 26 pages; 1.5B model, so nearest to our scale) shows boundary
expansion with "more than 2k training steps," roughly 16k GPU hours, 136K
prompts across five domains, n=16 rollouts per prompt, batch 256, plus KL
control and hard reference-policy resets on plateau; its abstract states
"reasoning boundary improvements correlates strongly with task competence of
base model and training duration." (b) arXiv 2508.10751 shows conversion
happens "through continual training" over 400 to 500+ steps but never
quantifies the lag. (c) arXiv 2511.16231, "Pass@k Metric for RLVR: A
Diagnostic Tool of Exploration, But Not an Objective" (single-author
theoretical note, Nanjing University, preprint, no experiments), derives that
the pass@k gradient is a scalar reweighting of the pass@1 gradient,
grad Jk = alpha_k * grad J1 with alpha_k = k(1-J1)^(k-1), meaning pass@k and
pass@1 improvement are gradient-coupled rather than sequential, with signal
vanishing when per-prompt success is near 0 or near 1. That last point is a
theoretical endorsement of the success-band deck curriculum: mid-difficulty
prompts are where any conversion signal is largest.

## Implications for the pre-registered gate

- Direction supported, rate unsupported. The strongest recent agent-setting
  evidence (arXiv 2604.14877) says RL expands capability exactly where SFT
  regresses it in compositional multi-turn tasks, and the metric critique
  (arXiv 2506.14245) implies our pass@5 doubling is real capability, not
  pass@k artifact. But every precedent that converted coverage into greedy
  gains used more steps (400 to 2,000+) and vastly more prompts (hundreds to
  136K, versus 8) than round 2's budget. Nothing external says 200 steps is
  enough; nothing says it is impossible either.
- The round-1 signature (pass@5 from 18 to 40 percent, greedy flat 5v5) is
  the coverage-first profile, which the literature says pays off through
  CONTINUAL training (arXiv 2508.10751, ProRL), not the sharpening profile
  that pays immediately. That cuts both ways: it argues the policy has
  genuinely new winning trajectories to consolidate, and it argues one more
  200-step round may undershoot the consolidation time.
- Blunt arithmetic on the bar: if round 2 lifted the true per-deck greedy win
  probability from ~42 percent (5/12) to 50 percent, a binomial read gives
  only ~39 percent chance of showing 7+ of 12; even a lift to 58 percent
  gives ~61 percent. Greedy decoding on a fixed panel is deterministic, so
  the real currency is deck flips (round 1 flipped one in each direction),
  but the calibration stands: 7/12 demands roughly a 15-point true
  improvement to be better than even money. The gate is a long-odds bet by
  construction, which is what a hard stop is for.
- Entropy collapse is the externally corroborated failure mode for 8 prompts
  (arXiv 2604.17928 at ACL 2026: fewer prompts, more severe collapse), and
  its signature is measurable mid-run (falling sequence entropy, shrinking
  rollout diversity, pass@N contracting toward pass@1). Round 1 showed no
  collapse markers; if round 2's extra 200 steps start showing them, that is
  mechanism information a 12-deck greedy score cannot give.
- A FAIL would prove: this recipe, at this step and prompt budget, on this
  hardware, does not convert its coverage gains into greedy wins, which is
  the operationally binding question and the pre-registered stop should
  execute as written. A FAIL would NOT prove: that RLVR cannot expand this
  model's Solitaire capability (external precedent locates conversion at
  budgets this project has chosen, reasonably, not to spend), nor that the
  pass@5 gains were illusory (the exact trajectory-level verifier makes them
  the best-attested capability gain the project has produced).
- If the gate PASSES, the external evidence says the result is
  mechanistically credible, not a fluke reading: agentic-RL expansion (arXiv
  2604.14877), early correct-reasoning incentives (arXiv 2506.14245), and
  gradient coupling of pass@k and pass@1 (arXiv 2511.16231) all permit greedy
  movement at modest budgets; the surprise would be the efficiency, not the
  sign.

## Sources

- arXiv 2510.04028 https://arxiv.org/abs/2510.04028 (preprint)
- arXiv 2604.14877 https://arxiv.org/abs/2604.14877 (preprint)
- arXiv 2506.14245 https://arxiv.org/abs/2506.14245 (preprint)
- arXiv 2512.16912 https://arxiv.org/abs/2512.16912 (accepted ICLR 2026)
- arXiv 2604.17928 https://arxiv.org/abs/2604.17928 (accepted ACL 2026 main)
- arXiv 2508.10751 https://arxiv.org/abs/2508.10751 (preprint)
- arXiv 2511.16231 https://arxiv.org/html/2511.16231v1 (preprint, theory note)
- arXiv 2505.24864 ProRL https://arxiv.org/abs/2505.24864 (preprint)

NOT-FETCHED: OpenReview pages for 2510.04028 (claimed ICLR 2026 status
unverified; treat as preprint) and 2508.10751.
