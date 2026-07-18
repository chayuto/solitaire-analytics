# External strategy stress-test: synthesis (2026-07-18)

Operator request 2026-07-17: "do external research to support our strategy
or otherwise", no compute. Execution: a first-attempt deep-research
workflow (2026-07-17) died on a session limit after its search phase (90
agents spawned, 10 completed, 0 claims verified); its lead lists were
salvaged from the run journal and re-run 2026-07-18 as five SEQUENTIAL
verification strands, one agent at a time, each fetching primary sources
and confirming or refuting the search-phase characterizations. Per-strand
notes (all 2026-07-18, this directory):

1. `20260718_extresearch_gate_calibration.md` (10 fetches, 2 searches)
2. `20260718_extresearch_novelty_ceiling.md` (18 fetches, 5 searches)
3. `20260718_extresearch_inference_layer.md` (12 fetches, 4 searches, gh)
4. `20260718_extresearch_round3_recipe.md` (9 fetches, 3 searches)
5. `20260718_extresearch_stack_watch.md` (registry checks, no agent)

Strategy under test: `20260703_long_term_project_strategy.md` as revised by
`20260704_next_move_deep_research.md`, `20260705_grpo_round2_preregistration.md`
(the 7+/12 hard-stop gate), and `20260705_progress_and_diminishing_returns.md`
(the bounded finish list). Today's program position: round-2 Phase R paused
at 50/64 rollouts, resume state in `20260707_grpo_round2_resume_state.md`.

## 1. Verdict table

| # | Strategy element | Verdict | Strand |
|---|---|---|---|
| 1 | Finish round 2 to its pre-registered gate | SUPPORTED (direction); rate UNRESOLVED | 1 |
| 2 | The hard stop itself (no retries on FAIL) | SUPPORTED | 1 |
| 3 | One-sided TR collapse fix, per-round reference refreeze | SUPPORTED | 1, 4 |
| 4 | Round-2 recipe (N=8, mean-only, zero-variance skip) | MOSTLY SUPPORTED; one named weakness | 4 |
| 5 | Round-3 sketch (per-token KL + re-banded decks) | CHALLENGED; better options identified | 4 |
| 6 | Ship path: volume + knowledge-gated endgame oracle | SUPPORTED | 3 |
| 7 | Constrained JSON decoding build | SUPPORTED WITH CORRECTION (fallback mode) | 3 |
| 8 | Per-move solver veto/force over determinized worlds | SUPPORTED, with design inputs | 3 |
| 9 | Write-up novelty claim | CHALLENGED AS PHRASED; SUPPORTED after narrowing | 2 |
| 10 | Ceiling planning numbers | MIXED (one upgraded, two weakened) | 2 |
| 11 | Stack (mlx-lm pin, patch, QAT re-baseline) | STABLE, no forced action | 5 |

## 2. The load-bearing findings, element by element

**1-2. Gate and hard stop.** The direction of the bet is externally
supported: the closest-setting evidence (arXiv 2604.14877, pass@(k,T) on
multi-turn agents) finds RL genuinely enlarges the capability boundary on
compositional agentic tasks, with the gap WIDENING at large k, while
matched-data SFT regresses the boundary; that is this project's internal
record (on-policy SFT closed at parity, RFT moving) reproduced externally.
The metric critique of Yue et al. (arXiv 2506.14245, CoT-Pass@K) implies
our pass@5 doubling is real capability, not artifact, because a Klondike
win is an exact 100-250-decision construction that cannot be lucky-guessed.
But NO fetched source demonstrates coverage-to-greedy conversion inside a
~200-step, 8-prompt budget; quantified precedents used 400 to 2,000+ steps
and hundreds to 136K prompts. A binomial read puts 7+/12 under a
+8-point true lift at roughly 39 percent. Conclusion: run it, expect FAIL
more often than PASS, and read a FAIL as "not at this budget" rather than
"RLVR cannot"; the pre-registered stop executes either way.

**3-4. Trainer.** Round 2's core choices are validated: N=8 matches the
long-horizon recipe default (arXiv 2603.21972), zero-variance skip matches
ProRL v2's dynamic sampling, and the per-round reference refreeze matches
ProRL v2's 200-500-step reset practice (vendor-grade, NVIDIA blog). The
one named weakness: smearing a single trajectory scalar across 100-250
mostly-routine decisions is the low-decision-density regime where
trajectory-level estimators provably lose signal-to-noise (arXiv
2606.22164, "signal dilution", SNR proportional to rho^(-1/2),
workshop-grade but analytic); three independent 2026 method lines
(Turn-PPO EACL 2026, GAGPO, GTPO) exist to fix exactly this. This is the
best available mechanistic hypothesis for why our round moves have been
parity-sized. Also flagged: the recipe borrows Dr.GRPO's mean-only
advantage while keeping a per-trajectory length normalization Dr.GRPO
itself removes; a self-consistency to test, not a settled choice.

**5. Round-3 (only if PASS).** The sketched per-token KL is demoted: the
one multi-turn recipe paper that tested KL removed it and widened clipping
instead, and ProRL v2 stays healthy through resets, not more regularizers.
The evidence-ranked round-3 list (strand 4): (i) GAGPO-style state-matched
step-level advantages (critic-free, validated at 1.5B with +11.7 success
points over GRPO, natural fit to our deterministic engine; needs gamma
near 0.99-1.0 and a shared-state density measurement first); (ii) cheap
fallback: down-weight routine forced-draw turns in extraction to raise
effective decision density; (iii) STEP-style success-rate-aware rollout
allocation (Findings of ACL 2026), which subsumes re-banded deck pools;
(iv) keep the one-sided TR with a formalized reset/stop trigger.

**6-8. Inference layer (ship path on FAIL).** The
LLM-proposes/solver-adjudicates division of labor is now peer-reviewed SOTA
on an imperfect-information card game (ToolPoker, ICLR 2026), which also
found behavior cloning and step-level RL improved reasoning style but NOT
game-theoretic accuracy, i.e. calling the solver at inference beat training
toward it. The solver-heavy contrarians (Code World Models 9/10 vs Gemini
2.5 Pro; Obscuro superhuman Fog-of-War chess with no LLM) push MORE
authority to the solver, not less, and mandate one eval addition: a
solver-only control arm (veto/force with a dumb tie-break) to measure
whether the LLM adds wins at all. The determinization critique behind
Obscuro is adversarial-specific (strategy fusion, common knowledge) and
does not transfer to single-player Klondike; Bjarnason HOP remains the
right tool class, and clairvoyance bias fails safe for a veto (unanimous
loss across worlds is near-certainly bad). PAC-MCTS supplies a margin
frame: hard veto/force only on unanimous verdicts; on splits, widen K or
leave it to the LLM. Constrained decoding: build it, but deploy as a
PARSE-FAILURE FALLBACK, not always-on; the verified small-model constraint
tax is severe (always-on JSON grammar cratered 7-9B instruct models to
15-52 percent output accuracy, worst on a Gemma), while our
SFT-model-emits-this-schema-natively case sits on the favorable side
(JSONSchemaBench found CD helps on schema-filling tasks); A/B always-on
before ever adopting it. Both scare numbers circulating in the leads were
wrong; the corrected numbers and conditions are in the strand-3 note.
Outlines remains the only first-class MLX path (mlx-lm PR #845 closed
unmerged; no xgrammar/llguidance MLX port found).

**9. Novelty.** The blanket "no published LLM-plays-Klondike work" is no
longer printable. Two near-misses found and verified: Bateni, Pratt,
Whitehead (AIIDE 2025, peer-reviewed) fine-tune LLMs on Klondike-family
solitaire RULES via a Solitaire GDL, explicitly excluding game playing (no
win rates, play traces from a heuristic DFS bot); Game-RL/GameQA (arXiv
2505.13886) uses Spider Solitaire states as VQA training data. The claim
that survives, verified by fresh sweep 2026-07-18: no published work in
which an LLM plays COMPLETE Klondike games and reports a WIN RATE. The
mandatory citation list (10 items with differentiators) is banked in the
strand-2 note; the project's own artifacts are already the top
discoverable items in the niche.

**10. Ceilings.** Upgraded: Solvitaire's 81.945 +/- 0.084 percent
perfect-information winnability is now JAIR-published (Vol 85, 2026, DOI
10.1613/jair.1.17167); cite the journal version. Weakened: the 43-52
percent imperfect-info ceiling had no primary source verified in this pass
(keep only with its original citation trail or label as project estimate),
and the "human expert draw-1 ~43%" attribution was REFUTED (the vendor page
does not contain it; the page's own numbers are typical-player 11-33
percent in one place and ~40 percent in another, internally inconsistent).
The strongest number in the write-up remains our own: ~33 percent pooled
fresh-deck (16/48) against the 81.9 percent perfect-info ceiling.

**11. Stack.** mlx-lm still 0.31.3 (released 2026-04-22; sanitize fix
merged 2026-05-04 remains unreleased; patch and guard stay). Official
Gemma 4 E2B QAT checkpoints confirmed live (575k downloads); one MLX
conversion exists (mlx-community OptiQ-4bit, full multimodal,
mixed-precision) but no Text-only QAT MLX conversion yet; re-baseline stays
a next-cycle item. Our pinned base revision equals upstream HEAD (no
drift).

## 3. Operational deltas (what actually changes)

1. **The next three runs are unchanged.** Finish Phase R (14 games), Phase
   T extract+train, Phase E gate, exactly per the preregistration. Nothing
   found overturns the design; the research only calibrates expectations
   (long-odds) and sharpens post-gate execution.
2. **Write-up edits, banked now** (zero GPU): narrow the novelty claim to
   "no published LLM plays complete Klondike with a reported win rate
   (checked 2026-07-18)"; add the strand-2 mandatory citation list; upgrade
   the Solvitaire cite to JAIR; drop or re-source the 43 percent
   human-expert figure; label the 43-52 percent band a project estimate
   unless its original trail is recovered.
3. **Constrained decoding build spec amended**: fallback-mode first
   (unconstrained decode, grammar engaged only on parse failure), always-on
   only after an A/B; prompt-side quote-hardening as the zeroth lever;
   schema constrains structure and integer move_index only.
4. **Veto/force eval design amended**: add the solver-only control arm;
   unanimity rule for hard veto/force; adaptive best-of-N (spend samples on
   split verdicts).
5. **If the gate passes**: round-3 design starts from the strand-4 ranked
   list (state-matched step advantages, routine-turn down-weighting, STEP
   allocation; per-token KL demoted to fallback).
6. **Either branch**: when diagnosing the round-2 result, apply the
   signal-dilution lens (decision density), and note the
   traj-normalize-vs-Dr.GRPO self-inconsistency as a one-variable test if
   any further training ever happens.

## 4. Method caution recorded

Four of the salvaged search-phase snippets were materially WRONG on fetch
(the pass@(k,T) paper's direction was inverted; the 92.5-to-35.0
constraint-tax number does not exist; the Tree-GRPO arXiv id was conflated
with GTPO; several vendor-stat attributions were to the wrong page). Search
snippets are leads, never citations; every number that reaches a doc must
be read off the fetched source. This session's notes follow that rule.
