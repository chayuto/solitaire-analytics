# External research 3 of 5: inference-layer branch (2026-07-18)

Strand 3 of the sequential external research pass requested 2026-07-17.
Method: a single verification agent fetched each lead from the interrupted
2026-07-17 deep-research sweep, confirmed or corrected the characterizations
against primary sources (12 fetches, 4 searches, plus gh CLI for the mlx-lm
PR question), no usage-limit errors. Companions: strands 1, 2, 4, 5 (same
date) and `20260718_external_strategy_stresstest.md`. Bears on: the
post-gate ship path (endgame oracle banked; constrained decoding and
per-move solver veto/force planned; see `20260704_next_move_deep_research.md`
strand B/C and `20260705_endgame_oracle_ab.md`).

Report body below is the verification agent's output, verbatim except that
HTML entity escapes were replaced with plain ASCII ">".

---

## Verdict summary

The inference-layer branch is SUPPORTED, with one demanded design
correction. The LLM-proposes/solver-adjudicates pattern is the current state
of the art for imperfect-information card games (ToolPoker, ICLR 2026:
external solvers supply the actions, the LLM supplies the interface;
behavior cloning and step-level RL improved reasoning style but not
game-theoretic accuracy, which mirrors this project's RFT experience), the
solver-heavy contrarian architectures (Code World Models, Obscuro) argue for
more solver weight rather than less, and PAC-MCTS's bounded-bias pruning
rule independently endorses the unanimity-based veto/force design. The
single most design-changing finding is in the constraint-tax literature: the
two scare numbers in the leads were both wrong (92.5-to-35.0 does not exist
in the snowballing paper; the real drop is 50.0 to 38.0 in a multi-round
reflection loop), but the verified small-model evidence is actually harsher
than the lead claimed: always-on JSON-grammar decoding cratered 7-9B models
to 15-52 percent output accuracy on reasoning tasks (Gemma 2-9B worst at
15.31 percent with 52.4 percent duplicate outputs), while both papers'
mitigations converge on the same architecture: generate unconstrained first,
apply the Outlines grammar only as a parse-failure fallback (or A/B
always-on before trusting it). The Outlines-on-MLX build itself is confirmed
viable: official MLXLM support exists, mlx-lm's maintainer explicitly
rejected native Outlines integration (PR #845, closed unmerged), so a
user-side logits processor is the sanctioned path, and Outlines' benchmarked
pathologies (timeouts, 40 s to 10 min compiles) are specific to complex
schemas (minItems, enum, arrays) that a 3-key schema does not touch.

## Source-by-source findings

1. arXiv 2602.00528, "How Far Are LLMs from Professional Poker Players?
Revisiting Game-Theoretic Reasoning with Agentic Tool Use" (Lin et al.,
submitted 2026-01-31). Status: accepted at ICLR 2026 (per abs page), so
peer-reviewed. Lead confirmed. SHOWS: baseline LLMs have three systemic
weaknesses (shortcut reliance, factual errors, reasoning/decision
misalignment); behavior cloning and step-level RL "improved reasoning
patterns but not game-theoretic accuracy". ToolPoker integrates "external
solvers for GTO-consistent actions with more precise professional-style
explanations" and claims "state-of-the-art gameplay while producing
reasoning traces that closely reflect game-theoretic principles". Division
of labor: solver guarantees the mathematically sound decision, LLM generates
the justification. Bearing on our design: direct validation of the advisor
architecture (solver decides where competent, LLM explains and covers the
rest); also independent evidence that training a model toward solver
behavior underdelivers versus calling the solver at inference, which is
exactly the branch condition this program sits on.

2. arXiv 2605.30094, "PokerSkill: LLMs Can Play Expert-Level Poker without
Training or Solvers" (Li, Wang, Huang, submitted 2026-05-28). Status:
preprint, not peer-reviewed. Lead partially confirmed, with two corrections.
SHOWS: agents built on GPT-5.5 XHigh (-57 +/- 21 mbb/hand), Claude Opus 4.6
(-80 +/- 29), Claude Opus 4.7 (-87 +/- 64) still LOSE to GTOWizard; the
claim is "reducing losses by 49-61% compared to default-prompt baselines"
and "outperforming the strong bot Slumbot". Claims: "first demonstration of
an LLM achieving competitive performance in a complex imperfect-information
game without game-specific training or solver queries". Corrections: (a)
"expert-level" means beats Slumbot while still losing to the GTO benchmark;
(b) it is not scaffold-free: it uses "a deterministic context engine" over
"a layered skill library, which is entirely designed by human poker
experts", i.e. a hand-built knowledge artifact standing in for the solver.
Bearing on our design: the contrarian case holds only at frontier scale and
even then needs expert scaffolding and still loses to exact computation; it
provides no support for dropping solver layers at 2B scale.

3. arXiv 2510.04542, "Code World Models for General Game Playing" (Lehrach
et al., incl. Lanctot, Singh, Murphy; submitted 2025-10-06). Status:
preprint. Lead confirmed. SHOWS: LLM "translate[s] natural language rules
and game trajectories into a formal, executable world model represented as
Python code" (transitions, legal moves, termination, heuristic value
functions, and "inference functions (for imperfect information games)");
MCTS does all the playing; "outperforms or matches Gemini 2.5 Pro in 9 out
of the 10 considered games", 5 of which are imperfect-information. Bearing
on our design: this is the competitor architecture, and it says classical
search over a correct model beats direct frontier-LLM play even under
imperfect information; since this repo already has the exact engine and
solver, the finding pushes toward giving the solver layer as much authority
as honesty (no hidden-state reads) permits, and it demands a solver-only
control arm in the eval.

4. arXiv 2604.14345, "PAC-MCTS: Bias-Aware Pruning for Robust LLM-Guided
Search and Planning" (Tianhao Qian, v1 2026-04-15, v3 2026-05-09). Status:
preprint, single author, not peer-reviewed (weigh accordingly). Lead
confirmed in substance. SHOWS: node expansion cast as Best-Arm
Identification with an LLM as a biased surrogate evaluator; guarantee form
is a PAC-style safe-elimination bound: sample complexity "O((Delta-4L)^-2)"
against a lower bound "Omega((Delta-2L)^-2)", with "Delta > 4L as the regime
where safe elimination is feasible" (Delta = value gap, L = bound on
evaluator bias); empirically "up to 78% fewer API evaluations" and "over 3x
higher sample efficiency" on Blocksworld and ALFWorld. Correction to lead:
the guarantee is conditional on a bounded-bias assumption, not an
unconditional "LLM cannot make search worse". Bearing on our design: the
transfer is the margin rule, prune only when the observed gap exceeds the
evaluator's bias bound. Our filter is safer than PAC-MCTS's setting (the
pruner is an exact solver, the biased LLM only breaks ties), but
determinization bias plays the role of L: unanimity across K worlds is the
margin condition, and near-split verdicts should widen K rather than trigger
a hard veto or force.

5. arXiv 2506.01242, "General search techniques without common knowledge for
imperfect-information games, and application to superhuman Fog of War chess"
(Zhang and Sandholm, CMU; v1 2025-06-02, v2 2026-02-28). Status: preprint on
arXiv (the abs page lists no venue). Lead confirmed. SHOWS: Obscuro,
common-knowledge-free search, no LLM anywhere; "Experiments against the
prior state-of-the-art AI and human players -- including the world's best --
show that Obscuro is significantly stronger"; claims FoW chess is "the
largest (by amount of imperfect information) turn-based zero-sum game in
which superhuman performance has been achieved". Method class: belief-state
search handling "the opponent's knowledge, signaling, etc.", i.e. explicitly
NOT determinization; this research line exists because determinization is
unsound in adversarial games (strategy fusion, non-locality). Bearing on our
design: the anti-determinization critique largely does not transfer to
single-player Klondike (no opponent, no signaling, no common-knowledge
problem); Bjarnason 2009's HOP remains the right tool class there, with
clairvoyance bias as the one residual defect. The transferable contrarian
lesson: in solver-strong domains search alone can be superhuman, so the eval
must measure whether the LLM tie-breaker adds wins over the solver layer
with a dumb tie-break.

6. Outlines docs, dottxt-ai.github.io/outlines/latest/features/models/mlxlm/.
Status: vendor docs, current. Lead confirmed with one refinement. SHOWS:
official MLXLM model class via `outlines.from_mlxlm(*mlx_lm.load(...))`;
"As a local model, MLXLM supports all forms of structured generation
available in Outlines" (JSON schema via Pydantic, regex, CFG, choices);
"You need a device that supports Metal"; install `outlines[mlxlm]`.
Refinement: this page does not mention logits_processors or any version
pins; the public API wraps mlx_lm.generate, with the logits-processor wiring
internal. The lower-level JSONLogitsProcessor-into-logits_processors route
the plan describes is the same mechanism used by mlx-lm PR #845 and by LM
Studio's MLX structured output, and the operator has already verified the
hook locally on 0.31.3. Bearing on our design: the planned build is the
canonical integration on this stack; no doc-level blocker.

7. arXiv 2604.06066, "From Hallucination to Structure Snowballing: The
Alignment Tax of Constrained Decoding in LLM Reflection" (Hongxu Zhou,
submitted 2026-04-07). Status: preprint, single author, not peer-reviewed.
Lead's headline number FALSIFIED: "The claimed 92.5%-to-35.0% degradation
does not appear in this paper's results tables or narrative." What it
actually SHOWS: Qwen3-8B in a Reflexion self-correction loop, with Outlines
enforcing a 5-category error taxonomy on the reflection output; "The
baseline accuracy of 50.0% dropped to 38.0% under constrained conditions"
(12 points, not 57.5); 23 previously-correct samples flipped to failure; 96
percent of diagnoses collapsed into FORMATTING_MISMATCH "death loops"; the
degraded group averaged 4,005.5 tokens. Mitigations proposed: "dynamic
fallback mechanisms" that "temporarily lift the decoding constraints" plus
rollback to a previous reasoning node. Field order is not the studied
variable; the harm mechanism is constraining the semantic diagnosis channel
to an enum inside a multi-round loop. Bearing on our design: refines rather
than undermines; our setting (single turn, free prose field, integer answer)
is the opposite shape of the harm case, but the paper's mitigation,
constrain-as-fallback with an escape hatch, is the deployment pattern to
adopt.

8. arXiv 2605.02363, "When Correct Isn't Usable: Improving Structured Output
Reliability in Small Language Models" (Galeone, Park, Ettorre, Ligorio;
submitted 2026-05-04). Status: preprint. Lead's specific numbers (0.69-0.83
value accuracy, 0.38-0.53 perfect rate) NOT FOUND; the real numbers are
worse. SHOWS: Llama 3.1-8B, Gemma 2-9B IT, Qwen 2.5-7B on GSM8K/MATH with
JSON output; NAIVE prompting reaches "up to 85% task accuracy on GSM8K but
0% output accuracy" (correct math, unusable format); constrained decoding
(vLLM JSON grammar) yields output accuracy of 52.46 (Llama), 15.31 (Gemma),
32.83 (Qwen) percent, with Gemma producing "52.4% of outputs are exact
duplicates" (degenerate generation under grammar) and "inference latency
relative to NAIVE is 3.6x (Llama CONSTRAINED) to 8.2x (Qwen REF+CONSTRAINED)";
their fix, AloLab, an iterative system-prompt optimizer (meta-agent Claude
Sonnet 4.5, black-box, no fine-tuning), reaches 84-87 percent output
accuracy on GSM8K at 0.63x-1.06x NAIVE latency. Mechanism statement: a
"format-task tension" where structured output "constrains its generation in
ways that can slightly reduce underlying task performance". Bearing on our
design: the strongest evidence that always-on grammar CD can be net harmful
for small models on reasoning-shaped tasks, and that prompt/format-side work
is the cheaper first lever; the Gemma-family model being the worst CD case
is a weak but pointed caution for a Gemma 4 student. Key difference in our
favor: their models were forced into an out-of-distribution format, while
our SFT student already emits this exact 3-key JSON natively, so the grammar
mostly coincides with the model's fluent path.

9. arXiv 2501.10868, "JSONSchemaBench: A Rigorous Benchmark of Structured
Outputs for Language Models" (Geng et al., incl. Moskal, Horvitz, Nori; v1
2025-01-18, v3 2025-02-27). Status: preprint (Microsoft/EPFL authorship).
Lead half right, half wrong. Confirmed: Outlines weakest of the six engines
(Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini): "Outlines has a
comparatively lower compliance rate, primarily due to timeouts during
generation"; compile time 3.48 s vs Guidance 0.00 s (GlaiveAI); 30.33 ms vs
6.37 ms per output token; schema features like "'minItems', 'maxItems',
'enum', and 'Array'... often take 40 seconds to 10 minutes for Outlines to
process". CORRECTION: the leader is Guidance (llguidance engine), not
XGrammar: "Guidance consistently demonstrates the highest compliance rate
across all datasets". Second correction, load-bearing and in CD's favor:
"constrained decoding, regardless of the framework, achieves higher
performance than the unconstrained setting" (about 3 percent for Guidance
over LM-only) on their downstream tasks. Bearing on our design: Outlines'
pathologies are complex-schema pathologies; a 3-key, no-enum, no-array
schema avoids all of them (compile once, cache the processor), and on this
benchmark's task family CD helped quality. The absence claim held up: no
xgrammar or llguidance MLX integration found (see Searched-not-found), so
Outlines remains the only first-class MLX option.

10. mlx-lm GitHub (checked via gh CLI, repo ml-explore/mlx-lm). PR #845
"Feat: Structured outputs" proposed exactly the planned mechanism (Outlines'
stateful OutlinesCoreLogitsProcessor built from a JSON schema, wired into
server and BatchGenerator via logits_processors). It was CLOSED UNMERGED on
2026-02-08; maintainer awni: "Indeed I don't think it makes sense to take
outlines as a dependency here", requesting only the incidental
logits_processors plumbing fix as a separate PR. No other open or merged PR
adds native structured output or a guided-decoding API (searches for
structured, grammar, constrained, guided). Several logits_processors
maintenance PRs exist (#1157 stale-token-data under lazy evaluation, #1230
None entries in merged processors, #1225 open per-slot bookkeeping fix for
batch generation); merge status not individually verified, relevant mainly
if batch generation is used. Bearing on our design: no native API is coming;
the user-side Outlines processor over the logits_processors hook is both the
sanctioned extension point and validated by an implementation the
maintainers reviewed.

## Constraint-tax assessment

For our exact case (single turn, 3 keys, prose field first, integer
move_index, tiny grammar, loss mode = unescaped inner quotes mid-prose), the
verified literature predicts a CONDITIONAL NET WIN, with the sign depending
on deployment mode:

- The two harm results do not match our shape. The snowballing result (50.0
  to 38.0) required a multi-round reflection loop with the semantic content
  forced into a 5-category enum; the small-LM result (15-52 percent output
  accuracy) forced instruct models into a JSON format they were not tuned to
  emit, on math benchmarks, always-on. Our student was SFT-trained to emit
  this exact JSON, so the constraint is near-coincident with its fluent
  distribution; JSONSchemaBench, whose tasks look more like ours (fill a
  known small schema), found CD "achieves higher performance than the
  unconstrained setting" for every framework.
- The specific failure the grammar fixes, and what it turns into. A grammar
  cannot make the model escape an intended inner quote; it will parse the
  unescaped quote as string-close and then force the next legal token
  (comma, next key). So the parse-death converts into a truncated
  board_analysis followed by a forced early transition to "move_index". That
  is strictly better than a dead turn (move_index still gets emitted and the
  engine validates legality downstream), but it is the one place our
  prose-first order interacts badly with CD: the forced transition shortens
  the in-context reasoning that precedes the answer. This is the
  "format-task tension" mechanism, in miniature and only on the turns that
  would otherwise have died.
- What the literature demands, concretely: (1) Fallback architecture, not
  always-on: attempt unconstrained decode first (preserves the reasoning
  distribution on the roughly 97 percent of turns that parse), engage the
  Outlines processor only on parse failure, per the snowballing paper's own
  mitigation ("temporarily lift the decoding constraints" inverted:
  temporarily impose them); this dominates the current retry-N-times design
  because the constrained retry cannot fail to parse. (2) A/B before
  trusting always-on: the small-LM paper shows the always-on tax can be
  large for 7-9B and worst on a Gemma; measure heldout12 win rate
  unconstrained-with-fallback vs always-on before adopting the latter. (3)
  Prompt/format-side hardening first: AloLab's 84-87 percent via
  system-prompt optimization at 1x latency says cheap non-CD levers (e.g.
  instructing single-quote or quote-free prose) may shrink the
  unescaped-quote rate before any grammar runs. (4) Compile once and cache:
  Outlines' 40 s to 10 min compiles are enum/array/minItems pathologies; our
  schema avoids them, but build the processor at session start, not per
  move. (5) Keep escape hatches: never let the grammar force an out-of-range
  move_index; the schema should constrain to integer, with legality left to
  the engine/solver layer that already validates it.

## Veto/force design inputs

- Division of labor is settled in this direction: ToolPoker (peer-reviewed)
  gets SOTA by letting the solver own the action where it is competent and
  the LLM own the interface, and found training (BC, step-level RL) could
  not substitute; Code World Models shows classical search over a correct
  model beats direct frontier-LLM play in 9/10 games including
  imperfect-info. Both say: when in doubt, move authority toward the solver,
  which is what veto/force does.
- The unanimity rule is the right safety condition, now with a theory frame:
  PAC-MCTS's "safe elimination iff Delta > 4L" transfers as: hard veto/force
  only on unanimous verdicts across all K worlds (maximum margin), because
  determinization bias (clairvoyance) is the L of this design; on split
  verdicts, do not prune, either widen K or leave it to the LLM tie-break.
  Our setting is strictly safer than PAC-MCTS's (exact solver prunes, biased
  LLM only tie-breaks), so "cannot be worse than the base policy" is closer
  to holding here than in the paper.
- Determinization is domain-appropriate here despite Obscuro: the
  common-knowledge/strategy-fusion critique that made CMU abandon
  determinization is adversarial-specific; single-player Klondike has no
  opponent, and the only known HOP defect (clairvoyance optimism) biases
  toward under-vetoing, i.e. the filter fails safe (a move that loses even
  under clairvoyant play in every sampled world is near-certainly bad).
- Mandate a solver-only control arm: Obscuro (superhuman, no LLM) and Code
  World Models both raise the honest possibility that the LLM adds nothing
  once the solver layer is strong; run veto/force with a trivial tie-break
  (random or first-legal) as a control against veto/force+LLM, the same A/B
  discipline already used for the oracle (volume+oracle 7/12 vs
  grpo_tr+oracle 6/12).
- The contrarian solver-free result does not apply at 2B: PokerSkill needed
  GPT-5.5/Opus-class models plus an expert-authored skill library and still
  lost to the GTO benchmark (-57 to -87 mbb/hand); nothing in it licenses
  removing solver layers from a 2B student, and its real lesson (structured
  domain scaffolding around the LLM) is what the oracle + veto/force stack
  already is.
- Per-move best-of-N composes cleanly: none of the sources argue against it;
  PAC-MCTS's budget results (78 percent fewer evaluations under
  confidence-based allocation) suggest spending the N samples adaptively,
  more samples only when the solver layer reports a split verdict, rather
  than uniformly.

## Searched-not-found

- 2025-2026 work applying determinized search, HOP, or LLM+solver hybrids to
  solitaire/patience games: NOT FOUND. Queries: "Klondike solitaire LLM
  solver hindsight optimization determinization 2025 2026" (returned only
  Bjarnason and Fern's 2009 ICAPS HOP/UCT paper, the project's existing
  citation, with its greater-than-35-percent Klondike lower bound) and
  ""solitaire" OR "patience game" language model agent playing 2026 arxiv"
  (returned LLM game-agent surveys and the Solvitaire winnability work, now
  listed as JAIR-published with thoughtful-Klondike winnability 81.945 +/-
  0.084 percent, a pure solver result, no LLM). The inference-layer program
  appears to have no direct prior art in this domain.
- xgrammar or llguidance MLX/mlx-lm integration: NOT FOUND. Query: "xgrammar
  OR llguidance MLX mlx-lm Apple Silicon integration structured output
  2026". JSONSchemaBench's absence claim stands; Outlines (plus the
  third-party otriscon/llm-structured-output acceptor library, surfaced but
  not fetched) remains the MLX option set.
- Native structured output in mlx-lm: resolved as a definitive no (PR #845
  closed unmerged, maintainer declined the Outlines dependency), so this is
  answered rather than not-found.
- Not fetched for budget: AdapTrack (arXiv 2510.17376, "Constrained Decoding
  without Distorting LLM's Output Intent"), surfaced incidentally in search
  results; title-relevant to the fallback-mode recommendation but
  unverified. The ToolPoker full PDF (model list, solver call protocol
  details) and PokerSkill full PDF were verified at abs-page depth only.
