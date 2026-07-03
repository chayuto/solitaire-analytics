# Deep experiment retrospective: every arc, every learning point (2026-07-03)

Companion to `20260703_long_term_project_strategy.md`. That doc says where we
go; this one is the deep review of where we have been: eleven experimental
eras across ~6 weeks, 76 dated reports, what each era tried, what it measured,
and the learning point it left behind. Sections 3-5 distill the catalog:
object-level findings about the problem, meta-level lessons about method, and
the costly-mistakes ledger.

## 1. Era timeline

| # | Era | Dates | Driving question | Outcome |
|---|---|---|---|---|
| 0 | Repo foundations | 2025-11 | build the analytics engine | infrastructure, no experiments |
| 1 | Handover + proxy prompt science | 05-22..24 | which prompt changes help the teacher? | A/B machinery built; proxy results mostly noise |
| 2 | First distillation (gemma-3n) | 05-24..25 | can a 2B student learn anything? | yes: gap halved, v1 published |
| 3 | Gemma 4 pivot + corpus poisoning | 05-25..26 | is the newer base better? | base better untuned; training REGRESSED it |
| 4 | Full-game truth | 05-27..29 | does the bench predict play? | NO; doom-loop is base-deep; temp is not a fix |
| 5 | Filter closure + prompt v1.3/1.4 | 05-30..06-05 | can filtering or prompt-state fix loops? | both NO (v4-A regressed; stall probe 0/40) |
| 6 | Prompt fidelity + win-data research | 06-06..08 | where is the teacher win-rate lost? | v1.5/v1.6 shipped; loops mostly sees-and-ignores |
| 7 | ORPO + the eval-fidelity reset | 06-08..11 | preference training? and is the harness honest? | harness was NOT honest; reset voided prior numbers; ORPO refuted |
| 8 | Won-only gate -> volume | 06-12..17 | does training beat untuned under a TRUSTED eval? | YES, first time; volume named, published |
| 9 | Corpus engineering | 06-17..23 | can corpus shape beat volume? | in-dist yes, generalization NO (all forms collapse) |
| 10 | On-policy SFT (best-of-N, STaR) | 06-23..07-01 | can self-wins beat volume? | NO: parity at best; diversity explains the failures |
| 11 | RFT / GRPO iteration-0 | 07-01..02 | can the game's reward beat volume? | collapse found+fixed; eval PENDING |

## 2. Era-by-era review

### Era 0 (2025-11): foundations

The analytics engine (models/engine/strategies/solvers/analysis/game/MCP) and
a set of improvement task docs. No measurements. Everything later stands on
this: the engine's pure-rules `apply_move` contract is what makes byte-exact
replay, drift gates, and exact adjudication possible.

### Era 1 (2026-05-22..24): handover and prompt science on a proxy

- The prompt-template audit found TWO templates in the wild and a natural
  experiment: calibration guidance shifted mean confidence -0.023 and cut
  0.95-saturation by 18pp. Recurring failures (oscillation, chain reversal,
  deadlock-but-draws) appeared in BOTH templates: baked into the prompt
  system, not any build.
- Five-arm Haiku proxy A/B (drop reasoningTrail, drop prose CoT, hoist
  notation): dropping prose CoT cut output ~19x with 100% JSON validity, but
  at N=15-20 every CI crossed zero. A4's Phase-1 +0.27 evaporated to -0.12
  (CI [-0.45, +0.18]) in the pre-registered Phase 1.5.
- Learning points: (a) proxy models answer proxy questions; the
  harvester-fidelity rule (31B only for teacher experiments) dates here;
  (b) small-N A/B deltas are noise until proven otherwise; (c) the
  reasoning-trail-as-few-shot-of-own-bad-behavior mechanism first surfaced
  here and kept being confirmed later (era 4's PRIOR REASONING amplifier).

### Era 2 (2026-05-24..25): first distillation, gemma-3n

- Untuned gemma-3n E2B baseline: -1.32 tier gap to teacher, 55% agreement.
- First LoRA (T5): tier 2.10 -> 2.75, gap halved, illegal moves eliminated,
  JSON held. Checkpoint selection mattered: iter-750 (3.15) beat iter-1000
  (2.75) and was promoted/published as the v1 line.
- Learning points: distillation works at all at 2B; later checkpoints are not
  automatically better; publish the measured best, not the last.

### Era 3 (2026-05-25..26): the Gemma 4 pivot and corpus poisoning

- mlx-lm loader bug worked around with a 6-line local sanitize patch (the
  patch is still load-bearing today).
- Untuned Gemma 4 E2B beat untuned 3n (2.55 vs 2.10). But v2 (same recipe as
  the successful v1) REGRESSED on every checkpoint: best 2.45 < v1's 3.15,
  with doom-loop confabulation traced to the corpus containing the teacher's
  own lost-game oscillation responses. First statement of CORPUS POISONING:
  the corpus teaches the failure as effectively as the skill.
- Learning point: a better base does not survive a poisoned imitation signal;
  and "same recipe" does not transfer across bases.

### Era 4 (2026-05-27..29): full-game truth

- The first full-game runs reframed everything: v1.1 wins-or-dooms within ~35
  turns; Gemma 4 falls into the QS col5/col7 loop from turn ~6 REGARDLESS of
  which LoRA is attached; a corpus filter that gained +0.40 tier on the
  single-turn bench produced fc=0 in play. THE BENCH LIES became the
  standing rule: single-turn accuracy does not measure play.
- Temperature micro-probe: the loop move was chosen 20/20 at temps
  0.4/0.7/1.0. Not a greedy-decoding artifact; not fixable at inference time.
- Late-game prompt audit: PRIOR REASONING was 37% of the prompt and
  byte-identical during loops, a positive-feedback amplifier (the era-1
  mechanism, confirmed on the real system). The INTENT-field idea was
  evaluated and REJECTED on degeneration-of-thought grounds before any spend:
  the first documented "cheap analysis kills a lever before GPU does".
- Learning points: play is the only trusted metric; loops are base-model-deep;
  fix-by-prompt has a narrow lane (state, not logic).

### Era 5 (2026-05-30..06-05): filter closure, v1.3/v1.4, the stall probe

- v4-A (gemma-3n + reversal-filtered corpus): best tier 2.80 < v1.1's 3.15,
  and in play the filter PUSHED gemma-3n INTO the Gemma-4-style loop
  (collapsed v1.1's fc=3 to 0). Corpus filtering closed on BOTH bases. Also
  the era of the three fabrication incidents (results written before
  measuring), which produced the hard rule that still governs every report.
- Stall-field probe: rendering explicit stall/repetition STATE to the student
  changed nothing (0/40 escapes vs 0/40 baseline). The obedience-trap
  diagnosis (model sees the signal and ignores it, or freezes) was confirmed
  behaviorally.
- v1.3's hard anti-undo predicate exposed the design hole (31B
  obeys-and-freezes, 26B ignores-and-loops); v1.4 shipped as text-only
  STRATEGY GUIDANCE cleanup.
- Learning points: filtering is not the lever; prompt-state is not the lever
  for entrenched behavior; hard predicates in prompts backfire asymmetrically
  across model sizes.

### Era 6 (2026-06-06..08): prompt fidelity, infra, and the win-data map

- v1.5 (draw-directive deleted, stall counters added) and v1.6 (DRAW TIMELINE
  doc-bug fixed, recycle logged into RECENT MOVES) shipped. The
  informational fixes WORKED (the invisible-recycle false-loop died); the
  behavioral hopes did NOT (stall counters cited 3/277 turns, 0 resigns on
  the first dead board; the resign lever eventually tallied 1 fold vs 28+
  proven-dead no-folds).
- HF storage migrated to append-only Parquet shards (974 MB pushes -> ~100 MB
  repo, 6.4x compression, incremental uploads).
- The win-data research mined 32.8k interactions: teacher 31% (average-human)
  vs a 43-52% imperfect-info ceiling; dead deals only ~10% of losses; tight
  loops are 65-77% VISIBLE in the existing 10-move window (sees-and-ignores,
  so rendering more history attacks the minority diffuse case only);
  post-recycle draws justified by "discovery" language 97.9% of the time even
  when nothing new can appear.
- Learning points: close information gaps, expect no behavioral miracles;
  the teacher's loss profile is mostly behavioral, not informational; measure
  where the win-rate actually leaks before proposing levers.

### Era 7 (2026-06-08..11): ORPO and the eval-fidelity reset

- ORPO preference trainer built (the memory-safe slice-trick head projection
  dates here and now powers GRPO). The pilot looked directional.
- Then the fidelity audit: the play harness had NO recycle move, sent forced
  positions to the model, used a stale v1.0 prompt, rendered corrupt
  foundations, and emitted duplicate ace moves. On the FIXED harness every
  prior absolute number was void: untuned base jumped 2.83 -> 7.92 mean fc
  (loop onset t4 -> t68), and ORPO v7-300 INVERTED to -4.42 paired, 0/24
  decks better. Mechanism: the preference pairs never contained draws, so the
  tuned policy drew 0.5% of the time and starved.
- Learning points: EVAL FIDELITY PRECEDES CONCLUSIONS (the most expensive
  lesson of the project, ~2 weeks of verdicts voided); preference/RL data
  must span the full action space; parse deaths are a first-class loss mode
  (2 of the base's best games died on JSON, not play).

### Era 8 (2026-06-12..17): the won-only gate and the naming of volume

- On the now-faithful harness: won-only SFT beat untuned base 12/13 decks
  (+13.4 mean fc, 4 wins vs 1), stochasticity confound refuted by a
  base-at-temperature control. First time training beat untuned under a
  TRUSTED eval. First full student win banked (seed 495097115, adjudicated
  cap-truncation).
- The matched-size ablation (all-successes at the same row count) TIED the
  won-only gate: the lever was VOLUME, not the won-filter. The full-corpus
  "volume" arm became the lead and was published to HF.
- Learning points: always run the matched-size control before crediting a
  filter; adjudicate final positions (raw 2 vs adjudicated 4 on loopcompress
  showed caps hide wins); name arms after the proven lever.

### Era 9 (2026-06-17..23): corpus engineering, in-dist vs generalization

- The family: closeout (harsh) fixed false-resigns but regressed mid-game
  reach, net flat; volcloseout (gentle) hit 8/13 in-dist TWICE (vs volume
  5/13) and was published with an honest card; volstrategy (27 strategy-text
  rows) 7/13 with learned-but-imperfect resign; volcombo showed the two
  levers INTERFERE (6/13, resign behavior erased); volsolver (solver-grounded
  rows) tied 8/13 in-dist with a footprint control (volsolver_lite, 324 rows,
  7/13) proving the lift was the FORM not the row count.
- Then the generalization pass collapsed all of it: on fresh decks volcloseout
  4/12, v0620 3/12, volsolver 3/12, vs volume's 5/12. And +12% more harvest
  data was FLAT on both recipes. The static-corpus line closed.
- Learning points: in-dist lift is cheap and misleading; generalization is
  the only promotion currency; levers that each work can interfere when
  combined; more teacher data stopped mattering (the imitation ceiling made
  visible).

### Era 10 (2026-06-23..07-01): on-policy SFT

- Best-of-N gate: temp-0.7 sampling solved 3/7 winnable decks greedy volume
  lost (~8.6%/sample): the reward signal EXISTS but is sparse. The corrective
  harvest on deep near-misses recovered 5/6 (30%/sample; the one miss was
  resign-blocked at fc50). Resign calibration finding: greedy volume folds
  correctly (3 of 4 resigns were on proven-dead boards); TEMPERATURE breaks
  the calibration.
- STaR: x4 oversample of 3 correlated self-wins collapsed generalization
  (0 wins paired vs volume 3); x1 halved the damage (2 v 5, fc -94) proving
  dose was not the cause; the diverse balanced 8-deck retrain erased the
  catastrophe (fc -2) and reached PARITY (4 v 5). Every configuration of
  imitating your own wins landed at-or-below volume.
- Learning points: sampling finds wins the greedy policy misses (the
  diagnosis that justified RFT); trajectory DIVERSITY, not mixing ratio, is
  the binding constraint for self-training; and even fixed, SFT-on-self-wins
  cannot exceed the policy that generated the wins. Also the early-read trap:
  the x1 result looked like a rescue on the subset chosen from x4's failures
  and inverted on the full paired set.

### Era 11 (2026-07-01..02): RFT / GRPO iteration-0

- Built in one window on era-7's slice-trick and era-10's rollouts: an
  advantage corpus of 12,516 decision turns across 12 decks (group-relative
  advantage per rollout, resign turns excluded, 0 drift failures), a
  REINFORCE trainer that continue-trains volume, and chained runners.
- Run 1 (KL-free): monotonic collapse, logp -0.5 to -23, all 12 heldout decks
  parse_failure, 0 wins vs volume 5. Mechanism: -A*logp is unbounded below
  for A<0; ~half the corpus is negative-advantage; the gradient norms stayed
  ~0.3-0.5 (slow DRIFT, so clipping cannot bind); and because negative and
  positive completions share the JSON scaffold, suppressing the negatives
  suppressed the format itself (the "shared-scaffold gradients cancel"
  hypothesis was written down and falsified within 12 hours).
- The fix: a one-sided downward trust region against the cached volume
  reference logp, proven in a CPU self-test (-3.55 -> -3.68 with TR vs
  -> -2646 without), plus a double-forward bug fix (~2x step time). Run 2
  (run 1 + TR, one variable) trained with the exact target signature:
  pos-adv logp RISING (-0.53 -> -0.47) while neg-adv fell boundedly
  (-0.55 -> -0.74), separation 0.02 -> 0.27 by step 150. Killed at ~step
  190/200 at the compute-window boundary; EVAL PENDING. This is the open
  measurement the strategy doc branches on.
- Learning points: RL losses need trust regions even at LoRA scale and tiny
  LR; diagnose collapse mode (drift vs spikes) before choosing the fix; CPU
  self-tests that prove gradient SIGN and BOUND catch failures a loss curve
  hides.

## 3. Object-level learning points (about the problem)

1. **Play is the only trusted metric.** Single-turn benches mispredicted play
   in both directions (era 4: +0.40 tier / fc=0; era 7: healthy SFT loss /
   broken play; era 11: clean training curve / total parse collapse).
2. **The imitation ceiling is real and reached.** Volume (~33% fresh) sits at
   teacher level (~31%); more data flat, reweighting non-transferable,
   self-imitation parity. Only non-imitative signal (game reward) can exceed
   it.
3. **Corpora poison as well as teach.** Teacher loops trained into students
   (era 3); filtering them out regressed other capabilities (era 5); the cure
   was volume plus, later, reward-weighting, never surgery alone.
4. **Doom-loops are base-model-deep.** LoRA-invariant, temperature-invariant,
   state-rendering-invariant. Behavioral fixes require training signal
   (reward), not information.
5. **In-dist lift does not transfer by default.** Every corpus-engineered
   +3 in-dist collapsed to -1/-2 on fresh decks; volume alone held.
6. **Training data must span the action space.** ORPO's drawless pairs
   starved draws (0.5% rate); full-game trajectories avoid this by
   construction, which is why the GRPO corpus is turn-complete.
7. **JSON discipline is a capability.** It degrades under tuning (6x
   regression), dies under naive RL (run 1), and kills otherwise-winning
   games. Constrained decoding is the structural fix and is still unbuilt.
8. **Resign calibration is policy-fragile.** Well-calibrated greedy, broken
   at temp 0.7; unteachable via prompt (1 fold vs 28+); teachable via data
   but interferes with other levers (volcombo).
9. **Prompts close information gaps, not logic gaps.** DRAW TIMELINE and
   recycle-logging worked (they added true state); stall counters, draw
   directives, and hard anti-undo predicates failed or backfired.
10. **Self-referential context amplifies failure.** Reasoning trails and
    PRIOR REASONING act as few-shot examples of the model's own worst
    behavior; v1.6-era prompts dropped them.
11. **Adjudication changes verdicts.** Monte-Carlo winnability erred BOTH
    ways; the exact solver flipped 4+ corpus verdicts and separates
    cap-truncated wins from losses. Every terminal position gets solved now.
12. **Diversity binds before dose** for self-training data; and neither is
    sufficient to beat the generating policy.
13. **Sampling reveals headroom greedy hides**: ~8.6%/sample on hard decks,
    30% on near-misses. That headroom is exactly what RFT is built to
    harvest.
14. **Checkpoint selection is an experiment**, not a convention: iter-750
    beat iter-1000 once (era 2), and iter-1000 won once (era 8). Measure.

## 4. Meta-level learning points (about method)

1. **Eval fidelity precedes conclusions.** The 06-11 reset (missing recycle,
   stale prompt, forced-position sends, corrupt renders) voided two weeks of
   verdicts including a whole training program's. Byte-fidelity to the
   production render plus drift gates on every replay path are now
   non-negotiable.
2. **Never write a result before reading it off the run.** Instituted after
   three fabrication incidents (2026-05-30); has held since; the reason every
   number in this document has a report behind it.
3. **Pre-register gates and both branches.** The A4 arc (Phase-1 +0.27 noise
   -> Phase-1.5 -0.12), the x1 early-read trap, and the run-2 gate all show
   why: post-hoc reads on subsets chosen by prior failures invert.
4. **Run the matched control before crediting the treatment.** Won-only vs
   allsucc (volume was the lever); volsolver vs volsolver_lite (form was the
   lever); base-at-temperature (stochasticity was not).
5. **Validate tools before trusting verdicts.** pyksolve returned "winnable"
   for every input for weeks; the parser returned None and nearly produced a
   wrong kill call. Fix the instrument, then judge.
6. **One variable per run.** Run 2 = run 1 + trust region. The x1 ablation =
   x4 with only the mix changed. This is what makes single-window experiments
   decisive.
7. **Cheap analysis and CPU self-tests kill bad levers before GPU does.**
   INTENT rejected on paper; grpo_loss sign/bound proven on a stub model;
   probe modes measure worst-case memory before committing a night.
8. **Bounded compute windows, resumable chains.** Window-end kills lose
   minutes, not runs (checkpoints, skip-if-done, idempotent phases). GPU only
   inside a declared window.
9. **Paired evals against cached baselines**, exact adjudication, fixed deck
   sets with governed reuse (iteration vs promotion vs ship benches).
10. **Small-N honesty.** n=12 deltas of +/-1 are noise; pooled-48 for public
    claims; CIs on cards.
11. **Proxy fidelity.** Teacher experiments on the teacher model only; the
    Haiku arms and the 26B counterexample both showed thresholds do not
    transfer across models.
12. **Negative results are deliverables.** Ten closed levers with dated
    evidence is why the current strategy fits on one page and why nothing is
    being re-litigated.
13. **Full-stream ingest keeps denominators true.** Win rates read off the
    corpus are real because everything the harvester makes is ingested,
    including kills, which get adjudicated and counted.

## 5. The costly-mistakes ledger

| Mistake | Cost | Rule it produced |
|---|---|---|
| Unfaithful play harness (era 7) | ~2 weeks of void verdicts, one refuted training program | byte-fidelity + drift gates everywhere |
| Results written before measuring (3x, 05-30) | credibility, rework | the hard rule; every number traceable to run output |
| pyksolve trusted while broken | wrong winnability verdicts, 4+ flipped later | validate instruments; exact solver default |
| KL-free RFT run 1 | one training window + one eval window | trust region; self-test gradient sign AND bound first |
| x4 STaR without dose/diversity controls | one window + a misleading regression story | ablate dose, then diversity; matched controls |
| Early read on x1's failure-chosen subset | near-wrong verdict | full paired set before any adjudication |
| Daytime GPU burn past the window (07-02) | a day of laptop thermals, a killed run | bounded windows; kill-or-ask at wrap-up |
| a1d118 WATCH call on a one-card blip | a wasted continuation | breakout requires faceDownTotal falling + tripwire |
| 9000237 at 41% of a planned corpus (caught pre-train) | none (caught) | composition share gates in every fuse script |
| Trusting window-overlap loop counts | inflated severity readings | count exact reversals on the stitched stream |

## 6. What survived everything (the keep-list)

- The **volume recipe**: full clean corpus, no filter, no reweighting,
  iter-1000, published with honest cards. Beat or matched every challenger
  across ~15 head-to-heads.
- The **v1.6 prompt system** and its byte-faithful local twin (harness,
  extractor, adjudicator): the platform every trusted number stands on.
- The **exact solver** and adjudication discipline.
- The **slice-trick training stack** (ORPO's one lasting contribution),
  now carrying GRPO within 11 GB on a 16 GB laptop.
- The **report + pre-registration culture**: 76 dated docs; every strategic
  claim in the companion doc cites one.
- The **best-of-N machinery**: the diagnostic that both closed SFT (parity
  proof material) and opened RFT (the advantage corpus).

## 7. Bridge to the strategy

The strategy doc's pillars are these learnings, operationalized: Pillar 1
exists because learnings 2, 13, and era-11's healthy-but-unevaluated trust
region say reward is the only unfalsified lever; Pillar 2 because learning 7
(JSON as capability) is the largest measured non-play loss; Pillar 3 because
learnings 2 and 9 reprice the harvest to steady state; Pillar 4 because
section 5 above is publishable; Pillar 5 because sections 4-5 are the method
that made six weeks of solo work converge instead of thrash.
