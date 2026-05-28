# Cross-Turn Plan Continuity: INTENT Field Hypothesis and Bench Design

**Date**: 2026-05-28
**Status**: HYPOTHESIS, NOT COMMITTED. Out of v1.3 scope. Documented now to preserve the design rationale and to define what the v1.4 decision will be conditioned on. Do not implement without an explicit go decision after the v1.3 bench results land.
**Audited against**: `prompt_engineering_expert` deep-audit conducted 2026-05-28 (in conversation; see Sources section for the cited research).
**Related**:
- `docs/reports/20260528_prompt_v1_3_candidate_spec.md` (the prompt revision currently shipped)
- `docs/reports/20260527_late_game_prompt_audit.md` (the audit that surfaced the doom-loop pathology)

## 0. Why this document exists

After the v1.3 spec shipped, the question came up: "how can the model continue a plan across turns?" The model already writes coherent 2-4 move plans in its `strategic_plan` field every turn, but each turn re-derives a new plan because the prompt is stateless. The natural design instinct is to add an INTENT field carried forward by the harvester.

This instinct is correct in shape and dangerous in detail. The naive INTENT design reproduces the same anti-pattern (in-context imitation of prior model output) that v1.3 just removed by dropping PRIOR REASONING. The 2025 literature has a name for it: "degeneration-of-thought" in the Reflexion line of agent research [15][16]. The model commits to a plan, the plan becomes wrong, and the verification predicate is too weak to catch it.

This doc captures the hypothesis, the load-bearing audit finding, and the bench design needed to test whether a stronger predicate (state-delta verification) avoids the failure mode. It explicitly does not commit to shipping INTENT in v1.4.

## 1. The observation that motivates INTENT

From sampled `solitaire-ai-log-6b491a-1779877547228.json` turn 79:

> *"Move [2] moves the 9S-8D sequence from Column 3 to Column 1 (TD)... This move exposes the TH (Red 10) and JS (Black J) in Column 3. Following this, the JS-TH sequence can be moved onto the JH (Red J) in Column 2, leaving only the QH (Red Q) on top of the hidden card..."*

That is a coherent 3-step forward plan. The model executes step 1 correctly. Turn 80, the harvester sends a fresh stateless prompt. The model re-reads the new board, writes a different plan that ends in moving 9S-8D back to col 3. Across the session this cycle repeats 165 times.

Conclusion: the bottleneck is not the model's ability to plan within a turn. The bottleneck is the lack of any persistence mechanism for what was just decided. A v1.4 INTENT design is the natural answer.

## 2. The naive design (and why it is at risk)

### 2.1 Naive INTENT schema

Add to the response schema:

```
plan_next_steps: [string, string]
```

Two forward moves the model intends to execute on the next two turns.

### 2.2 Naive prompt rendering

Turn N+1 prompt gains an INTENT block, derived by the harvester from turn N's response:

```
INTENT FROM PREVIOUS TURN (your committed plan):
  1. <step 1>
  2. <step 2>

Rule: if step 1 is in the current LEGAL MOVES, return its move_index.
If step 1 is no longer legal, write a new plan and execute its first step.
```

### 2.3 The load-bearing audit finding

The verification predicate ("if step 1 is in LEGAL MOVES") is too weak. It catches the case where the plan became syntactically illegal but not the case where the plan became semantically suboptimal. The suboptimal-but-legal case is the dominant Klondike failure path.

Concrete worked example. Model commits plan `[move JS-TH col 3 to col 2, expose face-down col 3]` on turn N. Executes step 1. Turn N+1 the stock draws `AS` to the waste. Under the naive predicate, step 1 of the remaining INTENT (`expose face-down col 3` is no longer needed because the JS-TH move already exposes; let us use a different example) is still legal, so the model executes it. The objectively better move (`AS waste to spades foundation`) is bypassed. The model continues its now-stale plan and compounds divergence for several turns.

This is the canonical "state drift" failure mode in the agent-memory literature [16][17]: the agent's belief about the optimal action diverges from the true state because the verification check is too permissive. Reflexion (Shinn et al. 2023) and its 2025 follow-ups [15] all document this; the explicit term is "degeneration-of-thought."

The label change from PRIOR REASONING to INTENT is real (it does change what the model imitates) but it does not eliminate the underlying problem that a model-generated text re-entering the context window is an in-context example of the model's prior reasoning, and Halawi et al. [4] is clear that models imitate in-context patterns even when they "know" they're wrong.

## 3. The mitigated design (state-delta verification)

The predicate needs to be semantic, not syntactic. The model writes the expected post-plan state metrics alongside the plan, and the predicate compares against current state before executing.

### 3.1 Augmented schema

```
plan_next_steps: ["move JS-TH col 3 to col 2", "expose face-down col 3"]
plan_expected_delta: {"foundationCards": "+0", "faceDownTotal": "-1"}
```

### 3.2 Augmented prompt rendering

```
INTENT FROM PREVIOUS TURN (your committed plan):
  1. move JS-TH col 3 to col 2
  2. expose face-down col 3
  Expected state delta after step 1: foundationCards +0, faceDownTotal -1

Rule (priority order, highest first):
  1. If LEGAL MOVES contains any move to a foundation, execute it
     (overrides intent).
  2. Else if step 1 of INTENT is in LEGAL MOVES AND no move in LEGAL MOVES
     would produce a strictly larger delta on
     (foundationCards, faceDownTotal), execute it.
  3. Else write a new plan and execute its first step.
```

Priority 1 catches the Ace-appeared case. Priority 2 catches the "intent step is still legal but now suboptimal" case. The model has to compute the delta comparison but every piece of state needed is already in the prompt.

### 3.3 Why the priority list is itself a risk

The override list at priority 1 ("any foundation move") is a hand-coded policy fragment. If a different override matters (e.g., "if a face-down card was just revealed and a productive move on it is now available"), the predicate misses it. You cannot enumerate all overrides without re-implementing the playing policy in the rule. Hence the alternative below.

## 4. The stronger design (solver-verified commitment)

Replace the LLM-side state-delta predicate with a harvester-side solver check.

When the model writes a plan, the harvester runs pyksolve on the projected board state after the 2-step plan and checks whether the projected state strictly dominates the current state on `(foundationCards, faceDownTotal)`. If yes, the plan is committed and the next turn's INTENT block is enabled. If no, the plan is rejected and no INTENT block appears on the next turn.

This converts plan-commitment into a solver-mediated gate. The LLM still decides; the solver gates persistence. Adds 100-500 ms per turn for the lookahead.

Defensible if you can afford the latency. The state-delta predicate (Section 3) is the cheaper approximation.

## 5. The distillation interaction (potentially disqualifying)

The harvester corpus becomes training data for a student LoRA. If the teacher learns to output `plan_next_steps`, the student inherits this. The student is deployed without the harvester's INTENT-rendering harness (unless the harness is replicated at inference time, which adds complexity).

Two opposing effects:
- **Attenuation hypothesis**: INTENT reduces teacher doom-loops; student inherits fewer doom-loops. Net positive.
- **Propagation hypothesis**: student learns "writing a plan means committing to it" and produces hyper-rigid execution at inference time without the verification predicate. Net negative.

You cannot predict which dominates from prompt design alone. The Voyager skill-library result [18] establishes that persistent memory structures are load-bearing for agent performance (15.3x lift in their benchmark), but the structure has to match between training and deployment for the lift to transfer.

Memory at `memory/v2-distillation-teacher-doom-loop.md` already notes the student inherits the teacher's doom-loop pathology in the v1.x corpus. INTENT might attenuate this or propagate it in a new form; both are plausible.

**Implication**: a v1.4 INTENT decision cannot be made on teacher-side bench results alone. The distillation A/B (Section 6.3) is the load-bearing experiment, not the teacher bench.

## 6. The bench design (if and only if we proceed to v1.4)

### 6.1 Pre-condition

This bench is conditional on the v1.3 bench (in `docs/reports/20260528_prompt_v1_3_candidate_spec.md` Section 4) producing positive results. If v1.3 alone fixes the doom-loops (median plateau drops from 35 turns to 15 turns or better on fresh seeds), the cross-turn plan persistence may not be necessary. The model planning from scratch each turn given current state may be sufficient once the PRIOR REASONING anti-pattern is gone.

Decision gate: only proceed to this v1.4 bench if the v1.3 bench shows clear gains AND the residual failure mode is still "model writes good plan, abandons it, no continuity." If the residual is something else (e.g., "model is just bad at picking moves period"), INTENT does not address it.

### 6.2 Four-arm teacher bench

| Arm | Prompt | Tests |
|---|---|---|
| E | v1.3 (post-bench winner) | baseline |
| F | v1.3 + INTENT with legality predicate (Section 2.2) | the naive design; predicted to regress or null |
| G | v1.3 + INTENT with state-delta predicate (Section 3.2) | the audited design |
| H | v1.3 + solver-verified commitment (Section 4) | solver-mediated persistence |

Pairwise diffs:
- F vs E: does plan persistence help at all when the predicate is weak?
- G vs F: does semantic verification beat syntactic verification?
- H vs G: does solver verification beat heuristic verification?

If F regresses and G recovers, the audit prediction is confirmed and G is the recommended teacher prompt.
If F is neutral and G shows small gains, the verification strength matters less than expected.
If H beats G by a wide margin, the LLM cannot self-verify even with state-delta and the solver is load-bearing.

### 6.3 Distillation A/B

For whichever teacher arm wins (G or H), collect a full harvester corpus. Train a student LoRA. Compare against the v1.3 baseline student (trained on a corpus without INTENT).

Benchmark both students on the standard 20-state eval set. Ship the harness configuration that produces the better student, even if it is not the better teacher.

This is the load-bearing experiment per Section 5. Skipping it would commit v1.4 based on teacher performance and inherit unknown student-side risks.

### 6.4 Seed set and metrics

Same as v1.3 bench (Section 4.2 and 4.3 of the v1.3 spec). 10 seeds total: the 2 known-winnable anchors (`2967897202`, `3263196305`) plus 8 fresh seeds. Metrics per session: moveCount, finalProgress, max-plateau-length, session-wide top-3 oscillation pair counts, outcome.

### 6.5 Cost estimate

- Teacher bench: 4 arms x 10 seeds x ~200 moves x ~2 s/turn x ~50% error rate x ~2 retries = roughly 16 hours wall-clock.
- Distillation: ~6 hours per student LoRA on the local rig.
- Total: ~28 hours wall-clock, runnable across one weekend if the harness supports parallel arms.

## 7. Pre-registered predictions

Logged here so the bench result can be evaluated against falsifiable claims rather than narrative-fitted ex post.

| Prediction | Confidence | Falsifies if |
|---|---|---|
| Arm F (naive INTENT) regresses on max-plateau-length vs Arm E | 70% | F matches or beats E on plateau |
| Arm G (state-delta INTENT) beats Arm E on plateau by 20% or more | 55% | G matches or regresses vs E |
| Arm H (solver-verified) beats Arm G on plateau by 10% or more | 50% | H matches or regresses vs G |
| Student trained on the G corpus does NOT clearly beat student trained on E corpus | 60% | G-student wins decisively on 20-state eval |
| The dominant residual failure mode in Arm G is "committed to a plan that was right at commit-time but wrong by execution-time" | 65% | Residual failures are not commit-stale-plan |

If the residual failure mode prediction is wrong, the diagnosis of the v1.3 outcome was probably wrong too, and the v1.4 design should be reconsidered from observed failures rather than this hypothesis.

## 8. Decision gates

- **Abandon INTENT if**: v1.3 bench shows large gains AND residual failure mode is not "abandoned plans." In that case the model is planning fine; persistence was not the bottleneck. Move on to v1.4 inference-config investigation (sampling temperature, thinking budget) rather than more prompt revisions.
- **Commit to G if**: v1.3 gains are partial, F regresses, G recovers, AND the G-student matches or beats the E-student in distillation A/B.
- **Commit to H if**: G underperforms expectations but H produces clear lift. Accept the latency cost.
- **Pivot away from INTENT entirely if**: G regresses too. The hypothesis was wrong; persistence is not the right mechanism. Reconsider from observed failures.

## 9. Items explicitly OUT of scope

- v1.4 prompt revisions other than INTENT. If the v1.3 bench surfaces other findings, those are tracked in a separate doc.
- Solver-as-decision-maker (Alternative (d) from the audit, where pyksolve picks moves and the LLM just types). That is a production-deployment option, not a research/distillation option. Out of scope.
- Two-stage prompt within one turn (Alternative (e)). Does not address cross-turn persistence; rejected in the audit.
- Single-step intent `next_move` (Alternative (b)). Too shallow; redundant with `move_index`. Rejected in the audit.
- Appending plan to RECENT MOVES as pseudo-entries (Alternative (c)). Blurs taken-vs-planned distinction; worse anti-pattern risk. Rejected in the audit.

## 10. Open questions for the harvester team (only if v1.4 proceeds)

1. Can the harvester carry state across turns for INTENT, or is the per-turn prompt construction strictly stateless? If stateless, the harness needs a session-level cache for `plan_next_steps`.
2. Can the harvester compute state deltas (Section 3.2) on the fly, or does the predicate need to be entirely model-side? Server-side delta computation is cleaner but adds harness complexity.
3. Can the harvester run pyksolve in-line per turn (Section 4)? If not, solver-verified commitment is out of reach regardless of audit findings.

## 11. What ships if INTENT is committed

- Updated prompt template with the INTENT block rendering.
- Schema addition: `plan_next_steps`, `plan_expected_delta` (or whatever variant the bench validates).
- Harvester-side session cache for cross-turn state.
- Updated `data/DATASET_NOTES.md` with the bench result and a note that the corpus now contains plan-commitment training data.
- A short addendum to `memory/v2-distillation-teacher-doom-loop.md` noting the propagation-vs-attenuation result.

## Sources

- [4] Halawi et al., "Overthinking the Truth": https://arxiv.org/pdf/2307.09476
- [5] "Understanding In-Context Learning from Repetitions": https://arxiv.org/pdf/2310.00297
- [15] MAR: Multi-Agent Reflexion (degeneration-of-thought): https://arxiv.org/html/2512.20845
- [16] Memory for Autonomous LLM Agents (state drift): https://arxiv.org/html/2603.07670v1
- [17] CogMem (state-drift in multi-turn reasoning): https://arxiv.org/pdf/2512.14118
- [18] Voyager skill-library performance result: https://towardsdatascience.com/a-practical-guide-to-memory-for-autonomous-llm-agents/
- [19] Verify Before You Commit / SAVER: https://arxiv.org/pdf/2604.08401
- [20] Plan Verification for LLM-Based Embodied Task Completion Agents: https://arxiv.org/pdf/2509.02761
