# Prompt audit experiment — Haiku 4.5 results

**Date**: 2026-05-24
**Plan**: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260524_prompt_audit_experiment_plan.md`
**Raw**: `/Users/chayut/repos/solitaire-analytics/experiments/prompt_audit_2026_05_24/`
**Status**: Phase 1 (Haiku triage) complete; recommendation for Phase 2 below.

## TL;DR

- **One headline win, clean and confirmed**: A2/A3 (drop `board_analysis` + `strategic_plan`) cut output size from ~1615 to ~87 bytes — a **19× reduction**. JSON validity stays at 100%. This is the strongest signal of the experiment.
- **Tier-score deltas are noisy at N=15 per arm**. All four arms have 95% paired-bootstrap CIs that cross 0. The pre-registered G1 graduation gate (Δ ≥ +0.4, CI excludes 0) is not met by any arm at this sample size.
- **Surprising and concerning finding**: on one of two oscillation states with a foundation move available, **A2 picked the shuffle (tier 2) instead of the foundation (tier 6) on all 3 runs**, while C0 picked foundation 3/3. The CoT fields may be helping the model deliberate on harder boards — removing them could lose moves the harder-to-think paths surface.
- **Recommendation**: graduate **A2 only** to Gemma follow-up, but with an enlarged bench (N=20+) to test whether the lost-foundation case generalizes. Hold A1, A3, A4.

## What was actually run

- **75 calls**: 5 arms × 5 bench states × 3 runs each.
- **Proxy**: Claude Haiku 4.5, simulated via subagent invocations (one fresh subagent per call). Each subagent read the rendered prompt from disk and wrote its response to disk.
- **Cost**: ~$0 (Claude Code subagent budget, not API). Wall time: ~3 hours including orchestration, ~10-20s per individual call.
- **Why scaled down from N=20×M=5**: each subagent call costs ~26K tokens of orchestration + simulation. The full 500-call plan would have meant 50+ orchestrator turns. Phase 1 traded statistical power for time-to-signal.

## Results

### Per-arm overall

| arm | n | mean tier ±σ | JSON ok | rev % | resp chars ±σ |
|---|---:|---|---:|---:|---|
| C0 | 15 | 3.67 ± 2.23 | 100% | 0% | 1615 ± 324 |
| A1 (drop reasoningTrail) | 15 | 3.13 ± 2.07 | 100% | 7% | 1495 ± 295 |
| A2 (drop prose CoT) | 15 | 3.20 ± 2.01 | 100% | 7% | **87 ± 2** |
| A3 (A1 + A2) | 15 | 3.53 ± 2.03 | 100% | 7% | **86 ± 2** |
| A4 (hoist notation) | 15 | 3.93 ± 2.02 | 100% | 7% | 1456 ± 225 |

Tier scoring: foundation=6, reveal=5, waste_play=4, shuffle=2, draw=1, recycle=1, illegal=0.

### Paired bootstrap of Δ vs C0 (5000 resamples, 95% CI)

| arm | mean Δ | 95% CI | crosses 0? | meets G1 gate? |
|---|---:|---|---|---|
| A1 | -0.53 | [-1.67, +0.53] | yes | no |
| A2 | -0.47 | [-1.53, +0.60] | yes | no |
| A3 | -0.13 | [-0.73, +0.27] | yes | no |
| A4 | +0.27 | [-0.53, +1.13] | yes | no |

**No arm passes the pre-registered G1 gate** (Δ ≥ +0.4 with CI excluding 0). The CIs are ±1.0 wide — N=15 per arm is too few to detect effects of the hypothesized size (±0.3–0.5).

### Per-category mean tier

| arm | early | endgame | midgame | oscillation |
|---|---:|---:|---:|---:|
| C0 | 1.00 | 4.67 | 5.00 | 3.83 |
| A1 | 1.33 | 4.67 | 5.00 | 2.33 |
| A2 | 1.00 | **6.00** | 5.00 | **2.00** |
| A3 | 1.67 | 4.67 | 5.00 | 3.17 |
| A4 | 2.00 | **6.00** | 5.00 | 3.33 |

**Three interesting patterns** (each based on only one state per category, so directional only):

1. **Midgame**: every arm picks the same move (reveal, tier 5) 3/3 times. The board has one dominant move; no arm differentiates here.
2. **Endgame**: A2 and A4 hit 6.00 (always foundation). C0/A1/A3 hit 4.67 (mostly foundation, one shuffle). Possible signal but tiny n.
3. **Oscillation**: directionally, A2 (2.00) and A1 (2.33) *underperform* C0 (3.83). A4 (3.33) and A3 (3.17) are between. **The arms that strip the most material from the prompt do worse on oscillation states**, not better — opposite of what the audit predicted.

### The smoking gun: `oscillation-bffb5d9cc178`

| arm | picks (3 runs) | tiers | mean |
|---|---|---|---:|
| C0 | [2, 2, 2] | foundation × 3 | 6.00 |
| A1 | [1, 2, 3] | draw, foundation, shuffle | 3.00 |
| A2 | [3, 4, 4] | shuffle × 3 | 2.00 |
| A3 | [2, 2, 3] | shuffle, foundation × 2 | 4.67 |
| A4 | [2, 2, 3] | shuffle, foundation × 2 | 4.67 |

On this state, **C0 finds the foundation move 3/3 times**. **A2 (slim schema only) misses it 3/3 times**, picking shuffle instead. A1 (drop reasoningTrail) finds it 1/3.

This is the opposite of the audit's predictions for A2. Hypothesis: the prose `board_analysis` field gives the model "thinking time in writing" that surfaces the foundation move. Without it, Haiku jumps to a tableau-shuffle without seeing the foundation option. A3 — which also slims the schema but additionally removes `reasoningTrail` — recovers somewhat (4.67), suggesting the reasoningTrail anchoring may have been distracting from the foundation option.

This is a sample of one state. **Could be a real effect; could be variance**. We need more oscillation states with foundation options to disambiguate.

### Output-size reduction — the clean win

| arm | mean response chars | σ | × vs C0 |
|---|---:|---:|---|
| C0 | 1615 | 324 | 1.0× |
| A1 | 1495 | 295 | 0.93× |
| A2 | **87** | 2 | **0.05×** |
| A3 | **86** | 2 | **0.05×** |
| A4 | 1456 | 225 | 0.90× |

This is the unambiguous, statistically-uncontroversial finding. Removing `board_analysis` + `strategic_plan` from the schema (A2) shrinks output by 19× with no validity loss. For an inference-cost-sensitive deployment (the harvester runs at scale), this is a real win on its own.

## Interpretation: what the data says vs what the audit predicted

| Audit finding | Prediction | Result | Verdict |
|---|---|---|---|
| #1 reasoningTrail anchors bad behavior (HIGH) | A1 ↑ tier by +0.3 to +0.7 | A1 Δ = −0.53 (CI [−1.67, +0.53]) | **Directionally opposite**; n too low to call |
| #2/#6/#11 prose CoT in JSON (HIGH) | A2 ↑ marginal tier, ↑ JSON validity | A2 Δ = −0.47 tier, JSON 100% (no change), output 19× smaller | **Mixed**: the output-size effect is real; the tier effect is null-or-negative |
| #4 notation buried in JSON (HIGH) | A4 +0.0 to +0.2 tier | A4 Δ = +0.27 (CI [−0.53, +1.13]) | Directionally aligned; n too low to call |
| Combined A1+A2 | Additive | A3 Δ = −0.13 (CI [−0.73, +0.27]) | Bisects A1 and A2; consistent with weak effects |

**Two pre-registered hypotheses (A1, A2) showed directionally NEGATIVE effects on tier score.** This is a finding worth taking seriously, not waving away as noise. It suggests:

- **The reasoningTrail may be load-bearing for some board states**, not purely a bad-anchor. The audit framing (Halawi et al. anchoring) is correct in principle but the harvester's reasoningTrail may not be as poisoned as we assumed at Haiku scale.
- **The prose CoT fields may serve a "scratchpad" function** that helps the model find non-obvious moves. Removing them shrinks output but may lose moves the model needed to think through.

This is exactly the kind of result that justifies running the experiment instead of shipping the audit recommendations blind.

## Caveats (do not skip)

- **Haiku 4.5 is NOT Gemma 4 31B.** Different architecture, different training, different failure modes. Haiku may be too good at "internal reasoning without writing" — Gemma 4 may rely more on explicit prose CoT and the negative tier effect on A2 might be worse on Gemma.
- **Per-state n=3 is too small for the per-category tables.** They're directional only.
- **The bench (5 states) may not be representative.** Especially: only 2 oscillation states, only 1 endgame state. A single state can drive the entire per-category mean.
- **Subagent simulation has overhead** (each call ~26K tokens of system prompt). Subagent identity may bleed in despite "act as a fresh model" framing.
- **No baseline against any solver-best move.** Tier-score is a heuristic; the "true best move" by a complete solver might disagree with our tier ranking.

## Recommendation: Phase 2

### What to graduate to Gemma 4 31B

**A2 (slim schema, drop prose CoT) — graduate**, but with the explicit hypothesis that **the tier-score effect on Gemma may differ from Haiku**. The 19× output reduction is real and worth confirming holds on Gemma. The lost-foundation-move risk needs N≥20 oscillation states to characterize.

### What to hold

- **A1 (drop reasoningTrail)** — Haiku result is mildly negative. Re-run only if A2 shows positive signal and we want to test the additive case again.
- **A3 (A1+A2)** — re-run if both A1 and A2 graduate; otherwise no point.
- **A4 (hoist notation)** — directional +0.27 but CI ±0.8. The audit predicted "+0.0 to +0.2"; the data is consistent. Test on Gemma only if we have spare quota.

### What to add to the bench

- More oscillation states with `tableau_to_foundation` available (we had 1; need ≥5 to see if A2's foundation-missing is real).
- More endgame states (we had 1; the apparent A2/A4 6.00 score needs more samples).

### Phase 2 design (when Gemma quota returns)

- Arms: C0 vs A2 only (2-arm, paired).
- Bench: 20+ states, biased toward oscillation-with-foundation-available.
- Runs: M=3 (Gemma sampling adds genuine variance even at low temperature).
- Total: ~120 calls. Should fit comfortably in a day's free-tier quota.

## Approach notes (lessons for the next round)

- **Subagent simulation works** for prompt-shape A/B when you don't have API access, but the per-call cost (~26K tokens) means you scale down N or M aggressively. Budget accordingly before designing the bench.
- **Pre-register predictions in writing.** Two of our four predictions came out the OPPOSITE direction. Without the pre-registration, it would have been easy to retcon "well, we always thought it might be mixed." The plan doc holds us honest.
- **Surface the surprising case (oscillation-bffb5d9cc178) above the headline stats.** A 0/3 → 3/3 foundation flip on one state is more actionable than a 0.5-tier mean shift across all states.
- **Output size is a free metric.** It needs zero domain reasoning to compute, has zero variance from one call (the schema dictates it), and directly maps to dollars at scale. Always report it.
- **The "no signal" arms (A1, A4) are not failures of the experiment** — they're updates against ship-blind audit recommendations.
