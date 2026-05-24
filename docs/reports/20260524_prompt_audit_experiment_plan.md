# Harvester prompt audit — experimental plan

**Date**: 2026-05-24
**Owner**: Chayut
**Status**: draft — pending arm-selection confirmation
**Related**: `20260523_prompt_format_ab_status.md` (layout A/B), `20260524_harvester_team_layout_ask.md` (pending TS change)

## Question we're trying to answer

The audit of the harvester prompt to `gemma-4-31b-it` produced ~15 findings across HIGH/MEDIUM/LOW severity. We don't have the Gemini quota to test all of them, and we shouldn't ask the harvester team to ship anything we haven't validated.

**Goal**: triage the candidate fixes on a cheap local proxy (Claude Haiku 4.5) so we only spend Gemma 4 31B budget on arms that show real signal.

## Scientific framing

We treat this as a multi-arm A/B with one control and four orthogonal treatments. Each arm is a modification of the rendered prompt — provider, model, decoding params, and bench remain identical across arms.

We pre-register hypotheses with directional predictions on primary metrics. Arms that fail to move the proxy needle by a minimum effect size do not graduate to the Gemma 4 31B follow-up.

### Why Haiku 4.5 as the proxy

- **Cheapest local option**. ~$0.001/call vs Gemma 4 31B's free-tier-exhausted reality.
- **Already calibrated**. Prior session (20260523 report) used Haiku as one of two proxies; we have intuition for where it agrees and disagrees with Gemma.
- **Reasoning model**. Same anti-pattern surface area as Gemma 4 IT for findings #2, #3, #6, #11 (asking thinking model for visible CoT).

### Known proxy caveats (document up front, don't pretend they don't exist)

| Risk | Mitigation |
|------|------------|
| Haiku is from a different family — may not share Gemma's specific failure modes (e.g. draw-loop fixation). | Treat Haiku results as a *floor*, not ground truth. Arms that don't move Haiku metrics still get vetoed; arms that move them earn a Gemma callback. |
| Haiku is stronger overall — may mask issues that hurt Gemma more. | Pick bench states where Haiku itself struggles (oscillation contexts, ambiguous midgame). |
| Different tokenizer / context handling. | Report token deltas with caveat; don't use as gate. |
| Haiku has stable adaptive thinking; Gemma 4 IT has explicit `<\|think\|>` channel. | Use Haiku's `thinking` enabled; strip thinking from history if any arm needs multi-turn. |

## Candidate arms

**Control (C0): current production prompt** — verbatim render from `compare_prompt_formats.py` (original format), with `reasoningTrail`, `recentMoves`, full prose schema, `notation` field buried in JSON.

**Arm A1: drop `reasoningTrail`** — single-field removal from the JSON payload. Tests audit finding #1 (in-context bad-example anchoring; Halawi et al.).

**Arm A2: drop prose CoT fields from schema** — remove `board_analysis` and `strategic_plan` from the response schema; final_decision only. Tests findings #2, #3, #6, #11 (asking thinking model for visible CoT; redundant prose; CoT inside JSON).

**Arm A3: combined A1 + A2** — tests interaction. If A1 and A2 both win, does the combo compound or saturate?

**Arm A4: hoist `notation` into rules preamble** — move the "Cards: rank then suit..." string from inside the JSON state object to the top-of-prompt rules section. Tests finding #4 (rules buried in data payload; lost-in-the-middle).

### Arms we are *not* testing this round (with reasons)

| Finding | Why deferred |
|---------|--------------|
| #5 Structured Outputs (Gemini `responseSchema`) | Provider-side, not prompt content. Worth a separate experiment with real Gemini quota. |
| #7 Hedged heuristics → hard predicates | Requires real heuristic design work; out of scope for prompt-shape A/B. |
| #8 Unranked rules → numbered priority | Same — content design, not structural. |
| #9 `recentMoves` semantics fix | Already in the 20260524 harvester ask (combined with layout change). Don't re-litigate. |
| #10 System/user split | Provider-side, requires API call shape change. |
| #11 XML structural blocks | Cosmetic — none of our metrics will detect a difference. |
| #13–15 LOW | Skip; noise-level. |

## Hypotheses (pre-registered)

For each arm, the prediction on the **primary metric** (mean move-tier score, scale 0–5) is stated with direction and effect size:

| Arm | Hypothesis (H1) | Predicted Δ vs C0 | Reasoning |
|-----|-----------------|--------------------|-----------|
| A1 | Removing the bad-rationale anchor reduces draw-loop / shuffle moves; chosen-move tier improves. | +0.3 to +0.7 | Halawi et al. effect size in their experiments was substantial; we expect smaller here because we have only ~7 examples of bad reasoning being shown vs their longer chains. |
| A2 | Removing forced prose CoT lets the model spend more output budget on the structured decision; fewer truncations, slightly better moves. | +0.0 to +0.3 | Haiku has plenty of output budget so the budget-pressure effect is weak; expect tiny direct gain. Primary value is JSON validity ↑. |
| A3 | A1 + A2 stack but with diminishing returns. | +0.3 to +0.8 | The two interventions hit different failure modes (anchoring vs format-switching), so additive but not synergistic. |
| A4 | Hoisting `notation` reduces column-numbering errors (1-based vs 0-based) and improves rule adherence. | +0.0 to +0.2 | The `notation` field is short; LITM penalty likely small at our context length (~5–8KB). Effect should be subtle. |

**Null (H0) for every arm**: Δ tier score is within ±0.15 of zero (smaller than typical between-run variance).

## Primary and secondary metrics

**Primary**
1. **Mean move-tier score** per arm, averaged across bench states × runs.
   Tier mapping (already implemented in `ab_test_prompt_formats.py`):
   - foundation: 5.0
   - reveal (face-down flipped): 4.0
   - useful_tableau (sequence build, no reveal): 3.0
   - waste_play: 2.0
   - shuffle (tableau move, no reveal, no progress): 1.0
   - draw_card: 0.5
   - recycle_stock: 0.0
   - illegal / parse_fail: -1.0

**Secondary**
2. **Oscillation rate**: did the chosen move recreate the state from 2 turns ago? (Use the actual `recentMoves` log to detect.)
3. **JSON validity rate**: parses + has required keys + `move_index` is a valid index into `legalMoves`.
4. **Output tokens** (mean): proxy for cost / truncation risk.
5. **Latency p50/p95** (ms): secondary cost signal.

## Bench design

- **N = 20 board states**, hand-picked from `data/raw/*.json` exports.
- **Composition**:
  - 6 oscillation contexts (states where the model previously got into a draw-loop)
  - 6 midgame (face-down count 10–15, multiple tier options available)
  - 4 early-game (face-down count 18–21, mostly draw/reveal choices)
  - 4 endgame (face-down count ≤4, foundation-heavy)
- **Filter requirements**:
  - ≥3 entries in `recentMoves` (so the anchoring effect has surface area)
  - ≥2 tier classes represented in `legalMoves` (so the choice is informative)
- **Reproducibility**: bench saved as `experiments/prompt_audit_2026_05_24/bench.json` with state hashes; same bench used for any Gemma follow-up.

## Run plan

- **N = 20 board states**, **M = 5 runs per (arm, state)** — tighter CIs (~0.15-0.25 stddev) so we can call hypothesis tests with confidence.
- **Total calls**: 5 arms × 20 states × 5 runs = **500 Haiku calls**.
- **Estimated cost**: ~$1.00 at Haiku 4.5 input ~$1/MTok, output ~$5/MTok.
- **Estimated wall time**: ~30 min with parallel batches of 5–10.

**Confirmed via user, 2026-05-24**: all 5 arms, N=20, M=5, Haiku-only triage; Gemma 4 31B reserved for arms that pass the success gates.

## Success gates (graduate to Gemma follow-up)

An arm graduates IF either:
- (G1) Δ tier-score vs C0 ≥ **+0.4** AND the run-level 95% CI does not include 0
- (G2) Oscillation rate is reduced by ≥ **50%** vs C0 (regardless of tier-score)

Arms that don't graduate are documented as null results (still useful) and shelved.

## Analysis

- **Per-arm summary**: mean ± stddev for each metric, with paired comparisons (each state evaluated under all arms).
- **Per-state breakdown**: per-arm tier on each state — looking for arms that win on average but lose on a specific class of states.
- **Failure-mode audit**: read 5 random transcripts from each arm to surface qualitative regressions the tier-score misses.

## Deliverables

1. This plan doc (you're reading it).
2. `experiments/prompt_audit_2026_05_24/bench.json` — fixed bench states.
3. `experiments/prompt_audit_2026_05_24/render_arms.py` — renderer for all 5 arms.
4. `experiments/prompt_audit_2026_05_24/run_haiku_sweep.py` — eval harness.
5. `experiments/prompt_audit_2026_05_24/raw/<arm>/<state>/run<N>.json` — every call captured.
6. `experiments/prompt_audit_2026_05_24/results.md` — final report with hypothesis tests, graduation recommendations.

## Open questions for the user (before running)

1. **Bench size N = 10, runs M = 3, total 150 calls** — comfortable? Can go bigger (N=20, M=5 = 500 calls) if we want tighter CIs.
2. **Arm selection** — keep all five (C0, A1, A2, A3, A4) or prune? A4 is the lowest-prior arm; could drop.
3. **Haiku vs Sonnet** — Haiku is cheaper but Sonnet is closer to Gemma 4 31B's capability tier. Stick with Haiku as planned, or use Sonnet for the "would-have-graduated" arms?

## Approach notes (for the lab notebook)

- **Pre-registration > post-hoc**: hypotheses logged here before running. Resist the urge to update predictions after seeing results — that's where p-hacking lives.
- **Bench is fixed, single seed selection**: changing the bench mid-experiment invalidates between-arm comparisons. If we want a bigger bench, we re-run *all* arms.
- **One change per arm**: A3 is the only "combined" arm and exists specifically to test interaction. Don't be tempted to add A5 with three changes — we won't be able to attribute the delta.
- **Document null results**: if A4 produces zero signal, that's still publishable info — it tells us the notation-buried-in-JSON finding is not load-bearing at this context length.
