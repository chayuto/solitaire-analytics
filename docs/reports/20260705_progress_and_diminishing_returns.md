# Progress and diminishing-returns analysis (2026-07-05)

Operator question: "how is our project progress overall - are we at the point
of diminishing returns?" This doc formalizes the answer with the measured
record. Companions: `20260703_long_term_project_strategy.md` (goals/pillars),
`20260703_experiment_retrospective.md` (the full experimental history),
`20260705_endgame_oracle_ab.md` and `20260702_grpo_rft_iter0.md` (the two
freshest results).

## 1. Answer in one paragraph

The project is at diminishing returns on its ORIGINAL axis (improving the 2B
policy via training), which has returned +0 strict generalization wins across
~12 windows and 10+ arms since volume (mid-June), and that dryness is now a
measured fact rather than a suspicion. It is simultaneously at ACCELERATING
returns on the inference-layer axis, where ~1.5 windows (mostly CPU) bought
+2/12 honest wins, with two of three levers still unbuilt. The correct frame
is not "stop vs continue" but axis rotation, which the project has already
executed, plus a now-visible finish line: roughly 5-8 windows of
finite, mostly cheap, high-value work to a designed v1 endpoint.

## 2. The capability curve, by axis

| Axis | Spend | Return | Marginal return today |
|---|---|---|---|
| Policy training, SFT era (May 24 - Jun 17) | ~8-10 windows | untuned ~8% to volume ~33% fresh (16/48); published | was ~+2.5pp/window; era closed |
| Policy training, post-volume (Jun 17 - Jul 4: corpus engineering, STaR x3, RFT iter-0) | ~12 windows | +0 strict generalization wins across 10+ arms; bought the do-not-revisit fence + the RFT stability result (TR fix, first converted deck) | ~zero unless round 2 passes its gate |
| Teacher / data | steady state since June | more data measured FLAT for the student; prompt levers exhausted (resign 1 fold vs 28+ no-folds) | zero by design (harvest continues for the public dataset, not for lift) |
| Inference layer (endgame oracle) | ~1.5 windows, mostly CPU (Jul 4) | volume+oracle 7/12 honest (+2), 13/13 when fired, ~0 latency | HIGHEST since volume; constrained decoding and determinized veto still unbuilt |
| Evidence / publication | writing only | 78+ dated reports; confirmed unpublished territory (no LLM-Klondike work exists) | high and UNREALIZED; one write-up converts 6.5 weeks into a durable public artifact |

Reading: returns did not decline smoothly; they collapsed on the training
axis (the imitation ceiling is real: volume sits at teacher level ~31-33%,
and imitation-shaped training cannot exceed its source) and spiked on the
inference axis (the game has an exact solver; composing with it is cheap and
measurable). The three "dry" weeks were not waste: they produced the fence
(10 closed levers with evidence) that authorizes concentrating all remaining
spend on the two live axes without second-guessing.

## 3. Ladder status (strategy-doc targets)

| Tier | Claim | Status 2026-07-05 |
|---|---|---|
| T0 | beat untuned base | DONE (volume ~4x base, published) |
| T1 | beat volume by +2 paired | BANKED BY THE COMPOSITE (volume+oracle 7/12 = +2, honest); raw-policy T1 rides on RFT round 2's hard-stopped gate |
| T2 | beat volume on fresh decks | unmeasured; one ship bench away |
| T3 | beat the teacher (>=38% pooled) | plausibly in reach: composite expectation ~40% fresh; same ship bench answers it |
| T4 | approach the 43-52% ceiling | needs round-2 success or the determinized veto lever |

## 4. The diminishing-returns test

The test that matters: is unrealized value still LARGE relative to remaining
cost? Yes, for a bounded list:

1. **RFT round 2 to its hard stop** (~3 windows, in flight; Phase R round 1/8
   already shows 4 wins / 8 samples = strong group contrast). A null costs
   nothing further by construction (pre-registered permanent close).
2. **Constrained JSON decoding + a 24-deck fresh SHIP BENCH of
   volume+oracle** (~2 windows). Converts "expected ~40%" into a claimable
   number; likely banks T2 and possibly T3. The bench must use fresh deals
   with a solver-depth floor (heldout12 is now known solver-shallow).
3. **The consolidated write-up** (~0 GPU). Highest value-per-hour item
   remaining in the entire project.
4. Optional stretch: **determinized veto/force prototype** (~2-3 windows),
   the one remaining big-upside lever (Bjarnason-class search lifted draw-3
   Klondike 13% to 23-37%), if T4 is wanted.

After items 1-3 (and optionally 4), the answer flips: every axis is then
measured-dry or shipped at this hardware scale, and the correct move is
ship v1, publish, steady-state harvest, stop. The project ends by design,
not by fatigue.

## 5. If we stopped today (the counterfactual)

Standing artifacts already hold up: the ~29k-row public dataset, two
published adapters, the pinned ~33% claim, the oracle module, 78 reports.
What stopping today forfeits: the ship number (one bench), the write-up
(one sitting), round 2's answer (already paid for in setup), and the
constrained-decoding fix for the known JSON regression. That forfeited set
is the cheapest, highest-certainty value in the whole backlog, which is
exactly the wrong place on the curve to stop. Two weeks earlier or later,
the calculus differs; today it says: finish the bounded list.

## 6. Risks to this read

- Round 2 passing its gate would EXTEND the capability axis (a good
  problem; the loop design is pre-specified and still window-bounded).
- The ship bench could come in below ~40% (retry-noise and deck-mix
  sensitivity are real at n=24); the claim ships with CIs either way.
- Solo-operator bandwidth remains the binding resource; the WIP-1 rule and
  resumable chains are the mitigations, and both have held for six weeks.
