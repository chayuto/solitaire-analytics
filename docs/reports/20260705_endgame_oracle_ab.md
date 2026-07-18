# Endgame oracle: build, ceiling A/B, honest A/B (runs 2026-07-04)

The inference-layer track's first lever, taken from research to measured
product value inside one day. Covers: the winning-line solver module, the
harness flag with its knowledge gate, the OMNISCIENT ceiling A/B (and the
design error it exposed), the HONEST A/B, and the verified pairing caveat.
Pre-registration lineage: `20260704_next_move_deep_research.md` strand C
(highest-certainty inference-time items); decision context in
`20260703_long_term_project_strategy.md` Pillar 2.

## TL;DR

- **Honest (knowledge-gated) result: volume+oracle 7/12 on heldout12 (+2 vs
  volume's 5), grpo_tr+oracle 6/12 (+1). The oracle went 13/13 when it
  fired, at ~zero cost (0.5s total probe overhead per 12 games).** This is
  the best honest advisor configuration measured to date.
- The omniscient variant (probe every turn on the true state) hit 11/12 and
  12/12 but is NOT an advisor number: it reads hidden cards. Its real
  findings: heldout12 is SOLVER-SHALLOW (8/12 deals perfect-info-solvable in
  3k nodes from turn 0), the composite machinery is flawless (21/21 proven
  lines, zero engine violations), and the oracle doubles as an in-run
  kill-diagnostic (folded volume's 9000102 at the exact turn the policy
  killed the board).
- Verified caveat: parse-retry sampling (temp 0.3) breaks strict trajectory
  pairing, so per-deck deltas mix oracle effects with retry luck; the
  clean, provable same-trajectory rescue is volume's 9000112 (fc=40 fd=0
  stall converted at t191). Games where the oracle never fired cannot be
  oracle effects (verified on the anomalies).

## 1. The module

`gemma4_finetune/endgame_oracle.py`: best-first winning-LINE search over the
HARNESS action space (visible legal moves + synthesized recycle, flips
auto-fired), sharing winnability_solver's heuristic, safe-autoplay forcing,
and transposition keying, plus parent-pointer line reconstruction. The
engine never generates foundation-to-tableau pullbacks (undo-only MoveType),
so every returned line is playable verbatim by the harness AND the
production UI. Verdicts: SOLVED (constructive line), UNSOLVABLE (sound
disproof), UNKNOWN (cap hit, caller falls back).

Validation (offline, 2026-07-04): four truncated recorded positions;
constructive replay-to-52 on all three SOLVED cases (13/21/46-move lines at
drop-last 12/25/60), verdict agreement with solve_winnable on all four
including an UNSOLVABLE loss final; every call under 0.05s / 100 nodes.

## 2. The harness flag and the knowledge gate

`play_deck_with_student.py --endgame-oracle N` probes decision turns under an
N-node budget: SOLVED plays the line to the win, UNSOLVABLE folds, UNKNOWN
falls through to the model (prompts byte-identical when the oracle does not
fire).

**Knowledge gate (default): the probe fires only when the player-visible
information determines the true position: zero face-down tableau cards AND
every card still in the stock already seen in the waste.** A perfect-memory
player then knows every card location and the exact draw order (recycling
preserves order), so solver play is legal under the project's imperfect-info
constraint: the chess-tablebase analogue. `--oracle-omniscient` lifts the
gate for eval tooling only.

The gate exists because the first A/B (below) was run WITHOUT it: the
original "probe every turn" design read hidden cards from turn 0, which is
adjudication, not play. The error was caught in the readout (solved@t0
tags), the gate was added the same hour, and the honest A/B re-run.

## 3. Omniscient A/B (ceiling; eval tooling, not an advisor result)

Both students, heldout12, cap 200, greedy, oracle cap 3000, NO knowledge
gate, vs cached no-oracle baselines. Result: volume+oracle 11/12 (+6),
grpo_tr+oracle 12/12 (+7); 1 proven-dead fold; probe overhead 68.8s / 17.6s.

What it actually established:

1. **heldout12 is solver-shallow.** 8 of 12 deals solve within 3000 nodes
   from turn 0 under perfect information (identical t0 set across both arms,
   as it must be). Every past comparison on this bench should be read with
   that in mind; fresh ship benches should prefer deeper deals.
2. **The composite machinery is production-grade.** 21/21 oracle lines
   executed through the live harness with zero engine violations across both
   arms.
3. **The dead-fold is a real-time kill diagnostic.** On 9000102 the oracle
   folded at t58: the exact turn volume's play made the board structurally
   dead, detected in-run rather than by post-mortem adjudication.
4. The 11-12/12 numbers are the perfect-information CEILING of the composite
   on this bench, useful as an upper reference line only.

## 4. Honest A/B (the product number)

Same protocol, knowledge gate ON (the default). Runs finished 2026-07-04
23:43.

### volume + oracle vs volume

| seed | volume | +oracle | oracle |
|---|---|---|---|
| 9000101 | won | won | solved@t115 |
| 9000102 | fc10 | fc10 | - |
| 9000105 | won | won | solved@t134 |
| 9000107 | won | won | solved@t124 |
| 9000108 | fc7 | WON | solved@t125 (gain) |
| 9000112 | fc40 | WON | solved@t191 (gain) |
| 9000118 | fc17 | fc14 | - |
| 9000119 | fc7 | fc7 | - |
| 9000120 | won | won | solved@t117 |
| 9000121 | fc3 | fc6 | - |
| 9000122 | fc16 | fc14 | - |
| 9000124 | won | won | solved@t91 |

**volume 5 -> volume_oracle_h 7 (delta +2); 0 folds; probe overhead 0.5s.**

### grpo_iter1_tr + oracle vs grpo_iter1_tr

Wins 5 -> 6 (delta +1): gains 9000108 (solved@t172) and 9000120
(solved@t151); one apparent loss (9000112) and one fc drop (9000122) in
games where the oracle NEVER fired (see section 5). Probe overhead ~0s.

Across both arms: **the oracle fired in 13 games and all 13 were won.** On
already-winning games it fired at t91-146 and closed them out, removing
late-game fumble risk for free.

## 5. Pairing caveat, verified

Baseline and oracle arms share greedy decoding, but parse-failure retries
sample at temp 0.3, so trajectories DIVERGE after any retry; per-deck deltas
therefore mix oracle effects with retry luck. Verified on the anomalies:

- volume 9000108: both runs retried at t36 (recorded sampled_at_temp) and
  became different games; the +1 there is divergence-assisted, not a
  same-trajectory rescue.
- grpo 9000112 ("loss") and 9000122 (fc drop): ZERO oracle moves in those
  games; the oracle cannot have caused either. Pure retry noise.
- The provably clean rescue: volume 9000112. The 2026-07-04 warm-start smoke
  independently proved volume's own baseline trajectory sits at a solvable
  fd=0 position near t195 and stalls; the honest oracle converts it at t191.

Correct reading: the honest oracle's certain value = the endgame class
(fd=0 stalls, false-resigns, late fumbles), worth ~+1-2 per 12 on this
bench; per-deck attribution beyond that is inside retry noise at n=12.

## 6. Product and program implications

- **Ship configuration: volume + knowledge-gated oracle** (7/12 heldout,
  zero information cheating, ~zero latency cost). Expected fresh-deck lift
  is the endgame class: the 20260628 winrate study found ~2 of 24 losses
  recoverable (resign+cap class), i.e. roughly 33% -> ~40%; the ship bench
  should measure this, not assume it.
- The oracle stacks on ANY policy, so it is orthogonal to the RFT round-2
  outcome (prereg `20260705_grpo_round2_preregistration.md`).
- Remaining inference-layer levers, in order: constrained JSON decoding
  (build recipe in `20260704_next_move_deep_research.md` strand B), then
  determinized per-move veto/force (the Bjarnason-class mid-game lever, the
  one remaining big-upside item).
- Eval governance addition: heldout12 is solver-shallow; retire it for ship
  claims (iteration bench only, as already governed) and build fresh
  benches with a solver-depth floor.

## 7. Artifacts

- Module + validation CLI: `gemma4_finetune/endgame_oracle.py` (4/4 offline
  tests). Harness flag + gate: `play_deck_with_student.py`
  (`--endgame-oracle`, `--oracle-omniscient`; summary fields oracle_mode/
  probes/probe_seconds/result/trigger_turn/line_len).
- Runners: `run_oracle_ab.sh` (omniscient) / `run_oracle_ab_honest.sh`
  (honest). Logs `play_runs/oracle_ab.log` / `oracle_ab_honest.log`.
- Eval dirs: `play_runs/star_iter1_eval/{volume_oracle,grpo_tr_oracle,`
  `volume_oracle_h,grpo_tr_oracle_h}/seed*/`.
- Smokes: `/tmp/oracle_smoke_9000112` (omniscient), `/tmp/
  oracle_smoke_honest_9000112` (gated; converts the same endgame).
