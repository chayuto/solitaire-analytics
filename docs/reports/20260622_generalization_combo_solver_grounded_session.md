# Session: generalization, combo, and solver-grounded rationales

Date: 2026-06-21 to 2026-06-22
Scope: a four-phase overnight GPU chain plus CPU tooling, continuing the
distillation work from `20260621_data_volume_and_strategy_text_eval.md`. All
play numbers are full-game adjudicated (zero-drift engine replay + sound
best-first solver), greedy decoding, production v1.6 prompt, temp-0.3 parse
rescue. In-distribution = the 13 held-out winnable decks @ cap300.
Generalization = 12 fresh solver-winnable decks (seeds 9000002..9000026) @
cap200, every deck winnable by construction so any resign or non-win is a failure.

## TL;DR

1. Generalization pass (the priority gate): the close-out recipe's dramatic
   in-distribution win advantage does NOT transfer. On fresh decks volume 5/12,
   volcloseout 4/12, volcloseout_v0620 3/12. The recipe is an in-distribution
   lever plus a behavior fix, not a generalization win lever; only its
   false-resign cure transfers (both recipe arms 0 resigns vs volume's 2 false).
   v0620 is the worst generalizer, so the tiebreaker is resolved: do not
   republish with it. Volume remains the best raw-win generalizer.
2. Combo (close-out recipe + strategy text): does NOT stack. volcombo 6/13,
   below volcloseout's 8, and the strategy's dead-board resign behavior is erased
   (0 resigns vs volstrategy's 5). The two levers interfere; pick one.
3. Solver-grounded rationales (form 3, the ceiling-breaker): new tooling built
   and validated; eval COMPLETE and strong. volsolver 8/13, meanFC 37.8, 1
   correct resign, 0 false. It TIES volcloseout (8) via an entirely different
   mechanism (grounded play, not corpus reweighting), beats volume/volstrategy/
   volcombo, and has the cleanest resign profile of any arm. The first
   non-reweighting lever to reach the top. The decisive generalization run is the
   open next step (does this 8 hold on fresh decks where the recipe's 8 collapsed
   to 4?).

The throughline: three independent results (recipe does not generalize, data is
saturated, recipe+strategy interfere) say corpus reweighting and mixing have hit
their ceiling. Solver-grounded play is the contrasting lever and the early signal
is the first to challenge the recipe.

## 1. Generalization (priority gate, COMPLETE)

| arm | gen wins/12 | gen meanFC | gen resigns | in-dist wins/13 |
|---|---:|---:|---:|---:|
| base | 1 | 15.6 | 0 | - |
| volume (flagship) | 5 | 28.5 | 2 false | 5 |
| wononly-gate | 3 | 21.2 | 0 | - |
| volcloseout (published) | 4 | 26.8 | 0 | 8 |
| volcloseout_v0620 | 3 | 20.0 | 0 | 8 |

- The in-distribution +3 (volcloseout 8 vs volume 5) collapses on fresh decks:
  volcloseout 8 -> 4 (halved), while volume 5 -> 5 (held). The close-out
  oversample overfits the in-distribution deck structure.
- The false-resign cure DOES transfer: both recipe arms resign 0 of the 12
  winnable decks, where volume false-resigns 2.
- v0620 (recipe + 12 percent more winning data) is the worst generalizer
  (3/12, meanFC 20.0) and only ties in-distribution (8=8). Decisive: volcloseout
  (old pool) stays the recipe lead; no republish with v0620.
- Implication: the published volcloseout's honest in-distribution-only card was
  the right call. It is not a strict upgrade over volume on fresh decks; volume
  remains the generalization flagship.

## 2. Close-out plus strategy combo (COMPLETE)

| arm | wins/13 | meanFC | resigns |
|---|---:|---:|---:|
| volume | 5 | 29.9 | 2 |
| volcloseout | 8 | 40.8 | 0 |
| volstrategy | 7 | 33.8 | 5 (4 correct) |
| volcombo (recipe + strategy) | 6 | 34.8 | 0 |

Hypothesis was 8 wins from the recipe plus correct dead-board resigns from the
strategy text. Instead the levers interfered: wins fell to 6 (below volcloseout's
8, meanFC 40.8 -> 34.8) and the strategy's resign behavior was erased (0 resigns
vs volstrategy's 5). Mechanism: the close-out oversample is all winning and
near-win trajectories that never resign, so it swamps the strategy rows'
resign-on-dead signal, while the strategy rows dilute the close-out gradient.
The 8 -> 6 win drop is partly small-n noise, but the 5 -> 0 resign collapse is a
clear real interaction. Conclusion: do not combine; volcloseout alone stays the
in-distribution lead.

## 3. Solver-grounded rationales (form 3, METHOD + PARTIAL result)

The escalation of the strategy-text lever from hand-authored declarative Q&A
(form 1, volstrategy 7/13) to solver-grounded decisions in the exact v1.6 play
format. Each row is a byte-faithful v1.6 board prompt whose target picks a move
on a concrete solver-proven WINNING line, with a natural-strategy rationale (no
mention of a solver, to avoid teaching solver-talk the model cannot back up).
Unlike the teacher rows (Gemma 31B, ~31 percent ceiling), the move is on a proven
win, so the rows aim to teach play above the teacher ceiling.

Method notes (the load-bearing engineering):
- Added a sound `solve_first_move` to the analyst solver, then a harness-space
  winning-PATH solver in the generator. The search runs in the harness move space
  (visible_legal_moves + auto_flip + recycle), NOT raw engine moves, because raw
  generate_moves includes foundation-to-tableau pullbacks and flip-as-choice that
  the live harness never offers, so a raw-space win can be unreproducible.
- A greedy "pick any winning-preserving move" approach was built first and
  REJECTED: it wanders the winnable set forever (measured: foundation stuck at 1
  for 300 moves) because reveals and builds stay technically winnable without
  ever completing the foundations. The fix is to search once for a full winning
  path and walk that acyclic sequence, which strictly converges to a win
  (verified: full games reach 52/52).
- For diversity, rows are an evenly-spaced subsample of each game (25 per game),
  so 700 rows come from 28 distinct deals spanning opening, mid-game and endgame
  (fc 0 to 48, median 12), not a handful of correlated games.
- dataset_volsolver = volume train + 700 solver rows (6363 rows, 11 percent);
  valid/test byte-identical to volume. Deal seeds are 8000001+, disjoint from
  every eval, benchmark and training seed (no leakage).

Final result (COMPLETE, 13 decks @ cap300):

| arm | wins/13 | meanFC | resigns |
|---|---:|---:|---:|
| volume | 5 | 29.9 | 2 (false) |
| volcloseout | 8 | 40.8 | 0 |
| volstrategy | 7 | 33.8 | 5 (1 false) |
| volcombo | 6 | 34.8 | 0 |
| volsolver | 8 | 37.8 | 1 (correct) |

volsolver TIES the recipe (8) and beats every other arm, via grounded play rather
than corpus reweighting. All 5 of its non-wins adjudicate UNSOLVABLE (dead
positions it played into), and its single resign (seed4250754298) was CORRECT
(the board was dead), so it has 0 false resigns -- the cleanest resign profile of
any arm, despite the solver rows containing no resign examples (the base model's
latent resign survived and fired correctly once).

Notably the win profile is COMPLEMENTARY to the recipe, not identical: volsolver
WINS seeds 1388178981 and 3263196305 (which volcloseout played into dead ends)
but LOSES seeds 350743738 and 4197389931 (which volcloseout wins). Each wins 8,
trading 2-for-2. So the two levers win different decks -- an ensemble or a careful
combine could in principle exceed 8. volcloseout reaches slightly further on
average (meanFC 40.8 vs 37.8; its near-win 3841057237 hit fc45).

This is the first lever to match the recipe without reweighting the teacher's
distribution. The decisive open question is generalization: the recipe's 8
collapsed to 4 on fresh decks, so the whole thesis -- that grounded play
generalizes where reweighting does not -- rests on the pending volsolver
generalization run.

## Artifacts (this session)

- Solver tooling: `winnability_solver.py` (+`solve_first_move`),
  `build_solver_grounded_corpus.py` (harness-space path solver + generator),
  `solver_grounded_rows.jsonl` (700 rows), `build_volsolver_corpus.py`.
- Generalization: `run_phase5_generalization_recipe.sh`; results in
  `play_runs/genTest/{volcloseout,volcloseout_v0620}` and `leaderboard.json`.
- Combo: `build_volcombo` via `build_volstrategy_corpus.py --src dataset_volcloseout`,
  `lora_config_volcombo.yaml`, `run_phase6_volcombo.sh`; results in
  `play_runs/phase6_volcombo/volcombo`.
- Solver arm: `lora_config_volsolver.yaml`, `run_phase7_volsolver.sh`,
  `volsolver` arm in `tournament_A.py`; results landing in
  `play_runs/phase7_volsolver/volsolver`.

## Next steps (ranked, for pickup)

1. DONE this session: volsolver in-distribution = 8/13 (ties the recipe, 1
   correct resign, 0 false). It is the first non-reweighting lever to reach the top.
2. THE PRIORITY NEXT RUN: volsolver on the 12 generalization decks. Run
   `tournament_A.py --arms volsolver --deck-path data/benchmarks/generalization_decks.json
   --seeds 9000002,9000003,9000005,9000008,9000010,9000013,9000020,9000021,9000023,9000024,9000025,9000026
   --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 --max-illegal-moves 10
   --prompt-version v1.6 --out-name genTest` (same out-name resumes; only the new
   arm runs). Compare to volume 5, volcloseout 4 on those decks. This is the
   decisive test of the whole thesis: solver-grounded play should GENERALIZE
   where corpus reweighting (recipe 8 -> 4) did not. ~6h on a free GPU.
3. Footprint control: volsolver mixes 11 percent solver rows vs volstrategy's
   5.4 percent, so a win could be the row count, not the form. If volsolver wins,
   confirm with a footprint-matched arm (~324 solver rows) to isolate form 3 from
   volume.
4. Resign gap: the solver rows never resign, so volsolver cannot resign dead
   boards. Add a small set of solver-confirmed-dead boards with a resign target
   (move_index -1) to teach correct resignation without the over-resign
   volstrategy showed.
5. If volsolver beats the recipe and generalizes, the next combine to try is
   volcloseout corpus + solver rows (recipe wins + grounded play), watching for
   the same interference seen in the strategy combo.

## Caveats

- Small n (13 in-distribution, 12 generalization) with high per-deck variance;
  read win-count deltas, not rates. Differences of 1 to 2 decks are within noise;
  the robust signals are the 8 -> 4 generalization collapse of the recipe, the
  5 -> 0 resign collapse in the combo, and volsolver winning decks other arms
  played dead.
- The generalization decks are biased to easy-to-moderate winnable deals (only
  those the solver cracked under a 200k-node cap), so absolute win rates are
  inflated; the paired comparison is the valid read.
- volsolver in-distribution is complete (8/13); its GENERALIZATION is not yet
  measured and is the load-bearing open question -- the recipe also looked great
  in-distribution (8) and then collapsed on fresh decks (4), so no
  generalization claim for volsolver can be made until that run lands.
