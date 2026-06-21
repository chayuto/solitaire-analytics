# Data-volume saturation and strategy-text transfer

Date: 2026-06-21
Scope: a 2x2 isolating the close-out recipe from harvest-data growth, plus a
first probe of training-time strategy text. All numbers are full-game
adjudicated (zero-drift engine replay + sound best-first solver) on the same 13
held-out winnable decks at a 300-turn cap, greedy decoding with the production
v1.6 prompt and a temp-0.3 JSON parse-rescue.

## TL;DR

1. The gentle close-out recipe (`volcloseout`) is CONFIRMED: 8/13 wins on two
   independent runs (meanFC 40.8 both), versus plain volume's stable 5/13. The
   +3 is real, not run-to-run variance.
2. Growing the harvest by +12% winning trajectories did NOT help: on both the
   plain and the close-out recipe, the larger corpus tied the smaller one
   (5 -> 5, 8 -> 8). Data volume has plateaued past ~6859 rows; the lever is the
   reweighting, not the raw volume.
3. Training-time strategy text WORKS as a complementary lever: `volstrategy`
   reached 7/13 (+2 over its plain-volume base) and, more importantly,
   transferred to behavior, learning to recognize and resign dead boards (4 of 5
   resigns correct). It over-resigned once on a winnable deck and did not, alone,
   beat the recipe. This refutes the worry that declarative knowledge would not
   transfer to procedural play.

Ship decision: `volcloseout` (already published) stays the lead model. Data
growth is not a lever to chase. Strategy text is a real direction to develop.

## Background

The lead student, the `volume` LoRA, beats the untuned base and generalizes, but
false-resigns winnable endgames (emergent foundation-blindness: it reaches a
fd=0 board with a legal foundation play and quits). The earlier `closeout` arm
(loopcompress + 2x oversample of won and fd<=2 rows) eliminated the resign but
regressed mid-game reach and did not beat volume, so it was rejected
(`docs/reports/20260619_closeout_augmentation_eval.md`). The successor,
`volcloseout`, applies a GENTLE 1x close-out oversample to the intact volume
pool. Run 1 gave 8/13; this report confirms it and then asks two follow-on
questions raised during the window:

- Does the harvest growth since the corpus was frozen (four new 31B wins,
  #bd2080 / #211fcc / #fbfdf8 / #adc679, +560 winning rows, 33 -> 37 winning
  sessions, +12%) move the win rate?
- Does mixing declarative Klondike strategy text into training improve play?

## Method

Five LoRA arms, all identical hyperparameters (rank 16, scale 2.0, dropout 0.05,
q/k/v/o + mlp gate/up/down, 16 layers, lr 2e-4, iters 1000, batch 1), so every
arm-to-arm difference is the corpus, not the optimiser. Base is
`mlx-community/Gemma4-E2B-IT-Text-int4`.

| arm | corpus | train rows |
|---|---|---:|
| volume | full 31B success pool (old, frozen 2026-06-14) | 5663 |
| volume_v0620 | same recipe, current 31B pool (+12% wins) | 8115 |
| volcloseout | volume + 1x oversample of won and fd<=2 rows | 6444 |
| volcloseout_v0620 | volume_v0620 + the same close-out oversample | 9792 |
| volstrategy | volume + 27 strategy Q&A rows oversampled x12 | 5987 |

For every arm, valid/test are byte-identical to its base so the train-side change
is isolated. The 13 eval decks are held out by seed; the four new wins folded
into the v0620 pool were verified NOT to be among them (no leakage).

## Results

### 1. volcloseout confirmation (recipe, old pool)

Two independent eval runs, the only difference being the harness parse-retry
nondeterminism:

| run | wins | meanFC | medFC | resigns |
|---|---:|---:|---:|---:|
| run 1 (tonightEval2) | 8 | 40.8 | 52 | 0 |
| run 2 (volcloseoutConfirm) | 8 | 40.8 | 52 | 0 |

Same 8 winning decks both runs, including #4221577640, the deck plain volume
false-resigns. Reach intact: the 5 non-wins adjudicate as 4 structurally dead
(UNSOLVABLE: #1388178981, #3263196305, #4161700176, #4250754298) and 1 winnable
near-win (#3841057237 reached fc45 fd0, SOLVED, cap-truncated). The +3 over
volume is confirmed.

### 2. The 2x2 (recipe vs data)

| | old pool | +data pool (+12% wins) |
|---|---|---|
| plain volume | 5 / 13, meanFC ~30 | 5 / 13, meanFC 30.8 |
| + close-out recipe | 8 / 13, meanFC 40.8 | 8 / 13, meanFC 38.4 |

- Recipe effect: +3 wins on BOTH pools. Robust.
- Data effect: +0 wins on BOTH recipes. Flat.

`volume_v0620` not only failed to add wins, it still false-resigned a winnable
fd=0 board (#405489085, fc35 fd0 -> SOLVED), the exact failure the recipe fixes.
So the false-resign cure comes from the close-out reweighting, not from feeding
the model more winning games. Data volume helped earlier (the volume card showed
volume beating a matched 2500-row arm, 5 vs 3), so the lever is real but
saturating; 6859 -> 9937 rows buys nothing more at this scale and gradient
budget.

### 3. The strategy-text probe

`volstrategy` = volume + 27 hand-authored draw-1 imperfect-information strategy
Q&A rows (oversampled x12, 5.4% of train), targeting our measured failure modes
(reveal pass-up, loops, foundation over-eagerness, both sides of the resign gap).

| arm | wins | meanFC | resigns | resign correctness |
|---|---:|---:|---:|---|
| volume (base) | 5 | ~30 | 0-2 | - |
| volstrategy | 7 | 33.8 | 5 | 4 correct (dead), 1 false |

Two findings:

- Wins: 7/13, which is +2 over its plain-volume base and below volcloseout's 8.
- Behavior: the strategy principles TRANSFERRED to play. The model learned to
  recognize and resign dead boards: 4 of its 5 resigns are on non-winnable
  positions (#3263196305, #4161700176, #4250754298 UNSOLVABLE, plus #4197389931
  UNKNOWN/early-dead). This is the first reliable dead-board-resign behavior in
  the project; the resign action had previously fired unreliably. The single
  false resign was #4221577640 (fc12 fd8 -> SOLVED, winnable), the litmus deck
  the close-out recipe instead wins.

JSON output was not broken (all 13 games played to completion); the precise
parse-rescue rate was not recoverable from the run summaries and should be
instrumented next time.

### Full arm comparison

| arm | wins/13 | meanFC | resigns | one-line read |
|---|---:|---:|---:|---|
| volume | 5 | ~30 | 0-2 | baseline |
| volume_v0620 | 5 | 30.8 | 2 (1 false) | +data is flat |
| volcloseout | 8 | 40.8 | 0 | the win lever (LEAD) |
| volcloseout_v0620 | 8 | 38.4 | 0 | data adds nothing on top |
| volstrategy | 7 | 33.8 | 5 (4 correct) | strategy transfers, over-resigns |

## Interpretation

- The corpus lever at this scale is TARGETED REWEIGHTING, not raw volume. The
  close-out oversample converts near-win stalls and kills the false-resign;
  +12% more winning games does neither. This mirrors the v7 ORPO lesson that how
  the corpus is weighted matters more than its size.
- Strategy-in-weights changes procedural play, where strategy-in-prompt (the
  v1.6 STRATEGY GUIDANCE block) does not. The model under-applies in-context
  strategy but altered its resign behavior after seeing strategy in training.
  The transfer was strong enough to over-apply (1 false resign), which is a
  calibration problem, not a transfer failure.
- Recipe and strategy are COMPLEMENTARY, not competing: the recipe wins more and
  removes the false resign; strategy adds correct dead-board resigns (budget
  savings on truly-dead decks). Neither alone is the whole answer.

## Ship decision

`volcloseout` (old pool, 8/13, meanFC 40.8, 0 resigns) remains the lead and was
published 2026-06-20 as
`chayuto/gemma-4-e2b-it-solitaire-advisor-volcloseout-lora` with an honest
in-distribution-only card. `volcloseout_v0620` ties it on wins (8) at slightly
lower meanFC (38.4), so it does not justify a republish on in-distribution
evidence. The one open item on the publish is the fresh-deck GENERALIZATION pass,
still pending; because v0620 trains on a larger, more diverse corpus, it remains
the candidate to compare against volcloseout specifically on generalization.

## Next steps

1. Generalization pass (priority): run volcloseout, and volcloseout_v0620, on the
   12 fresh solver-winnable decks the volume lead was scored on. This closes the
   open gate on the published model and is the tiebreaker between the two recipe
   arms.
2. Close-out + strategy combo: train volcloseout's corpus plus the strategy rows.
   Hypothesis: 8 wins and 0 false resigns from the recipe, plus correct
   dead-board resigns from strategy, i.e. win conversion AND budget efficiency.
3. Resign calibration: soften the "resign when dead" principles and strengthen
   "never resign a winnable fd<=2 endgame with a legal foundation play" to remove
   the over-resign seen in volstrategy.
4. Escalate strategy forms: move from declarative Q&A (form 1) to
   rationale-augmented decisions (form 2), then to SOLVER-GROUNDED rationales
   (form 3), which fuses strategy text with the solver-as-teacher lever and is
   the path to break the ~31% teacher imitation ceiling.
5. Stop chasing data volume as a win lever; keep ingesting for coverage and for
   the winning trajectories that future on-policy or solver-grounded work needs.

## Caveats

- Small n (13 decks) with high per-deck variance; read win-count deltas, not
  precise rates. The harness has parse-retry nondeterminism (decisions are
  greedy; only parse failures retry at temp 0.3), so volume's 5 and
  volcloseout's 8 are trusted because they reproduced across runs.
- Generalization is not yet measured for any arm except the older volume lead.
- The strategy probe used 27 hand-authored principles; breadth and sourcing
  (hand-written vs solver-grounded) are themselves variables to test.
- JSON parse-rescue rate was not captured this round; instrument it before
  reading strategy-text arms for discipline effects.

## Artifacts

- Corpora: `build_volcloseout_corpus.py` (now with `--src`/`--raw`),
  `build_volstrategy_corpus.py`, `prepare_dataset.py`.
- Configs: `lora_config_{volume_v0620,volcloseout,volcloseout_v0620,volstrategy}.yaml`.
- Runners: `run_volcloseout_confirm.sh`, `run_phase2_volume_v0620.sh`,
  `run_phase3_volcloseout_v0620.sh`, `run_phase4_volstrategy.sh`; arms added to
  `tournament_A.py`.
- Run outputs: `gemma4_finetune/play_runs/{volcloseoutConfirm,phase2_v0620,phase3_v0620,phase4_volstrategy}/`
  (gitignored).
