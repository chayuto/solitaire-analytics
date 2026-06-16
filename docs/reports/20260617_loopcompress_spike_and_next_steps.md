# Loop-compression spike, checkpoint-selection, and the resign-calibration lever

Date: 2026-06-17
Scope: session record + next-steps plan. Follows the generalization verdict
(`docs/reports/20260614_generalization_run_plan.md` section 8, committed
b2627c9). All play numbers are read from `play_runs/*/summary.json`; cap-
truncated and resigned finals are exact-adjudicated with the sound solver.

## 1. Session summary (what this window established)

1. GENERALIZES (committed, pushed). On 12 fresh solver-winnable decks, volume
   +12.9 fc / 5 wins and gate +5.6 / 3 wins beat untuned base (1 win). The
   student learned to play Klondike, not the harvester deck distribution.
   Volume-1000's 2 resigns there were adjudicated correct (dead boards).
2. Checkpoint-selection (run `volCkptSel`): publish volume-1000. No earlier
   checkpoint is cleaner on JSON; the regression is not a late-training
   artifact (section 2).
3. Resign adjudication is now a first-class metric, judged at the board the
   model resigned on, never at the deck-at-deal (the recurring trap that
   produced two earlier wrong "false resign" calls).
4. Loop-compression spike (runs `loopEval`, `loopEval300`): promising, gated on
   a single fixable flaw (section 3). This was the most informative training
   result of the window, and only the adjudication revealed it.

Cross-cutting finding: the dominant remaining lever on win rate is RESIGN
CALIBRATION, not corpus volume or filter (section 4).

## 2. Checkpoint-selection: publish volume-1000

volume is the best adapter but carries a JSON-discipline regression (more
temp-0.3 parse-retries). We asked whether an earlier checkpoint keeps the wins
with cleaner JSON. It does not. All three checkpoints on the 13 held-out decks:

| checkpoint | wins | meanFC | temp parse-rescues |
|---|---|---|---|
| volume-250 | 5 | 31.7 | 102 |
| volume-500 | 2 | 21.3 | 126 |
| volume-1000 (final) | 5 | 27.7 | 34 |

Both earlier checkpoints are 3-4x dirtier on JSON for no win gain, and quality
is non-monotonic (500 is a deep trough). So volume-1000 is the publish
candidate, and the real JSON fix is constrained decoding at inference, not
checkpoint choice. Resign adjudication of the earlier checkpoints was mixed:
volume-250 false-resigned a winnable fd=0 near-win (239901548) and a correct
dead board (405489085); volume-500 false-resigned 350743738; volume-1000's
in-dist resign (4221577640) was inconclusive at the 300k node cap. So false
resigns are real and corpus-dependent, but not (so far) a volume-1000 problem.

## 3. Loop-compression spike: promising, gated on false resigns

Build (`build_loopcompress_corpus.py`): the volume corpus minus EXACT-state
doom-loop cycle bodies, keeping the escape move with its real loop-context.
Exact state (not a looser key) is deliberate: the gap between exact (5.5% cut)
and a tableau-only key (51% cut) is almost entirely DRAW decisions, and cutting
those would reproduce the v7 ORPO draw-starvation failure. The cut is 5.5%
(379 rows; 301 from lost games), draw-safe. Hypers identical to volume.

Result on the 13 held-out decks (run `loopEval`), raw vs adjudicated:

- RAW: loopcompress 2 wins / meanFC 32.3 / 6 parse-rescues / 3 resigns, vs
  volume-1000 5 wins / 27.7 / 34 / 1. Looks worse.
- ADJUDICATED: every loopcompress resign and high-fc stall is SOLVED
  (winnable), all at fd=0 (fully revealed). 3 cap-truncated wins (239901548
  fc32, 405489085 fc38, 3263196305 fc46) plus 3 false resigns (2703165610
  fc36, 3123337720 fc47, 4221577640 fc40).

So loopcompress reached a winnable, fully-revealed endgame on 8 of 13 decks
versus volume's 5 wins, with ~6x cleaner JSON. Its non-wins stall HIGH (cards
revealed); volume's non-wins stall LOW (cards still buried). Its mid-game play
is as good or better; it just cannot finish.

cap-300 re-eval (run `loopEval300`) on the 3 cap-truncated stalls:
- 239901548 stall32 -> W52 at turn 231 (converted; progressing, not looping)
- 405489085 stall38 -> W52 at turn 215 (converted)
- 3263196305 stall46 -> R@47 at turn 212 (did NOT convert; false-resigned with
  more budget, so a resign case not a budget case)

Net: loopcompress = 4 confirmed wins at cap 300, and its 4 remaining
winnable-reached losses are ALL false resigns. The deficit is the false-resign
reflex (4 decks), not the cap (2 decks, fixed by budget) and not play quality.
Mechanism hypothesis: dropping the loop-body rows (which taught "grind when
stuck") makes the model reach winnable endgames cleanly but then resign instead
of grinding them out. Verdict: a promising recipe, gated on the resign flaw;
with it fixed, loopcompress projects to ~8 wins vs volume's 5.

## 4. The lever: resign calibration

Across this window, the thing most often standing between the student and a win
is a resignation on a winnable board:
- loopcompress: 4 false resigns on winnable fd=0 boards.
- volume-250 / volume-500: false resigns too.
- volume-1000: correct on generalization (dead boards), inconclusive in-dist.

Open question that gates the fix: WHY does the student false-resign? The teacher
resigns very rarely, so the corpus has few resign examples; the behavior may be
EMERGENT (the v1.6 prompt offers move_index -1, and in a stuck/long endgame the
model without the grind pattern may default to it) rather than imitated. That
changes the fix, so we diagnose before building.

## 5. Next steps (prioritized)

### 5.1 Resign calibration (the lever) -- next experiment
- Step 0 (cheap, CPU): read the `thinkingText` / `decision.reasoning` on the 4
  loopcompress false-resign turns. Does the model believe the board is lost, or
  is it bailing out of a long endgame? This tells us whether the fix is corpus,
  prompt, or preference, before we spend a training window.
- Step 1 (by diagnosis):
  - If imitated: strip resign-into-loss turns from the corpus (resign-strip),
    retrain loopcompress, eval at cap 300 + adjudicate.
  - If emergent / close-out gap: AUGMENT endgame close-out examples (the won
    games' final grind) rather than stripping, or add a small resign penalty.
- Decision rule: loopcompress-no-false-resign keeps the mid-game reach AND
  converts the 4 false-resign decks -> the recipe is loop-compress + resign-fix
  (projected ~8 wins). If it just flails on dead boards instead, that is an
  acceptable cost (dead boards are lost anyway).

### 5.2 Bank the generalization result
- Publish volume-1000 to HF with a model card (needs the operator's go and the
  HF token). Draft blog part 3 (the generalization payoff + the resign
  correction as the honesty beat).

### 5.3 JSON robustness
- Implement constrained decoding / a JSON grammar at inference so the published
  adapter emits valid JSON without the retry harness. Verify on volume-1000.

### 5.4 Fairness controls (cheap, before any headline claim)
- Re-run volume at cap 300 (apples-to-apples vs loopcompress@300; expected to
  change little since volume's non-wins stall low, but unproven).
- Optionally re-run the full loopcompress 13 at cap 300 for a clean arm number.

### 5.5 Longer horizon
- Solver-as-teacher to break the ~31% teacher imitation ceiling (distribution-
  independent, pathology-free trajectories rendered through the v1.6 template).

## 6. Artifacts produced this window

Committed here: `build_loopcompress_corpus.py`, `lora_config_loopcompress.yaml`,
`run_loopcompress.sh`, `run_loopcompress_eval.sh`, `run_volume_ckptsel.sh`, the
`tournament_A.py` checkpoint + loopcompress arms, and this report. Gitignored
(derived/large): `adapters_loopcompress` and `adapters_volume_ckpt0{250,500,750}`
(+ checkpoints), `dataset_loopcompress`, `training_loopcompress.jsonl`, and the
`play_runs/{volCkptSel,loopEval,loopEval300}` run dirs.
