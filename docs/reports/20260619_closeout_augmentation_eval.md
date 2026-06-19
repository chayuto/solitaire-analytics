# Close-out augmentation: fixes the false-resign, does not raise win rate

Date: 2026-06-19
Scope: training + eval record for the `closeout` arm, the experiment proposed in
`docs/reports/20260617_loopcompress_spike_and_next_steps.md` section 5.1 (the
"emergent / close-out gap -> AUGMENT" branch of the resign-calibration lever).
All play numbers are read from `play_runs/tonightEval/leaderboard.txt` and the
per-game lines in `play_runs/tonight_run.log`; cap-truncated and resigned finals
are exact-adjudicated with the sound solver (`play_runs/tonightEval/adjudication.log`).

## 1. What this window established

1. The close-out augmentation ELIMINATES the false-resign reflex. loopcompress
   resigned 5 of 13 decks; closeout resigns 0 of 13 (volume also resigns 0). The
   targeted failure is gone.
2. It CONVERTS 2 of the 4 diagnosed false-resign decks to wins (3263196305,
   4221577640). 4221577640 is a win no other arm achieved (volume stalled it,
   loopcompress resigned it).
3. But it does NOT raise the win rate. closeout = 4 wins = loopcompress's 4, and
   below volume's 5. The 2 conversions are paid for by 2 regressions (239901548,
   405489085: both win under another arm -> low-fc cap-stall under closeout).
4. It did NOT keep loopcompress's mid-game reach (the property section 5.1's
   decision rule required). meanFC fell to 25.5 (loopcompress 32.7, volume 31.8)
   and medFC to 17.0 (loopcompress 36.0); closeout introduces new LOW-fc stalls
   (fc5/7/9/10/11). The augmentation traded a resign reflex for a mid-game-stall
   reflex.

Verdict in one line: the recipe fixes the thing it targeted but is not a net
improvement and does not beat the incumbent volume student. Section 5 settles
5.1's open question against it: the new stalls are NOT on dead boards. 5 of
closeout's 9 stalls are on WINNABLE boards, and they sit DEEP in face-down cards
(fd 3-20, mean ~11) where loopcompress and volume fail only after reaching
fully-revealed fd<=1 endgames. closeout did not swap a resign for an acceptable
dead-board flail; it regressed mid-game excavation and now stalls early on boards
it should win. The predecessor's projection (loopcompress + resign-fix -> ~8
wins) assumed converting all 4 false-resign decks while keeping the reach;
instead only 2 converted, the reach regressed, and the net stayed at 4.

## 2. The arm under test

`closeout` = the PINNED loopcompress split with a TRAIN-ONLY 2x oversample of
(won-session AND faceDown<=2) rows. Built by `build_closeout_corpus.py` ->
`dataset_closeout`; valid/test byte-identical to dataset_loopcompress so the only
variable vs the loopcompress arm is the augmentation.

- train 7086 = 5458 base + 1628 copies of 814 close-out rows.
- draw-safe: post-aug train is 39.0% draws (base 45.9%; the v7 ORPO starvation
  that collapsed play was 0.5%), foundation share 22% -> 33.7%.
- hypers identical to volume / loopcompress (rank 16, scale 2.0, lr 2e-4, iters
  1000, 16 layers, max_seq 2048); see `lora_config_closeout.yaml`.

Eval: `tournament_A.py --arms closeout,loopcompress,volume` on the 13 held-out
decks at cap 300 turns, prompt v1.6, parse-retry temp 0.3. loopcompress and
volume are re-run here as same-window, same-cap baselines (not reused numbers).

## 3. Leaderboard (measured, cap 300)

| arm | wins | meanFC | medFC | resigns | stalls (max_turns) |
|---|---|---|---|---|---|
| volume (incumbent lead) | 5 | 31.8 | 27.0 | 0 | 8 |
| loopcompress | 4 | 32.7 | 36.0 | 5 | 4 |
| closeout (experiment) | 4 | 25.5 | 17.0 | 0 | 9 |

Two facts jump out: closeout zeroes the resigns (loopcompress's distinctive
failure), and closeout has the LOWEST meanFC/medFC of the three. The resign
column moved to the stall column, and the stalls are shallower.

## 4. Per-deck outcomes (all three arms, same cap)

| seed | volume | loopcompress | closeout |
|---|---|---|---|
| 1388178981 | won | won | won |
| 239901548 | won | won | stall fc18 |
| 2703165610 | won | resign fc36 | stall fc30 |
| 3123337720 | stall fc18 | resign fc47 | stall fc9 |
| 3263196305 | won | resign fc17 | **won** |
| 350743738 | stall fc31 | stall fc19 | stall fc17 |
| 3841057237 | stall fc14 | resign fc14 | stall fc11 |
| 405489085 | stall fc27 | won | stall fc5 |
| 4161700176 | stall fc14 | stall fc12 | stall fc10 |
| 4197389931 | stall fc15 | stall fc14 | stall fc7 |
| 4221577640 | stall fc16 | resign fc40 | **won** |
| 4250754298 | stall fc19 | stall fc18 | stall fc17 |
| 495097115 | won | won | won |

wins: volume {1388178981, 239901548, 2703165610, 3263196305, 495097115};
loopcompress {1388178981, 239901548, 405489085, 495097115};
closeout {1388178981, 3263196305, 4221577640, 495097115}.

The four diagnosed false-resign decks, isolated:

| seed | volume | loopcompress | closeout | converted? |
|---|---|---|---|---|
| 2703165610 | won | resign | stall fc30 | resign removed, not won |
| 3123337720 | stall | resign | stall fc9 | resign removed, not won |
| 3263196305 | won | resign | won | yes |
| 4221577640 | stall | resign | won | yes (unique to closeout) |

closeout vs loopcompress is a TRADE, not a strict gain:
- gained (resign -> win): 3263196305, 4221577640.
- lost (win -> low stall): 239901548 (fc52 -> fc18), 405489085 (fc52 -> fc5).
- net wins 4 = 4; resigns 5 -> 0.

closeout vs volume (the model we would actually ship): closeout wins 4221577640
that volume cannot, but loses 2703165610 and 239901548 that volume wins. Net 4 < 5.

## 5. Adjudication: closeout's stalls are mostly on WINNABLE boards

Bounded sound-solver verdict on every non-win final (the exact position each arm
reached, replayed zero-drift), from `adjudication.log`. 11 of the 13 decks are
winnable (some arm won or reached a SOLVED final); 4161700176 and 4250754298 were
driven dead by all three arms.

loopcompress resigns, judged at the board the model resigned on:
- FALSE resigns (winnable, revealed fd<=3): 2703165610 fc36, 3123337720 fc47,
  3263196305 fc17, 4221577640 fc40. Four, matching the diagnosis.
- CORRECT resign (dead): 3841057237 (fc14 fd10 UNSOLVABLE). So loopcompress had
  4 false resigns, not 5; one was a correct give-up on a board it had driven dead.

closeout's 9 stalls, by final-position verdict:
- WINNABLE (reached a winnable position, did not finish): 239901548 (fd3),
  3123337720 (fd9), 3841057237 (fd12), 405489085 (fd20), 4197389931 (fd13). Five.
- dead final (UNSOLVABLE): 350743738, 4161700176, 4250754298. Three.
- UNKNOWN (replay cycle hit the 90s bound): 2703165610 (fd7; deck is winnable,
  volume won it).

The decisive number is the face-down count at the winnable non-win finals:

| arm | winnable boards not won | fd at those finals |
|---|---|---|
| volume | 4 (+2 driven dead) | 0, 0, 1, 1 |
| loopcompress | 4 false resigns (+3 driven dead) | 0, 0, 0, 3 |
| closeout | 5 stalls (+1 unknown, +2 driven dead) | 3, 9, 12, 13, 20 |

volume and loopcompress fail on winnable boards only after digging out every (or
nearly every) face-down card and reaching the endgame. closeout fails on winnable
boards with 9-20 cards still buried. The augmentation made the model better at the
fd<=2 endgame it was oversampled on (it won 3263196305 and 4221577640 by closing
fully-revealed boards loopcompress had resigned) and worse at the mid-game
excavation that gets you to that endgame. That is why the wins did not move: the 2
endgame conversions are paid for by mid-game stalls.

Mechanism: this is the v7 draw-starvation lesson one level up. We guarded the draw
fraction (39%, healthy) so the model still draws, but the 2x oversample of fd<=2
rows still skewed the game-PHASE distribution toward the endgame and degraded the
earlier game. Preserving the draw axis was necessary but not sufficient.

## 6. Tooling: a replay hang fixed in the adjudicator

The adjudicator's deterministic replay has an auto-forced loop that advances any
position with exactly one legal move. On a dead cap-stall whose only remaining
moves are draw and recycle, that is a non-progressing cycle
(draw..draw..recycle.. forever, never reaching 52 foundations), so replay never
terminates and never reaches the solver. This wedged the first overnight
adjudication for ~12h on closeout/seed2703165610. Fix committed here:
`adjudicate_final_position.py` gains `--timeout-s` / `--node-cap`, and the
wall-clock bound now wraps the WHOLE per-board job (replay AND solve), not just
the solve, so a replay cycle is bounded too. Boards that hit the bound are
reported UNKNOWN(timeout). The real loop bug (auto-forced should not treat
draw/recycle as forced progress) is logged as a follow-up in section 7.

## 7. Next steps (prioritized)

### 7.1 Reject the blunt close-out augmentation; volume stays the lead
closeout (4 wins) does not beat volume (5 wins) and regresses mid-game reach, so
it is not a release candidate. The resign that motivated this whole line is
INDUCED by loopcompress's filtering (removing the grind / loop-body rows); the
unfiltered volume corpus does not resign at all (0/13) and reaches fully-revealed
endgames on more decks. The cheapest "resign fix" is therefore to keep the grind
rows, i.e. ship volume, not to filter and then patch. The loop-compress -> close-
out branch has now failed to beat volume twice.

### 7.2 If the close-out idea is kept, apply it GENTLY to volume, not loopcompress
volume's 4 non-win winnable boards are near-win cap-stalls at fd0/1, high fc
(350743738 fc31, 405489085 fc27, 3123337720, 4197389931). Those are what a LIGHT
close-out nudge could convert without a mid-game cost, because volume's mid-game
reach is already intact. Concretely: 1.25-1.5x (not 2x) oversample, widen the band
beyond fd<=2 so the endgame is not over-weighted, and GATE on meanFC not dropping
(the metric that exposed this failure).

### 7.3 Cap sensitivity on volume's near-win stalls (cheap, decisive)
volume's 4 winnable stalls are fd0/1 at turn 300. Re-run just those 4 at cap 500
to separate "needs budget" from "genuinely stuck". If they convert, the lever is
turns (free), not corpus.

### 7.4 JSON robustness (unchanged from 5.3)
Constrained decoding / a JSON grammar at inference for the published volume
adapter, so it does not depend on the parse-retry harness.

### 7.5 Ceiling lever (unchanged from 5.5)
Solver-as-teacher trajectories rendered through v1.6, to break the ~31% teacher
imitation ceiling that caps every imitation arm here.

### 7.6 Adjudicator follow-up
The timeout (section 6) bounds the hang but the underlying loop bug remains: fix
the auto-forced loop so a draw/recycle-only dead position is not treated as forced
progress (break when the only legal move is draw or recycle, or detect a repeated
state), so dead cap-stalls resolve UNSOLVABLE instead of UNKNOWN(timeout). Then
re-adjudicate closeout/seed2703165610.

## 8. Artifacts produced this window

Committed here: `build_closeout_corpus.py`, `lora_config_closeout.yaml`,
`run_tonight.sh`, the `tournament_A.py` closeout arm, the
`adjudicate_final_position.py` timeout/bound fix, and this report. Gitignored
(derived/large): `adapters_closeout`, `dataset_closeout`, and the
`play_runs/tonightEval` run dir.
