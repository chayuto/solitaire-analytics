# Stall-field A/B probe: negative result on the student doom-loop

**Date**: 2026-05-31
**Status**: RESULT (measured). Single-board probe complete; full-game prevention run appended below.
**Scope**: tests the harvester stall/repetition STATE recommendation
(`docs/reports/20260531_harvester_recommendation_doomloop_temporal_state.md`) on the
distilled student, the most reliably reproducible instance of the doom-loop.
**Pre-registration**: decision rule fixed before the run (section 3).

## 0. One-line result

Rendering explicit stall/repetition STATE in the prompt does NOT break the entrenched
QS doom-loop on the untuned Gemma 4 E2B student: 0/40 escape, byte-for-byte the same
behaviour as baseline. This confirms the obedience-trap prediction at the student scale
and moves the lever off prompt-state and onto preference training.

## 1. Why this probe

The v1.3 inventory review (this session) showed v1.3's hard anti-undo predicate solved
oscillation on the 31B teacher (max de-inflated reversals across the whole 31B v1.3
cohort = 6, on a win; zero of the 4 v1.3 losses oscillate). The failure migrated to a
zero-reversal, high-activity, zero-yield stall: the "obedience trap" from memory
`v1-3-anti-undo-predicate-design-hole` (b2d946 / 783780). The standing recommendation
to the harvester team is to add stall/repetition STATE so the model can perceive the
no-progress condition it is currently blind to.

The objection, raised in the v1.4 notes and decisive here: the model already
under-uses signals it has (the obedience trap is the model ignoring its own correct
reasoning). If it ignores conclusions it derived itself, it may ignore a STALL line too.
The only way to settle it is to test. The student QS loop (seed 3263196305) is the
cheapest, most reliable bench: the temperature probe showed it picks the loop move 20/20
at temp 0.4 / 0.7 / 1.0, so a single board fully exercises the pathology.

## 2. Method

`gemma4_finetune/stall_field_probe.py`. Mirrors the temperature probe:

1. Hydrate seed 3263196305, replay greedily (temp 0.0) to turn 12, self-checking each
   move_index against the recorded untuned run. While replaying, track per-turn
   board signature and (foundation, face-down) counts to compute the two stall STATE
   values a player would perceive at the probed board.
2. Render the baseline board (no stall block) and the +stall board (STALL/REPEAT lines
   added). The +stall arm is rendered through the SAME `render_prompt` path the
   full-game runner uses, so the two arms differ only by the stall block.
3. Each arm: temp-0 sanity (1 sample) + N=20 samples at temp 0.7 and 1.0. Escape =
   sampled move_index other than the loop move (the baseline greedy pick).

Faithfulness: replay self-check 12/12 match, 0 mismatch. The probed board is
byte-identical to the one the original run saw at turn 12.

Stall STATE at the probed board (the strongest possible form of the signal):
- `no_progress_moves = 11` -> "STALL: no foundation play and no new card revealed in the last 11 moves."
- `position_seen_before = 5` -> "REPEAT: this exact board position has already occurred 5 time(s) earlier this game."

Loop move = index 0 (`Move QS from column 7 to column 5`); the only escape is index 1
(`Draw the next card from the stock onto the waste`).

## 3. Pre-registered decision rule

- Baseline escape ~0 AND +stall escape materially > 0 -> the stall STATE is
  perception-addressable on the student; validates the v1.4 P0 and yields a student
  inference lever at once.
- Both ~0 -> the model ignores the new signal the way it ignores its own reasoning
  (obedience-trap); kill the stall-field as a student fix.

## 4. Results

| arm | temp 0.0 sanity | temp 0.7 (N=20) | temp 1.0 (N=20) |
|---|---|---|---|
| baseline | move 0 | {0: 20}, escape 0.00 | {0: 20}, escape 0.00 |
| +stall field | move 0 | {0: 20}, escape 0.00 | {0: 20}, escape 0.00 |

0 parse failures across all 84 generations. Result artefact:
`gemma4_finetune/play_runs/stall_field_probe_result.json`.

Both baseline arms reproduce the temperature probe exactly (40/40 loop move). With the
stall block present, the model still selected the loop move 40/40, temp-0 greedy
included. The two arms are behaviourally identical.

## 5. Interpretation

The obedience-trap prediction is confirmed at the student scale. Told in plain text that
it has been in this identical position 5 times and has revealed nothing and played
nothing to a foundation in 11 moves, the model takes the loop move every time. It does
not act on explicit, quantified no-progress STATE, exactly as it does not act on its own
reasoning. This is a third independent negative control on the loop, alongside corpus
filtering (v2 / v3 / v4-A / v5) and sampling temperature: the pathology is not
addressable by what the model is shown, on this model.

## 6. Scope boundary (what this does and does not kill)

- KILLS: the stall field as a STUDENT inference fix for an ENTRENCHED loop. Dead.
- Does NOT decisively kill the v1.4 harvester P0. That targets the 31B teacher's
  obedience-freeze, and 31B is far more capable than this ~2B student; a 31B may act on
  a signal a 2B cannot. The student null is strong suggestive evidence against (the
  ignore-no-progress mechanism looks real and Gemma-family-wide), but the teacher-side
  question can only be settled by the 31B A/B in the recommendation doc (its section 7),
  which is harvester-coordinated.
- Tested ESCAPE from an entrenched loop, not PREVENTION from turn 0. Prevention is also
  unlikely (the signal is strongest at the entrenched board and is ignored; earlier it
  is weaker), but the full-game run below closes that question directly.

## 7. Bug found and fixed (independent of the result)

The full-game `--stall-field` path committed in `1e6be25` was dead code. `render_prompt`
referenced `stall_info` in its body and `main()` passed it as a 7th argument, but the
parameter was never added to the function signature, so every call raised
`NameError: name 'stall_info' is not defined`. The `--stall-field` flag had never
actually run. Fixed: added `stall_info: Optional[dict] = None` to the signature;
verified the baseline (stall_info=None) output is byte-identical to before, and the
+stall arm adds exactly the STALL/REPEAT block. This also unblocks the full-game run.

## 8. Where the lever goes

The teacher is at 63% (7/11 on 31B v1.3) and climbing; the loop is a student / small-model
problem that no prompt field touches. This pushes the remaining lever firmly to
preference training (explicit loop-penalty: DPO / ORPO), the one approach that can teach
"do not loop" from a signal the imitation corpus lacks. See
`do-not-ship-untuned-gemma4` and `corpus-filter-program-closed`.

## 9. Reproducibility

```bash
# Single-board A/B probe (this result)
.venv/bin/python gemma4_finetune/stall_field_probe.py
# -> gemma4_finetune/play_runs/stall_field_probe_result.json

# Full-game prevention run (stall field on from turn 0)
.venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed 3263196305 --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
    --stall-field --out-dir gemma4_finetune/play_runs/stall_field_fullgame_seed3263196305 \
    --max-turns 120
```

## 10. Full-game prevention run (RESULT)

The single-board probe tested ESCAPE from an entrenched loop. This run tests PREVENTION:
stall field ON from turn 0, full game, so the model sees the stall/repeat counts
accumulate from the very start rather than meeting them already-large.

```
.venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed 3263196305 --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
    --stall-field --out-dir gemma4_finetune/play_runs/stall_field_fullgame_seed3263196305 \
    --max-turns 120
```

Outcome: `max_turns`, 120 turns, final fc=0 / fd=20, a 119-turn plateau at fc=0
(foundation count and face-down count were 0 and 20 on every single turn). The model
played a QS swap on all 120 turns: col5<->col7 on 119 of them (60 one direction, 59 the
other), with the opening move the col4->col5 variant. It reached the foundation zero
times and revealed zero face-down cards. Behaviourally identical to the recorded no-stall
baseline (300 turns, fc=0, fd=20, 299-turn plateau). 0 parse failures, 0 illegal moves.

Verification the stall block actually rendered (not a silent no-op): the run's own
recorded move sequence was replayed through `render_prompt` with the same stall_info the
runner computed, and the rendered prompt was inspected for the lines. The STALL line
appears on 118 of 120 turns and the REPEAT line on 117 of 120 (both absent only at turns
0-1, the intended early no-op before any stall exists). Sampled rendered lines, quoted
verbatim:
- turn 3: `STALL: no foundation play and no new card revealed in the last 2 moves.` /
  `REPEAT: this exact board position has already occurred 1 time(s) earlier this game.`
- turn 15: `STALL: ... in the last 14 moves.` / `REPEAT: ... occurred 7 time(s) ...`
- turn 40: `STALL: ... in the last 39 moves.` / `REPEAT: ... occurred 19 time(s) ...`

So the signal was present and escalating in the prompt for essentially the whole run and
was ignored every turn.

Confidence while looping under the escalating signal: mean 0.919 (min 0.85, max 0.95,
n=120). The model is not hesitating; it reports high confidence the loop move is best
while being told in plain text that it has made no progress in dozens of moves and
revisited this exact position dozens of times.

PREVENTION fails exactly as ESCAPE failed. Both directions of the stall-field hypothesis
are now negative on the student. Section 6's scope boundary is unchanged: this is a
student / small-model result; the 31B teacher question is still only settleable by the
harvester-coordinated 31B A/B, now with a strong prior against.

