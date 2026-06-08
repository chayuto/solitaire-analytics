# ORPO loop-penalty pilot and the full-game evaluation reframing

Date: 2026-06-08

## TL;DR

The ORPO preference trainer is built (it was the blocker since 2026-06-02) and
piloted on Gemma 4 E2B Text. First full-game result on the canonical winnable
deck (seed 3263196305):

- **Untuned E2B and the v3 SFT checkpoint are byte-identical failures**: 300/300
  moves are `tableau_to_tableau`, they never draw, never reach a foundation, and
  oscillate `QS col5 <-> col7` for the entire game. SFT moved the full-game
  needle by exactly zero.
- **ORPO-450 partially escaped, on all three winnable decks tested.** It broke
  the QS shuffle loop everywhere (zero oscillation), and on two of three decks it
  played real cards to a foundation (max fc 2 and 3) which untuned/SFT never do.
  It is capped not by looping but by a move-index grounding bug (out-of-range
  picks) and early stalls, ending at turns 19 to 28. So it changes the policy
  toward progress, but does not yet sustain it.

This is directional (small deck pool, no untuned baseline yet on the two new
decks), but it is the first evidence that the preference signal moves the policy
toward progress where SFT moves nothing. The next experiment, a move-contrast
retrain (v7), is running to engage the contrastive term that this pilot left
nearly inactive.

## 1. Where the training path stood

The SFT distillation line is exhausted: v2 (raw), v3 (shuffle-filtered), v4-A
(gemma-3n + filtered), v5/v5b (won-only) all HOLD. Root cause (see
`docs/reports/` history and the `v2-distillation-teacher-doom-loop` finding):
the teacher's loop responses live inside even the won games, SFT can only
imitate, so it has no gradient that says "do not loop". The planned fix was
preference training with an explicit loop penalty (ORPO). The blocker was that
`mlx_lm 0.31.3` ships no DPO/ORPO loss, so the trainer had to be built. It now
is.

## 2. The ORPO trainer (engineering reference)

Two new files, committed in `a71720b`:

- `gemma4_finetune/orpo_loss.py` - reference-free odds-ratio loss matching TRL's
  ORPOTrainer: `L = NLL(chosen) + beta * softplus(-(logodds_chosen -
  logodds_rejected))`, length-normalised, with a weight-free self-test.
- `gemma4_finetune/train_orpo.py` - a custom loop over `mlx_lm`'s LoRA +
  `grad_checkpoint` machinery (since the library has no preference trainer).

**Memory technique, reusable for any preference or distillation loss on 16 GB.**
ORPO forwards both the chosen and rejected sequence in one autograd graph, so a
naive version peaked at ~10 GB and lagged the macOS UI (a smoke on 2026-06-02
OOM-froze the laptop). The fix: the loss only needs logits at the response
positions, but the prompt is ~1500 of ~2000 tokens and is masked out. So the
loss runs the transformer body, slices the hidden states down to the ~250
response positions, and only then applies the ~256k-wide output head plus
Gemma's logit-softcap. That halved peak to ~5.2 GB (loss bit-identical, because
the head and softcap are per-position) and ran ~40% faster. Crash-safety
defaults: probe-by-default (a real run needs `--train`), a hard MLX memory cap,
batch size 1, gradient checkpointing, cache cleared each step.

## 3. Why more data or more SFT will not fix it

Measured 2026-06-08 against the current store. The won-only 31B pool grew from 7
sessions / 1923 turns (the v5b training set) to 25 sessions / 3593 turns, a
roughly 3x increase in distinct winning trajectories. But the *shape* did not
improve:

- The teacher's moves inside won games are 35.7% foundation, 34.7% draw, 9.8%
  reveal, and **11.8% no-progress shuffles, of which 4.3% are outright
  reversals** (a lower bound on loop contamination).
- The newest wins are not cleaner. In time order the most recent wins are among
  the worst: v1.5 `#6eb393` 23.1% shuffle, v1.6 `#4c73b8` 23.5%, `#fdc52f`
  16.7%. The "clean" label in DATASET_NOTES referred to zero immediate
  consecutive reversals, a narrower metric; the diffuse shuffle rate is high.

SFT imitates the shape, not the volume, so 3x more of the same-shaped data does
not add an anti-loop gradient. The full-game result below confirms it directly:
v3 (SFT) played identically to untuned.

## 4. The pilot training run

`gemma4_finetune/adapters_orpo_v6/`, 600 iters, beta 0.1, lr 2e-4, seq 2048,
batch 1. Peak held flat at 5.22 GB for the full ~85 minutes (zero creep).
Checkpoints at 150/300/450/600. Validation loss bottomed near step 500 (0.54)
with a slight uptick after, so the best-generalising checkpoint is ~450, not the
final 600 (mild overfit on 371 pairs).

**Caveat that matters for interpretation.** Preference accuracy was 1.00 and the
odds-ratio term L_OR was already ~0.006 by step 100, collapsing to ~0.0015. That
means the model already assigned higher whole-sequence likelihood to the chosen
(progress) response than the rejected (shuffle) response, almost from the start.
So the contrastive term, the whole reason ORPO was chosen over SFT, was nearly
inactive; almost all the gradient went into the NLL-on-chosen term, which is
just SFT on the progress responses. Likely cause: chosen and rejected are whole
JSON responses differing in their entire text, so the model scores the progress
response higher because it reads more teacher-like overall, not specifically
because it has internalised "do not shuffle here". So this pilot is a weak test
of the contrastive mechanism, and the next iteration should harden the rejected
to a single-token move swap (same prompt, same text, only the move index
differs) so the gradient is about the move, not fluency.

## 5. The evaluation reframing (methodological)

This is the most transferable learning of the session.

**The single-turn 20-state bench is near-worthless for the doom-loop.** It
ranked untuned Gemma 4 E2B *above* the 31B teacher on oscillation states, yet in
full-game play E2B is far worse (it never reaches even one foundation card on
the canonical deck). When a metric is that wrong in that direction, it cannot be
trusted in the other direction either. So the SFT "HOLD" verdicts, which leaned
on this bench, were judged on a proxy that does not predict the thing we care
about.

**The fundamental metric is graded full-game progress on winnable decks**: max
and mean foundation cards reached, turns-to-loop-onset, no-progress-shuffle
fraction, and win rate, measured as a paired tuned-versus-untuned delta on
shared seeds. Binary "did it win" and single-turn tier are both too blunt: a
partial improvement (shuffles less, loops later, reaches fc=4 instead of 1)
shows up only in the graded metrics.

**Distillation regresses toward the teacher.** It helps a base that is below the
teacher (gemma-3n improved with v1.1) and hurts a base that is above the teacher
on some axis (E2B's oscillation discipline regressed, because the 31B teacher
rationalises loop moves and the untuned 4B did not). This is not a capacity
story: the 31B teacher (eight times larger) and gemini-flash-lite both
doom-loop the same way, and the loop sits in the visible 10-move window when it
happens. The failure is the training objective, not the parameter count, which
is the optimistic read for the 4B: the raw capability is present, we were
training against the wrong target.

A related confusion worth recording: "26B is worse than 31B" (a play-rate
observation) and "E2B beat 31B on oscillation" (a single-turn bench observation)
are not contradictory, because they are on different axes. On the axis that
matters, full-game play, the order is the intuitive 31B > 26B > E2B. The "E2B
beat 31B" result was only on the reflex bench, where a smaller model is less
able to talk itself into a clever bad move.

## 6. Full-game results on seed 3263196305

All three runs use the same deck, same `play_deck_with_student.py`, max 300
turns. (ORPO ran with `.venv/bin/python`, which has both `mlx_lm` and
`solitaire_analytics`; the `gemma4_finetune/venv` lacks the latter.)

| Arm | turns | max fc | final fc | move mix | oscillation | outcome |
|---|---|---|---|---|---|---|
| untuned E2B | 300 | 0 | 0 | 300 tableau_to_tableau | QS 5<->7 14x | max_turns, 100% no-progress |
| v3 SFT (iter750) | 300 | 0 | 0 | 300 tableau_to_tableau | QS 5<->7 14x | max_turns, identical to untuned |
| ORPO-450 | 28 | 0 | 0 | 24 draw, 1 reveal, 3 illegal | none | illegal_move at turn 28 |

Detail on ORPO-450: it drew from the stock for turns 0 to 23, then at turn 24
took `Move QS from column 4 to column 5 (reveals a hidden card)`, dropping
face-down 21 to 20, a reveal that untuned and SFT never produce. At turn 25 the
board reached a state with a single legal move (index 0), but the model picked
`move_index=5` (out of range) three times in a row and the run terminated on the
illegal-move guard.

**Interpretation.** ORPO shifted the policy off the deterministic `QS col5<->col7`
shuffle that traps untuned and SFT, toward drawing and a reveal, which is the
behaviour it was trained to prefer. That is a real change SFT does not produce.
But it did not convert to foundation progress (fc=0), and it exposed a new
bottleneck: move-index grounding. When the legal set collapsed to one move, the
model emitted a learned index rather than selecting from the presented legal
list. The illegal-move guard then ended the game early, so this is not a clean
300-turn comparison.

## 7. Next steps

1. **Diagnose the grounding failure.** Re-run ORPO-450 with a higher
   illegal-move tolerance, or auto-take the single legal move, to see whether the
   illegal pick is a transient glitch the model recovers from or a hard stop.
   This decides whether the reveal-then-die trajectory is closer to a win than
   it looks.
2. **More winnable decks with dynamic range.** The canonical seed is a floor
   test (untuned scores zero). Decks where untuned reaches fc>0 then loops give
   the range needed to detect graded improvement. Run untuned and ORPO paired on
   3 to 4 such seeds.
3. **Engage the contrastive term.** Re-mint pairs so the rejected is a
   single-token move swap of the chosen (same prompt and text, only the move
   index differs), forcing the odds-ratio gradient onto the move choice rather
   than whole-text fluency. The pilot's L_OR near zero shows the current pairs do
   not exercise it.
4. **SFT-on-chosen control.** Train an SFT baseline on the chosen responses only
   and full-game-bench it, to learn whether ORPO adds anything over plain SFT on
   the progress moves given the contrastive term barely fired.
5. Gate everything on graded full-game metrics, not the single-turn bench.

## Artifacts

- Code: `gemma4_finetune/orpo_loss.py`, `gemma4_finetune/train_orpo.py` (commit `a71720b`).
- Checkpoints: `gemma4_finetune/adapters_orpo_v6/` (150/300/450/600) and the
  prepped `gemma4_finetune/adapters_orpo_v6_at450/`.
- Pilot data: `dataset_orpo_pilot/` (437 train / 75 valid pairs; 371 fit at
  seq 2048). Note: `mint_preference_pairs.py` source is lost (only the `.pyc`
  and output survive); reconstruct before re-minting.
- Full-game runs: `gemma4_finetune/play_runs/{gemma4_untuned,v3_iter750,orpo_v6_at450}_seed3263196305/`.
