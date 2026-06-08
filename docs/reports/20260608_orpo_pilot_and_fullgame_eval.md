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

This is directional (small deck pool), but it is the first evidence that the
preference signal moves the policy toward progress where SFT moves nothing.

- **The move-contrast retrain (v7) gives the best play we have seen, but does not
  cure the loop.** Its early checkpoint reached fc=8 on one deck (4x the fc=2 that
  untuned/SFT/v6 cap at there) with real multi-step tableau play, yet it still
  loops at fc=0 on the canonical deck. The honest reading: move-contrast ORPO
  raises the achievable progress ceiling where progress moves exist, but it does
  not confer the planning to escape loops where the model must first create
  opportunities. Details in section 8.

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

## 8. Move-contrast retrain (v7): the contrastive term engages, best play yet, still loops

The pilot's odds-ratio term was inert (pref_acc 1.0 from step 1), so we re-minted
pairs that differ ONLY in the final `move_index` (chosen = an available
foundation or reveal, rejected = a no-progress shuffle, often the exact loop
move; `mint_move_swap_pairs.py`, 1195 pairs from 2216 mintable states) and
retrained as v7 with identical hyperparameters.

Training: the odds-ratio term ENGAGED this time. L_OR held ~0.50 to 0.68 and
sharpened (vs v6's collapse to ~0.0015), pref_acc bounced 0.80 to 1.00. The
move-contrast fix worked at the training level.

Full-game, two findings:

1. **The final checkpoint (600) overcooked into a format bug.** It writes a
   content identifier into the integer field, `{"move_index": 8C_to_9H}`, which
   is invalid JSON and parse-fails within a few turns. The model is spontaneously
   trying to select moves by *content*, and the integer schema breaks it. This is
   a late-training artifact (v6's whole-response pairs never induced it).
2. **The early checkpoint (300) has no format bug and gives the best play yet,
   but is inconsistent.** Across the three decks, max fc = 0 / 8 / 3:

   | deck | v7-300 max fc | end state | untuned max fc |
   |---|---|---|---|
   | 3263196305 | 0 | QS 5-7 loop | 0 (loop) |
   | 2853966634 | **8** | 7D 3-6 loop | 2 (loop) |
   | 2967897202 | 3 | 9S 1-4 loop | pending |

On 2853966634, v7-300 reached fc=8 with genuine multi-step play (8 foundation
plays plus waste maneuvering), 4x the fc=2 that untuned, v6, and v7-600 cap at
there. But it does not cure the loop: it makes more progress *before* looping
where progress moves are available, and on the canonical deck (no easy progress,
the model must create its own opportunities) it loops immediately at fc=0 like
untuned. So move-contrast ORPO raises the achievable progress ceiling on amenable
decks but does not confer the multi-step planning needed to escape loops in
general.

Two method notes: (a) checkpoint selection matters and validation loss
*mispredicts* play, v7-300 plays far better than the val-best v7-600, another
instance of the proxy-vs-play divergence; (b) the content identifier the model
emits motivates a content-based action representation as a next experiment.

## Artifacts

- Code: `gemma4_finetune/orpo_loss.py`, `gemma4_finetune/train_orpo.py` (commit `a71720b`); `gemma4_finetune/mint_move_swap_pairs.py` (commit `4010dee`).
- Checkpoints: `gemma4_finetune/adapters_orpo_v6/` and `adapters_orpo_v7/` (both 150/300/450/600); prepped single-checkpoint dirs `adapters_orpo_v6_at450/`, `adapters_orpo_v7_at300/`.
- Pilot data: `dataset_orpo_pilot/` (whole-response pairs); move-contrast `dataset_orpo_moveswap/` (1195/185) and `dataset_orpo_moveswap_big/` (1852/266).
- Full-game runs under `gemma4_finetune/play_runs/`: `{gemma4_untuned,v3_iter750}_seed3263196305`, `orpo_v6_at450_seed{3263196305,2853966634,2967897202}`, `orpo_v7_seed{...}`, `orpo_v7_at300_seed{...}`.
