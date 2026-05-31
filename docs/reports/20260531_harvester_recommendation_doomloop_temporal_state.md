# Recommendation to the harvester team: add temporal / stall STATE to the prompt

**Date**: 2026-05-31
**From**: distillation side (Chayut, local Gemma 4 / gemma-3n training + full-game eval)
**To**: harvester prompt team
**Status**: recommendation, not a shipped spec. The core change is a hypothesis with strong
rationale, not a proven fix; validate before committing (section 7).
**Scope**: prompting system only. No harness, model, or corpus change requested.

## 0. The ask in three lines

1. Add a small set of **stall / repetition STATE fields** to the game prompt (section 5):
   a position-repetition count, a no-progress counter, and a longer or summarized move
   history. Render them as observed state, the way FOUNDATIONS and PROGRESS already are.
2. Do this as state, NOT as a new rule or heuristic. In particular, do NOT respond to the
   loop by hardening the existing "do not undo your own work" instruction (section 6).
3. This is the single highest-leverage prompt change available for the doom-loop. It is
   complementary to, and higher priority than, the E1/E2/E3 cleanups already drafted in
   `docs/reports/20260530_v1_4_harvester_side_notes.md`.

## 1. The problem: a deterministic doom-loop the training side cannot fix

The local distilled students fall into a deterministic oscillation and never recover. On
the solver-confirmed-winnable seed 3263196305, the trained student repeats a two-move
tableau swap (QS between col5 and col7) every turn, sends zero cards to the foundations,
and reveals zero face-down cards. Verified this week: a 65-turn run held foundation count
at 0 and face-down count at 20 for the entire run, every turn at confidence 0.9.

This is now confirmed to be deep and not addressable from the training corpus:

- **Survives every corpus variation.** Raw corpus (v2), move-level reversal-filtered corpus
  (v3, v4-A), and won-games-only corpus (v5) all leave the loop intact in full-game play.
- **Survives sampling temperature.** A probe on the exact loop state chose the loop move
  20 out of 20 times at temperature 0.4, 0.7, and 1.0. It is not a greedy-decoding artifact.
- **Survives a base swap.** It appears on both Gemma 4 E2B and gemma-3n.

The conclusion the training side has reached: imitation learning cannot teach "do not loop,"
because the demonstrations do not contain that lesson (section 2). The remaining levers are
prompt-side and explicit-loop-penalty training. This doc is the prompt-side lever.

## 2. Why the corpus cannot fix it (the part that points at the prompt)

The 31B teacher that generates the corpus does not produce these loops. Its losing games in
our ingested set are short (3 to 138 turns, typically dead-deal or early-end situations),
not 300-turn oscillations. So the corpus contains neither looping behaviour to avoid nor
explicit loop-breaking behaviour to copy. A student trained to imitate a teacher that never
loops has no gradient signal pointing away from the loop. That is the structural reason
every SFT attempt has failed on this specific pathology.

What that leaves: change what the model SEES at decision time so that a loop is perceivable.
That is a prompt-template change, which the harvester team owns.

## 3. Why this is an INFORMATION gap, not a logic or rule gap

This matters because it determines the SHAPE of the fix. The model is not failing to reason;
it is reasoning correctly over an incomplete picture.

The current prompt gives the model complete SPATIAL state (every column, face-down counts,
legal moves, the reveal tag on moves that turn a card) but incomplete TEMPORAL state:

- **RECENT MOVES is capped at the last 10 moves.** A loop longer than 10 turns is therefore
  invisible. Every prompt inside a 50-turn loop looks to the model like "I have made a few
  reasonable-looking tableau moves recently."
- **PROGRESS reports current values, not trend.** It shows `face-down remaining=20` and a
  foundation count on every turn, but never that those numbers have not changed in 40 turns.
- **There is already a soft rule that fails.** The RECENT MOVES block literally says
  "review before picking, do not undo your own work," and the model loops anyway. This is
  direct evidence that another instruction will not help. The model is not disobeying a rule;
  it cannot see that it is breaking it, because the window is too short to show the pattern.

A human player at the same board would instantly notice "I have shuffled this one card back
and forth dozens of times and nothing has changed." The prompt hides exactly that. Closing
that gap is squarely within the prompt's job (render the state a player would perceive) and
does not cross into injecting heuristics.

## 4. The principle this stays inside

Per the standing prompt-design rule (render state and observations a human player would have;
never inject decision rules, heuristics, or "ONLY IF" predicates): the fix must be STATE the
model reads, not LOGIC the model is told to follow. A repetition count is state. "If you have
repeated a position 3 times, draw instead" is a heuristic and is out of scope. The
recommendations in section 5 are all the former.

## 5. Recommended fields (in descending confidence)

Render these in or near the existing PROGRESS / RECENT MOVES blocks, in the same plain-text
style. All are facts a player perceives; none instruct the model what to do.

1. **No-progress counter (highest confidence, lowest cost).** One line, for example:
   `STALL: 0 foundation plays and 0 reveals in the last N moves` where N is the turns since
   either the foundation count or the face-down count last changed. This is the single most
   direct rendering of the thing the model is currently blind to, and it is cheap: the
   harvester already tracks both counts.

2. **Position-repetition count.** One line, for example:
   `This board position has occurred 7 times before this game.` Requires hashing the board
   state (tableau + foundations + stock position) per turn and counting recurrences. This is
   the most decisive loop signal but costs a state-hash; medium effort.

3. **Longer or summarized move history.** Either widen RECENT MOVES beyond 10, or add a
   compact summary line such as `last 40 moves: QS col5<->col7 x19, draw x2`. Lowest-risk
   but also weakest on its own (a wider raw window may just delay the loop); most useful as a
   complement to 1 and 2. Note: this also resolves E2 in the v1.4 notes (the prompt text says
   "last 5 moves" while the block renders 10; pick one and make it the loop-relevant window).

Suggested minimum viable change: ship field 1 alone first. It is the cheapest, it is pure
state, and it directly targets the blindness. Add 2 if 1 alone does not move the metric.

## 6. What NOT to do (an explicit anti-recommendation)

Do not respond to the loop by strengthening the anti-undo instruction. We have a separate,
verified finding (the v1.3 anti-undo "obedience trap," from 31B sessions b2d946 / 783780
versus 26B cbced2): under the same v1.3 prompt, a model told more forcefully not to undo can
obey and FREEZE, declining a productive move and drawing into a dead stock, which is a
different failure that is worse, not better. A harder predicate is net-negative on 31B. The
stall signal avoids this because it adds perception, not obligation: it tells the model where
it is, and lets the model's own reasoning decide what to do about it.

## 7. How to validate before shipping

This is a recommendation, not a proven fix. Suggested check, cheap on your side:

- A/B the prompt with and without field 1 on the seeds where the loop reproduces (3263196305
  is the canonical one; we can supply others). Measure: median loop length (consecutive
  same-card reversals on the move stream) and whether any run that previously looped now
  reaches a higher foundation count.
- We can run the same A/B locally on the distilled student, which is where the loop is most
  reliably reproducible, and report back. Coordinating the prompt-template change with us lets
  both sides measure the same intervention.

A reasonable success bar: the stall field shortens or breaks the loop on at least the
canonical seed without introducing the obedience-freeze from section 6. If it does neither,
the conclusion is that the loop is not perception-addressable and the lever moves to
explicit loop-penalty training (preference learning), which is our side to build.

## 8. Secondary benefit (honestly scoped)

If the harvester adds these fields to the template, the corpus generated under it would begin
to carry the stall signal as context. For the 31B teacher this is a weak benefit (it rarely
loops, so there is little loop-then-recover behaviour to capture). For weaker harvested models
(the 26B cohort, or any future self-play with the distilled student) it is a real benefit: it
would seed the corpus with the loop-recognition-then-recovery demonstrations that imitation
training currently lacks entirely. The primary benefit is at inference; treat the corpus
benefit as a bonus, not a justification.

## 9. What we are NOT asking for

- No change to the harness, the move engine, or the resign mechanism (resign is tracked
  separately).
- No new heuristics, decision rules, or conditional predicates in the prompt.
- No model or corpus change.
- Not blocking on the E1/E3 STRATEGY GUIDANCE cleanups in the v1.4 notes; those are
  independent and can proceed in parallel. The stall field is the loop-specific item.

## References

- `docs/reports/20260530_v1_4_harvester_side_notes.md` (E1/E2/E3 cleanup; temporal awareness
  was a parked candidate there, this doc elevates it)
- `docs/reports/20260530_v4a_training_and_bench_session.md` (loop survives the gemma-3n base)
- `docs/reports/20260529_compute_window_session_report.md` (temperature probe: loop survives
  sampling)
- `docs/reports/20260530_v5_wononly_preregistration.md` (won-only SFT: helped the bench, did
  not fix the loop)
