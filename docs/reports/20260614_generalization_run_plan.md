# Plan: the generalization run (for later tonight, 2026-06-14)

Status: PLAN. Nothing launched. Compute is paused. This is the design for the
next run, to be started later tonight on the user's go.

## 1. Where we are (what the last runs established)

Four arms have now been graded on the faithful held-out harness, same 13 decks,
cap 200, exact adjudication. All numbers read from summary files.

| arm | corpus | meanFC | wins | resigns | beats base |
|---|---|---|---|---|---|
| base | none (untrained) | 14.2 | 1 | 0 | --- |
| gate | 2492 rows, 100% won | 27.5 | 4 | 0 | 12/13 |
| allsucc | 2500 rows, 38% won | 25.5 | 3 | 1 | 10/13 |
| volume | 6859 rows, 36% won | 27.7 | 5 | 1 | 12/13 |

Established facts:
1. Training beats untuned base under trusted eval (gate PROMOTE, stochasticity
   control refuted it being luck). First time in the project.
2. The won-only FILTER is not the lever; data VOLUME is (gate ~= allsucc at
   matched size).
3. MORE volume helps modestly: full 6859 beat the matched 2500 on wins (5 vs 3)
   and meanFC (27.7 vs 25.5), and ties the won-only gate on meanFC with one
   more win. Best single student so far = volume.
4. NEW cost: lost-game data teaches a false-resign reflex. allsucc and volume
   each resigned a winnable deck (volume quit #4221577640, which the gate won
   52). The gate never resigns (no losses in its corpus).

## 2. The decisive open question

Every eval deck so far is held out BY SEED but drawn from the SAME harvester win
pool as the training data. So we have shown the student beats base on decks LIKE
its training, not on Klondike in general. The single unanswered question, on
which publish / scale / ship all depend:

  Did the student learn to play Klondike, or memorize the harvester's deck
  distribution?

Tonight's run answers this.

## 3. Tonight's primary run: generalization on fresh decks

### 3.1 Decks (already built, CPU, validated)

`data/benchmarks/generalization_decks.json`: 12 fresh, randomly-dealt,
solver-confirmed-winnable decks (seeds 9000002..9000026). Zero overlap with the
benchmark or any training corpus (verified). Each hydrates to a valid 52-card
standard deal and solves to SOLVED. These are guaranteed unseen by teacher and
student.

Documented bias: only deals the engine solver cracked under a 200k-node cap were
kept, so the set skews easy-to-moderate winnable. Harder-but-winnable deals are
under-represented. Read the PAIRED delta, not absolute win counts.

### 3.2 Arms (recommended: 3 arms)

base + gate + volume on the 12 fresh decks.
- base is the control.
- volume is the best student (test whether the strongest config generalizes).
- gate (won-only, 0 resigns) tests whether generalization depends on corpus
  type, and whether the false-resign reflex is a volume-specific artifact.

Minimal version if time-constrained: base + volume (the core decisive test).

### 3.3 Method

Faithful harness, cap 200, tiered parse rescue, temp-0.3 retry, greedy policy.
Then adjudicate EVERY game (fresh decks are harder, so expect more cap
truncations; adjudication distinguishes a cap-truncated win from a real
failure). Paired by deck against base.

### 3.4 Decision rule (pre-committed)

- Student clearly beats base on the fresh decks (positive mean paired delta,
  better on a clear majority, at least a couple of wins): GENERALIZES. The model
  learned to play. Publish is justified; "collect and train on more data"
  validated end to end.
- Student approximately equals base on the fresh decks (delta within noise,
  near-zero wins for both): MEMORIZED. The in-distribution wins were the deck
  pool, not skill. Pivot to solver-as-teacher (distribution-independent) as the
  next training direction.
- Read the paired delta, not absolute fc (fresh decks are harder and easy-biased).

### 3.5 Side measurement (free)

The fresh decks are all winnable, so any resign on them is wrong. The run also
measures whether volume's false-resign reflex generalizes (resigns on fresh
winnable boards). If volume resigns several fresh winnable decks, the
resign-strip recipe (section 5) gets more urgent.

### 3.6 Threats to validity

- Statistical power: n=12 with high per-deck variance (in-distribution deltas
  ranged -40 to +35). A small true effect will not resolve at n=12. A clear
  signal needs a sizable mean delta. If the result is muddy, the follow-up is
  more fresh decks (slow to solve).
- Easy-deck bias could lift both arms and compress the gap.
- Difficulty mismatch vs the benchmark: handled by the paired design.

### 3.7 Cost

~50 min/game worst case, but many games end early. 3 arms x 12 = 36 games, est
~15-22 h. 2-arm minimal = 24 games, est ~10-15 h. One overnight window.

## 4. Pre-launch checklist (do these first, all CPU, no GPU)

1. Commit the current uncommitted batch: deck-path wiring
   (`play_deck_with_student.py`, `tournament_A.py`), `build_generalization_decks.py`,
   the volume config/arm/scripts, the generalization builder/launcher, and the
   two blog posts.
2. Document the volume result (section 1 table) into the study report
   `docs/reports/20260614_wononly_gate_sft_study.md` and the strategic-review
   memory, including the false-resign finding.
3. Edit `run_generalization.sh` to the chosen arms. It currently runs
   `--arms base,wononly-gate`; for the 3-arm plan change to
   `--arms base,wononly-gate,volume`.
4. Confirm GPU idle (no leftover play/train procs) and laptop on mains
   (the 16 GB machine has crashed under memory pressure before; one GPU job at
   a time, subprocess-per-game already enforces this).
5. Arm a monitor that does NOT contain the literal strings
   `play_deck_with_student.py` / `train_v2.py` / `tournament_A.py` (the
   gpu_busy-matches-the-watcher footgun stalled a run on 2026-06-14).

## 5. Secondary / next-night (NOT tonight unless the night has room)

Resign-strip recipe ("best of both"). Volume's only regression is the
false-resign from lost-game data. Test the hypothesized best recipe: keep all
the data but strip the resign-into-loss turns (and maybe the dead-flail tails),
retrain, eval on the 13 in-distribution decks. If it keeps volume's wins and
drops the resign, that is the recipe to ship and to scale. Build is CPU
(filter the corpus), then ~80 min train + ~10 h eval. Lower priority than
generalization, which is the gating question.

## 6. Exact command (after the pre-launch checklist)

    zsh gemma4_finetune/run_generalization.sh    # after editing --arms to base,wononly-gate,volume

run_generalization.sh waits for the GPU to free, then runs the tournament with
`--deck-path data/benchmarks/generalization_decks.json --out-name genTest`. It
is resumable (resume-skips completed games). Adjudicate afterward with
`adjudicate_final_position.py` over `play_runs/genTest/<arm>/seed*`.

## 7. What each outcome unlocks

- Generalizes: this becomes a real result. Publish the best adapter (after a
  checkpoint-selection pass for the JSON-discipline regression), write blog
  part 3 as the payoff, and the project pivots from "can a small model learn
  this" (answered yes) to "how good can it get" (more data, resign-strip,
  solver-teacher to break the teacher ceiling).
- Memorized: equally publishable as an honest negative, and the next training
  direction is solver-as-teacher, whose data does not depend on the harvester
  distribution. Blog part 3 becomes "it memorized, here is what that taught me."
