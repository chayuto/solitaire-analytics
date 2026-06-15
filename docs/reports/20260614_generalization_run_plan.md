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

## 5. Clean-trajectory spikes (NOT tonight unless the night has room)

Two cheap, time-boxed corpus spikes. Both are SFT-only (no contrastive
objective, which is what killed the ORPO move-contrast attempt via action
collapse). Both are gated behind generalization. Treat each as a spike: build
the v1 (exact cases only, no over-engineering), train one arm, eval on the 13
in-distribution decks against base / gate / volume, then decide. Time-box the
corpus-build; if it gets gnarly, ship the simple version and measure.

### 5.1 Spike: loop-compression corpus

Idea (operator, 2026-06-14): loops are the core pathology, so compress each
game down to its progressing spine. Detect exact board-state cycles by
replaying the game through the engine and hashing each state; a state that
repeats means the moves in between were a no-op cycle (A -> B -> C -> A). Remove
the loop body, but KEEP the escape move (A -> D) with its real loop-context, so
the model learns "when you have been going in circles, here is how you break
out." The escape rows are the single most valuable signal in the corpus and
must be preserved, not deleted. This reframes the idea from loop-DELETION to
loop-COMPRESSION-that-keeps-escapes.

Why it might work: it pushes the corpus toward clean, solver-like progressing
lines while keeping real model reasoning, and it removes "go in circles"
examples the student can imitate. Why it might not: we tried a cousin (v4-A
reversal-strip) and it collapsed a working config from fc=3 to fc=0
(`docs/reports/20260530_v4a...`). Three failure modes to dodge: (a) volume loss
(loops can be most of a doom-game's turns, and volume is the proven lever, so
log how much data the compression cuts and compare against the volume arm
specifically), (b) context corruption (kept rows still show the loop in their
RECENT MOVES history; keep that real rather than re-rendering synthetic
history, since the loop-context is what makes the escape teachable), (c)
connective-tissue moves wrongly cut. Scope v1 to exact state-cycle detection
(catches the tight-loop majority; diffuse thrash without an exact repeat is out
of scope for the spike).

Build: CPU, a script that replays each game, finds repeated state hashes,
splices out the cycle bodies, keeps escapes. Then ~80 min train + ~10 h eval as
one arm (`loopcompress`). Decision: beats the volume arm -> loops were
poisoning imitation, this is the recipe; ties or regresses -> loops were not the
bottleneck, the student lacks the positive progress skill (a useful negative,
and consistent with v4-A).

### 5.2 Spike: resign-strip recipe

Volume's only regression is the false-resign from lost-game data. Keep all the
data but strip the resign-into-loss turns (and maybe the dead-flail tails),
retrain, eval. If it keeps volume's wins and drops the wrong resigns, that is a
shippable refinement. Build is CPU, then ~80 min train + ~10 h eval.

### 5.3 Combined

If either spike helps, the natural endpoint is one "clean trajectory" corpus:
all the data, loop bodies compressed (escapes kept), resign-into-loss tails
stripped. Build it only after the spikes show which filters actually move the
number, so we are not stacking unvalidated filters (the mistake the won-only
program made).

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
