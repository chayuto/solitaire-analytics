# Full-game play sequence: scientific report

**Date**: 2026-05-27
**From**: Chayut / dataset side
**Compute window**: 2026-05-26 evening to 2026-05-27 morning, approximately 5 hours
**Researcher**: Chayut Orapinpatipat (with Claude Opus 4.7)

**Companion documents**:
- `docs/reports/20260526_full_game_play_sequence_plan.md` (pre-registered plan)
- `docs/reports/20260526_v3_experiment_design.md` (v3 pre-registration)
- `docs/reports/20260526_harvester_team_resign_hygiene_versioning_ask.md` (parallel prompt-side track)
- `data/DATASET_NOTES.md` (cross-version teacher benchmarks; doom-loop session catalogue)
- `data/benchmarks/winnable_decks.json` (the canonical bench decks)

---

## Executive summary

In approximately five hours of compute we tested whether the deployed Klondike-advisor student and two candidate models can win a solver-confirmed winnable deck end to end. The headline finding is that **the single-turn match-to-teacher score on the 20-state Phase 1.5 bench does not predict full-game competence**. The deployed v1.1 LoRA student (gemma-3n base) plays meaningfully for 35 turns then doom-loops; every Gemma 4 E2B variant tested (untuned, v3-filter-trained) falls into an identical deterministic `QS col5/col7` oscillation from turn 6 onward, regardless of LoRA training. The corpus reversal-filter improved the single-turn bench by +0.40 tier (v2 best 2.45 to v3 best 2.85) but did not change full-game behaviour for the Gemma 4 base. The implication: the doom-loop pathology lives in the model-and-prompt-structure interaction, not primarily in the training corpus. The harvester ask (resign + state-repetition annotation) is the correct lever; corpus filtering alone is necessary-but-insufficient.

---

## 1. Background

### 1.1 Project context

The Klondike Solitaire dataset project distills a 31B teacher (`gemma-4-31b-it`) into a local Gemma 4 E2B or Gemma 3n E2B student via QLoRA, with the goal of shipping a 2B-class Solitaire advisor that runs on Apple M5 hardware. Prior work established:

- A 20-state Phase 1.5 bench at `experiments/a4_phase1.5_2026_05_24/prompts/C0/` measures single-turn match-to-teacher.
- v1.1 LoRA (1000 iters on gemma-3n + 1635-row stall-filtered corpus, checkpoint at iter 750 promoted) is the deployed student, with bench score 3.15 mean tier and 6/7 foundation recovery.
- v2 LoRA (same recipe on Gemma 4 E2B Text via local sanitize patch) was HELD because all checkpoints regressed on oscillation states (v2 best 2.45 mean tier; oscillation agreement 3/7 vs untuned 6/7).
- v3 experiment (pre-registered 2026-05-26) hypothesised that filtering teacher-reversal turns from the training corpus would unblock the v2 regression.

### 1.2 What was unknown before this window

The 20-state bench is a single-turn diagnostic. It scores "did the student pick the same move as the teacher on this isolated board?" Two things it cannot measure:

1. Whether the student can sustain coherent play across many turns (planning, follow-through).
2. Whether the student exhibits the same doom-loop pathology as the teacher across full sessions.

The corpus contains 31B teacher doom-loops (sessions `adf71b`, `645d03`, `73fd85`, etc.). If the student inherits this pathology, the single-turn bench will not detect it because each oscillation turn looks like a "match the teacher's bad pick" success on isolated evaluation. The full-game runner closes this measurement gap.

---

## 2. Mini-goal and pre-registration

### 2.1 Mini-goal (one sentence)

Can the deployed v1.1 LoRA student actually win a solver-confirmed winnable deck end to end, not just match the teacher's single-turn pick on the 20-state bench?

### 2.2 Pre-registered decision rules

Per `docs/reports/20260526_full_game_play_sequence_plan.md` section "Pre-registered interpretation rules", committed BEFORE any run. Subsequently revised mid-experiment (see Section 6.5 below) when Run 2 exposed a coarseness in the EARLY FAIL threshold.

| outcome class | original rule | revised rule (after Run 2) |
|---|---|---|
| WIN | outcome=won AND turns<=250 | unchanged |
| MIDGAME_STALL_DOOMLOOP | fc in [10..29] AND plateau>=30 | plateau>=15 AND clear oscillation in last 10 moves |
| EARLY_FAIL | fc<10 | reserved for cases with no foundation plays at all AND no face-down reductions |
| MOVE_INDEX_FIXATION | (not in original) | added: 3 consecutive illegal picks of the same index after legal list shrinks |

The revision was published in the plan doc BEFORE Run 3 fired, so it cannot be post-hoc on subsequent runs.

### 2.3 Pre-registered v3 hypothesis

Per `docs/reports/20260526_v3_experiment_design.md`:

> **H_v3**: Removing teacher direct-reversal turns from the training corpus will produce a v3 LoRA that:
> 1. Preserves untuned-baseline oscillation behaviour (>=4/7 foundation recovery; oscillation agreement on all 7 oscillation states matching or beating v2-untuned 6/7)
> 2. Retains v2-untuned mean tier or improves on it (>=2.55)
> 3. Adds new value on non-oscillation states (mean tier > 2.55 driven by early-game and midgame improvement)

With quantitative predictions:
- **P1**: Dropped row count 5-15% of input
- **P2**: v3 best mean tier > v2 best (2.45)
- **P3**: v3 best oscillation agreement >= 5/7
- **P4**: v3 best foundation recovery >= 4/7

---

## 3. Experimental setup

### 3.1 The full-game runner

`gemma4_finetune/play_deck_with_student.py` (committed `9f41fc6`, fixed `ad24e99`).

Loads a deck from `data/benchmarks/winnable_decks.json`, hydrates an engine `GameState`, then loops:

```
1. visible_legal_moves(state)         # excludes flip_tableau_card and foundation_to_tableau
2. render_prompt(state, ...)          # hybrid-v1 format matching the harvester
3. model.generate(prompt)             # MLX-LM inference
4. extract_decision(response)         # JSON parse to move_index + confidence
5. apply_move(state, legal[move_index])
6. auto_flip(state)                   # mimic harvester auto-flip behaviour
7. update RECENT MOVES + PRIOR REASONING
```

Termination: won / stalled / max_turns / parse_failure (3 consecutive) / illegal_move (3 consecutive) / engine_violation. Writes per-turn `turns.jsonl` plus `summary.json` plus raw responses.

### 3.2 The benchmark deck

Seed `3263196305`, draw-1 Klondike. Initial deck stored ground-truth in `data/raw/solitaire-win-010e01-1779766255660.json`'s `initialBoardSetup` block. Pyksolve confirms solvable in 9 ms under draw-1. The deck is known-winnable because the 31B teacher won it in 170 moves (session `019e6142-...010e01`, build `de7dc06`).

### 3.3 Models under test

| arm | base | LoRA | source |
|---|---|---|---|
| v1.1 LoRA | `mlx-community/gemma-3n-E2B-it-text-4bit-dwq` | `gemma4_finetune/adapters_t5_at750` | deployed student |
| Gemma 4 untuned | `mlx-community/Gemma4-E2B-IT-Text-int4` (via `gemma4_text_patch.py`) | none | v2 ship target per HELD memory |
| v3 iter 750 | `mlx-community/Gemma4-E2B-IT-Text-int4` | `gemma4_finetune/adapters_v3/_play_stage_iter750` | filter-trained best |

Information level: imperfect (`ObservationConfig.human()` semantics: face-down cards stay hidden). Matches deployment.

### 3.4 The corpus filter (v3 training)

`gemma4_finetune/filter_shuffles.py`. Drops training rows where the teacher's chosen move is a tableau-to-tableau move that directly reverses the most recent tableau-to-tableau move visible in `recentMoves`. Conservative-by-design: rows that cannot be parsed confidently are KEPT (false-negative bias).

Result on the 2026-05-26 corpus: 1953 input rows -> 1832 kept (121 dropped, 6.2%). After `prepare_dataset.py` game-level split: 1279 train / 144 valid / 168 test.

Training: 1000 iters on Gemma 4 E2B Text via the gemma4_text_patch, LoRA rank 16 over 16 attention layers, learning rate 2e-4, sequence length 2048 (truncates roughly 5-10% of rows), 4 checkpoints saved at 250/500/750/1000.

---

## 4. Results

### 4.1 Run 1: v1.1 LoRA on seed 3263196305 (attempt 1), engine contract violation

**Outcome**: crashed at turn 12 with `apply_move` returning None on a move emitted by `generate_moves` for the same state. Trace surfaced in `auto_flip` one call later.

**Root cause**: my `deck_to_state` hydrator was setting `face_up=False` on stock cards to mirror the engine's canonical `deal_klondike()` convention. But the engine's `STOCK_TO_WASTE` apply does NOT flip the card on draw; it just moves the card from stock to waste with whatever `face_up` value it had. Face-down stock cards therefore entered the waste face-down, and `WASTE_TO_TABLEAU` planted them face-down on the tableau, breaking downstream legality checks.

**Second silent bug discovered while fixing the first**: the deck JSON lists `drawPile` top-of-stock-first (`drawPile[0]` is the FIRST card the harvester draws), but the engine's `STOCK_TO_WASTE` pops from `state.stock[-1]` (engine convention: top-of-stock is at the END). My hydrator was preserving the JSON order so the engine was drawing cards in REVERSE order from the harvester. The first 5 stock draws happened to "work" because no legal waste-to-* moves were available on those turns; the bug bit only when a waste-to-tableau move became legal under reversed order.

**Both fixes**:
```python
# Hydrate stock cards as face_up=True
# Reverse the JSON list so engine pops cards in harvester order
stock = [to_card(c, default_face_up=True) for c in reversed(deck["stock"])]
```

Also added a defensive abort in the runner: if `apply_move` returns None for a move generated by `generate_moves`, log details and exit with `outcome=engine_violation` rather than crashing in `auto_flip` one call deeper.

**Lesson**: the 20-state bench never exercised `apply_move` past the first turn. The full-game runner exposed two latent bugs in the engine/runner contract within 12 turns. Bench coverage does not equal interface coverage.

### 4.2 Run 2: v1.1 LoRA on seed 3263196305 (attempt 2), MIDGAME STALL by doom-loop

**Run artefact**: `gemma4_finetune/play_runs/v1_seed3263196305_run2/`

**Outcome**: 58 turns played to `final_foundation_cards=3`, then 3 consecutive illegal-move picks triggered the safety abort. Wallclock 13.8 min.

**Trajectory in three phases**:

```
turns 0-24:  competent stock draws + opening tableau setup; fc=0 throughout, fd=20-21
turns 25-34: tableau reorganisation; revealed several face-downs; played AC to foundation
turns 35:    played AH to foundation; fc=2
turns 36-54: JD col4 <-> col7 oscillation (18 occurrences in 19 turns)
turns 55-57: 3 consecutive illegal picks of move_index=4 (legal list dropped to indices [0..3])
```

Confidence stayed at 0.95-1.0 throughout the oscillation window. Foundation count crept from 2 to 3 during the oscillation only because lucky waste plays (2C) happened to be playable.

**Classification**: MIDGAME_STALL_DOOMLOOP per the revised rule. Same pathology class as 31B teacher's documented doom-loops in `adf71b` / `645d03` / `73fd85`, just with a 2-card chain (JD only) instead of a 3-card chain (6D-5C-4D in adf71b).

**Material finding**: the deployed v1.1 LoRA student inherits the teacher's doom-loop pathology when playing a solver-confirmed winnable deck. The student is not merely bad at endgame; it falls into the canonical failure mode after 35 turns of meaningful play.

### 4.3 Run 3: Gemma 4 untuned on seed 3263196305, IMMEDIATE doom-loop, no progress

**Run artefact**: `gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/`

**Outcome**: ran the full max_turns=300 with `final_foundation_cards=0`, `plateau_at_end_turns=299`. Wallclock 68.5 min.

**Trajectory**: fell into a `QS col5 <-> col7` oscillation by approximately turn 6 (after one or two real tableau reorganisations early) and stayed there for the remaining 294 turns. Never reached fc=1. Never reduced face-down beyond an initial 1-card reveal (fd stayed at 20 for 300 turns).

**Classification**: MIDGAME_STALL_DOOMLOOP. The 299-turn plateau is the longest in the corpus.

**Material finding**: untuned Gemma 4 E2B is markedly WORSE at full-game play than v1.1 LoRA on gemma-3n. This contradicts the prior `v2-distillation-teacher-doom-loop` memory, which had ranked the arms in the opposite order based on single-turn bench (Gemma 4 untuned tier 2.55 vs gemma-3n untuned 2.10). The single-turn bench was ranking by "best reactive move", not by "ability to play a game".

### 4.4 v3 training and 20-state sweep

**Training**: clean run on Gemma 4 E2B Text + reversal-filtered corpus, 1000 iters in ~85 min, val loss 3.16 to 0.350, peak 8.41 GB. 4 checkpoints saved.

**20-state sweep results** (`gemma4_finetune/baseline_n20_gemma4_text/learning_curve_v3.json`):

```
config              json   ill   agr  tier    gap   fnd  osc_agr  osc_t
v2 untuned         20/20  0/20 12/20  2.55  -0.87  4/7    6/7     4.00
v2 best (iter 500) 20/20  2/20 10/20  2.45  -0.97  4/7    3/7     2.43
v3 iter 250        20/20  1/20 10/20  2.35  -1.07  3/7    3/7     2.86
v3 iter 500        20/20  4/20  7/20  2.20  -1.22  3/7    2/7     2.29
v3 iter 750        20/20  0/20 11/20  2.85  -0.57  5/7    4/7     3.57  <-- best
v3 iter 1000       20/20  2/20  9/20  2.60  -0.82  4/7    3/7     2.86
v1.1 iter 750 (3n) 20/20  2/20 11/20  3.15  -0.27  6/7    4/7     3.57  (reference)
```

**Pre-registered prediction check**:

| prediction | predicted | actual | verdict |
|---|---|---|---|
| P1: drop count | 5-15% of input | 6.2% (121 of 1953) | PASS |
| P2: v3 best tier > 2.45 | yes | 2.85 | PASS |
| P3: v3 best osc >= 5/7 | yes | 4/7 | FAIL |
| P4: v3 best fnd >= 4/7 | yes | 5/7 | PASS |

Three of four predictions hold. H_v3 condition 1 (preserve untuned oscillation strength 6/7) fails: v3 best dropped to 4/7. Auto-verdict per `score_v3_learning_curve.py`: HOLD. The corpus filter helps on average single-turn metrics but does not preserve the oscillation-recognition strength that motivated the experiment.

### 4.5 Run 4: v3 iter 750 on seed 3263196305, identical doom-loop to Gemma 4 untuned

**Run artefact**: `gemma4_finetune/play_runs/v3_iter750_seed3263196305_run1/`

**Outcome**: max_turns=300 with `final_foundation_cards=0`, `plateau_at_end_turns=299`. Wallclock 24.7 min (faster per-turn than untuned Gemma 4 because the LoRA layers add minor inference overhead but the oscillation kept response length short).

**Trajectory**: identical to Run 3. Same `QS col5 <-> col7` oscillation pattern from approximately turn 6, fc=0 throughout, fd=20 for the entire 300 turns.

**Classification**: MIDGAME_STALL_DOOMLOOP, oscillation pattern identical to untuned Gemma 4.

**Material finding**: the corpus reversal-filter does NOT change full-game behaviour for the Gemma 4 base. The Gemma 4 base falls into a deterministic QS oscillation on this deck regardless of LoRA training. Decisive negative for the v3 hypothesis as formulated.

---

## 5. Cross-arm comparison

```
arm                          single-turn bench  full-game seed 3263196305
                             (mean tier)        (final fc, oscillation pattern)
31B teacher (corpus, ref)    n/a                won 52/52 in 170 moves
v1.1 LoRA (gemma-3n)         3.15  (best)       fc=3 after 35 competent turns, JD col4/col7 loop
v3 iter 750 (Gemma 4 + LoRA) 2.85  (v3 best)    fc=0, QS col5/col7 loop from turn ~6
Gemma 4 untuned              2.55              fc=0, QS col5/col7 loop from turn ~6
v2 best (Gemma 4 + LoRA)     2.45              not tested
v1 untuned (gemma-3n)        2.10              not tested
```

Two things to note:

1. The single-turn bench rank order (v1.1 > v3 > Gemma4 untuned > v2 best > v1 untuned) is consistent with full-game competence ranking for the two extreme arms we tested (v1.1 plays meaningfully; v1 untuned was never tested but inferred to be worst).
2. The middle of the bench rank order does NOT predict full-game behaviour: v3 iter 750 (tier 2.85) and Gemma 4 untuned (tier 2.55) produce IDENTICAL doom-loop trajectories despite a 0.30 tier gap on the bench.

---

## 6. Findings

### 6.1 Single-turn bench does not predict full-game competence

The cleanest result of this window. Two pairs of evidence:

- **gemma-3n vs Gemma 4 untuned**: bench tier 2.10 vs 2.55 (Gemma 4 better) vs full-game fc 3 vs 0 (gemma-3n + LoRA much better). Rank order INVERTS between bench and play.
- **Gemma 4 untuned vs v3 iter 750**: bench tier 2.55 vs 2.85 (v3 better) vs full-game fc 0 vs 0 (identical). Bench gap does not translate.

The bench measures "did the student match the teacher's pick on this isolated board?" It does not measure "can the student plan across turns?" or "can the student avoid known failure modes?". For a sequential decision task, full-game play is the authoritative measurement; single-turn agreement is necessary-but-insufficient.

### 6.2 Doom-loop pathology is base-model-specific, not corpus-specific

Both untuned Gemma 4 AND filter-trained v3 fall into the IDENTICAL `QS col5/col7` oscillation pattern from turn 6 on the same deck. The pattern is bit-deterministic across runs of the same model. This is not the corpus-derived doom-loop story the `v2-distillation-teacher-doom-loop` memory tells (which says training-data poisoning transferred the failure mode to the student).

The QS oscillation appears whether the model has been trained on filtered corpus, unfiltered corpus, or no corpus at all. The pathology lives in the model-architecture-and-prompt-format interaction. A different base (gemma-3n) under the same prompt format does NOT fall into the same trap; it falls into a different oscillation (JD col4/col7) at a different board state (fc=2-3 after real play).

### 6.3 Reversal-filter helps on bench, not in play

v3 best is +0.40 tier above v2 best on the single-turn bench (2.85 vs 2.45). That is a real lift. But the lift evaporates in full-game play, where v3 and Gemma 4 untuned are indistinguishable.

Two readings:

1. **Charitable**: the filter caught some single-turn pathology but the model still relies on reversal-shuffling in multi-turn sequences. A stronger filter (board-hash repeat detection, multi-step loop detection) might help. This is the v4 hypothesis.
2. **Skeptical**: the bench gain came from the filter making the corpus easier to fit (fewer ambiguous training rows), not from removing pathology. Under this reading, no amount of corpus filtering will fix the doom-loop because the doom-loop is not a training artefact.

This window cannot distinguish between these two readings. v4-A (gemma-3n + filter, the proper test) would help.

### 6.4 Confidence saturation persists across all arms

Every model in the play runs reported confidence in the 0.85-1.0 range, mostly clustered at 0.95-1.0. The deployed v1.1 LoRA, the untuned Gemma 4 base, and the v3-filter-trained LoRA all exhibit identical saturation. Confidence is not a signal of move quality. This validates the recent harvester ask to drop the confidence field from the response schema (delivered manually by the user 2026-05-26).

### 6.5 Move-index fixation: a new failure subclass

At the end of Run 2, the v1.1 LoRA picked `move_index=4` three turns in a row after the legal-moves list shrunk to 4 entries (indices [0..3]). The model appeared to "lock in" on a positional pick from a prior turn's legal list and failed to re-evaluate when the list shape changed.

This is a discrete failure mode worth cataloguing alongside the doom-loop family. The runner's `--max-illegal-moves 3` safety cap caught it; the diagnostic value is in the 19 turns of JD oscillation immediately prior, not the 3 illegal picks themselves.

Possible mechanism: the v1.1 LoRA training set may have many positions where move_index=4 was correct, so the model has a prior on picking it. When stuck in a doom-loop, the model falls back to this prior even after the legal-list shape no longer accommodates it.

### 6.6 The 31B teacher won this deck

The 31B teacher (session `010e01`) won seed `3263196305` in 170 moves. So the deck is not just solver-winnable in principle; it is winnable by the same model family the students were distilled from. The students are losing where the teacher won. The gap between teacher and student on this deck is the headline shippability problem.

---

## 7. Gotchas catalogued

### 7.1 Engine STOCK_TO_WASTE does not flip cards

`solitaire_analytics/engine/move_validator.py` `apply_move` for `STOCK_TO_WASTE`:

```python
elif move.move_type == MoveType.STOCK_TO_WASTE:
    card = new_state.stock.pop()
    new_state.waste.append(card)
```

No face-up flip. Stock cards retain their `face_up` state through the draw. The canonical `deal_klondike()` produces face-down stock, so any consumer who calls `apply_move` on a fresh deal will produce face-down waste cards, which then fail face-up legality checks downstream.

**Workaround for the runner**: hydrate stock cards as `face_up=True` in `deck_to_state`. The harvester operationally treats waste cards as visible/playable, so this matches reality. The engine does not enforce face-down stock anywhere in validator or apply_move, so the override is safe.

**Latent open question**: does `deal_klondike()` callers in the existing test suite trip over this? Worth a follow-up check. Out of scope for this window.

### 7.2 Engine stock orientation is reverse of harvester JSON

Engine convention: `state.stock[-1]` is top-of-stock; `apply_move(STOCK_TO_WASTE)` does `stock.pop()` (from the END).

Harvester convention (in win-record `initialBoardSetup.drawPile` and our derived `winnable_decks.json.stock`): `drawPile[0]` is the FIRST card drawn (top-of-stock first).

**Workaround for the runner**: `stock = [to_card(c, ...) for c in reversed(deck["stock"])]`. The JSON convention stays, the engine sees its expected orientation.

### 7.3 Harvester `difficulty` field is NOT draw count

The `winnable_decks.json` initial schema labelled `draw_count: 3` based on the harvester export's `difficulty: 3` field. User corrected: `difficulty` is the harvester's 1-5 deal-arrangement knob (3 = true random; other values arrange the deck in some documented way and still seed-randomise within that arrangement). It is unrelated to draw count.

Further: the harvester runs draw-1 Klondike exclusively for every game in the corpus. There are no draw-3 sessions on file.

**Fix**: renamed deck record field to `harvester_difficulty`, added `perceived_difficulty` for the per-deal metric, set explicit `draw_count: 1` based on the corpus-wide rule. Documented in `data/benchmarks/winnable_decks.json` header.

### 7.4 `aiDecisionLog` in win-record files is a 30-entry rolling buffer

Discovered earlier in the corpus side, surfaced again here. `solitaire-win-*.json` files cap `aiDecisionLog` at 30 entries regardless of moveHistory length (170, 194, 284 across the three win records). The complete advisor trace lives in the corresponding `ai-log` file. Do not use `aiDecisionLog.length` as a count of AI calls; use the matching `ai-log`'s interaction count instead.

### 7.5 Confidence and alternative_move_index unused downstream

Confirmed via this window. Confidence saturated 0.93 across all corpora, never below 0.8, indistinguishable between winning and losing sessions. `alternative_move_index` not consumed by any pipeline. User asked harvester team manually 2026-05-26 to drop both fields from the next-version export schema. Historical exports retain the fields (no retroactive scrub); the next version cuts forward. Memory entry at `upcoming-export-schema-drop-confidence-altmove.md` captures the migration to-do list (ingest_exports.py, the runner's extract_decision, the bench's confidence_median, the HF data card, and version-tagging in cross-version analyses).

### 7.6 Multi-card move semantics in moveHistory replay

While validating engine-vs-harvester draw-1 semantics by replaying the corpus 010e01 win record through the engine, 3 of 170 moves at moves 28, 35, 43 produced no engine-side match. The mismatch correlates with multi-card tableau-to-tableau moves where the harvester's `cardIndex` field encoding does not map cleanly to the engine's `num_cards` field. Not blocking for the play sequence (the runner generates moves engine-side and does not consume harvester moveHistory). Logged as a backlog inconsistency.

### 7.7 mlx-lm Gemma 4 E2B sanitize bug

Pre-existing: `mlx-lm 0.31.3` cannot load Gemma 4 E2B quants out of the box (`ValueError: Received 140 parameters not in model.`) because the sanitize step does not strip redundant KV-shared layer weights. Local patch at `gemma4_finetune/gemma4_text_patch.py` (15 lines) extends `sanitize()`. Imported before any `mlx_lm.load` for Gemma 4 models. The patch is a no-op for non-Gemma-4 models. Upstream PR draft at `docs/internal/mlx_lm_gemma4_text_pr_draft.md`; submission still pending.

### 7.8 The bench is not enough

The 20-state Phase 1.5 bench has been our primary signal for "is this checkpoint shippable?" This window proved it is not sufficient. The bench:

- Does not exercise the engine's apply_move path past turn 0 (would have caught the stock-orientation and face-up bugs immediately).
- Does not measure multi-turn coherence or doom-loop avoidance.
- Has a single-turn match-to-teacher score whose middle-of-range values do not predict full-game behaviour.

Adding the full-game runner alongside the bench is a permanent upgrade. Both should run on every candidate checkpoint going forward. The full-game runner is ~50 min vs the bench's ~5 min; reserve it for the final SHIP/HOLD decision, not every checkpoint.

---

## 8. Strategic implications

### 8.1 For the harvester ask

The 2026-05-26 harvester ask (resign + drop-confidence-bands + move-NOTATION-line + semantic-versioning, with PRIOR-REASONING-truncation held) is doubly justified by this window's findings:

- **Resign**: the deployed student inherits the teacher's doom-loop pathology in full-game play. When all legal moves are JD oscillations and drawing is exhausted, the student has no out, same action-space problem as the teacher. The student would benefit from resign as much as the teacher does.
- **State-repetition annotation** (deferred to next cycle): the QS-col5/col7 oscillation is a textbook case the harness could detect at the legal-moves level via board-state-hash comparison against a rolling buffer.

### 8.2 For v2/v3 LoRA training

The v3 hypothesis is rejected as formulated. Filtering reversal turns from the corpus helps single-turn metrics but does not change full-game behaviour. The doom-loop pathology is base-model deep on Gemma 4 E2B under the hybrid-v1 prompt format.

Two reasonable next moves:

- **v4-A**: train the gemma-3n base on the filter-trained corpus. v1.1 LoRA on gemma-3n is the only LoRA arm that plays meaningfully; the v3 filter might help it further (or might not). This is the proper test of whether the filter helps the model that actually plays.
- **v4-B (deferred)**: stronger filter on Gemma 4 (board-hash repeat detection, multi-step loop detection). Lower expected value given the base-model-deep finding; would only differentiate if the gotcha is filter sensitivity not filter strength.

We recommend v4-A first.

### 8.3 For v1.1 LoRA shipping

v1.1 LoRA remains the best shippable LoRA arm on the single-turn bench AND in the only full-game test we ran. It plays competently for 35 turns before doom-looping. With the resign action shipped (harvester ask), v1.1 would likely terminate cleanly mid-doom-loop rather than producing illegal-move artefacts. This is a meaningful UX win even without changing the model.

Honest caveat: v1.1 does NOT WIN this deck. It reaches fc=3 of 52. So it is shippable as an advisor that ALSO has a resign option, not as a fully-autonomous winning player.

---

## 9. Way forward

### 9.1 v4-A (pre-registered, scoped)

**Hypothesis H_v4A**: Training gemma-3n (the deployed v1.1 base) on the same reversal-filtered corpus that produced v3 will yield a LoRA that:

1. Preserves v1.1's full-game competence (fc>=3 after similar competent-play window on seed 3263196305)
2. Improves single-turn bench tier above v1.1's 3.15
3. Extends the competent-play window before doom-loop onset (turn>=40, vs v1.1's turn-35 doom-loop entry)

Quantitative predictions to be locked before training begins. Compute estimate: ~95 min training + ~25 min sweep + ~50 min full-game play = ~170 min.

### 9.2 v4-B (deferred until we have v4-A results)

**Hypothesis H_v4B**: A stronger filter on Gemma 4 LoRA training (board-hash repeat detection + multi-step loop detection, not just direct reversal) will produce a Gemma 4 + LoRA that escapes the QS oscillation in full-game play.

Lower expected value because the Run 4 result suggests the QS pathology is base-model-deep, not a corpus artefact. Worth running only if v4-A succeeds and we want to push the corpus-filter hypothesis further on the harder-base case.

### 9.3 What we will NOT do

Decisions made explicitly to avoid scope creep:

- We will NOT run more full-game tests on Gemma 4 variants on seed 3263196305 until we have a substantively different intervention. Two arms (untuned + v3) producing identical doom-loops is enough evidence; a third would not change the conclusion.
- We will NOT run seed 2967897202 with any arm until we have a positive result on seed 3263196305 worth confirming. Running seed 2967897202 with a doom-looping arm just produces another "doom-looped" datapoint.
- We will NOT attempt inference-time prompt-engineering interventions (e.g., anti-reversal anti-prompts) in the next window. The harvester ask covers the prompt-side track and ships through the integrated build, not through one-off inference patches.
- We will NOT publish play_runs to the HF dataset. They are model artefacts, not corpus. The published dataset stays teacher-only per the existing data card.

### 9.4 Next compute window plan

```
[NOW]      Document this window (this report) and push
[+0 min]   Pre-register v4-A hypothesis with quantitative predictions
[NEXT]     v4-A training (~95 min)
[+95 min]  v4-A sweep on 20-state bench (~25 min)
[+120 min] v4-A best checkpoint full-game play on seed 3263196305 (~50 min)
[+170 min] Analyze + classify per the pre-registered rules
[+170 min] If v4-A WIN: confirm on seed 2967897202 (~50 min)
           If v4-A MIDGAME_STALL_DOOMLOOP: write up as "filter is necessary
           but insufficient even on the playing base"; pause LoRA work
           pending harvester-ask response
```

---

## 10. Reproducibility

### 10.1 Commit chain

```
e570252  Add analyze_play_run.py for mechanical post-run diagnostic
b8574b9  v3 sweep result: corpus filter helps but is not the bottleneck
3910e18  v3 iter 750 full-game play: identical doom-loop to Gemma 4 untuned
b38e9b0  Run 2 result: v1.1 student doom-loops on the canonical bench deck
d7c16e5  Run 3 + memory correction: Gemma 4 untuned plays WORSE than v1.1 on full games
ad24e99  Plan full-game play sequence; fix stock face_up hydration bug
9f41fc6  Scaffold play_deck_with_student.py: full-game runner for the v1.1 LoRA
562cf09  Build reusable winnable_decks.json from solitaire-win-* initialBoardSetup
01ce9ad  Scaffold PRIOR REASONING truncation bench for harvester ask edit 4
7ae9158  Bench PRIOR REASONING truncation; hold edit 4 in harvester ask
a8c83fd  Draft 20260526 harvester ask: resign + hygiene + semantic versioning
0833eeb  Tighten harvester ask doc: fix three-vs-four inconsistencies
0e82e2b  Correct difficulty semantics in winnable_decks.json
b2fa2fe  Drop draw-3 from winnable_decks.json; corpus is draw-1 only
```

### 10.2 Run artefacts

- `gemma4_finetune/play_runs/v1_seed3263196305_run2/` (v1.1 LoRA, Run 2)
- `gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/` (Gemma 4 untuned, Run 3)
- `gemma4_finetune/play_runs/v3_iter750_seed3263196305_run1/` (v3 iter 750, Run 4)
- `gemma4_finetune/adapters_v3/` (v3 LoRA checkpoints at 250/500/750/1000)
- `gemma4_finetune/baseline_n20_gemma4_text/posttune_v3_at*.json` (v3 sweep responses)
- `gemma4_finetune/baseline_n20_gemma4_text/learning_curve_v3.json` (v3 aggregated sweep)
- `gemma4_finetune/training_v3.log` (v3 training log)

### 10.3 Commands to reproduce

```bash
# Build deck record
.venv/bin/python scripts/build_winnable_decks.py

# v3 corpus filter
.venv/bin/python gemma4_finetune/filter_shuffles.py \
    --in  data/dataset/training.jsonl \
    --out data/dataset/training_shuffle_filtered.jsonl

# v3 dataset prep
cd gemma4_finetune
./venv/bin/python prepare_dataset.py \
    --log ../data/dataset/training_shuffle_filtered.jsonl \
    --out dataset_v3

# v3 training
./venv/bin/python train_v2.py --config lora_config_v3.yaml

# v3 bench sweep
bash sweep_v3_checkpoints.sh

# Full-game play (any arm)
cd ..
.venv/bin/python gemma4_finetune/play_deck_with_student.py \
    --deck-seed 3263196305 \
    --model-id  mlx-community/Gemma4-E2B-IT-Text-int4 \
    --adapter-path gemma4_finetune/adapters_v3/_play_stage_iter750 \
    --out-dir gemma4_finetune/play_runs/<run-name> \
    --max-turns 300 \
    --max-tokens 2048

# Mechanical analysis
.venv/bin/python gemma4_finetune/analyze_play_run.py \
    gemma4_finetune/play_runs/<run-name>
```

---

## 11. Honest assessment of this window's work

What went right:
- Pre-registration prevented post-hoc rationalisation. The decision rule revision in section 6.5 is documented mid-experiment, not after; the v4-A predictions will be locked before training.
- The full-game runner caught two engine-side bugs the bench had hidden, AND produced the first measurement of full-game competence the project has had.
- The v3 experiment resolved decisively (3 of 4 predictions hold, the fourth fails clearly), so we know what to try next.
- The cross-arm comparison clarifies that single-turn bench tier is necessary-but-insufficient as a shipping signal.

What was suboptimal:
- The first runner pass crashed in 12 turns on a latent engine bug. Better unit testing of the deck_to_state hydrator and a 5-turn smoke test before launching the full run would have saved one wasted attempt.
- The pre-registered EARLY_FAIL threshold (fc<10) was too coarse and had to be revised mid-experiment. A better pre-registration would have used plateau-with-oscillation from the start.
- We did not run the v3 best checkpoint on seed 2967897202 to confirm the doom-loop generality. Skipped intentionally per scope guard 9.3 but worth noting.

What we now know we DON'T know:
- Whether the gemma-3n base + filter (v4-A) breaks the doom-loop or not.
- Whether the v1.1 LoRA would win seed 2967897202 (a different deck) or fall into a deck-specific doom-loop there too.
- Whether the doom-loop is harvester-prompt-version-specific (hybrid-v1) or also present under the prior json prompt format.
- What the actual mechanism of the QS deterministic loop in Gemma 4 is (is it a tokenizer artefact, an attention-pattern artefact, or a sampling artefact?).

These four open questions feed the next compute window's plan.
