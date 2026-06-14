# Won-only SFT teaches Klondike policy under faithful evaluation: a gate, a stochasticity control, and a filter-vs-volume ablation

**Date:** 2026-06-14
**Authors:** solitaire-analytics (distillation track)
**Status:** COMPLETE and verified (gate, stochasticity control, and filter-vs-volume ablation all finished; 13/13 held-out decks per trained arm)
**Companion:** `docs/reports/20260611_tourA_v16_eval_fidelity_rerun.md` (the eval-fidelity reversal that motivated this study) and `docs/reports/20260611_next_compute_window_plan.md` sections 11.6 to 11.8 (the execution log this report formalizes).

## Abstract

We test whether supervised fine-tuning (SFT) of a small local model on a teacher's winning Klondike trajectories produces a policy that plays better than the untuned base, under an evaluation harness validated to be byte- and behavior-faithful to the production prompt. A LoRA adapter trained on 36 won teacher sessions (the "gate"), with the 13 evaluation decks held out of training by seed, beats the untuned base on 12 of 13 held-out decks (4 outright wins to the base's 1; mean paired foundation-card delta +13.4; gate mean foundation cards 27.5 versus base 14.2). The adapter triggers the harness's temperature retry path far more often than the base (163 versus roughly 12 turns across 13 games) because training degraded its JSON output discipline, which confounds learned policy against self-induced stochasticity. A five-deck control running the untuned base at the same temperature refutes the confound: base-at-temperature wins 0 of 5 decks where the gate won 4 and forced the 5th, often scoring below greedy. We conclude the gate learned transferable policy. This is the first time in this project that training beats untuned base under trusted evaluation, reversing a prior "distillation is falsified" conclusion that we now attribute to broken evaluation proxies. A matched-size ablation (all-success natural mix, 38% won, versus the gate's 100% won) separates the won-only filter from raw data volume: both trained arms beat the base by roughly 12 foundation cards on average, while the gate's edge over all-success (+2.08 mean paired, 7 of 13 decks, deck deltas spanning -40 to +35) is within deck-to-deck noise and both arms convert 6 of 13 decks to a won or near-forced win. The gain is therefore attributable to SFT on the data volume, not specifically to the won-only filter, which weakens the case for an expensive win-maximizing harvest and favors collecting and training on more data of either outcome.

## 1. Background and motivation

Prior full-game evaluations in this project concluded that every trained checkpoint (v2, v3, v4a, v5, v5b, v7, v7b) lost to the untuned Gemma 4 E2B base, and that distillation-as-executed was falsified. The companion report established that those conclusions rested on broken evaluation proxies: validation loss, a 20-state single-turn benchmark, and a full-game harness that used a stale prompt, lacked the stock-recycle move, fed forced positions to the model, and could corrupt the foundation render. After the harness was made faithful, the untuned base was shown to win a full game, and the "trained loses to untuned" verdicts became suspect because the training arms had never been graded on a trusted instrument.

The won-only SFT idea (keep only turns from games the teacher won) had a prior HOLD verdict (v5), but that verdict was graded on the broken proxies and is therefore void. The pre-registered next step was to re-test it as a gate: "validate the data consumer before scaling the data producer." If teacher wins are learnable signal, scaling the win harvest is justified; if not, the harvesting strategy needs rethinking. This report executes that gate, controls its primary confound, and begins to identify the active ingredient.

## 2. Methods

### 2.1 Corpus construction and held-out design

The training corpus is the game-level won/lost filter over the decision log: every turn of a game the teacher (Gemma 4 31B) won, no turns from any other game. As of this run there are 36 won sessions.

Critically, the 13 evaluation decks are held out of training by seed (`build_wononly_corpus.py --exclude-seeds`). This matters because the evaluation benchmark (`data/benchmarks/winnable_decks.json`) is itself constructed from teacher win records, so without an explicit exclusion every evaluation deck would also appear in training and the measurement would be a memorization probe. After holdout the gate corpus is 2492 rows across 18 won games, split 1881 train / 148 valid / 463 test by game.

### 2.2 Training

All training is LoRA on `mlx-community/Gemma4-E2B-IT-Text-int4` (the same int4 base the harness evaluates), rank 16, scale 2.0, dropout 0.05, all attention and MLP projections, 16 layers, max sequence length 2048, batch size 1 with gradient checkpointing, learning rate 2e-4, 1000 iterations, checkpoints every 250. Hyperparameters are identical to the prior v2/v5 runs so the only deliberate variable is the corpus. Wall time about 80 minutes, peak memory 8.4 GB on a 16 GB unified-memory machine. Validation loss bottomed at iteration 300 (0.212) and drifted to 0.233 by iteration 1000, the pre-registered overfit signature; the final checkpoint is the pre-registered evaluation arm, with earlier checkpoints retained for a later checkpoint-selection experiment.

### 2.3 Evaluation harness and adjudication

Full games are played on the faithful v1.6 harness (`play_deck_with_student.py`): the byte-verbatim v1.6 prompt header, the dynamic v1.6 render, the stock-recycle move, auto-play of forced single-legal-move positions (as in production), and a greedy base sampling policy (temperature 0). Decision parsing is tiered (strict JSON, then an inner-quote repair, then a `move_index` field extraction); parse failures and out-of-range move indexes arm a single temperature-0.3 retry, matching production's stochastic retry budget, and reset to greedy on the next valid parse. Games run to a 200-turn cap.

Because the cap truncates games mid-cascade, every `max_turns` game is adjudicated exactly: the recorded decision stream is replayed through the engine with zero-drift assertion at every turn, and the exact final position is handed to a sound best-first solver returning SOLVED, UNSOLVABLE, or UNKNOWN (`adjudicate_final_position.py`). A SOLVED final at small explored-state count is a banked win truncated by the cap; UNSOLVABLE means the policy converted a winnable deal into a dead one.

Comparison is paired by deck against the untuned base, whose 13-deck numbers come from the same harness (`play_runs/tourA_v16_rescue`). The primary metric is final foundation-card count (0 to 52); the secondary is outright win rate.

### 2.4 The stochasticity control

The gate triggers the temperature-0.3 retry path 163 times across its 13 games versus roughly 12 for the base, because won-only training degraded JSON output discipline about sixfold. A separate experiment had shown that temperature injection alone can break deterministic stalls. The gate's advantage is therefore confounded between learned policy and self-induced stochasticity. To separate them, the control runs the untuned base (no adapter) at full temperature 0.3 with a fixed sampling seed on the five decks the gate won or forced, one sample each, same harness and cap. Decision rule: if base-at-temperature also wins these, the gate's edge is stochasticity (HOLD); if base-at-temperature still stalls where the gate won, the gate learned policy (PROMOTE).

### 2.5 The filter-vs-volume ablation

The gate changed two things at once relative to prior all-success SFTs: the won/lost filter and scale (36 won sessions). To isolate the filter, a matched-size all-success arm is trained on 2500 rows across 32 games (7 won, 25 lost, 38% won-rows) drawn from the natural success mix with the same 13 decks held out and identical hyperparameters (`lora_config_allsucc.yaml`). Decision rule: gate much greater than all-success on the held-out decks means the won-only filter is the lever (scale wins); gate approximately equal to all-success means the gain is volume or recency, not the filter.

## 3. Results

### 3.1 The gate beats the base on held-out decks (R1)

Across all 13 held-out decks the gate is better on 12, worse on 1 (by a single card), with 4 outright wins to the base's 1. Mean paired foundation-card delta +13.4; gate mean foundation cards 27.5 versus base 14.2. Exact adjudication of the gate's truncated games adds two cap-truncated wins (seed 3123337720 reaching 44 with a 10-state solve; seed 405489085 reaching 25 with a 30-state solve), so 6 of 13 gate games are won or near-forced wins. On seed 2703165610 the gate preserved winnability (final SOLVED) where the base threw the game (final UNSOLVABLE), a qualitative improvement masked by a one-card lower score.

### 3.2 The stochasticity confound is refuted; verdict PROMOTE (R2)

| deck | base greedy | base temp-0.3 | gate |
|---|---|---|---|
| 495097115 | 40 | 20 | won 52 |
| 1388178981 | 18 | 20 | won 52 |
| 239901548 | 13 | 19 | won 52 |
| 4221577640 | 11 | 10 | won 52 |
| 3123337720 | 22 | 5 | 44 (forced) |

Base-at-temperature won 0 of 5; the gate won 4 and forced the 5th. Temperature alone not only failed to reproduce the wins, it frequently scored at or below greedy (40 to 20 on seed 495097115; 22 to 5 on seed 3123337720). The confound is refuted: the gate's advantage is learned policy, not an artifact of its own degraded-discipline retries. Verdict: PROMOTE.

### 3.3 Filter-vs-volume ablation: the filter is not the lever; volume is (R3)

Across all 13 held-out decks, the matched all-success arm (38% won) and the won-only gate (100% won) are statistically indistinguishable, and both are far above the base:

| arm | mean foundation cards | wins | won-or-near-forced | beats base |
|---|---|---|---|---|
| base | 14.2 | 1 | 1 | --- |
| gate (100% won) | 27.5 | 4 | 6 of 13 | 12 of 13 |
| all-success (38% won) | 25.5 | 3 | 6 of 13 | 10 of 13 |

Per-deck (gate / all-success foundation cards, "W" = win):

| seed | base | gate | all-success | gate - all-succ |
|---|---|---|---|---|
| 495097115 | 40 | W52 | W52 | 0 |
| 1388178981 | 18 | W52 | W52 | 0 |
| 3123337720 | 22 | 44 | 9 | +35 |
| 4250754298 | 15 | 16 | 16 | 0 |
| 239901548 | 13 | W52 | 19 | +33 |
| 350743738 | 10 | 12 | W52 | -40 |
| 3263196305 | 11 | 14 | 7 (resigned) | +7 |
| 2703165610 | 10 | 9 | 35 | -26 |
| 405489085 | 10 | 25 | 22 | +3 |
| 4221577640 | 11 | W52 | 43 | +9 |
| 4161700176 | 8 | 12 | 9 | +3 |
| 3841057237 | 12 | 13 | 10 | +3 |
| 4197389931 | 4 | 5 | 5 | 0 |

The gate's mean paired edge over all-success is +2.08 foundation cards (median +3), but the deck-level deltas span -40 to +35: all-success won seed 350743738 (52) where the gate reached only 12, and reached 35 on seed 2703165610 where the gate stalled at 9, while the gate dominated seeds 3123337720 and 239901548. A sign test over the 9 non-tied decks (7 favor the gate, 2 favor all-success) gives p of about 0.18, so the filter's edge is not statistically distinguishable from zero at n=13. Exact adjudication equalizes the win picture further: both arms convert 6 of 13 decks to a won-or-near-forced win (the all-success cap-truncated wins are seeds 4221577640 SOLVED in 10 states, 2703165610 SOLVED in 25, and 405489085 SOLVED in 54).

Conclusion: the large effect is SFT on the data volume (both arms beat the base by roughly 12 foundation cards on average), and the won-only filter adds at most a small, noise-level increment. The won-only framing is not the active ingredient. One behavioral note: the all-success arm fired the resign action (move_index -1) on seed 3263196305, a winnable deck, so the resign was incorrect, but it is the first time any student adapter has emitted a resign, plausibly because its corpus included lost games.

### 3.4 Volume scaling: more data helps modestly, and the resign reflex appears (R4)

If volume is the lever, does more of it help? A fourth arm trained on the
ENTIRE non-eval success pool (6859 rows / 77 games, 36% won, the full natural
mix) at the same iters and holdout (`lora_config_volume.yaml`).

| arm | corpus | meanFC | wins | resigns | beats base |
|---|---|---|---|---|---|
| base | none | 14.2 | 1 | 0 | --- |
| gate | 2492 / 100% won | 27.5 | 4 | 0 | 12/13 |
| all-success | 2500 / 38% won | 25.5 | 3 | 1 | 10/13 |
| volume | 6859 / 36% won | 27.7 | 5 | 1 | 12/13 |

Full volume beat the matched all-success arm on wins (5 vs 3) and meanFC
(27.7 vs 25.5), and tied the won-only gate on meanFC (27.7 vs 27.5) with one
more win. So more data helps, modestly, beyond the matched 2500. Volume won
the canonical doom deck 3263196305 outright (52), which every prior measurement
scored at 0 and which both other trained arms failed.

The cost is the resign reflex. Volume fired a wrong resign on seed 4221577640,
a winnable deck the gate won outright. Combined with all-success's wrong resign
on 3263196305, the pattern is clear: lost-game turns teach the model to quit,
and it misfires on winnable boards. The won-only gate, with no losses in its
corpus, never resigns. This suggests a "best of both" recipe (section 5): keep
all the data for the volume gain, but strip the resign-into-loss turns to remove
the quitting reflex. Untested as of this writing.

Data-scale note: 6859 is most of the clean 31B data that exists. The store
holds 40,284 raw 31B interaction attempts, but only ~9,134 (about 22%) became
clean usable decisions; the rest were lost to provider errors and timeouts
(under 44% of moves per game are logged). So the dominant "more data" lever is
recovering logging yield from games already played, not playing more games. A
26B cohort (7,372 raw interactions) and a gemini cohort (2,684) exist but are
excluded by the 31B-only training filter and would each be a separate
model-mix experiment.

## 4. Threats to validity

- **Distributional, not just seed, holdout.** The 13 evaluation decks are held out by seed but drawn from the same harvester win pool as the training corpus, so held-out-from-training is not held-out-from-distribution. Whether the gate learned Klondike or the harvester's deck distribution is untested here and is the designated next experiment (fresh solver-confirmed-winnable random deals).
- **Sampling-regime mismatch in the control.** Base-at-temperature is full-temperature on every move, whereas the gate is greedy with temperature only on retries. The control is therefore strong evidence against the stochasticity confound, not an identical-regime proof.
- **Mechanism unidentified.** PROMOTE shows the gain is real and not stochasticity; it does not explain whether the gain is better move selection, more fluent reproduction of the teacher's output cadence, or longer coherent play before drift.
- **Output-discipline regression.** The gate degraded JSON validity about sixfold. The result holds under production-faithful retry tolerance, but the regression is real and matters for any deployment; checkpoint selection (the iteration-300/500 checkpoints) is an open mitigation.
- **Single benchmark, small n.** Thirteen decks, one base, one corpus snapshot. Absolute win rate (4 of 13 on solver-confirmed-winnable decks) is low; the teacher itself wins about 31% of arbitrary deals. The gate beats base, it does not solve Klondike.
- **Cap truncation.** Adjudication recovers the truth of truncated games, but the win-versus-foundation-count comparison still mixes outright wins with truncated near-wins.

## 5. Discussion

This is the first result in the project where training beats the untuned base under trusted evaluation. It reverses the standing "distillation-as-executed is falsified" conclusion and marks the fourth broken proxy caught in sequence: validation loss, the single-turn benchmark, the unfaithful harness, and "training never helps." The pattern is methodological: each reversal came from refusing a convenient number in favor of graded, faithful, held-out, paired full-game evidence, and from controlling the most plausible confound before claiming a mechanism.

The ablation is the more consequential thread for strategy, and it resolved against the filter: the won-only and natural-mix corpora are statistically indistinguishable, both beating the base by roughly 12 foundation cards. The expensive winnable-to-won harvest-scaling program (best-of-N replay, win-yield multipliers) is therefore not justified by this evidence; the cheaper "collect and train on more data of either outcome" path is favored. The caveat is that the natural-mix corpus still shares the harvester win games as a subset and the same deck distribution, so this separates "filter to wins" from "use the natural mix" at matched volume, not "wins" from "a corpus with zero wins."

## 6. Next experiments (pre-registered)

1. Generalization on fresh solver-confirmed-winnable random deals never seen by teacher or student, base versus gate paired. This is the decisive test of whether the gate learned Klondike or the deck distribution, and it is now the top experiment since the filter-vs-volume question is resolved.
3. Checkpoint selection: evaluate the iteration-250 and 500 checkpoints on the five decisive decks to find a checkpoint that keeps the wins with less JSON-discipline regression, before any public release.
4. Solver-as-teacher pilot: pathology-free demonstrations rendered through the v1.6 template, independent of the teacher's roughly 31% ceiling. Promoted if generalization shows the gate memorized.

## 7. Artifacts and reproducibility

- Adapters: `gemma4_finetune/adapters_gate` (gate), `gemma4_finetune/adapters_allsucc` (ablation). Gitignored; regenerate via `lora_config_gate.yaml` / `lora_config_allsucc.yaml` through `train_v2.py`.
- Corpora: `build_wononly_corpus.py --exclude-seeds <13 seeds>` and `build_allsucc_matched_corpus.py`, then `prepare_dataset.py`.
- Eval and adjudication: `tournament_A.py --arms <arm> --out-name tourA_v16_rescue`, `adjudicate_final_position.py`, `run_base_temp03_control.sh`, `run_allsucc_ablation.sh`.
- Per-game records and summaries: `gemma4_finetune/play_runs/tourA_v16_rescue/<arm>/seed<seed>/` (gitignored).
- All foundation-card and outcome numbers in this report were read from those summary files, not from any in-loop log or notification stream.
