# GRPO round 2, Phase R resume state (2026-07-07)

Handoff log for resuming the pre-registered round-2 RFT harvest across compute
windows. Pre-registration: `docs/reports/20260705_grpo_round2_preregistration.md`.
Recall memory: `grpo-rft-iter0`.

## Status: PAUSED at 50/64 rollouts. Nothing is running.

Second pause 2026-07-07 ~06:40 AEST at the operator's request (compute window
wrap-up). The detached runner, its `caffeinate`, and the in-flight game were all
killed (SIGTERM sufficed); a broad GPU/ML process sweep came back empty. The
in-flight `seed9000025/s7` directory (no `summary.json`) was removed so that slot
re-runs clean. The chain is fully resumable by construction (round-major runner
with a `[ -s summary.json ]` skip-if-done guard).

## Resume in one command (14 games left, ~4h to finish Phase R)

```
cd /Users/chayut/repos/solitaire-analytics
nohup caffeinate -i -s zsh gemma4_finetune/run_bestofn_r2.sh > gemma4_finetune/play_runs/bestofn_r2.launch.log 2>&1 &
```

Skip-if-done steps over the 50 banked games and runs only the 14 remaining:
round 7 for seeds 25, 209, 215, 216, 226, 207 (seeds 13 and 24 already have s7),
then all of round 8. Progress tallies land in `gemma4_finetune/play_runs/bestofn_r2.log`.

To stop cleanly at the next window end:
```
pkill -f run_bestofn_r2.sh; pkill -f play_deck_with_student.py
# then remove any partial s-dir with no summary.json before the next resume
```

## Banked state at 50/64 (authoritative, read off disk)

19 wins total. 7 of 8 decks carry within-group win/loss variance (usable
group-relative advantage signal); only 9000025 is flat. fc spreads are from the
round-6 tally (n=6); seeds 13 and 24 each added one loss at s7 to reach n=7.

| deck    | n | wins | variance | fc spread (through n=6)        |
|---------|---|------|----------|--------------------------------|
| 9000013 | 7 | 3    | yes      | 11, 13, 43, 52, 52, 52 (+loss) |
| 9000024 | 7 | 4    | yes      | 11, 11, 52, 52, 52, 52 (+loss) |
| 9000025 | 6 | 0    | no       | 3, 3, 3, 10, 10, 10            |
| 9000207 | 6 | 2    | yes      | 3, 24, 24, 25, 52, 52          |
| 9000209 | 6 | 1    | yes      | 5, 5, 9, 9, 37, 52             |
| 9000215 | 6 | 1    | yes      | 11, 21, 21, 28, 49, 52         |
| 9000216 | 6 | 3    | yes      | 27, 27, 28, 52, 52, 52         |
| 9000226 | 6 | 5    | yes      | 39, 52, 52, 52, 52, 52         |

9000025 is a consistent all-fail (hard cap at fc 3 or 10). If it finishes 0/8 it
becomes a zero-variance group and the extractor correctly skips it; the other
seven decks carry the training contrast.

## Headline readout: pass@5 lift over the volume selection prior

Every deck reached n>=5 this window, so the round-1 policy's pass@5 is now
measurable against the volume prior the pool was selected on.

| deck    | volume pass@5 (prior) | grpo_iter1_tr pass@5 | delta |
|---------|:---------------------:|:--------------------:|:-----:|
| 9000013 | 1/5                   | 2/5                  | +1    |
| 9000024 | 1/5                   | 3/5                  | +2    |
| 9000025 | 1/5                   | 0/5                  | -1    |
| 9000207 | 0/5                   | 2/5                  | +2    |
| 9000209 | 1/5                   | 1/5                  |  0    |
| 9000215 | 1/5                   | 1/5                  |  0    |
| 9000216 | 1/5                   | 3/5                  | +2    |
| 9000226 | 1/5                   | 4/5                  | +3    |
| total   | 7/40 (18%)            | 16/40 (40%)          | ~2.3x |

The round-1 TR policy roughly doubled pass@k on the success band while its greedy
heldout12 stayed at parity (5 vs 5). That is the textbook RLVR regime: coverage
rises before greedy@1 does, and converting that coverage into pass@1 is exactly
what round 2 is meant to do. One regression to keep honest: 9000025 went from
volume 1/5 to round-1 0/5 (the policy lost a line volume occasionally found).

## Night progress log (2026-07-06 to 2026-07-07, AEST)

- Session 1 (2026-07-05 06:06 START): rounds 1 (07:57) and 2 (10:28) complete,
  plus 4 games of round 3; paused ~12:00 at 20/64 (9 wins) to free memory.
- Session 2 (2026-07-06 22:10 RESUME, this window): skipped the banked 20, then
  - round 3 complete 23:31 (24/64, 11 wins)
  - round 4 complete 01:50 (32/64, 14 wins; 9000226 broke its all-win streak)
  - round 5 complete 04:17 (40/64, 16 wins; pass@5 readout above)
  - round 6 complete 06:30 (48/64, 19 wins)
  - round 7 partial (seeds 13, 24 done at s7, both losses) reaching 50/64
  - paused ~06:40 at operator request; job killed clean, sweep empty.

## Pipeline ahead (do NOT auto-launch past a window)

1. Finish Phase R (the resume command above), 14 games, ~4h.
2. Phase T, extract plus train (~3-4h GPU):
   - `extract_grpo_corpus.py --advantage-mode mean --traj-normalize`
     (Dr.GRPO mean-only advantages, per-trajectory length normalization,
     zero-variance group skip retained). Drift-gate the extraction.
   - `train_grpo.py` resuming from `adapters_grpo_iter1_tr`, TR beta 2.0 anchored
     to that policy, lr 8e-5, minibatch 8, 200 steps, `--max-rows 1600`
     stratified, checkpoint every 50. Invalidation check: diag pos/neg separation
     must open by step 100.
3. Phase E, eval plus gate (~3.5h): greedy cap 200 on heldout12 (9000101-24) vs
   cached volume (5/12) and grpo_iter1_tr (5/12), oracle OFF, exact adjudication.
   - GATE: grpo_r2 >= 7 strict paired wins (volume+2) is PASS, RFT is a confirmed
     lever, continue the loop.
   - Anything below 7 is FAIL: RFT CLOSES PERMANENTLY at parity-with-movement,
     no dose/LR/beta retries. Ship volume plus the knowledge-gated endgame oracle
     (best honest advisor measured, 7/12) and pivot to the inference layer and
     write-up.

Short-window alternative (operator call, not the pre-registered path): extract
Phase T now on the partial 50-sample pool (~6 per deck) rather than waiting for
the full N=8. The prereg default is the full harvest.

## File pointers

- Runner: `gemma4_finetune/run_bestofn_r2.sh`
- Banked rollouts: `gemma4_finetune/play_runs/bestofN_r2/seed*/s*/summary.json`
- Run log: `gemma4_finetune/play_runs/bestofn_r2.log`
- Policy adapter (round-1 output, on-policy for round 2):
  `gemma4_finetune/adapters_grpo_iter1_tr/` (step-50/100/150 plus the promoted
  `adapters.safetensors`)
- Deck pools: gate decks in `data/benchmarks/generalization_decks.json`,
  corrective decks in `data/benchmarks/winrate_decks.json`
- Pre-registration and gate: `docs/reports/20260705_grpo_round2_preregistration.md`
