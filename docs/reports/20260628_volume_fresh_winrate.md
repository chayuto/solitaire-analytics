# Volume greedy win-rate on 24 fresh decks: generalization is ~33%, not ~42%

Date: 2026-06-28
Context: with the on-policy SFT levers exhausted (see
`20260628_star_iter1_oversample_ablation.md`), this window measures the one
number that gates the harvest-vs-RFT-vs-ship fork: volume's true greedy
generalization win-rate on a broad fresh deck set, which had only two 12-deck
point estimates (both 5/12).

## TL;DR

The volume student (the published LEAD, trained 2026-06-14) wins **6 of 24**
fresh solver-winnable decks greedily (strict 52/52), meanFC 24.4. Exact
adjudication of the deep near-misses lifts this to an **effective 8/24 (33%)**:
one deck was a cap-truncated win (fc40, fd0, SOLVED n=13) and one was a
false-resign at fc50/52 (SOLVED n=3, the close-out phenotype). Pooling all three
fresh sets (general12 5/12 + heldout12 5/12 + this 6/24) gives **16/48 = 33%**
[approx 95% CI 20-47%]. So volume generalizes to about one in three fresh
winnable decks, below the ~42% the two small sets implied, but still roughly 4x
the untuned base (~8%). The 18 losses (especially two thrown boards and two
stalls on winnable) are the concrete corrective-signal pool for a best-of-N or
RFT step; two of them (the resign and the cap) are recoverable without new
training.

## Method

24 fresh decks dealt from seeds 9000201-9000239 (`build_generalization_decks.py
--seed-start 9000201`), each solver-confirmed winnable at node cap 200k, verified
disjoint from the heldout (9000101+) and generalization (9000001+) eval sets and
from all training corpora. Volume adapter (`adapters_volume`) played greedy
(temp 0, cap 200, prompt v1.6, the production-faithful retry rules), one game per
deck, via `run_volume_winrate.sh` on the same proven harness as the star_iter1
eval. Deep near-misses (final fc >= 20) were then adjudicated by
`adjudicate_final_position.py` (zero-drift engine replay of every decision, then
the sound best-first solver on the exact final position).

## Results

Strict: 6/24 won (25%), meanFC 24.4.

Wins: 9000205, 9000212, 9000217, 9000224, 9000227, 9000235 (all 52/52).

Deep near-miss adjudication:

| deck | strict outcome | final fc/fd | solver | reading |
|---|---|---|---|---|
| 9000237 | resigned | 50 / 0 | SOLVED n=3 | false-resign one cluster from a forced win |
| 9000226 | max_turns | 40 / 0 | SOLVED n=13 | cap-truncated win |
| 9000216 | max_turns | 23 / 6 | SOLVED n=80 | stall on winnable |
| 9000215 | max_turns | 21 / 3 | SOLVED n=1322 | stall on winnable |
| 9000207 | max_turns | 22 / 2 | UNSOLVABLE n=32 | threw a winnable board |
| 9000209 | max_turns | 20 / 5 | UNSOLVABLE n=16 | threw a winnable board |

Effective win-rate counting the two near-forced wins lost to a controllable
failure (9000237 false-resign, 9000226 cap): 8/24 = 33%.

Remaining losses (fc < 20, not adjudicated, stalled or thrown early): 9000201
(9), 9000202 (6), 9000203 (8), 9000206 (3), 9000210 (7), 9000214 (6), 9000221
(11), 9000231 (13), 9000232 (11), 9000236 (12), 9000238 (8), 9000239 (3).

## Findings

1. **The 5/12 headline was optimistic.** Two independent 12-deck sets both read
   5/12 (42%); the larger 24-deck set reads 6/24 (25%), pulling the pooled
   estimate to 16/48 = 33%. Volume generalizes, but to ~1 in 3, not ~2 in 5.
2. **The LEAD student false-resigns on fresh decks.** 9000237 resigned at 50/52
   with all face-down cards already revealed, a position the solver finishes in
   3 nodes. This is the same close-out false-resign the volcloseout recipe fixed
   in-distribution, now observed on plain volume on an unseen deck. It is a
   controllable loss (resign calibration / close-out oversample), not a policy
   ceiling.
3. **Two of 24 losses are recoverable without training** (the false-resign and
   the fc40/fd0 cap truncation); the genuine policy gaps are the two thrown
   boards (winnable converted to dead) and the two stalls on winnable.
4. **Effective rate (33%) equals the pooled strict rate (33%)** on this set,
   a consistent read of volume at about one third of fresh winnable decks.

## Implication for the fork

Volume's generalization is real but modest (~33%) and partly bottlenecked by a
fixable resign reflex. The 18-deck loss pool (with exact UNSOLVABLE/SOLVED labels
on the deepest six) is now the targeted input a best-of-N corrective harvest or
RFT would use; the thrown boards (9000207, 9000209) also carry solver-provable
win-preserving alternatives at their fatal move, which is the same preference
signal mechanism 4 of the strategic review mines.

## Reproducibility and artifacts

- Runner: `gemma4_finetune/run_volume_winrate.sh`; decks
  `data/benchmarks/winrate_decks.json` (24, seeds 9000201-9000239).
- Eval summaries: `gemma4_finetune/play_runs/volume_winrate_eval/seed*/`.
- Adjudicator fix: `adjudicate_final_position.py` now locates a run's deck by
  searching all `data/benchmarks/*.json` (the summary records only a generator
  tag, not a path), so fresh deck sets adjudicate; backward-compatible (tries the
  default benchmark first).
- Uncommitted: the runner, the deck file is gitignored (benchmarks dir), the
  adjudicator patch, this report.
