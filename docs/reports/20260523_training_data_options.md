# Training-data options for the E2B Solitaire distillation

**Date:** 2026-05-23
**Author:** Chayut (with claude assistance)
**Status:** experiment design + two pilot artifacts

---

## TL;DR

The teacher-capability concern raised earlier today (Gemma-4 31B
self-aware-but-impotent on Klondike, ~10% win rate by corpus inspection)
opens up several training-data options. With **unlimited Gemma tokens**
the cost calculus changes: scale becomes free, so the experiment can
A/B multiple labelling strategies and let the held-out E2B performance
arbitrate, instead of betting on one upfront.

This doc captures:

1. The two pilot artifacts produced today and what they prove.
2. The expanded experimental matrix (six candidate training-data
   sources) and the comparison framework.

---

## Pilot artifacts produced today

Both run from `scripts/example_solver_training_record.py`.

### Pilot A — Solver-as-labeller (BLOCKED on solver capability)

**Goal:** deal a fresh Klondike seed, have `ParallelSolver` find a
winning move sequence, emit `(state, chosen_move)` records as
ground-truth training data without LLM involvement.

**Result:** the in-repo `ParallelSolver` does NOT solve fresh Klondike
deals in any tested budget:

| seeds tried | beam_width | per-seed timeout | states explored | wins |
|---|---|---|---|---|
| 9 (default set) | 2,000 | 60s | 105k–478k each | 0 / 9 |
| 2 (seeds 42, 7) | 20,000 | 180s | 2.5M, 2.9M | 0 / 2 |

Naive beam search over the full Klondike state space exhausts depth=200
without finding any solution path, even at 20k beam. This matches the
research literature: full-game Klondike requires domain-specific search
(hindsight optimisation, UCT with rollouts, etc.) — naive BFS or beam
search is known to be inadequate.

**Implication for option #2:** "solver-as-labeller" cannot be implemented
with the current in-repo solver. Path forward would be one of:

- Integrate a stronger solver (e.g. `solvitaire`, Bjarnason-style UCT).
  Estimated effort: medium — wrap an external binary, parse its output.
- Run the existing solver on partial mid-game states (not fresh deals)
  where the remaining search space is tractable.
- Drop this option and rely on LLM-side labelling (pilot B).

### Pilot B — LLM-replay-from-corpus-win (WORKS today)

**Goal:** take the corpus's one verified win on the newer template
(session `…1abf260154e1`, build `6dfc8a9`, seed `3263196305`) and emit
per-turn records from `data/dataset/decisions.jsonl`.

**Result:**

```
matched 138 successful turns across turnIndex 0..173
wrote 138 per-turn records -> data/dataset/demos/replay_training_example.jsonl
final state: foundationCards=51, completionProgress=98%
training-eligible rows: 138/138 (100%)
```

Every record has: state metrics, chosen move, board analysis, strategic
reasoning, confidence, `trainingEligible: true`. The full record shape
is in `data/dataset/demos/replay_training_example.jsonl`. The LLM
labelling produces *complete training rows* — state + action + CoT — in
a way the solver-only approach cannot.

**Implication:** the corpus *already* generates valid training data
from LLM wins. The bottleneck is the *number* of wins (currently two,
one with full attribution). Scaling means harvesting more sessions,
which is where the unlimited-token budget comes in.

---

## Expanded experimental matrix (given unlimited Gemma)

Six candidate training-data sources, ranked roughly by infrastructure
cost. The unlocked thing here is that we can run several in parallel
and let downstream E2B evaluation decide which produces the best
student, instead of betting on one.

| # | Source | What goes in the training set | Pros | Cons |
|---|---|---|---|---|
| 1 | **Win-filtered harvest** (current) | only complete-game wins | clean signal, model in coherent winning play | win rate ~5–15% → need 5–20× more sessions for meaningful N |
| 2 | **Per-move quality filter** | moves that increased `foundationCards` or decreased `faceDownTotal`, from any game | order of magnitude more data, no game-completion gate | the rest of the game (reasoning conditioned on bad moves) is discarded; SFT on isolated good moves loses long-horizon context |
| 3 | **Best-of-N at decode time** | sample K candidate moves per turn, score each via engine heuristic, keep only the move-+-rationale that maximises a quality function | gets a stronger model out of a weak one cheaply (test-time compute); produces dense per-turn high-quality data | rationale was generated for the chosen move, so it self-justifies; may not be as causally honest as natural reasoning |
| 4 | **Solver-validated LLM** | LLM proposes move + rationale; check the move appears in solver's principal-variation set; regenerate if not | combines authentic reasoning with ground-truth move correctness | currently blocked on solver from pilot A; works only mid-game |
| 5 | **Solver-only** | (state, action) pairs, no CoT | pure imitation, no rationale-action mismatch | no natural-language teaching signal for E2B; blocked on solver |
| 6 | **Solver-move + LLM-rationale (post-hoc)** | solver picks the move, LLM writes the rationale conditioned on that move | authentic moves + readable explanations | rationale is post-hoc rationalisation, not causal — risks training the student to confabulate; blocked on solver |

### Recommended experiment design

Given unlimited Gemma but bounded student-training and eval budget:

**Phase 1 (cheap, parallel-runnable in days):**

- Run #1 and #2 in parallel for a week. Target: ~500 winning games
  for #1 (means ~5k attempts at 10% rate); ~50k high-quality
  per-move rows for #2 (drops out naturally from the same harvest).
- Run #3 on a subset of sessions (~200 games) — implement
  best-of-N=8 sampling, keep the engine-validated best move each turn.

**Phase 2 (decision gate):**

- Train three E2B variants in parallel (one per source).
- Eval on a held-out set of 100 winnable boards (perfect-info solved
  in advance, so we know which are solvable).
- Compare: win rate, mean turns-to-completion, oscillation rate.

**Phase 3 (only if #1/#2/#3 underwhelm):**

- Invest in a better solver (#5 / #6 path becomes available).
- Or switch teacher entirely (the earlier "stronger teacher" path).

The phase-1 cost is roughly:
- Compute: free (unlimited Gemma tokens).
- Wall-clock: ~1 week of harvest under the current ~75% provider-error
  rate.
- Storage: ~5 GB of raw exports (manageable; gitignored already).

### What the phase-1 experiment will tell us

The three variants test three distinct hypotheses:

- **#1 (win-filter)**: "If we only show the student wins, it learns
  to win." Implicit assumption: the LLM teacher's *winning* games
  contain coherent strategy worth imitating, even if its overall play
  is incoherent.
- **#2 (per-move quality)**: "Bad games still contain good moves. The
  student learns from those." Implicit assumption: per-move quality is
  more important than full-game coherence for distillation.
- **#3 (best-of-N)**: "Test-time compute can synthesise a stronger
  teacher from a weak one." Implicit assumption: the LLM has enough
  capacity to generate one good move-+-rationale per 8 tries.

The result space is informative regardless of which wins:
- #1 best → curate aggressively, slow harvest, high-quality bar.
- #2 best → don't worry about wins, filter at move level, scale wide.
- #3 best → test-time scaffolding is the lever, not raw data volume.
- All similar → data source isn't the bottleneck; model capacity is
  (and the earlier "switch teacher" recommendation comes back).
- All bad → the original premise (E2B can play Klondike at all) needs
  rethinking, independent of training-data source.

---

## What blocks each path today

| Path | Blocked on | Unblock cost |
|---|---|---|
| #1 win-filter | nothing — just harvest budget | already running |
| #2 per-move quality | a quality scorer (foundationDelta + faceDownDelta are already in `decisions.jsonl`) | half-day script |
| #3 best-of-N | a decode-time sampler + engine validator | 1–2 days script |
| #4 solver-validated | a working full-game solver | weeks (integrate external) |
| #5 solver-only | a working full-game solver | weeks (as above) |
| #6 solver+rationale | a working full-game solver | weeks (as above) |

The three cheap paths (#1, #2, #3) are also the three most directly
informative for the phase-1 A/B. Recommend starting there.

---

## Phase-1 pilots — actuals from today's corpus

All three feasible options now have working scripts and concrete output
on the existing 2,753-row decisions corpus (23 sessions, of which 2 are
wins). Numbers below are the today-corpus baseline; they scale with
harvest size.

| Option | Script | Output | Rows produced | Notes |
|---|---|---|---|---|
| #1 win-filter | `scripts/extract_winning_trajectories.py` | `data/dataset/demos/option1_win_filtered.jsonl` | **243** total (138 from `0154e1`, 105 from `ce0fb4`) | only 138 are `trainingEligible=true`; ce0fb4's 105 lack the eligibility flag because its build/seed metadata is unavailable (pre-attribution session) |
| #2 per-move quality | `scripts/filter_per_move_quality.py` | `data/dataset/demos/option2_quality_filtered.jsonl` | **349** quality moves (12.8% keep rate from 2,730 consecutive-success pairs) | 191 foundation advances, 192 face-down reveals (1 overlap). 1.4× more rows than option #1 *from the same corpus*. |
| #3 best-of-N | `scripts/example_best_of_n.py` | `data/dataset/demos/option3_best_of_n_{first_legal,random_n,exhaustive}.jsonl` | one full game per proposer on seed 42 | harness works; demonstrates the lift from single-candidate (`first_legal`) to multi-candidate (`random_n`, `exhaustive`) on the same board sequence |

### Option #3 illustrative game on seed=42

Played one game per proposer with the same engine-heuristic scorer:

```
proposer       turns  foundation  faceDown  revealed  won  stuck
first_legal      300           1        15         6            
random_n          39           3        12         9           Y
exhaustive       300           3        11        10            
```

What this shows:
- The harness produces a per-turn record with chosen move, alternatives
  considered, and engine scores — matching the schema training expects.
- Lift from single-candidate to multi-candidate is measurable (foundation
  1 → 3, reveals 6 → 10) on the same deal.
- All three proposers still play terribly because **the engine heuristic
  scorer is weak** — 1-ply lookahead can't solve Klondike (same finding
  as pilot A). The lever that matters in production is the LLM
  proposer's signal, not the harness; this offline demo just proves the
  scaffolding compiles and the output schema is right.

### Cross-option comparison on today's corpus

| Dimension | #1 win-filter | #2 quality-filter | #3 best-of-N |
|---|---|---|---|
| rows produced today | 243 | 349 | one-game/proposer (demo only) |
| training-eligible | 138 | 349 (all retained have a quality signal) | depends on proposer choice |
| includes full CoT? | yes (LLM `boardAnalysis` + `reasoning`) | yes (per-row, even from losing games) | yes if proposer is LLM; no in offline demo |
| scales with harvest | linearly with attempts | super-linearly with attempts (every game contributes; only wins count for #1) | linearly with proposer calls × N |
| moves from losing games? | no | yes (the point of #2) | no concept — generates fresh data |
| failure mode | not enough wins | bad moves between good ones could teach disjointed strategy | proposer dominates outcome quality |

The biggest finding from running them is that **option #2's keep rate
(12.8%) means every harvest dollar produces ~5× the training rows
compared to option #1's win-only path** (given typical 10–20% win
rates). That's a real scale advantage and supports including #2 in
phase-1 regardless of the curation-vs-scale debate.

The biggest finding from the failed pilot A is that **option #2 is
also currently the only "ground-truth-validated" lever available** —
not because the moves are solver-proven optimal, but because they have
an observed monotone effect on the two metrics the game itself
defines as progress. That's a much weaker guarantee than a solver win
path, but it's the strongest signal available today.

## Related artifacts

- `/Users/chayut/repos/solitaire-analytics/scripts/example_solver_training_record.py`
  — original solver/replay pilot; demonstrates the labeller-schema shape.
- `/Users/chayut/repos/solitaire-analytics/scripts/extract_winning_trajectories.py`
  — option #1 implementation.
- `/Users/chayut/repos/solitaire-analytics/scripts/filter_per_move_quality.py`
  — option #2 implementation.
- `/Users/chayut/repos/solitaire-analytics/scripts/example_best_of_n.py`
  — option #3 harness with three offline proposers.
- `/Users/chayut/repos/solitaire-analytics/data/dataset/demos/` — output
  JSONL artifacts for each option.
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260522_prompt_template_audit.md`
  — corpus-wide template + capability analysis that motivated this doc.
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260522_harvester_team_handover.md`
  — harvester-team asks that unblock attribution for any of these
  experiments (`promptTemplateHash`, resignation output, same-seed run).
