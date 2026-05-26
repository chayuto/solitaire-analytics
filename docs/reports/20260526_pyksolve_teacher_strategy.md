# Pyksolve PIMC as a parallel teacher track

**Date**: 2026-05-26
**Status**: Strategic decision recorded. Track B (pyksolve PIMC teacher) approved
to run in parallel with Track A (continued 31B teacher distillation, free
harvest). Track C (public publication of the Gemma 4 31B Klondike play corpus
and tooling) approved as a first-class concurrent goal. v1.1 LoRA remains
canonical; no production change.
**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.7)

## 1. Context

The Gemma 4 E2B distillation project has shipped v1.1 (Gemma 3n base, LoRA
trained on `gemma-4-31b-it` decisions) and HELD v2 (Gemma 4 E2B base, same
recipe, doom-loop regression on every checkpoint, see
`20260526_v2_gemma4_distillation_lab_log.md`). v3 (per-turn shuffle filter on
the same corpus) is scaffolded and ready to fire when compute opens.

The same-day pyksolve integration commit (`6a02b88`) revealed that the
in-repo `ParallelSolver` was too weak to test deal solvability (returned 0/5
on the corpus's confirmed-winnable seed). pyksolve (Cython wrapper around
ShootMe/Klondike-Solver) returns 10/10 in ~10 ms each on the same input.
That capability raises a strategic question for the distillation project,
not just for the analyst skill it was originally added for.

## 2. The strategic question

> Should the teacher signal for the next distillation generation be the
> `gemma-4-31b-it` model, as today, or pyksolve under Perfect-Information
> Monte Carlo (PIMC) determinisation?

The two answers correspond to fundamentally different research programs.

### Track A: LLM-imitation distillation (status quo)

- Teacher: `gemma-4-31b-it` decisions harvested from
  `https://solitaire.chayuto.com/`. Harvest is **free** (the
  self-hosted endpoint has no per-call cost), so this track can keep
  growing in the background at zero marginal spend.
- Student goal: "match the 31B teacher's decisions."
- Training data: prompt with `CURRENT GAME (JSON)` (or hybrid-v1 plain text)
  plus the teacher's full JSON output (move, boardAnalysis, strategicPlan,
  confidence, thinking trail).
- Failure modes inherited from the teacher: confidence saturation at 0.91
  mean, doom-loop confabulation, ~75% provider-error/timeout rate (cost is
  wall-time and storage only, not money), ~9% corpus win rate.
- Current state: v1.1 shipped, v2 HELD, v3 scaffolded. The corpus continues
  to grow as the harvester runs; each ingest cycle adds rows for free.

### Track B: Search-grounded policy distillation (new)

- Teacher: pyksolve under PIMC. For each board state, sample K consistent
  worlds (face-down identities determinised from the user-visible cards),
  solve each with pyksolve, aggregate to a single move label.
- Student goal: "play Klondike well under the user's information set."
- Training data: same prompt format as today (so the student input matches
  what a real user sees), plus a synthesised move label (and optionally a
  synthesised reasoning trail derived from pyksolve's winning continuation).
- Failure modes: PIMC has known limitations (strategy fusion, non-locality)
  but cannot reproduce the doom-loop confabulation pattern the LLM teacher
  exhibits. No provider errors. No API budget.
- Current state: design only, not started.

## 3. The honest view that informed this decision

Captured verbatim from the in-session reasoning at the point of decision,
so the rationale is preserved with full force rather than retroactively
softened.

> Stop distilling from gemma-4-31b. It's a mediocre teacher (~9% win rate,
> 0.91-saturated confidence, doom-loop confabulation that just put v2 on
> HOLD). Pyksolve PIMC is a strictly better move teacher: faster,
> optimal-under-uncertainty, and structurally incapable of the failure
> mode that killed v2.
>
> (Cost is not the differentiator: the harvest endpoint is free. The
> argument is purely about teacher *quality*, not budget.)
>
> The "hybrid teacher" suggested earlier (PIMC move + 31B rationale) is a
> hedge to make the change feel safer. The cleaner move:
>
> 1. Drop the 31B from the move signal entirely. Pyksolve PIMC produces
>    the move label.
> 2. Synthesize reasoning text from pyksolve's solution structure, not
>    from the 31B. "This move wins in 17/20 consistent worlds; the winning
>    continuation is foundation 5H, reveal col 4, foundation 6S, ..." is
>    honest and grounded. The 31B's verbal reasoning is wrong about half
>    the time (e11e3b is a clean example: it stated the correct plan and
>    then ignored it).
> 3. Change the success criterion from "matches 31B" to "wins games."
>    Which is what you actually wanted from a Solitaire advisor anyway.
> 4. Throw out the current `data/dataset/training.jsonl`. It's structurally
>    contaminated with the 31B's doom-loop responses. That's literally why
>    v2 retraining reproduced the failure: you trained on the failure.
>
> The harder honest question: if the deployed goal is "advisor that plays
> Klondike well," ship pyksolve itself behind a thin LLM-as-interface
> wrapper and skip the distillation entirely. Distillation is interesting
> research, but it's making the result objectively worse than the teacher
> could be on its own. The distillation only justifies itself if the
> deployment constraint is "must be a single forward pass on E2B with no
> external tools" (e.g., shipping inside a mobile app that can't run
> pyksolve). Be honest with yourself about whether that constraint is
> real or aesthetic.

## 4. The user's response: run three tracks in parallel

Rather than abandoning Track A, the decision is to **run Tracks A, B,
and C in parallel** as independent goals. Rationale:

- Track A has months of sunk investment (corpus, prompt-template audits,
  harvest team co-ordination, shipped v1.1). The infrastructure exists,
  the v3 experiment is one-command-away from firing, and **harvest is
  free**, so the background cost of continuing to grow the corpus is
  ~zero. Even if Track B succeeds, Track A's data has independent value
  (Track C below) and the comparison between them is itself a result.
- Track B is genuinely novel; the comparison between Track A's best and
  Track B's first run is itself a valuable result, regardless of which
  wins.
- Track C (public publication) is independently valuable: nobody else
  has a public corpus of how a frontier-ish LLM actually plays Klondike
  with full reasoning trails, calibration data, and a documented failure
  taxonomy. Even if neither distillation track ships a winning model,
  the corpus and analysis tooling are useful to others.
- The three tracks share infrastructure (the harvester, the corpus, the
  Gemma 4 E2B base and patch, MLX-LoRA pipeline, the 20-state bench at
  `/Users/chayut/repos/solitaire-analytics/experiments/a4_phase1.5_2026_05_24/`)
  so the marginal cost of adding B and C alongside A is small.
- Running all three lets success criteria be settled empirically rather
  than declared.

## 5. Track A: continue as planned (background, free)

Unchanged. The active item is the v3 experiment scaffolded at
`/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md`
(per-turn shuffle filter applied to the v2 corpus). Pre-registered
hypothesis and one-shot recipe already in place. ~105 min compute when
window opens.

The current `data/dataset/training.jsonl` (1635 rows of teacher decisions
including known doom-loop responses) is kept as the Track A corpus. The
fact that it's "contaminated" is the whole point: that's the input
hypothesis Track A is testing, that filtering at the per-turn level can
de-poison it.

**Background mode**: the harvester continues to run with no human in the
loop. New exports land in `/Users/chayut/Downloads/`, get ingested via
`/Users/chayut/repos/solitaire-analytics/scripts/ingest_exports.py` (which
now handles the hybrid-v1 plain-text prompt format), and the corpus grows
session by session. Cost is wall-time and disk only. No prompt to the
operator unless an export is genuinely interesting (a new failure mode,
a win, a same-seed comparison arm). This is the steady-state for Track A
between training experiments.

## 6. Track B: PIMC pyksolve teacher

### 6.1 What needs to be built

| component | what | rough effort |
|---|---|---|
| PIMC labeller | function that takes a (state + legalMoves) input and returns the move that wins in the most pyksolve samples, plus the win count | ~3 hrs |
| Reasoning synthesiser | template that converts pyksolve's solution path into a `boardAnalysis` / `strategicPlan` / `move` JSON matching the existing prompt-output contract | ~2 hrs |
| Corpus generator | tool that walks every state in the existing corpus (or a freshly-dealt set of N games) and emits a PIMC-labelled training row in the same JSONL shape as `data/dataset/training.jsonl` | ~2 hrs |
| Training run | reuse existing `train_v2.py` and `lora_config_v2.yaml` against a `dataset_b1` directory built from the new labels | 85 min compute |
| Evaluation | reuse the 20-state Phase 1.5 bench at `experiments/a4_phase1.5_2026_05_24/`. Score against the same teacher-pick reference used by v1.1 and v2 | 20 min compute |

Total: ~7 hours of build time plus ~105 minutes of compute. Same cost
budget as a v3 run.

### 6.2 Two corpus options for Track B

Option B1: **PIMC-relabel the existing 1635 rows.** Same prompts, same
state distribution, only the labels change. Cleanest controlled
comparison against Track A because the input distribution is held
constant.

Option B2: **Generate a fresh corpus.** Use pyksolve's own
`shuffle1(seed)` to deal N games, drive them with a known-good policy
(or random + PIMC-correct play), capture (state, PIMC-label) at every
turn. Much larger corpus possible; closer to RL-style on-policy data.
Different distribution from the 31B harvest.

Recommendation: start with B1 (cheaper, faster, cleaner comparison).
B2 is the natural follow-up if B1 looks promising.

### 6.3 Pre-registered hypothesis for B1

> H_B1: A LoRA trained on PIMC pyksolve labels over the same 1635 prompts
> will (a) outperform v1.1 on the 20-state bench on mean tier, AND
> (b) preserve oscillation-resistance at the level of v2-untuned
> (osc_correct >= 5/7), AND (c) not exhibit the doom-loop confabulation
> pattern v2 trained showed.

Success at all three conditions: PIMC distillation is the new mainline.
Success at (b) and (c) but not (a): the move signal is correct but the
training rows are too few to drive a large lift; trigger B2.
Failure at (b) or (c): the imperfect-info-to-perfect-info gap is real
and PIMC's strategy-fusion limitation matters in practice; document and
revisit with a richer labelling scheme.

### 6.4 The deployment-shape question

Today's E2B deployment is a JSON-emitting advisor with verbal reasoning.
If Track B succeeds, the verbal reasoning is templated from pyksolve's
solution structure rather than learned from a human-like LLM. That is
**different** in character even when the surface shape matches:

- Templated reasoning is always truthful (it reflects an actual search
  result) but stylistically uniform.
- 31B-derived reasoning is stylistically varied but factually unreliable.

If end-users care about the verbal style and not just the correctness,
Track A may retain value purely as a "voice" donor. The hybrid teacher
(PIMC move + 31B rationale) becomes a real third option in that case,
not just a hedge. Defer the decision to after the B1 result lands.

## 7. Track C: public publication of the Gemma 4 31B Klondike corpus

**Goal**: make the corpus, the analysis tooling, and the failure-mode
taxonomy publicly available so anyone curious about how a frontier-ish
LLM actually plays Klondike (and where it fails) can build on top of it.
This is independent of whether either distillation track ships.

### 7.1 What exists already

The Hugging Face dataset `chayuto/klondike-llm-decisions` (CC-BY-4.0)
already publishes three configs derived from the corpus:

- `full`: every successful decision across all models and schema versions,
  including stalled and doom-loop turns. Research baseline.
- `client_v1_teacher_clean_raw`: filtered to the teacher model on the
  current schema, full record. Training-friendly subset.
- `client_v1_teacher_clean_lean`: same as clean_raw with a flatter
  schema (Arrow-friendly).

Per memory `[[hf-dataset-published]]` this was first published 2026-05-25
and regenerates from `scripts/ingest_exports.py` plus an `HfApi().upload_folder`
push. Each new ingest cycle can re-publish.

### 7.2 What to add for Track C

The dataset alone is a row dump. To make it useful to outside readers,
add a publication layer over the top:

| artefact | what | where it lands |
|---|---|---|
| **Dataset card improvements** | Expand the existing HF README with the failure-mode taxonomy from `data/DATASET_NOTES.md`, the same-seed validation methodology, and a "how to interpret the rows" section. Currently the card is auto-generated and machine-tone. | `data/publish/README.md`, pushed to HF on next ingest |
| **Companion writeup** | A "How Gemma 4 31B plays Klondike" report: win rate, plateau lengths, the doom-loop confabulation pattern, the missing-Ace lock pattern, the calibration-bands prompt comparison. Aimed at someone deciding whether this corpus is useful for their work. | New `docs/public/` subdir, mirrored to a public-facing markdown (HF Spaces, blog, or a GitHub Pages slot) |
| **Replay tooling** | Document the `solitaire.chayuto.com/?seed=<seed>&session=<uuid>` URL pattern so outsiders can replay specific corpus seeds in their own browser and compare their own AI's decisions. | Section in the companion writeup |
| **Analysis skill (excerpt)** | A redacted version of `.claude/skills/solitaire-analyst/` (drop internal-only references, keep the verdict template, dead-deal signatures, and the `check_winnability.py` script) so others can run the same kill-or-continue analysis on their own exports. | New `tools/solitaire-analyst-public/` subdir, or a separate small repo |
| **Bench publication** | The 20-state Phase 1.5 bench at `experiments/a4_phase1.5_2026_05_24/` is the standard scoring rig used by v1.1/v2 evals. Publish the prompts and the scoring methodology so others can score their own models on the same boards. | New HF dataset or a section in the existing one |

### 7.3 What's intentionally NOT published

- The harvester source code (lives in a separate repo not maintained
  here).
- The HF dataset auth token (`hf_*`) and any other secrets.
- The internal-only docs under `docs/internal/` (harvest-team comms,
  per-engineer pings).
- Working v2/v3/Track B adapters until they ship (avoids confusion with
  the v1.1 canonical model).

### 7.4 Cadence

Track C is a periodic-batch goal, not a continuous one. The natural
trigger is: every time the harvest corpus crosses a milestone (every
~+500 rows, or after a build/prompt-template change), regenerate and
re-push the HF dataset, refresh the companion writeup with the new
numbers, and announce the bump on whatever public surface the user
prefers (GitHub repo README, blog post, social post, etc.).

### 7.5 Why this matters

The corpus is a small but reasonably clean public record of a
specific failure class (long-horizon planning under uncertainty in a
fully-observable goal state with partial observability of the
intermediate state). That class is not well-represented in standard
benchmarks. The fact that the same teacher reaches 100% on simple
positions but 9% on full games, and that the failures are
self-aware-but-impotent rather than random, is genuinely interesting.
Researchers working on planning, RLHF, or imperfect-info game AI may
find the corpus useful as either a benchmark or a counterexample
generator. Track C makes that possible without requiring the
researcher to set up a Klondike harvester from scratch.

## 8. The non-distillation alternative

The harder honest view at the end of section 3 stands and must be
revisited explicitly at the next decision gate: if the deployment goal
is "advisor that plays Klondike well" and the constraint of "must be a
pure E2B forward pass" is aesthetic rather than real, the right ship is
pyksolve itself behind a small LLM-as-interface wrapper (a chat skin
that translates user requests into pyksolve calls and pyksolve outputs
into prose). That's much simpler and strictly better than any
distillation.

Track B at minimum validates the distillation premise. If even with the
strongest possible teacher signal the distilled student loses to
pyksolve-direct by a wide margin, the answer is "ship pyksolve, not the
distilled model."

## 9. Decision criteria for the next ship decision

Settle after B1's bench scores land:

- **Ship v1.1 (status quo)**: B1 fails at (a) and (b); v3 also fails or
  yields negligible lift. Track A and Track B both blocked; v1.1 stays
  canonical pending more harvest data.
- **Ship v3 (Track A wins)**: v3 lifts mean tier > 3.15 and B1 doesn't
  match or beat it. The corpus filtering hypothesis was the bottleneck.
  Track A is the mainline.
- **Ship B1 (Track B wins)**: B1 lifts mean tier > 3.15 AND preserves
  oscillation behaviour. The teacher signal was the bottleneck. Track A
  is deprecated for future generations; v1.1 remains in production for
  back-compat.
- **Ship hybrid (mixed result)**: B1 wins on move quality but loses on
  reasoning style metric (if measurable). Combine PIMC moves with
  31B-derived reasoning text and retrain.
- **Skip distillation (non-distillation wins)**: pyksolve-direct (no
  distillation) beats every distilled candidate on the bench. Ship the
  thin-wrapper option from section 7.

## 10. Risks and open questions

- **PIMC strategy fusion**: PIMC's known weakness is making moves that
  are good in expectation but irreversible if the actual world is
  different from the most-common consistent world. For Klondike this
  matters less than Bridge but is real; the 20-state bench may not
  surface it.
- **K (number of samples per PIMC call)**: K=20 is the smallest defensible
  number. Larger K reduces variance but linearly increases corpus-build
  time. Defer tuning to after the B1 result; if results are flaky on
  re-runs, raise K.
- **Draw-3 vs draw-1**: harvester is draw-3. pyksolve supports draw-3
  but solve rates differ; calibrate before the corpus build.
- **No reasoning trail in B1**: the templated reasoning is a known
  compromise on style. If the user cares about voice, this is a feature
  loss vs Track A.
- **Reproducibility of the corpus**: PIMC sampling has a random seed;
  the corpus build must record the seed so re-builds reproduce
  identical labels.

## 11. Memory updates this session

- `[[gemma4-e2b-distillation-project]]`: add Track B pointer noting the
  parallel-track decision; do not remove the Track A status which is
  unchanged.
- New memory `[[pyksolve-pimc-teacher-track]]`: project-type, describes
  Track B's purpose, the H_B1 hypothesis, and points back to this
  document for the full design.

## 12. Cross-references

- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260525_gemma4_e2b_v2_exploration_plan.md` (Track A plan and v2 results)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v2_gemma4_distillation_lab_log.md` (Track A scientist log; doom-loop diagnosis)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_v3_experiment_design.md` (Track A's next experiment, ready to fire)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260526_session_close_v2_gemma4_text_HELD.md` (v2 HOLD record)
- `/Users/chayut/repos/solitaire-analytics/.claude/skills/solitaire-analyst/scripts/check_winnability.py` (pyksolve adapter and pysol-format converter, reusable by Track B)
- `/Users/chayut/repos/solitaire-analytics/data/dataset/training.jsonl` (the contaminated corpus that motivates this decision)

## 13. Next concrete action

When the next compute window opens, choose whichever you want to run
first based on appetite:

- **Track A v3** (~105 min): tests "filter the existing corpus to
  remove doom-loop turns." Pre-registered; one shell command. Cleanest
  way to settle the corpus-content hypothesis Track A was built on.
- **Track B B1** (~7 hrs build + ~105 min compute): tests "swap the
  teacher signal entirely." Bigger ambition; bigger build cost; but the
  payoff is a categorically different student that is structurally
  immune to the failure that put v2 on HOLD.

Both are valid first moves. The decision between them is about whether
you want to first finish what's already in flight (v3) or first see
whether the new direction even works (B1). Either order is fine; both
will get run eventually under the parallel-tracks decision.
