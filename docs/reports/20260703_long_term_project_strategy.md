# Long-term project strategy (2026-07-03)

A deep strategy for the Klondike LLM program: where the project stands after
five weeks of measurement, what the durable goals are, which levers remain
live, and a pre-registered decision structure for the next quarter. Written at
the moment the project sits on a single cheap measurement (the trust-region
RFT eval) that forks its future.

## 0. TL;DR

- **Mission**: a fully local Klondike advisor (Gemma 4 E2B int4 + LoRA) that
  plays BETTER than the 31B teacher it was distilled from, with the public HF
  dataset, the public adapters, and a methods write-up as the durable
  artifacts.
- **Position**: the volume adapter is the shipped lead (~33% fresh-deck
  win-rate, ~4x untuned base, statistically at teacher level already). Every
  SFT-shaped improvement lever is CLOSED by measurement, not by opinion.
- **The one live capability lever** is on-policy RFT (GRPO/RLOO). Iteration-0
  proved the failure mode (KL-free collapse), proved the fix (one-sided trust
  region: discrimination without collapse in training), and is UNEVALUATED
  (killed at compute-window end). One ~2h eval resolves the fork.
- **Two futures, both acceptable**: RFT beats volume -> scale it into an
  iterated loop (per-token KL, fresh rollouts per round). RFT flat -> the
  capability program closes honestly; the project pivots to deployment
  quality, steady-state data, and publication.
- **Independent of the fork**: constrained JSON decoding is the
  highest-value deployment lever left (kills the parse-death class and the
  adapter's 6x JSON regression), compute-cheap, and helps BOTH futures.
- **Teacher/harvest moves to steady state**: freeze prompt v1.6, keep
  full-stream ingest + HF publishing cadence, push dead-board budget fixes to
  the harness side, stop teacher prompt R&D unless a measured gap reopens.

## 1. Mission and end-states

The project is three nested deliverables, in priority order:

1. **The advisor** (product): an on-device Solitaire advisor at E2B-int4
   scale that a laptop or phone can run, playing at or above average-human
   strength under honest imperfect information.
2. **The evidence** (research): a fully measured record of what does and does
   not work when distilling + RL-ing a small local model against a real
   sequential game: 76 dated reports, adjudicated evals, pre-registered gates.
3. **The public artifacts** (community): the CC-BY-4.0 decision dataset
   (~29k interactions at the 2026-06-27 push), the published adapters, honest
   model cards.

### The capability ladder (targets, each gated by measurement)

| Tier | Claim | Bench | Status |
|---|---|---|---|
| T0 | Beat untuned base | any | DONE (volume ~33% vs ~8%, ~4x) |
| T1 | Beat volume paired | heldout12, >=+2 wins | OPEN (RFT run-2 eval pending) |
| T2 | Beat volume fresh | 24 fresh decks, >= volume+2 | open |
| T3 | Beat the teacher | pooled 48+, >=38% strict | open (volume 33% ~ teacher 31%: parity, not exceedance) |
| T4 | Approach the ceiling | pooled 48+, >=42% | stretch (draw-1 imperfect-info ceiling ~43-52%) |

T3 is the strategic prize: a 2B student EXCEEDING its 31B teacher is the
thesis of the whole program. Imitation provably cannot get there (section 3);
only RL against the game can.

### Definition of done for v1 of the project

- A shipped adapter with T1+T2 confirmed (or a documented closure at T0 with
  volume as final), constrained decoding in the inference path, cards current.
- The consolidated methods write-up published (HF blog or arxiv preprint).
- Harvest/publish running as low-touch steady state.

## 2. Asset inventory (2026-07-03)

**Data**: ~29k-row public dataset (5 configs, append-only Parquet, integrity-
gated); 53 terminal win-record files; full-stream ingest with per-version
attribution; 4 model cohorts catalogued (31B teacher, 26B MoE, MiniMax-M3,
flash-lite), teacher-only filter for the clean training set.

**Students**: volume (flagship, on HF, ~33% fresh 16/48); volcloseout (in-dist
specialist, on HF, honest card); a graveyard of measured negatives (v2-v7b,
star_*, vol* arms); GRPO run-2 checkpoints 50/100/150 (healthy diag,
unevaluated).

**Machinery** (all committed): byte-faithful v1.6 play harness with drift
gates, warm-start, temp/best-of-N; exact final-position adjudicator; exact
winnability solver; SFT trainer (mlx_lm wrapper); ORPO trainer; GRPO trainer
with one-sided trust region + self-tested loss; advantage-corpus extractor
(replays ALL rollouts, 12,516 turns extracted at 0 drift); chained unattended
runners (idempotent, resumable, window-kill-safe); ingest/publish pipelines +
three skills (analyst, ingest, dataset-publish).

**Knowledge base**: 76 dated reports; ~45 memory files with a maintained
index. The negative-results map below is as valuable as the positives. The
full experiment-by-experiment review (eleven eras, learning-point catalog,
costly-mistakes ledger) is the companion doc
`20260703_experiment_retrospective.md`; section 3 here is its executive
summary.

## 3. What measurement has settled (the do-not-revisit fence)

Every line below is a measured closure with a dated report. Revisiting any of
them requires NEW evidence, not new enthusiasm.

| Lever | Verdict | Evidence anchor |
|---|---|---|
| Ship untuned Gemma 4 | NO (doom-loops in play) | 2026-05-31 directive + play evals |
| Corpus filtering (both bases) | FAILS (collapsed fc on play) | 20260530 v4a |
| Single-turn bench as proxy | INVALID (does not predict play) | full-game reframe, 2026-05-29 |
| Static-corpus engineering (reweight / declarative / solver-grounded) | in-dist lift, NO generalization (3-4/12 vs volume 5) | 20260623 form-3 |
| More teacher data (same recipe) | FLAT (volume_v0620 = volume) | 20260621 |
| ORPO move-contrast pairs | HARMFUL (draw starvation, -4.42 paired) | 20260611 rerun |
| SFT-on-self-wins (x4 / x1 / diverse) | at-or-below volume in every config | 20260628 + 20260701 |
| KL-free RFT | COLLAPSES (unbounded negative branch, slow drift) | 20260702 |
| Student stall-field prompting | no effect on entrenched loop (0/40) | stall-field probe |
| Teacher prompt as resign lever | unreliable (1 fold vs 28+ proven-dead no-folds) | v1.5/v1.6 record |

Two structural facts fall out of this map:

1. **The imitation ceiling is reached.** Volume (~33%) is statistically at
   teacher level (~31%). Teacher-shaped data cannot lift the student further:
   adding data is flat, reweighting it does not generalize, and imitating
   self-wins is parity at best. Any lift from here must come from a signal the
   teacher does not provide: the game's own reward.
2. **The marginal training value of new harvest rows is ~zero.** The data
   program's value is now the public dataset, eval/rollout raw material, and
   teacher research; it is not a student-improvement lever. This reprices the
   whole harvest pipeline (section 7).

## 4. The strategic picture

### Why RFT is the only live capability lever

RL against the game engine optimizes the true objective (win the deck), is
not bounded by the teacher's ~31%, and iteration-0 has already:
- proven the reward plumbing (advantage corpus, 0 drift, resign-safe),
- found and fixed the stability failure (trust region: pos-adv logp RISING
  while neg-adv falls boundedly, separation 0.02 -> 0.27 by step 150, vs the
  KL-free run's -24 collapse),
- cost almost nothing (rollouts were free: reused best-of-N pools sampled by
  volume itself).

What it has NOT yet shown is that logp discrimination converts to WINS. That
is the pending ~2h measurement.

### The economics of a compute window

Measured costs (2026-06/07, M-series 16GB, int4 base):

| Unit of work | Cost |
|---|---|
| RFT training step (minibatch 8, ~2k-tok prompts) | 50-100 s |
| 200-step RFT round | 3-5 h |
| Reference-logp pre-pass (1600 turns) | ~27 min |
| One greedy eval game (cap 200) | ~10 min |
| 12-deck paired eval | ~2 h |
| One temp-0.7 rollout sample (cap 250) | ~15-25 min |
| ~30-sample rollout harvest | one window |
| 1000-iter SFT (for reference) | ~3 h |

A window is one overnight (~8-10 h). Practical shapes: train+eval chain (run
2's design), a rollout harvest, or a 24-48 deck bench. One GPU program at a
time (solo operator, one machine).

### Eval governance (hard rules going forward)

- **heldout12** (9000101-24): ITERATION bench only, paired A/B vs the cached
  volume baseline. Signal threshold: |delta| >= 2 wins; +/-1 is noise.
- **fresh-24** (new seeds each time, solver-winnable): PROMOTION bench. Any
  adapter claiming to beat volume must win here; decks are then retired.
- **pooled-48+**: SHIP bench for public claims (the ~33% pin came from one).
- Exact adjudication always (cap-truncated near-wins are wins for analysis;
  strict wins for public claims). Never write results before reading them off
  the run.

## 5. Pillar 1: capability (the RFT program)

### Immediate: resolve the fork (next window, ~2-3 h GPU)

Promote the healthy step-150 checkpoint
(`cp adapters_grpo_iter1_tr/0000150_adapters.safetensors ...adapters.safetensors`)
and re-run `run_grpo_iter1_tr.sh`: it skips training and runs the 12-deck
paired eval + adjudication. Pre-registered gate (unchanged from 20260702):

- **A. grpo_iter1_tr >= volume+2**: the negative gradient converts to wins.
  Proceed to the iterated loop (below).
- **B. within +/-1 (parity)**: treat as flat, close per C, with ONE carve-out:
  if the failure mode is parse/JSON (not play quality), run the constrained-
  decoding re-eval once (that is a deployment fix, not a new lever).
- **C. <= volume-2**: close the capability program at T0. Ship volume as
  final. Pillars 2 and 4 become primary.

### If A: the iterated RFT loop (the T1->T3 path)

Round structure (each round ~3 windows):

1. **Rollout** (1 window): N=5, temp 0.7, ~20 decks, sampled by the CURRENT
   policy (on-policy per round). Deck pool: the winrate24 losses + fresh
   solver-winnable decks; rotate pools between rounds. Add a
   resign-suppressed arm (the 2026-06-29 harvest showed temp-0.7 resigns cap
   recovery; suppression would likely have taken 6/6).
2. **Update** (1 window): extract advantages (existing extractor), 150-200
   steps from the current adapter. Escalation already designed: per-token KL
   to the round's reference policy (cache per-token ref logp of completion
   tokens in the pre-pass, ~MBs; penalize per-token downward drift), replacing
   the mean-logp trust region if the coarser bound leaks under iteration.
3. **Eval + gate** (1 window): paired heldout12; promote to fresh-24 only on
   >=+2. Stop rule (loop-until-dry): two consecutive rounds with paired delta
   < +2 -> bank the best adapter, run the promotion bench, close.

Calendar honesty: at ~3 windows/round and 2-4 windows/week, each round is
about a week; a 3-4 round program is a month. T3 (beat teacher at >=38%
pooled) is the exit target; T2 (fresh-24) gates any HF publication of an RFT
adapter, with a new repo and an honest card (volume stays flagship until then).

### If B/C: closure discipline

Write the consolidated capability-closure report (the RFT arc completes the
story: imitation reaches the teacher, RL-at-2B-scale either lifts it or
measurably does not). No further capability levers exist at current scale; the
fence in section 3 holds. Re-opening requires a genuinely new resource
(bigger student, a stronger teacher policy, or search-augmented play a la
solver-in-the-loop, all out of scope for the current machine).

## 6. Pillar 2: deployment quality (the ship track)

**Constrained JSON decoding** is the standing highest-value item, fork-
independent:
- Problem class it kills: parse-failure deaths (run-1 eval died 12/12 on
  parse; tourA base's best games died on parse; volume carries a documented
  6x JSON-discipline regression; the harness papers over this with retry
  budgets production won't have).
- Build: an mlx_lm logits-processor state machine that constrains generation
  to the fixed 3-key schema (prose values free, structure forced). CPU-side
  build + one GPU validation window (paired heldout12: volume+constrained vs
  volume; expect stall-losses to stay, parse-losses to convert).
- It also raises RFT rollout efficiency (no wasted samples on malformed JSON).

**Packaging**: keep both HF cards current on every result that changes a
claim (the 2026-07-01 refinement is the template); add a "run it locally"
quickstart (mlx_lm + adapter + the sanitize patch); version policy: volume is
flagship until a challenger passes fresh-24, then new repo, never overwrite.

**Later (only after T1+)**: a prompt-native student (retrain on a
student-designed render, decoupling from the harvester's v1.6 prompt; the
drift-gated replay machinery makes corpus re-rendering possible), and an
advisor mode in the web UI as a demo.

## 7. Pillar 3: data and teacher (steady state, budget-protected)

- **Freeze the harvester prompt at v1.6.** The student, the extractor, the
  harness, and 100% of the RFT machinery are byte-coupled to the v1.6 render.
  Prompt churn now costs re-extraction, re-baselining, and distribution drift
  for at-best-marginal teacher gains (the last two teacher gaps, resign and
  loops, each resisted two prompt-lever attempts). If the harvester team ships
  v1.7 anyway: pin ingest attribution, re-extract, re-baseline before any new
  student comparison.
- **Keep full-stream ingest + publish cadence** (the two skills drive it):
  ingest every drop, publish to HF per batch or ~weekly. Value: public
  dataset growth and the versioned record, NOT student lift.
- **Push dead-board budget fixes harness-side.** The exact solver
  (true_world_winnability) already adjudicates kills; propose the harvester
  team auto-terminate on proven-dead boards instead of waiting for prompt-
  induced resigns (1 fold vs 28+ no-folds says the prompt lever is dead).
- **Cohorts stay catalog-only.** The 26B training-mix experiment is
  deprioritized indefinitely: with more-data-flat measured on the teacher's
  own rows, mixing a weaker cohort has no supported upside.

## 8. Pillar 4: evidence and publication

- **The write-up**: "Distilling and RL-ing a local game advisor at 2B scale"
  covering the full measured arc: teacher harvesting and its failure modes,
  the imitation ceiling, the generalization collapses of corpus engineering,
  SFT-on-self-wins parity, the KL-free RFT collapse and the trust-region fix,
  and whichever ending the RFT gate delivers. Both endings are publishable;
  negative results are the differentiator. Target: HF community blog first
  (links the dataset + models), arxiv preprint if the RFT arc lands T3.
- **Timing**: draft after the RFT fork resolves (one window of writing, no
  GPU). The 76 dated reports are the raw material; the strategy of dated,
  pre-registered reports continues as the project's lab notebook.

## 9. Pillar 5: method hygiene and infrastructure

- **Compute governance** (standing, learned 2026-07-02): GPU only inside a
  user-declared window; wrap-up means kill-or-ask for every detached job;
  chains stay resumable so window-end kills are cheap. Prefer overnight
  windows (observed daytime throughput degradation to 0.01 it/s).
- **Statistical honesty**: n=12 paired deltas of +/-1 are noise; promotion
  requires fresh decks; ship claims require pooled 48+. CIs on any public
  number.
- **Reproducibility**: every experiment = committed scaffolding + gitignored
  artifacts + a dated report; drift gates on every replay path; seeds pinned;
  "never write results before measuring" stays a hard rule.
- **WIP limit**: one GPU program at a time; build/CPU work may parallelize.

## 10. Pre-registered decision tree (next quarter)

```
Window N (next): promote s150 -> eval run-2 (~2-3h)
  A. >= volume+2  -> RFT loop round 2 (rollout window, KL update window,
                     eval window). Iterate with loop-until-dry (stop after
                     2 rounds of paired delta < +2).
                       - any round passes fresh-24 -> publish RFT adapter
                         (new repo), volume stays until then
                       - dry -> bank best, run promotion bench, close at
                         whatever tier held
  B. parity (+/-1) -> if failure mode is parse: constrained-decoding
                      re-eval once, then re-gate; else -> C
  C. <= volume-2   -> capability CLOSED at T0/T1-fail
Fork-independent (any order, CPU-heavy):
  - build constrained decoding (+1 GPU validation window)
  - draft the consolidated write-up after the fork resolves
  - steady-state ingest/publish continues throughout
Quarter exit states (both acceptable, both pre-registered):
  S-RL : an RFT adapter at T2/T3, published, write-up covers the lift
  S-Ship: volume final at T0, constrained decoding shipped, write-up covers
          the honest ceiling
```

## 11. Roadmap

| Horizon | Item | Pillar | Cost |
|---|---|---|---|
| Now (next window) | Run-2 eval gate (promote s150) | 1 | ~2-3h GPU |
| Now | Constrained-decoding builder (CPU) | 2 | build only |
| Next (1-2 weeks) | RFT loop rounds OR closure report | 1/4 | ~3 windows/round |
| Next | Constrained-decoding validation | 2 | 1 window |
| Next | Ingest + HF push on harvest batches | 3 | CPU + minutes |
| Later (month) | Promotion bench for any challenger | 1 | 1-2 windows |
| Later | Methods write-up + publication | 4 | writing |
| Later | Ship-bench refresh of the public ~33% claim | 2 | 2 windows |
| Optional | Prompt-native student; web-UI advisor demo | 2 | out of window budget |

## 12. Risks and kill criteria

| Risk | Mitigation / kill rule |
|---|---|
| Heldout12 overfit via iteration | promotion always on fresh decks; retire used promotion decks |
| RFT trust region leaks under iteration | per-token KL escalation designed and memory-budgeted; if round-2 diag collapses even with KL -> close |
| Chasing noise (n=12) | +/-1 = noise by rule; loop-until-dry stop |
| Harvester ships v1.7 | freeze request; if shipped anyway: pin, re-extract, re-baseline before comparisons |
| 16GB ceiling | slice-trick + B=1 + grad-checkpoint hold at 11GB peak; per-token KL adds MBs not GBs; no full second model ever resident |
| Provider instability on harvest | operator-managed; ingest is full-stream so rates stay true |
| Solo-bandwidth burnout | WIP limit 1; both quarter exits are pre-registered as acceptable |
| Scope creep back over the fence | section 3 table is binding; reopening needs new evidence or new resources |

## 13. Operating principles (codified)

1. Pre-register gates before spending GPU; both branches written down.
2. Paired evals against a cached baseline; adjudicate final positions.
3. Never write a result before reading it off the run output.
4. Negative results get first-class reports; the fence is maintenance-free
   only if it is documented.
5. Compute windows are explicit and bounded; every chain resumable.
6. One variable per experiment (run 2 = run 1 + trust region, nothing else).
7. Byte-fidelity to the training distribution (drift gates everywhere).
8. Public artifacts tell the truth (cards updated the day a claim changes).
9. The teacher renders state, never logic; the student's lift must come from
   training signal, not injected heuristics.
10. Ship the measured thing, not the hoped-for thing.
