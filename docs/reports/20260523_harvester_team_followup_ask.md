# Follow-up to harvester team — one ask at a time

**Date:** 2026-05-23
**From:** Chayut / dataset side
**Re:** Single next change request + acknowledgement of Ask 1 shipping
**Companion documents:**
- `docs/reports/20260522_harvester_team_handover.md` (the original three asks)
- `docs/reports/20260522_prompt_template_audit.md` (the audit that motivated them)

---

## Revision — 2026-05-23 (post harvester-team response)

The harvester team answered the four open coordination questions and
flagged a load-bearing misconception in this doc. Reframing the ask
before anything ships. **Full revised ask is in the new section
"Revised ask (post 0.3-baseline correction)" below; the original
sections are kept for the audit trail but should be read with that
revision in mind.**

### Their four answers, in summary

1. **Current temperature is `0.3`, not `~0.7`.** Set explicitly in
   `constants.ts:121` with the comment *"low, for consistent advice"*.
   Our entire framing of the original ask assumed they were at 0.7 —
   so "drop from 0.7 → 0.2" was a much larger delta in our heads than
   it would have been in production. **We were wrong.** Acknowledged.
2. **Per-section decoding (Option B) is not supported by the
   single-call architecture.** Achievable only by splitting into two
   inference calls, which changes the conditioning context the
   `final_decision` JSON sees. They're holding on that. So Option B
   is off the table without a refactor we should not push for in this
   experiment.
3. **No other inference parameters are sent.** Only temperature; no
   `topP`/`topK`/`maxOutputTokens` — all Gemini defaults. Nothing has
   varied across builds today. **The new `inferenceParams` field they
   plan to add will catch any future drift** — that's a useful
   side-benefit of this exchange.
4. **Backfill is not retroactive.** `promptTemplateHash` is stamped at
   `recordAIInteraction()` call time on new records only. Legacy
   exports in `data/raw/` that predate PR #179 stay `null` for the
   new fields. Mirrors the original `appCommit` rollout. **Workable on
   our side** — we'll partition the dataset by "has-hash" vs "legacy"
   and report comparisons within each cohort.

### What this changes for the ask

- The headline misconception (0.3 vs 0.7) **weakens the original
  decode-incoherence diagnosis somewhat**, because at T=0.3 the
  action distribution is already fairly deterministic. If
  oscillations are happening at T=0.3, the wrong move is much more
  likely to be in or very near the argmax of the action distribution
  — which would mean the problem isn't sampling noise but the
  model's underlying preference ordering. That's a stronger claim
  about model capability and a weaker case for temperature as the
  lever.
- Production temperature should **not** drop from 0.3 — they set
  this value deliberately for "consistent advice", and we have no
  data showing 0.3 is wrong in production.
- The right next experiment is a **same-seed cross-temperature side
  run, not a production change**. Specifically: replay a known
  borderline-winnable seed under greedy decoding (T=0.0) once, and
  compare the move sequence and outcome against the T=0.3 run on
  the same seed. This is a tiny experiment — one game, ~10-20 min
  wall-clock — and it cleanly tests whether greedy decoding moves
  the action distribution at all.

---

## Revised ask (post 0.3-baseline correction)

**Run one same-seed game under `temperature = 0.0` (greedy), one
time, as a comparison arm.** Don't change production. Don't change
the prompt. Don't change the build.

Recommended seed: **`2284386365`** (today's `…5ffb25` session). We
have:
- A 88-success-turn trace of T=0.3 play to compare against.
- Solver Monte Carlo on the turn-93 board showing ≥10% of consistent
  worlds are winnable — so a behavioural-difference test is
  defensible.
- Both runs would use the current template
  (`promptTemplateHash` `e2923795…2b91b2`) and the current build
  (`7f01833`).

Alternative if `2284386365` is awkward to schedule: seed
`3263196305` (the corpus's one win on `6dfc8a9`). Greedy on a
known-winning seed tests the reverse hypothesis — does greedy
break a working trajectory? Either seed is informative; pick
whichever fits your harvest cadence.

**What we'll measure on the comparison:**

- Whether the move at each turn matches the T=0.3 trace.
- The first turn at which the two traces diverge.
- Whether greedy reaches a doom-loop sooner, later, or not at all.
- Whether the model's `boardAnalysis` and `reasoning` text changes
  at all (it shouldn't if temperature only affects sampling — that's
  itself a worth-knowing sanity check).

**Cost on your side:** one game, maybe one harvester-config switch,
no prompt edit. **Cost on our side:** ~1 hour of analyst time to
write up the comparison.

If greedy produces measurably different (better or worse) outcomes,
we have evidence the action distribution is partially escaping the
argmax — and a production temperature drop becomes worth considering
with proper A/B. If greedy produces *identical* outcomes, we've
ruled out the decode-noise hypothesis with strong evidence; the
problem is the model's preference ordering, and the next ask shifts
toward prompt or fine-tuning.

---

## Other coordination items from the team's response

- **`inferenceParams` field**: please ship it. Stamp the current
  temperature + any other generation parameter on every successful
  interaction, even when they're constant. We'd rather have a column
  full of `{temperature: 0.3}` than discover later that a parameter
  drifted silently. The Ask 1 pattern (one field, always stamped)
  worked cleanly; the same shape for `inferenceParams` will too.
- **Cadence**: thanks for explicitly supporting one-ask-at-a-time —
  it's how we'll operate from here. Holding Asks 2 and 3 still feels
  right; revisit after the greedy comparison concludes.

---

## What follows is the original ask doc, kept for the audit trail

The headline ask in the original section below ("drop temperature
from ~0.7 to ~0.2") is **superseded** by the revision above. The
rationale (decode-incoherence diagnosis, evidence from `…5ffb25`,
six-session pattern across builds) is still relevant; it's only the
specific intervention that needed re-framing.

---

## TL;DR

1. **Thank you — Ask 1 is live.** `promptTemplateHash` and
   `promptTemplateFinalisedAt` are now stamped on every per-interaction
   record (verified on session `…5ffb25` today). You picked SHA256 over
   MD5 (the better choice) and the ISO datetime format
   (`2026-05-22T00:00:00Z`). This single change unblocks all future
   prompt-vs-build attribution analysis on the dataset side.

2. **One ask this round, not three.** The dataset side has decided that
   bundled requests obscure cause and effect: when several things
   change at once, we can't tell which one moved the needle. From here
   on we will ask for one change at a time and observe the corpus
   response before asking for the next.

3. **The one ask: lower the decoding temperature** (or move the
   `final_decision` JSON to greedy decoding) and let us watch one
   harvest cycle. **Do not upgrade the prompt this round.** The
   evidence — laid out below — is that the model's reasoning is
   already correct on the failing sessions; what's broken is the
   gap between reasoning and action, which is a decode-time
   phenomenon, not a prompt phenomenon.

---

## Status of standing asks

| | Ask | Status | Notes |
|---|---|---|---|
| 1 | Per-interaction `promptTemplateHash` + `promptTemplateFinalisedAt` | ✅ **shipped** | Backfill not yet verified — please confirm whether legacy `data/raw/` rows now carry the hash or if those remain `null`. |
| 2 | Resignation output (move_index = -1 → `outcome: resigned`) | ⏳ standing | Defer until after the decoding-temperature experiment; if temperature alone fixes the doom-loops, the resignation output becomes a smaller win. |
| 3 | Same-seed cross-build experiment on seed `3263196305` | ⏳ standing | Defer for the same reason; if we change decoding parameters first, the comparison frame changes. |

---

## This round's ask: lower the decoding temperature

### The change

Whatever Gemma decoding parameters you use today, **drop the
sampling temperature** for the next harvest cycle. Two options of
increasing aggressiveness:

- **Option A (minimal):** lower the overall temperature from current
  (we believe ~0.7) to ~0.2. Affects both the reasoning text and
  the final-decision JSON.
- **Option B (preferred):** keep the reasoning/analysis text at the
  current temperature; force the `final_decision` JSON portion to
  greedy decoding (temperature 0, or equivalent). This is more
  surgical — it preserves the model's natural-language exploration
  in `board_analysis` and `strategic_plan` while forcing the action
  to be the argmax over the model's own action distribution
  conditioned on what it just wrote.

Option B is harder to implement (requires segmented sampling per
output section) but is the cleanest test of the diagnosis. If your
inference path doesn't support per-section decoding parameters,
Option A is fine — both move in the same direction.

**Do not change the prompt template, the build, the seed pool, or
the harvester config in the same cycle.** The whole point is that
this is observable in isolation.

### Why this is the right next ask, not a prompt rewrite

The corpus evidence is now strong enough to support a confident
diagnosis. Session `…5ffb25` from today is the cleanest case:

- Seed `2284386365`, build `7f01833`, current template
  (`promptTemplateHash` `e2923795…2b91b2`,
  `promptTemplateFinalisedAt` `2026-05-22T00:00:00Z`).
- **Phase 1 (turns 0-92):** model brought foundations from 0 → 13
  and revealed 8 of 21 face-down cards. Above-average opening play,
  comparable to the corpus win `…0154e1`.
- **Phase 3 (turns 106-119):** 14 consecutive turns of zero
  productive moves; 7 of them are the same `TD col 2 ↔ col 4`
  oscillation.
- **Solver check on the turn-93 board (Monte Carlo over 10
  consistent worlds, beam=2000, timeout 30s each):** 1 sample
  solved with a 63-move winning continuation. **Lower bound 10%
  win rate on the position the model abandoned;** beam search's
  one-sidedness means the true rate is likely higher.
- **The model's own final-turn `boardAnalysis` literally states the
  correct plan:** *"Column 4 can be emptied in two moves: first,
  by moving the red Ten of Diamonds (TD) to the black Jack of
  Clubs (JC) in Column 2, and second, by moving the black Jack
  of Spades (JS) to..."* — and then the final decision is the TD
  oscillation that defeats that plan.

That last point is the load-bearing one for this ask. **The model's
text output already contains the right reasoning.** The action is
inconsistent with it. That's a sampling-noise problem at decode time,
not a prompting problem. No prompt edit will fix the fact that the
chosen-move token was sampled from a distribution that disagrees
with the reasoning the model just emitted.

The audit document spelled this out as the predicted ceiling. We now
have a concrete instance with solver-confirmed winnability, and the
pattern repeats across `645d03` (5C/4D loop), `73fd85` (TS/9D loop),
`1f2fd2` (8C/7D loop), `391920` (3C/4H loop), `3d03e5` (JD chain
reversal), `c7fdb9` (7H col 2/6 loop). Six distinct sessions across
five distinct builds, two templates, one consistent pattern.

### What we should see

If decode-incoherence is the bottleneck:

- **Doom-loop sessions stop reaching 50+ turns of pure
  oscillation.** Either they don't enter the loop, or the loop
  ends in 2-3 turns when the same conditional sample picks the
  correct argmax.
- **Win rate moves measurably.** Even doubling from the current
  corpus rate (one verifiable win on the new template across ~15
  attempts ≈ 7%) to 14% would be a strong positive signal.
- **Mean confidence shifts up** (greedy sampling biases toward
  higher-probability moves, which the model rates more confidently
  by design).
- **No change in `boardAnalysis` quality** — the reasoning section
  was never the problem.

If decode-incoherence is NOT the bottleneck:

- Win rate is unchanged or worse.
- Oscillations persist with similar frequency.
- We learn something important: the model's chosen-move distribution
  is already heavily peaked, and the wrong moves were the argmax all
  along. That redirects us toward prompt or fine-tuning interventions.

Either result is informative, which is why this is a good first
experiment.

### The experiment

Minimum-viable observation cycle:

- Run **one normal harvest cycle** (whatever your current weekly
  cadence is) under the new decoding parameters.
- Same model, same prompt template, same harvester config
  otherwise.
- The dataset side will produce a comparison report against the
  prior week's harvest (same template + build distribution) on:
  - Win rate
  - Mean plateau length
  - Per-session oscillation count
  - Confidence distribution
- Decision in two weeks: keep, revert, or escalate.

### Open coordination questions

1. **What temperature do you currently use?** We've been assuming
   ~0.7 based on Gemma defaults but haven't confirmed.
2. **Does your inference path support per-section decoding** (Option
   B above), or are we constrained to Option A?
3. **Is there any other inference parameter changing across builds**
   (top_p, top_k, repetition penalty, max_tokens) that we should
   know about? Stamp it on the per-interaction record too if it
   moves between builds.
4. **Backfill confirmation for Ask 1:** do legacy exports in
   `data/raw/` (those without `appCommit`) now carry
   `promptTemplateHash`, or are they left as `null`? Either is
   workable, just want to know.

---

## Cadence going forward

This document is the new pattern: **one ask, one observation
cycle, one decision.** It replaces the bundled-asks style of the
previous handover. The standing asks (resignation output, same-seed
cross-build experiment) remain on the queue and will return as
single-ask documents when the current experiment concludes.

The dataset side is happy to absorb the extra coordination overhead
of single-asks because the alternative — bundled changes that we
can't disentangle — produced six straight builds with no
attributable lift. Cleaner cause-and-effect signal is worth the
slower pace.

---

## Where to find the evidence

- `data/DATASET_NOTES.md` — full session catalogue, including
  `…5ffb25` (entry under "Known doom-loop sessions").
- `docs/reports/20260522_prompt_template_audit.md` — corpus-wide
  template + capability analysis; the "decode-incoherence ceiling"
  framing originates here.
- `docs/reports/20260523_training_data_options.md` — the
  parallel-track exploration of training-data sources, which also
  hinges on whether teacher-side capability can be improved.
- `data/store/interactions.jsonl` — every successful interaction
  now carries `promptTemplateHash` + `promptTemplateFinalisedAt`,
  so the comparison report will key off those fields directly.
