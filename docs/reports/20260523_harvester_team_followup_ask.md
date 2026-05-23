# Follow-up to harvester team — one ask at a time

**Date:** 2026-05-23
**From:** Chayut / dataset side
**Re:** Single next change request + acknowledgement of Ask 1 shipping
**Companion documents:**
- `docs/reports/20260522_harvester_team_handover.md` (the original three asks)
- `docs/reports/20260522_prompt_template_audit.md` (the audit that motivated them)

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
