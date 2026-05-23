# Prompt format A/B — work-in-progress status

**Date:** 2026-05-23
**Status:** paused — Gemini API key per-day quota exhausted
**Pick up by:** waiting for next-day quota reset, or wiring up a paid Vertex / production-key path

## The question

Does replacing the harvester's `CURRENT GAME (JSON)` block with a hybrid
ASCII-board format change Gemma 4 31B's move picks — specifically, does it
reduce the **33% oscillation-reversal take-rate** documented in
`20260523_training_data_options.md`?

The hypothesis (from same doc): the model fails on multi-turn-state-waste
patterns because `recentMoves` is buried inside a deeply-nested JSON object
competing with 60+ other fields. Reformatting for spatial / sequential
salience should lift the multi-turn signal without changing what the model
can do.

## What was built

Two scripts, both runnable from the venv (`.venv/bin/python`):

### 1. `/Users/chayut/repos/solitaire-analytics/scripts/compare_prompt_formats.py`

Pure format converter + side-by-side renderer. No API calls. For each
sampled corpus snapshot:

- Renders `original.txt` (the JSON block as the harvester ships)
- Renders `hybrid.txt` (ASCII board + numbered RECENT MOVES + numbered
  LEGAL MOVES — see the `render_hybrid` function for the exact layout)
- Writes a `summary.md` with character/token deltas

Filters:
- `--filter any` — uniform random sample
- `--filter oscillation` — turns where the most-recent tableau move's
  reverse is in `legalMoves` (the trap we care about)
- `--filter midgame` — 10-18 face-down cards remaining

Pilot output at
`/Users/chayut/repos/solitaire-analytics/data/dataset/demos/prompt_format_compare/`
— 6 oscillation samples. Hybrid is consistently **0.18×–0.24×** the
character count of the JSON form.

### 2. `/Users/chayut/repos/solitaire-analytics/scripts/ab_test_prompt_formats.py`

The actual A/B harness. For each sample, builds two prompts sharing the
same preamble (rules) but differing in the CURRENT GAME block. Sends both
to the Gemini API and compares picks. Requires `GEMINI_API_KEY` env var.

Plumbing currently handles:
- HTTP 500 / 503 retries (Gemma 4 31B is unstable on long prompts — same
  pattern the harvester sees)
- HTTP 429 with smart sleep parsed from the `retry in Ns` message
- Thought-vs-answer parts (Gemma 4 emits `"thought": true` parts and
  separate answer parts; `extract_text` skips thoughts)
- Both response shapes: `{move_index: N}` (simplified ask) and
  `{final_decision: {move_index: N}}` (harvester ask)
- Move classification into tiers: foundation / reveal / waste_play /
  shuffle / draw / recycle / illegal — for "is B's pick strictly better
  than A's pick" aggregation

### Default model — strict
`--model gemma-4-31b-it` is the default and must stay that way. Stronger
Gemini models (2.5-flash etc) don't reproduce the Gemma-specific failure
modes — running on them gives a confidently wrong answer about whether the
hybrid prompt helps the harvester. See
`feedback_harvester_fidelity_model.md` in the memory directory.

## What's currently valid

- The **format renderer is solid** — visually verified on
  `sample-00/hybrid.txt`: ASCII board reads at a glance, oscillation trap
  (legal move [2] undoing the recent moves 7-9) is obvious in hybrid form
  but structurally invisible in JSON form.
- The **A/B harness end-to-end works** against the Gemma 4 31B endpoint —
  verified via trivial-prompt curl. Plumbing is ready.

## Independent LLM read (subagent, 3 samples)

A general-purpose subagent — no conversation context, told to ignore the
"original" / "hybrid" filenames and judge purely on content — read three
sample pairs (00, 02, 04) and concluded **hybrid wins on decision-readiness
and trap-avoidance**. Two findings I had not raised:

### Finding A — `reasoningTrail` is actively harmful (not just clutter)

The harvester's `CURRENT GAME (JSON)` block ends with a `reasoningTrail`
array — the model's prior turns' written justifications. The subagent
flagged this as the **biggest single liability of the JSON format**, not
the JSON syntax itself.

In sample 02 the trail literally rehearses the move that started the
oscillation ("Move 5 is the essential first step") immediately before
asking the model to pick again. In sample 04 it justifies the just-made
move that the legal-moves list now offers to undo. The trail acts as an
anchor that biases the model toward continuing whatever sequence it was
already in — exactly the failure mode we're trying to fix.

The hybrid format drops `reasoningTrail` entirely, which the subagent
called out as a positive. **This is probably the single most impactful
change in the hybrid format, more than the ASCII board itself.**

### Finding B — `seenDrawPileCards` is real signal hybrid was dropping

The subagent flagged a legit regression in the first hybrid draft: I had
dropped `seenDrawPileCards` (the list of cards the model has already seen
flipped from stock to waste). For late-game / stock-recycling decisions
this is genuine signal — knowing "AC has not yet been seen" matters for
deadlock reasoning.

**Fixed in this commit**: the renderer at
`/Users/chayut/repos/solitaire-analytics/scripts/compare_prompt_formats.py`
now emits a single-line `SEEN IN WASTE THIS CYCLE: 9H 5S KS …` block.
Re-rendering bumped hybrid from 0.18-0.24× to 0.19-0.24× of original —
negligible cost, restored signal.

## Cross-model pick test (Haiku + Sonnet as Gemma proxies)

Since the Gemma 4 31B API quota was exhausted, ran a proxy A/B with two
subagents: Claude Haiku (weaker, closer to Gemma's capability) and Claude
Sonnet (stronger). Each was given the same 3 oscillation samples × 2
formats = 6 independent pick tasks, presented in shuffled order with no
A/B framing.

| sample | trap | Haiku JSON | Haiku Hybrid | Sonnet JSON | Sonnet Hybrid |
|---|---|---|---|---|---|
| 00 | `[2]` undoes moves 7-9 | `[1]` draw ✅ | **`[2]` reversal ❌** | `[1]` draw ✅ | `[1]` draw ✅ |
| 02 | `[5]` continues JS/TH loop | **`[5]` continue loop ❌** | `[1]` TC→col 5 ✅ **breaks loop** | `[1]` TC→col 5 ✅ | `[1]` TC→col 5 ✅ |
| 04 | `[7]` undoes recent 7H | **`[7]` reversal ❌** | **`[7]` reversal ❌** | `[0]` draw ✅ | `[0]` draw ✅ |

### What this told us

**Sonnet is too strong** — 0/6 traps taken, identical picks across formats.
Not a useful Gemma proxy. Same lesson as the earlier gemini-2.5-flash run:
stronger models don't reproduce Gemma's failure mode, so don't substitute
them for harvester-fidelity questions.

**Haiku is a partial proxy** — took 2/3 traps on JSON (samples 02 and 04).
But on hybrid: lost the sample-02 trap (good — broke the loop), gained the
sample-00 trap (bad — picked a useless shuffle without the trail telling
it to draw), tied on sample-04 (both formats failed the single-move
reversal equally). **Net for Haiku at N=3: format didn't clearly win**,
even though the qualitative read favored hybrid.

### What this changes about the recommendation

The Haiku result was more honest than the qualitative read alone
suggested. Specifically:

1. **The sample-02 win is real and matches the subagent's read** — when
   `reasoningTrail` was rehearsing the loop, hybrid (which drops it) freed
   Haiku to pick the productive break-out move.

2. **But sample-00 reveals a downside of dropping `reasoningTrail`** —
   there the trail was *correct* (it told Haiku to draw, which was right).
   Stripping it left Haiku to read the board and incorrectly conclude it
   could "expose QS in col 3" (QS is already face-up; the pick was useless).

3. **`reasoningTrail` is high-variance, not purely harmful** — helpful when
   correct, harmful when wrong. On oscillation specifically it tends to be
   wrong (because the model was already in a loop when it wrote the trail).
   On other turns it can be the right anchor.

4. **Sample-04 — both formats failed equally** on a single-move reversal
   trap. Suggests neither format reliably catches short-pattern reversals;
   that's a separate problem from the multi-turn loop case.

## Refined recommendation for the harvester ask

The single-ask-at-a-time cadence means we need to pick the highest-leverage
change. After the Haiku data, the framing softens:

1. **DEFINITELY ship — switch the CURRENT GAME block to a hybrid ASCII
   layout** (board as per-column text lines, recent moves and legal moves
   as numbered lists, `seenDrawPileCards` preserved as a single line).
   - Token cost: 4-5× reduction (objective)
   - Board readability: uncontroversial (both Haiku and Sonnet handled it)
   - One clear positive datapoint: Haiku broke a sustained oscillation
     loop under hybrid that it perpetuated under JSON

2. **NOT yet — don't blanket-drop `reasoningTrail`.** The Haiku data shows
   it cuts both ways. Better framing if we want to address its anchoring
   effect: **re-label and reposition** — rename to something like
   `PRIOR REASONING (may be obsolete; verify against current state)` and
   move it BELOW the legal moves so it doesn't anchor the decision
   pre-pick. Preserves the helpful case (sample 00) while reducing the
   anchoring effect (sample 02).

3. **Honest framing for the harvester team**: we have a qualitative case
   for the layout change plus one clear Haiku positive datapoint on a
   sustained-loop sample. We don't have Gemma A/B numbers yet. Layout
   change is low-risk because it doesn't drop any information (with the
   `seenDrawPileCards` fix) and improves decision-readiness across model
   sizes. Treat `reasoningTrail` separately once we have more data.

If we have to send only ONE thing as the next single-ask: send the layout
change. It's the lower-risk, higher-confidence improvement. The
`reasoningTrail` question needs the Gemma A/B that's blocked today.

## What's not yet valid

- **No Gemma 4 31B A/B numbers.** The actual measurement run was blocked by
  the free-tier per-day quota on the temporary key. We have no usable A/B
  data from the right model.

- **Discard any gemini-2.5-flash A/B output** that may sit in
  `/Users/chayut/repos/solitaire-analytics/data/dataset/demos/ab_prompt_format/`.
  It was a plumbing-validation run, not a measurement. The model picked 0/8
  reversals (vs Gemma's measured 33%) — too strong to reproduce the
  failure mode we're trying to measure.

## Next-session checklist

1. **Get a Gemma 4 31B-capable key with enough quota.** Either a paid AI
   Studio key or the harvester team's production Vertex AI credentials.
   Per-day free quota was the blocker, not per-minute.

2. **Run three A/B conditions instead of two**, to isolate which change
   matters most. Right now `ab_test_prompt_formats.py` only does
   {original JSON} vs {full hybrid}. Add a third arm: {original JSON
   minus reasoningTrail}. The three picks per sample tell us:
   - JSON-with-trail → JSON-without-trail: does dropping the trail alone
     reduce reversal take-rate?
   - JSON-without-trail → hybrid: does the ASCII layout add anything on
     top of the trail-drop?
   This is the right experiment if the subagent finding (Finding A above)
   holds — we want to know if the prompt-shape change is mostly
   "remove the harmful trail" or "rewrite the board layout."

3. **Run at N≥30, oscillation filter, on Gemma 4 31B:**
   ```bash
   GEMINI_API_KEY=... .venv/bin/python scripts/ab_test_prompt_formats.py \
       --n 30 --filter oscillation --model gemma-4-31b-it
   ```
   Expected duration: ~10-15 minutes including retry friction.

4. **Read the summary table.** Specifically:
   - `reversal picks` across the three arms — biggest drop is the
     load-bearing change.
   - `tier distribution` — does any arm pull picks up the ladder (more
     foundation/reveal/waste_play, fewer draw/shuffle)?
   - `B vs A: better=X worse=Y tied=Z` — net move-quality direction.

5. **Write the next single-ask** based on which arm wins:
   - If trail-drop alone explains most of the drop → ask the harvester team
     to remove `reasoningTrail` from the prompt. Surgical, easy to ship.
   - If layout adds significant lift on top → bundle both into a single ask
     "drop reasoningTrail AND switch the board to ASCII layout."
   - If neither moves the needle → format wasn't the bottleneck. Look at
     what actually differs in picks and pivot.

6. **Also run on the `any` filter** (not just oscillation) to see whether
   the format change has general lift or only fixes the specific trap.

## Files

- `/Users/chayut/repos/solitaire-analytics/scripts/compare_prompt_formats.py`
- `/Users/chayut/repos/solitaire-analytics/scripts/ab_test_prompt_formats.py`
- `/Users/chayut/repos/solitaire-analytics/data/dataset/demos/prompt_format_compare/` — six side-by-side sample pairs from the format renderer (qualitative inspection, and the same samples the Haiku/Sonnet test was run on)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260523_training_data_options.md` — the pattern-specific decode-incoherence finding that motivates this work (K-shuffle 0% take-rate vs oscillation 33% take-rate)

## Related memory

- `feedback_harvester_fidelity_model.md` — strict Gemma 4 31B rule for
  harvester-fidelity experiments
- `gemma4-e2b-distillation-project.md` — overall distillation project state
