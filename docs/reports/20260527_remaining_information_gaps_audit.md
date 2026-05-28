# Remaining Information Gaps Audit (Post v1.2)

**Date**: 2026-05-27
**Status**: Forward-looking inventory. v1.2 (DRAW TIMELINE + CYCLE counter) is currently being implemented field-side by the harvester team; this doc is a catalog of remaining gaps we have NOT yet asked for. Decisions on cycle ordering are deferred.
**Design principle applied throughout**: `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/prompt-closes-info-gap-not-logic.md`. Each candidate change is tested against "could a human player at the table observe or remember this?" Information gaps are ship-eligible; logic / heuristics / decision rules are not.
**Companion documents**:
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_harvester_team_v1_2_draw_timeline_ask.md` (the currently-in-flight ask)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_candidate_spec.md` (technical spec for v1.2)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` (separate audit on PRIOR REASONING amplifier and other late-game pathologies)

## 0. Purpose

Catalog every information item that a human Klondike player has access to (via senses + memory + permitted memory aids) that the LLM does NOT currently get from the harvester prompt. Decide nothing here. Surface the inventory so we can pick cycle priorities later with the full landscape in view.

This complements the late-game prompt audit (which inventoried prompt BLOAT) by inventorying prompt GAPS.

## 1. Audit methodology

For each candidate gap, evaluate:

| Field | What it means |
|---|---|
| **Symmetry** | Can a human player observe / remember / derive this at the table with standard Klondike rules and permitted memory aids? PASS = ship-eligible. FAIL = drop. |
| **Type** | Information (state, observation, memory) or Logic (heuristic, decision rule). Only Information passes the principle. |
| **Cost** | Approx prompt-chars to render. Low / Medium / High. |
| **Value** | Subjective assessment of how much this gap is hurting the model. Low / Medium / High. Backed by evidence where available. |
| **Status** | Open (not yet asked), in-flight (asked but not shipped), shipped, declined. |

## 2. What's already symmetric (so we don't re-litigate)

For context, the v1.1 + (in-flight v1.2) prompt already exposes these:

| Information item | How it's rendered | Adequate? |
|---|---|---|
| Current foundations | FOUNDATIONS line | Yes |
| Current tableau face-up cards | TABLEAU block with `??` for face-down | Yes |
| Current stock count | STOCK line | Yes |
| Current waste top | STOCK line `WASTE top:` | Yes |
| Whether recycle is available | STOCK line `recycle stock: yes/no` | Yes |
| Cards drawn this cycle (post-v1.2) | DRAW TIMELINE block | Once v1.2 ships, yes |
| Cycle number (post-v1.2) | STOCK line `CYCLE:` field | Once v1.2 ships, yes |
| Last 10 moves | RECENT MOVES block | PARTIAL, see gap G1 |
| Game rules | Static RULES section | Yes |
| Legal moves at current state | LEGAL MOVES block | Yes |
| Foundation progress percentage | PROGRESS line | Yes (derivable from foundations but cheap to surface) |

## 3. Remaining gaps (in order of estimated impact)

### G1: Extended move memory (currently capped at last 10)

**Symmetry**: PASS. A human player has perfect recall of every move they made, not just the last 10.

**Type**: Information.

**Current behavior**: `RECENT MOVES` shows last 10. A 200-turn game has 190 moves the model can't see.

**Why this matters**:
- Long-cycle oscillation detection. The 3-card oscillation in session `adf71b` (seed `114946100`) played out across ~30 moves. The model could not see the pattern because the window only held 10.
- "Have I tried this before?" reasoning. The model rationalizes moves as novel attempts when it has actually tried the same sequence multiple times.

**Caveat: naive "uncap" introduces noise**. Just dumping all 200 moves verbatim has real downsides:
- Prompt bloat. Even compressed, ~7 KB of history per prompt. The actually-current state has to compete with hundreds of historical lines for attention.
- Recency dilution. The legal moves and current board state may get drowned out by sheer volume.
- Doom-loop amplification. During a stuck state, most of the history is the same 2-move swap repeated 50 times. That is the pathology family of A1 (PRIOR REASONING amplifier): rendering more text of past behavior teaches the model what to imitate, not just what to remember.

So G1 is not "uncap RECENT MOVES". It is a design question with at least four sub-options to consider before we ask for anything:

**G1a: Modest cap bump (10 to 30 or 50)**.
- Smallest change. Catches medium-cycle patterns (the `adf71b` 30-move oscillation) without exploding prompt size.
- Easy to test incrementally. Bump to 30 first; if that helps, consider 50.
- Cost: low (3x to 5x the current size).

**G1b: Compressed full history**.
- All moves, one short line each: `t156: 5S col6->col3`.
- Drop labels, drop flip annotations, drop reasoning.
- Cost: medium-high (~7 KB worst case).
- Risk: still has the recency-dilution and amplification concerns above.

**G1c: Run-length compressed history**.
- Same as G1b but collapse consecutive identical signatures: `t140-t160: QS col5<->col7 x12 times`.
- A human player's memory naturally consolidates repetition; this mimics that.
- Information-preserving (no derivation, just rendering), so still PASS on the principle.
- Cost: medium (compressed during doom-loops, comparable to G1b otherwise).
- Probably the best of the four because it specifically reduces the noise that worries us most.

**G1d: Two-window approach**.
- Detailed last-K (e.g. last 10, like today) + heavily compressed older history (e.g. a one-line summary "earlier this game: 8 foundation pushes, 18 stock recycles, primary tableau activity in cols 5 and 7").
- Highest implementation complexity.
- Risk: the "summary" component verges on derived view; not clear it survives the principle audit.

**Cost / value resolution**:
- G1a (cap bump): low cost, modest value. Easy starting point.
- G1c (RLE): medium cost, likely best value. Targets the specific noise mode we worry about.
- G1b (full compressed): probably overkill once G1c exists.
- G1d (two-window): defer pending discussion.

**Recommended approach if/when we revisit**: try G1a (bump to 30) first as a cheap experiment. If it helps, consider G1c. Skip G1b. G1d requires more thought.

**Value**: still HIGH as a gap, but the right RENDER strategy is not obvious. Picking the wrong one (G1b naive uncap) could regress us.

**Status**: Open, with sub-design unresolved.

### G2: Absolute turn / move count

**Symmetry**: PASS. A human counts turns or watches the clock.

**Type**: Information.

**Current behavior**: PROGRESS line shows completion percent (e.g. "completion=12%"). The absolute turn count is nowhere in the prompt.

**Why this matters**:
- Pace awareness. A model can reason about "we're at turn 250, this game has gone on a long time" only if it knows the turn number.
- Pairs naturally with G1. If we ever ship full move history, the move-history needs turn-stamps to be useful, and at that point the absolute counter is implicit.
- The resign output (already shipped in v1.1) becomes more useful with explicit turn count; the model can reason "we've been on this board for 100 turns".

**Cost**: Trivial (one integer in an existing line).

**Value**: Medium standalone, HIGH when paired with G1.

**Status**: Open.

### G3: Foundation push sequence

**Symmetry**: PASS. Each foundation push is a visible event the human watched.

**Type**: Information.

**Current behavior**: FOUNDATIONS line shows current tops only. The chronological order of pushes is not surfaced.

**Why this matters**:
- Chain planning. The WON-game reasoning we documented in `010e01` shows the model spelling out chain reactions: "draw 9C, then TC, then JC...". A foundation push history would help this style of reasoning by showing what chains have already been executed.
- Recently-played-foundation cards are sometimes needed back in the tableau (the strategy guidance even warns about pushing too eagerly). Knowing which foundation cards just went up would inform that judgment.

**Cost**: Low (one line, e.g. "FOUNDATION PUSH ORDER: AS 2S AC 2C 3S 4S AH 2H 3H").

**Value**: Medium. The WON case shows the value of chain reasoning; the gap likely matters more in late game.

**Status**: Open.

### G4: Reveal history (when each face-up tableau card was uncovered)

**Symmetry**: PASS. Each flip is a visible event.

**Type**: Information.

**Current behavior**: When the model sees `col5: ?? ?? 9H 8S 7H`, it cannot distinguish "9H was dealt face-up" from "9H was revealed at turn 47 when I moved TS off it".

**Why this matters**:
- Mostly historical context for understanding HOW the board got to its current state.
- A human player implicitly remembers this and can reason about "the 9H has been waiting here since turn 47, I should use it".

**Cost**: Low. Could annotate inline (e.g. `9H(t47) 8S 7H`) or add a brief REVEAL LOG section.

**Value**: Low to medium. Niche but cheap.

**Status**: Open.

### G5: Per-card touch / move counts

**Symmetry**: PASS. A human remembers "I have moved 8D between col4 and col6 three times now".

**Type**: Information (borderline, see note below).

**Current behavior**: Only inferable from RECENT MOVES (last 10), and only if the moves fall in the window.

**Why this matters**:
- Oscillation self-awareness. The current 9H 6-card oscillation in `db1804` involved the same 6-card unit moving 7 times in 20 turns. A per-card counter would surface this.

**Cost**: Medium (a per-card table of ~10 lines, each showing card + move count).

**Value**: Medium for oscillation awareness, but mostly subsumed by G1 (full move history) if that lands first.

**Borderline concern**: a "touch count" verges on metric-as-instruction. The information ("how often was each card moved") is observation; whether the model uses it to detect oscillation is reasoning. Stays on the right side of the principle as long as we don't add commentary like "if touch count > 3, consider a different move".

**Status**: Open. Lower priority than G1 because G1 subsumes most of the value.

## 4. Asymmetries in the WRONG direction (model sees what human doesn't)

These are NEGATIVE gaps: information the model has that a human does NOT. Per the principle, these should be removed or reduced.

### A1: PRIOR REASONING (the model's own past internal thoughts rendered as text)

**Symmetry**: FAIL. A human player does not see their own past inner monologue rendered as text in front of them. They have a fuzzy memory of past reasoning, not a verbatim 524-character paragraph.

**Type**: Information that the model has and the human does not.

**Current behavior**: Up to 5 prior `strategic_plan` texts rendered each turn.

**Why this matters (and why it's documented)**:
- Late-game audit at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` found that 5 byte-identical copies of the same `why` text appear in PRIOR REASONING during doom-loop. The model emits same plan, sees it 5 times, becomes more confident in the same plan.
- Documented amplifier of doom-loop pathology. The audit pre-registered an H1 hypothesis test for dedupe.

**Fix candidates**:
- Dedupe by `why` text (only show distinct entries) - audit H1
- Cap at N=2 or N=3 (smaller window, less amplification) - audit H1 variant
- Remove entirely (most aggressive; let the model rely on RECENT MOVES + the new G1 if shipped)

**Status**: Open, separately tracked in the audit doc. Should be tested in isolation because the response distribution may change.

### A2: Move-type labels in LEGAL MOVES (snake_case identifiers)

**Symmetry**: Loose FAIL. A human doesn't think about moves as `tableau_to_foundation` strings.

**Type**: Information presentation; minor noise.

**Current behavior**: Each legal move line has format `[N] tableau_to_tableau   Move QS from column 7 to column 5`.

**Why this matters**:
- ~26 chars/move of redundancy with the prose description.
- Two parallel labels for the same move may amplify `MOVE_INDEX_FIXATION` failure class.
- Anti-pattern flagged as M4 in the late-game audit.

**Status**: Open, low severity. Could be a no-cost cleanup any time.

## 5. Suggested cadence (NOT a commitment)

Pre-registered guess at ordering for future cycles, assuming v1.2 ships and tests cleanly. We will decide based on field results, not this table.

| Cycle | Candidate change | Type | Rationale |
|---|---|---|---|
| v1.2 (in flight) | DRAW TIMELINE + CYCLE counter | Info | 97.9% prevalence evidence, biggest known gap |
| v1.3 candidate | G1a (bump RECENT MOVES cap from 10 to 30) + G2 (turn counter) | Info | G2 is trivially cheap; G1a is the lowest-risk way to test whether more memory helps. Defer G1c/G1b/G1d until we see whether G1a moves the metric. |
| v1.4 candidate | A1 PRIOR REASONING dedupe (audit H1) | Anti-asymmetry | Documented amplifier; needs isolated test |
| v1.5 candidate | G3 (foundation push sequence) | Info | Smaller value but cheap and clean |
| Later | G4, G5 | Info | Niche; revisit if specific failure modes surface |
| Later | A2 move-type label removal | Noise reduction | Cleanup |

Conservative spacing keeps each cycle to one variable change, so we can attribute outcome differences cleanly.

## 6. Explicit non-asks (things that look like gaps but aren't)

These would FAIL the symmetry test and should not be added:

- **Face-down tableau identities revealed by elimination** (a human can only DERIVE this; rendering it pre-derived is x-ray vision)
- **Pyksolve verdict on whether the current position is winnable** (a human at the table has no oracle)
- **Optimal next move recommendation** (this is the model's job)
- **A `should_resign` signal** (decision rule, not observation)
- **Annotation of which legal moves are "good" or "bad"** (logic, not information)
- **A summary of strategic principles applied so far** (synthesis, not observation)

And these would be questionable per the principle:

- **Stock content sorted by suit / rank for easier scanning** (the human sees stock in DRAW order, not sorted; sorted is a derived view that biases the model)
- **A "moves until empty stock" countdown** (derivable from stock count; surfacing it pre-derived is mild logic)
- **Best matching tableau-to-tableau opportunities ranked by `reveals_hidden`** (ranking is decision-making)

## 7. Decision deferred

This is a CATALOG, not a request. Decisions on which gap to address next (and in what order) are deferred until:

1. v1.2 lands in the harvester team's build
2. We observe 5+ traces on the new template
3. We assess whether the DRAW TIMELINE alone moves the 97.9% discovery-rationale metric

After those three conditions, we revisit this doc and pick the next priority. Re-evaluating based on what v1.2 did or did not fix is more informative than committing to v1.3 now.

## 8. Update protocol

When a gap moves status (e.g. open -> in-flight -> shipped, or declined), append an entry to a "Status changes" section at the bottom of this doc with a date. Do not edit the gap rows themselves in-place; preserve the original assessment as written.

When a new gap is discovered (e.g. through analysing future failures), add it to section 3 with the next available G-number. New asymmetries-in-wrong-direction get new A-numbers.
