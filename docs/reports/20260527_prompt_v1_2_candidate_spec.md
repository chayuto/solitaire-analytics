# Prompt v1.2 Candidate Spec + Local Test Plan (post-audit)

**Date**: 2026-05-27
**Status**: CANDIDATE SPEC. Implementation not started. Local A/B test pending the next compute window.
**Supersedes**: the draft plan at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_draw_timeline_local_test_plan.md` (kept for the audit trail; this doc is what the implementation should follow).
**Audited against**: `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/SKILL.md` checklist (2026-era best practices, model-calibrated to Gemma 4 IT internal reasoning).

## 0. Why this exists

`c99da9` (the third attempt on benchmark seed 2967897202) exposed two harvester-prompt design gaps that prevent the model from reasoning about partial observability over the stock pile:

1. During the first stock cycle, the prompt provides no list of cards already drawn. `RECENT MOVES (last 10)` is too short to cover a 24-card cycle.
2. After recycle, a `SEEN IN WASTE THIS CYCLE` list appears, but its label is misleading: it actually renders "cards currently in stock that have been observed before", and its contents SHRINK as the model draws.

Net effect: the model in c99da9 spent 30+ turns drawing for AD even though AD was provably not in the stock (the SEEN list at ti=44 listed every other card but AD). The information needed to make the correct deduction was present in the prompt; the label and formatting made it unreachable.

This document is the candidate spec for prompt v1.2, which addresses these two gaps with the minimum surface-area change. The candidate has been audited; all HIGH and MEDIUM severity audit findings are folded in.

## 1. Hard constraint: symmetry to human play

Every information item the prompt provides must be obtainable by a human playing the same deck under standard Klondike rules. Memory aids (e.g. writing down draws) are permitted; card-x-ray vision is not.

The v1.2 design respects this:

| Item | Human can observe? | v1.2 renders? | Symmetric? |
|---|---|---|---|
| Current waste top | Yes (face-up on waste pile) | Yes (`{NOW}`) | YES |
| Cards already drawn this cycle | Yes (visible in waste pile beneath top) | Yes (right of `{NOW}`) | YES |
| Identity of stock cards in cycle 1 | NO (face-down, not yet drawn) | `???` placeholder | YES |
| Identity of stock cards in cycle 2+ | Yes, IF perfect recall of cycle 1 | Yes (full identities) | YES (perfect-recall human) |
| Number of cards in stock | Yes (count is visible) | Yes (STOCK line) | YES |
| Identity of face-down tableau cards | NO | `??` placeholders | YES |
| Current cycle number | Yes (human knows how many times they recycled) | Yes (cycle counter) | YES |

**Caveat to document**: the cycle 2+ render gives the model perfect-recall over cycle 1's draw order, whereas a typical human has imperfect memory. This is still within game rules (memory aids are allowed). The model is playing as "perfect-recall human", not "model with x-ray vision". Worth flagging in the harvester ask if we get there.

## 2. The v1.2 changes (final spec, audit-incorporated)

### 2.1 New block: DRAW TIMELINE (replaces SEEN IN WASTE THIS CYCLE)

**Two pieces, in two different locations in the prompt:**

**Piece A, the rendering rule lives in the RULES preamble.** This is the audit's HIGH-severity fix (anti-pattern #2: rules buried in data payload). Add this paragraph immediately after the existing KLONDIKE SOLITAIRE RULES section:

```
INTERPRETING THE DRAW TIMELINE (when present in the game state):
The DRAW TIMELINE block renders the stock and waste piles as one linear sequence
of card identifiers. The current waste top is wrapped in {curly braces}. Tokens
LEFT of {NOW} are cards that will be drawn next in this stock cycle: the
immediate next draw is the token directly left of {NOW}, the draw after that is
the token two positions left, and so on. Tokens RIGHT of {NOW} are cards drawn
earlier in this cycle, still sitting in the waste pile beneath the current top.
The token ??? marks a card whose identity has not yet been observed. ??? can
only appear during the first stock cycle. After the first recycle every position
is a known identity, because every stock card has passed through the waste at
least once.
```

**Piece B, the data block contains only the line itself.** No re-explanation, no labels:

```
DRAW TIMELINE:
  ??? ??? ??? 8D QH 3C {7H} 4S KH 2D
```

Render rules (deterministic from runner state):

- During cycle 1: leftmost positions render as `???` until the model has seen that card pass through the waste at least once.
- After first recycle: every slot is a known identity.
- When stock is empty (just before recycle): `{NOW}` sits at the leftmost position; only waste-side cards render to its right.
- When the run has no draws yet (turn 0): SKIP the block entirely (do not render).
- Maximum render length: 24 cards (full stock size) plus the `{NOW}` token.

No `|` separator between sections. Pure whitespace. The `{}` braces around the current card are sufficient delimiter. (Audit fix M1.)

### 2.2 Stock cycle counter (inline in STOCK line)

The STOCK line gets a `CYCLE:` field, using the same colon-key pattern as the existing fields (avoids the parenthetical-embed misparse risk flagged in audit M2).

Before:
```
STOCK: 14 cards   WASTE top: 2H   recycle stock: no
```

After:
```
STOCK: 14 cards   CYCLE: 2   WASTE top: 2H   recycle stock: no
```

Semantics: `CYCLE = 1 + (number of times recycle has fired)`. Cycle 1 starts at game start. Cycle 2 begins the moment the first recycle action is applied.

No new section heading; no new line. One key-value pair appended to an existing line.

### 2.3 No strategy heuristic edits (dropped per design principle)

An earlier draft of this spec proposed rewriting the existing v1.1 STRATEGY GUIDANCE bullet "Drawing from the stock is reasonable when no productive tableau/foundation move exists" into a hardened, timeline-aware predicate. **Dropped.**

Reason: per the design principle captured in `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/prompt-closes-info-gap-not-logic.md`, the prompt's job is to render the state and observations a human player would have through memory. It is NOT to inject decision logic, heuristics, or thinking procedures. That is the LLM's training's job.

The audit's anti-pattern #8 concern (the existing heuristic now conflicts with the new timeline info) is real, but the principled fix is to REMOVE the existing line, not replace it with a sharper one. Removal is being requested out-of-band by Chayut, not in this written spec. If pursued, removal is a pure deletion (no new logic added).

For the local test plan, we keep the v1.1 strategy section UNCHANGED. The experiment becomes: "does adding the DRAW TIMELINE alone (without removing or rewriting any heuristic) move the metrics". That is the cleanest possible single-variable test.

### 2.4 Static header tweak

The existing format-description sentence reads:
```
You will receive the current game as plain-text blocks (NOTATION, FOUNDATIONS, STOCK,
TABLEAU, RECENT MOVES, SEEN IN WASTE, LEGAL MOVES, PROGRESS, some are optional).
```

Replace `SEEN IN WASTE` with `DRAW TIMELINE` in that enumeration. No other header changes.

### 2.5 Prompt-budget impact (recalculated after dropping the heuristic edit)

| Block | v1.1 chars (typ.) | v1.2 chars (typ.) | Delta |
|---|---:|---:|---:|
| SEEN IN WASTE THIS CYCLE block | ~70 | 0 | -70 |
| INTERPRETING DRAW TIMELINE paragraph (in RULES) | n/a | ~580 | +580 |
| DRAW TIMELINE data line | n/a | ~90 | +90 |
| Stock line CYCLE field | n/a | ~10 | +10 |
| Strategy heuristic | (unchanged) | (unchanged) | 0 |
| **Net** | | | **+610** |

The v1.1 cleanup (drop confidence + alternative_move_index + calibration paragraph) freed ~1200 chars. v1.2 spends ~610 of that. Net prompt is still ~590 chars SMALLER than v1.0.

## 3. Local implementation contract

Lands in `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py`. No engine changes; no LoRA changes.

### 3.1 New runner state

```python
draw_history: list[str]      # every card drawn, chronological, across all cycles
stock_cycle: int             # starts at 1, increments on recycle_stock
```

### 3.2 New helper: render_draw_timeline

Pseudocode (audit-incorporated, no `|` separator):

```
def render_draw_timeline(state, stock_cycle, draw_history) -> str | None:
    upcoming = list(state.stock)             # in draw order (next-to-draw first)
    past_in_waste = list(reversed(state.waste[:-1]))  # most-recent-first beneath {NOW}
    if not state.waste:
        return None     # no draws yet, skip rendering
    now_card = card_short(state.waste[-1])
    if stock_cycle == 1:
        seen_set = set(draw_history)
        upcoming_render = ["???" if card_short(c) not in seen_set else card_short(c)
                           for c in upcoming]
    else:
        upcoming_render = [card_short(c) for c in upcoming]
    line = " ".join(upcoming_render + ["{" + now_card + "}"] + [card_short(c) for c in past_in_waste])
    return "DRAW TIMELINE:\n  " + line
```

### 3.3 Where the orientation paragraph lives in render_prompt

Insert immediately after the existing KLONDIKE SOLITAIRE RULES bullet list, BEFORE `THE GOAL:`. The paragraph from section 2.1 piece A goes here verbatim. (This is the audit's H1 fix: orientation rule in preamble, not data.)

### 3.4 STOCK line change

```python
# was:
f"STOCK: {len(state.stock)} cards   WASTE top: {waste_top}   recycle stock: {recycle}"
# becomes:
f"STOCK: {len(state.stock)} cards   CYCLE: {stock_cycle}   WASTE top: {waste_top}   recycle stock: {recycle}"
```

### 3.5 Flag

Add `--prompt-version {v1.1,v1.2}` to `play_deck_with_student.py`. Default = v1.1 (avoid silent behaviour change for existing scripts).

### 3.6 Cycle counter increment

In the main loop, on a successful `recycle_stock` move apply, increment `stock_cycle += 1`. Track this BEFORE rendering the next turn's prompt (so the increment is visible in the same turn's STOCK line that follows the recycle).

### 3.7 Symmetry pre-test (must pass before any model call)

Write a unit test in `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/tests/` (or run inline) that asserts:

- At turn 5 of cycle 1 (after 5 draws), the rendered DRAW TIMELINE line contains exactly 5 known identities (the 5 drawn cards), plus 19 `???` placeholders, plus `{NOW}` (= last drawn).
- At cycle 2 turn 1 (immediately after recycle), the line contains 24 known identities (full stock + 1 `{NOW}`), zero `???`.
- The identities in the cycle-2 render match the EXPECTED order: stock_cycle_2[i] == cycle_1_draw[i] (Klondike recycle preserves draw order).

Audit fix #5: explicit symmetry check before any model is exposed to the v1.2 prompt. Prevents accidental info leak.

## 4. Test arms (unchanged from prior draft)

Phase A is a tight A/B isolating the prompt effect. Phase B expands only if Phase A passes.

### 4.1 Phase A

| Arm | Model | Adapter | Prompt | Deck |
|---|---|---|---|---|
| **A_v1.1** | Gemma 4 E2B untuned | none | v1.1 | seed 2967897202 |
| **A_v1.2** | Gemma 4 E2B untuned | none | v1.2 | seed 2967897202 |

Compute estimate: 2 runs x ~70 min = ~140 min.

### 4.2 Phase B (only if Phase A passes)

| Arm | Model | Adapter | Prompt | Decks |
|---|---|---|---|---|
| **B_v1.1** | v1.1 LoRA (gemma-3n + adapters_t5_at750) | LoRA | v1.1 | 2967897202, 3263196305 |
| **B_v1.2** | v1.1 LoRA | LoRA | v1.2 | 2967897202, 3263196305 |
| **C_v1.2** | Gemma 4 E2B untuned | none | v1.2 | 3263196305 |

Compute estimate: 5 runs x ~70 min = ~350 min.

## 5. Pre-registered hypotheses, audit-strengthened

Each prediction is falsifiable with a quantitative pass criterion set BEFORE the runs.

### H1: cycle-aware reasoning emerges (audit-strengthened measurement)

**Prediction**: in A_v1.2, the model's `strategic_plan` will demonstrate use of the new prompt blocks.

**Pass criterion (primary)**: code every turn where the model chose `draw_card` into one of these rationale categories by manual review of the strategic_plan field:

- (a) draw to reveal new info (legitimate in cycle 1; ILLEGITIMATE in cycle 2+ if timeline shows no useful upcoming card)
- (b) draw to reach specific card X (verify X appears in the timeline left of `{NOW}`; if not, this is a hallucinated rationale)
- (c) draw because no productive tableau move (the old heuristic; legitimate)
- (d) draw with an explicit timeline / cycle reference (new in v1.2)

**Pass**: in A_v1.2, category (b) hallucinated rationales drop to zero AND category (d) explicit references appear at least 5 times. In A_v1.1, category (d) is zero by construction.

**Pass criterion (secondary, automated)**: word-boundary grep on strategic_plan text:
```
grep -ciE '\b(cycle|timeline|upcoming|already drawn|known stock)\b' \
  gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_2/responses/turn_*.txt
```
(audit fix L2: word boundaries prevent matching "lifecycle" etc.)

At least 5 distinct turns must match. The primary criterion is the load-bearing one; the secondary is a sanity check.

**Fail criterion**: category (d) never appears AND category (b) does not drop. Means the v1.2 prompt is invisible to the model.

### H2: stock-search draw-spam shrinks

**Prediction**: draw_card share during the longest plateau will be lower in A_v1.2 than A_v1.1.

**Pass criterion**: A_v1.2's plateau draw_card share is at least 15 percentage points lower than A_v1.1's. (c99da9 v1.1 baseline was 82 percent draws during plateau.)

**Fail criterion**: draw_card share unchanged or higher in v1.2.

### H3: peak foundation gain (noisy, do not gate on alone)

**Prediction**: A_v1.2 peak `final_foundation_cards` exceeds A_v1.1 peak by at least 5 cards.

**Caveat**: with N=1 per arm, this metric varies ~10 cards run-to-run. Reported for completeness; not part of the go decision.

### H4: hidden-card reveal

**Prediction**: A_v1.2 reveals more face-down cards (lower minimum fd reached) than A_v1.1.

**Pass criterion**: A_v1.2 reveals at least 2 more hidden cards than A_v1.1.

**Why this is the cleanest metric**: hidden-card reveal is the necessary condition for any non-trivial win path. A prompt change that increases reveal rate is causally upstream of every other quality metric.

### H5: Phase A go decision

**Proceed to Phase B if any TWO of {H1 primary, H2, H4} pass.** H3 is too noisy. H1 secondary is a sanity check, not a gate.

**Stop and document if zero or one of {H1 primary, H2, H4} pass.**

## 6. Decision tree

```
Phase A complete
    |
    +-- >=2 of {H1, H2, H4} pass  -->  PROCEED TO PHASE B
    |
    +-- <=1 pass                  -->  STAY ON v1.1, no harvester ask
                                       Document the failure here

Phase B complete (if reached)
    |
    +-- v1.2 wins on >=2 arms     -->  FILE HARVESTER v1.2 ASK
    |
    +-- v1.2 mixed or loses       -->  Keep runner on v1.2 if local wins;
                                       no harvester ask
```

## 7. Risks (audit-incorporated, new risk 6 added)

1. **Model ignores new blocks**: if H1 fails, we cannot diagnose H2/H4 as "v1.2 doing nothing" vs "v1.2 actively confusing the model". Mitigation: manual review of 5 random turn responses per run regardless of automated grading.

2. **Renderer leaks unobserved card identities**: section 3.7 unit test catches this. Pre-test must pass before any model is exposed to v1.2.

3. **Prompt got longer, inference got slower**: tolerable up to 10 percent rise in mean call_seconds; more triggers a review.

4. **Training corpus is on v1.1**: B arms test how robust the LoRA is to a prompt format change. B_v1.2 regression while B_v1.1 holds = signal that future LoRA training must use v1.2 prompts. This is information, not a failure.

5. **Harvester ships its own v1.2 first**: watch the c99da9 monitor and new exports for a third template hash. Our local result still informs the design conversation.

6. **Perfect-recall asymmetry (NEW per audit)**: in cycle 2+, v1.2 gives the model perfect recall over cycle 1's draw order. A typical human has imperfect memory. This is rule-symmetric (memory aids are permitted in Klondike) but worth flagging when filing the harvester ask, so the team knows the model now has an information advantage over a "typical" human while staying within the "perfect-recall human" envelope.

## 8. Scope guards (what we WILL NOT do in this pass)

- Re-train any LoRA. Inference-time only.
- Modify the static header beyond the section 2.1 + 2.3 + 2.4 changes.
- Add a separate FACE-DOWN-by-elimination line (rejected by user; the timeline plus tableau lets the model derive it).
- Address the inherited anti-patterns NOT introduced by v1.2 (free-form CoT in JSON, PRIOR REASONING amplifier). Tracked in `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_late_game_prompt_audit.md` for separate experiments.
- File anything with the harvester team. That happens after Phase B at earliest.

## 9. Pre-flight checklist for the test compute window

Before starting Phase A:

- [ ] Section 2.1 piece A paragraph added to STATIC HEADER, immediately after KLONDIKE SOLITAIRE RULES
- [ ] Section 2.3 strategy-heuristic replacement applied in STRATEGY GUIDANCE
- [ ] `render_draw_timeline` implemented per section 3.2
- [ ] STOCK line cycle field per section 3.4
- [ ] `--prompt-version` flag added per section 3.5; default v1.1
- [ ] Symmetry unit test from section 3.7 passes
- [ ] Audit script `/tmp/audit_prompt.py` (or its in-repo location) updated to recognize the new DRAW TIMELINE section header
- [ ] Pass criteria from section 5 re-read and confirmed; LOCKED before any model call

## 10. After-action commitments

If Phase A passes:
- Append a "Phase A result (run on YYYY-MM-DD)" section to this document.
- Update `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` with the local v1.2 result.
- Draft a v1.2 harvester ask doc only after Phase B completes.

If Phase A fails:
- Append a "Phase A result" section documenting the failure.
- Leave the runner on v1.1.
- Note for future work: "the draw-timeline + cycle-counter approach did not move the metrics on Gemma 4 E2B untuned; if a different intervention is tried, do not re-litigate this design without new evidence."

## 11. Files this plan touches when executed

| Path | Change |
|---|---|
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py` | implement v1.2 per section 3 |
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/tests/test_draw_timeline.py` (new) | symmetry unit tests per section 3.7 |
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_1/` | new run output |
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_2/` | new run output |
| `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_candidate_spec.md` | this doc; append Phase A result after the run |
| `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` | update if Phase A passes |

## 12. Audit incorporation summary (changes from the prior draft)

For traceability, the changes in this doc relative to `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_draw_timeline_local_test_plan.md`:

| Audit finding | Severity | Resolution |
|---|---|---|
| H1: orientation rule in data block, not preamble | HIGH | Moved to RULES preamble per section 2.1 piece A |
| H2: conflicting heuristic with new timeline info | HIGH | NO heuristic rewrite. Per the "prompt closes info gap, not logic" principle, the fix is to remove the existing line, requested out-of-band. The candidate spec leaves the v1.1 STRATEGY GUIDANCE section unchanged. See section 2.3. |
| M1: `|` separator ambiguous next to card identifiers | MEDIUM | Dropped; pure whitespace + `{}` braces per section 2.1 piece B |
| M2: `(cycle K)` parenthetical embed could misparse | MEDIUM | Changed to `CYCLE: K` colon-key pattern per section 2.2 |
| M3: H1 measurement is keyword-based, under-counts attendance | MEDIUM | H1 primary criterion is now manual rationale-category coding; keyword grep is secondary sanity check |
| M4: inherited anti-pattern of free-form CoT in JSON | MEDIUM | Noted in scope guards; deferred to a separate experiment |
| L1: date-stamp the Phase A result section explicitly | LOW | After-action commitment updated |
| L2: word-boundary grep | LOW | Applied to H1 secondary criterion command |
| Risk 6: perfect-recall asymmetry | n/a | Added to section 7 |

The candidate is ready for implementation when the next compute window opens.
