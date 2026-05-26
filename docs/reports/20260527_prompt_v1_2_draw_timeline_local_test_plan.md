# v1.2 Prompt Test Plan: DRAW TIMELINE + Stock Cycle Counter (Local Bench Before Harvester Ask)

**Date**: 2026-05-27
**Status**: PRE-ROLLOUT. To be tested locally in the next compute window before any harvester ask is filed.
**Author**: Captured during the c99da9 ingest session
**Triggering context**: c99da9 (seed 2967897202, third attempt) showed the model searching the stock for AD even though AD is NOT in the stock contents (verified via SEEN IN WASTE list). The harvester prompt currently hides stock-cycle memory during cycle 1 and exposes a mislabeled `SEEN IN WASTE THIS CYCLE` list after recycle. This plan tests two minimal prompt changes that close the memory gap.

## 0. Scope

What this document is:
- Pre-registered design of the v1.2 prompt changes
- Full local test plan with arms, metrics, and pass / fail rules
- The implementation contract for the runner
- A decision tree for what to do after results land

What this document is NOT:
- Code yet. No changes have been rolled out.
- A harvester ask. We test locally first; if results justify, then file.

## 1. Background: what we found in c99da9

(Full diagnosis in `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_conversation_log_v1_1_discovery_and_ingest.md` sections 6 to 8.)

Two prompt-design gaps the c99da9 run exposed:

1. **Cycle 1 memory gap**: the harvester prompt renders no `SEEN IN WASTE` list during the first stock cycle. The only memory of what was drawn is `RECENT MOVES (last 10)`. A 24-card cycle has 14 draws fall off the back of that window before the cycle ends. The model is genuinely blind to what it has already drawn.

2. **Misleading section label**: after the first recycle, the prompt shows `SEEN IN WASTE THIS CYCLE: <list>`. The actual semantics are "cards currently in the stock pile that have been observed before". The list SHRINKS as the model draws (because drawn cards leave the stock). A reader of the label would expect the opposite.

Concrete consequence: at c99da9 ti=44, the SEEN list contained 17 cards. AD was NOT in the list. AD was also not in foundations or face-up tableau. By elimination AD must be face-down. The model's stated plan: "draw to find AD". That plan is unreachable from the current state. The information needed to reach the correct deduction is in the prompt; the label makes it nearly impossible to extract.

## 2. The v1.2 changes

Two minimal changes. No new fields. One rename + format change. One inline counter.

### 2.1 Change 1: DRAW TIMELINE block (replaces SEEN IN WASTE THIS CYCLE)

**Header line** (one sentence at the top of the block, in the rendered prompt):

```
DRAW TIMELINE (left = upcoming draws; {NOW} = current waste top; right = past draws still in waste):
```

**Body line** (single linear render):

```
  ??? ??? ??? 8D QH 3C | {7H} | 4S KH 2D
```

Reading rules captured in the header:
- Leftmost item: the card that will be drawn LAST in this stock cycle
- Each rightward step: one draw EARLIER in time
- `{NOW}`: current waste top (the card just drawn)
- After `{NOW}`: cards drawn earlier in this cycle, still in the waste pile beneath the current top
- `???`: identity not yet observed (cycle 1 only; vanishes after the first recycle)

**Render rules** (deterministic from runner state):
- During cycle 1: leftmost positions render as `???` until the model has seen that card pass through the waste at least once
- After first recycle: every slot is a known identity
- When stock is empty: `{NOW}` slides to the leftmost position; only waste-side cards render to its right
- Bounded length: at most 24 cards per cycle (the full stock size)

### 2.2 Change 2: stock-cycle counter inline in STOCK line

Before:
```
STOCK: 14 cards   WASTE top: 2H   recycle stock: no
```

After:
```
STOCK: 14 cards (cycle 2)   WASTE top: 2H   recycle stock: no
```

Cycle counter semantics: `cycle = 1 + (number of times recycle has fired)`. Cycle 1 starts at game start. Cycle 2 begins the moment the first recycle action is applied.

### 2.3 Estimated prompt-budget impact

| Block | v1.1 chars (worst case) | v1.2 chars (worst case) | Delta |
|---|---:|---:|---:|
| SEEN IN WASTE THIS CYCLE | ~70 (label + 17 cards) | 0 (removed) | -70 |
| DRAW TIMELINE | n/a | ~180 (header sentence + 24-card line) | +180 |
| STOCK line cycle annotation | n/a | ~10 | +10 |
| **Net** | | | **+120** |

The v1.1 cleanup (drop confidence + alternative_move_index + calibration paragraph) freed ~1200 chars. We are spending ~120 of that headroom.

## 3. Local implementation contract

The local runner at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py` is the test harness. The two changes land there.

### 3.1 New runner state

- `draw_history: list[str]`: every card drawn (in chronological order of draws across all cycles). Populated in the `chosen.move_type.value == "stock_to_waste"` branch with the rendered card short string.
- `stock_cycle: int`: starts at 1. Increments by 1 the moment a `recycle_stock` move is applied (already a distinct move type the engine surfaces).
- `current_cycle_first_draw_idx: int`: index into `draw_history` marking where the current cycle's first draw lives. Updated on recycle.

### 3.2 New renderer logic in `render_prompt`

Pseudocode:

```
upcoming = stock contents in DRAW ORDER (next to draw first); identities from runner state
past_in_waste = waste contents excluding top, in MOST-RECENT-FIRST order
now_token = "{<waste_top>}"

if stock_cycle == 1:
    # Mark upcoming slots whose card has NOT been seen yet as ???
    seen_set = set(draw_history)
    upcoming_render = ["???" if c not in seen_set else c for c in upcoming]
else:
    upcoming_render = list(upcoming)

line = " ".join(upcoming_render + [now_token] + past_in_waste)
```

Edge cases the runner must handle:
- Stock empty: render `[upcoming = empty list]` plus `{NOW}` and the waste tail
- First turn: no draws yet, no waste; SKIP rendering the block (or render `(no draws yet)` for explicitness)
- Recycle just fired this turn: stock contents are now reversed waste. `draw_history` does NOT clear (the model has seen all these cards before).

### 3.3 Cycle counter change

In the existing STOCK line render, change:
```
f"STOCK: {len(state.stock)} cards   WASTE top: {waste_top}   recycle stock: {recycle}"
```
to:
```
f"STOCK: {len(state.stock)} cards (cycle {stock_cycle})   WASTE top: {waste_top}   recycle stock: {recycle}"
```

### 3.4 Static header changes

The format-description sentence currently reads:
```
You will receive the current game as plain-text blocks (NOTATION, FOUNDATIONS, STOCK,
TABLEAU, RECENT MOVES, SEEN IN WASTE, LEGAL MOVES, PROGRESS — some are optional).
```

Replace `SEEN IN WASTE` with `DRAW TIMELINE` in that enumeration. No further header changes.

## 4. Test arms (kept minimal to control cost)

Phase A is a tight A/B isolating the prompt effect. Phase B expands only if Phase A shows benefit.

### 4.1 Phase A: minimum-viable A/B

| Arm | Model | Adapter | Prompt | Deck |
|---|---|---|---|---|
| **A_v1.1** | Gemma 4 E2B untuned (`mlx-community/Gemma4-E2B-IT-Text-int4` + sanitize patch) | none | v1.1 (current runner) | seed 2967897202 |
| **A_v1.2** | Gemma 4 E2B untuned | none | v1.2 (this plan) | seed 2967897202 |

Why this pair:
- Gemma 4 E2B untuned is the cheapest model in our stack (~70 min full-game wallclock per run)
- No LoRA adapter loading; avoids confounding with training-distribution shift
- Seed 2967897202 is the deck where c99da9 failed; we want to know if the v1.2 prompt would have changed the outcome
- The v1.1 baseline gives us the doom-loop / dead-deal-flailing comparison point we already cataloged

**Phase A compute estimate**: 2 runs x ~70 min = ~140 min wallclock. One compute window.

### 4.2 Phase B: expansion (only if Phase A passes)

If Phase A pass criterion is met (see section 6), expand to:

| Arm | Model | Adapter | Prompt | Decks |
|---|---|---|---|---|
| **B_v1.1** | v1.1 LoRA (`gemma-3n` + `adapters_t5_at750`) | LoRA | v1.1 | seeds 2967897202, 3263196305 |
| **B_v1.2** | v1.1 LoRA | LoRA | v1.2 | seeds 2967897202, 3263196305 |
| **C_v1.2** | Gemma 4 E2B untuned | none | v1.2 | seed 3263196305 |

**Phase B compute estimate**: 5 runs x ~70 min = ~350 min wallclock. ~6 hours, fits one large compute window.

## 5. Metrics and what we measure per run

Primary metrics (extracted from `summary.json` + `turns.jsonl` + `analyze_play_run.py`):

| Metric | Source | What it answers |
|---|---|---|
| `final_foundation_cards` (peak fc) | summary | did the model make more or less foundation progress |
| `final_face_down` (lowest fd reached) | derived from turns | did the model reveal more or fewer hidden cards |
| `plateau_at_end_turns` | summary | how long did it stall at its peak |
| classification (WIN / MIDGAME_STALL / DEAD_DEAL / etc.) | `analyze_play_run.py` | failure mode label |
| draw_card percentage during plateau | turns | was the model draw-spamming |
| recycle count | turns | how many times did the model loop the stock |
| mean call_seconds per turn | turns | did the prompt change slow inference |

Secondary metrics:

| Metric | Source | What it answers |
|---|---|---|
| `prompt_chars` per turn | turns | does the prompt grow as expected (~120 chars more) |
| count of `discard_to_tableau` moves | turns | did the model commit waste cards to tableau more often |
| Hidden cards revealed across the run | turns (fd delta) | proxy for "did the model unlock the board" |
| First reasoning that references "cycle" or "drawn" or "timeline" | response text | did the model attend to the new prompt blocks |

## 6. Pre-registered predictions and pass / fail rules

Each prediction is falsifiable. Pass criteria are set BEFORE the runs.

### H1: cycle-aware reasoning emerges

**Prediction**: in Phase A_v1.2, the model's `strategic_plan` will reference the cycle counter or the timeline at least once during the run. The model will write something like "since we are in cycle 2 and the timeline shows X" or "AD is not in the timeline so drawing will not produce it".

**Pass criterion**: at least 3 distinct turns in A_v1.2 contain a strategic_plan that quotes any of: "cycle", "timeline", "upcoming", "next draw", "already drawn", or "?" in a deductive sense. A_v1.1 baseline will have zero by construction.

**Fail criterion**: zero turns reference the new prompt blocks. This would mean the model is ignoring the new information; we need a bigger intervention.

### H2: stock-search draw-spam shrinks

**Prediction**: the draw_card percentage during the longest plateau will be lower in A_v1.2 than A_v1.1.

**Pass criterion**: A_v1.2's plateau draw_card share is at least 15 percentage points lower than A_v1.1's. (c99da9 baseline was 82 percent draws during plateau.)

**Fail criterion**: draw_card share unchanged or higher in v1.2. This would mean the model attends to the info but does not change its action distribution.

### H3: peak foundation gain

**Prediction**: A_v1.2 reaches a higher peak `final_foundation_cards` than A_v1.1.

**Pass criterion**: A_v1.2 peak fc is at least 5 cards higher than A_v1.1 peak fc.

**Fail criterion**: peak fc unchanged or lower.

**Caveat**: with N=1 trace per arm, this metric is noisy (the same arm can vary by ~10 cards run-to-run). H3 alone is weak; H1 + H2 are the load-bearing signals.

### H4: hidden-card reveal

**Prediction**: A_v1.2 reveals more face-down cards (lower minimum fd reached) than A_v1.1.

**Pass criterion**: A_v1.2 reveals at least 2 more hidden cards across the run than A_v1.1.

**Fail criterion**: reveals the same or fewer.

**Why this is the cleanest metric**: hidden-card reveal is the necessary condition for any non-trivial win path. A prompt change that increases reveal rate is causally upstream of every other quality metric.

### H5: cumulative phase-A go decision

**Pass to Phase B if any TWO of {H1, H2, H4} pass.** H3 is too noisy to gate on alone.

**Fail to Phase B if zero or one of {H1, H2, H4} pass.** Document the result, leave the runner on v1.1, no harvester ask.

## 7. Run protocol (mechanical)

The runner is invoked twice with a single flag toggling the prompt version. Implementation note: add a `--prompt-version {v1.1,v1.2}` argument to `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py`. Default keeps v1.1 to avoid silent behaviour change.

### 7.1 Setup (one-time, fits within the test window)

```
# 1. Implement v1.2 changes in play_deck_with_student.py:
#    - add stock_cycle counter
#    - add draw_history list
#    - extend render_prompt with v1.2 timeline + cycle annotation
#    - add --prompt-version flag
# 2. Unit-test the renderer on a fixture state (one cycle, mid-cycle, post-recycle)
```

### 7.2 Phase A execution

```
# Both runs in parallel if memory allows (32 GB+ helps), otherwise sequential
mkdir -p /Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs

.venv/bin/python gemma4_finetune/play_deck_with_student.py \
  --deck-seed 2967897202 \
  --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
  --prompt-version v1.1 \
  --out-dir gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_1

.venv/bin/python gemma4_finetune/play_deck_with_student.py \
  --deck-seed 2967897202 \
  --model-id mlx-community/Gemma4-E2B-IT-Text-int4 \
  --prompt-version v1.2 \
  --out-dir gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_2
```

### 7.3 Grading

```
.venv/bin/python gemma4_finetune/analyze_play_run.py \
  gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_1 --json > grade_v1_1.json
.venv/bin/python gemma4_finetune/analyze_play_run.py \
  gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_2 --json > grade_v1_2.json
```

H1 grading: `grep -ciE 'cycle|timeline|upcoming|already drawn' responses/turn_*.txt`. Manual review on flagged turns.

H2 grading: from `turns.jsonl`, compute draw_card share within the longest plateau window per `analyze_play_run.plateau_windows`.

H3 grading: compare `summary.final_foundation_cards`.

H4 grading: compute min(fd) over the run from `turns.jsonl`.

## 8. Decision tree after Phase A

```
                    Phase A complete
                          |
                  +-------+-------+
                  |               |
            >=2 of H1,H2,H4    <=1 of H1,H2,H4
                  |               |
            PROCEED TO         STAY ON v1.1
            PHASE B            (no harvester ask)
                  |
            +-----+-----+
            |           |
        Phase B all   Phase B mixed
        pass on        or fails
        2+ arms
            |              |
        FILE HARVESTER  KEEP RUNNER ON
        v1.2 ASK         v1.2 (local wins),
                         no harvester ask
```

## 9. Pre-committed risks and what to watch for

1. **Risk: model ignores new blocks.** If H1 fails (zero references to cycle / timeline), we cannot diagnose H2 / H3 / H4 as "v1.2 doing nothing" vs "v1.2 actively confusing the model". Mitigation: read 5 random turn responses from each run by hand, regardless of automated grading.

2. **Risk: the v1.2 render breaks legality.** The DRAW TIMELINE relies on knowing the future stock order. The harvester knows it (the deck is dealt deterministically). Our runner knows it (we control the engine). But the LLM should not be given info it would not normally see in cycle 1. Specifically: in cycle 1, identities of UNDRAWN stock cards must render as `???`. Pre-test: dump the v1.2 prompt at cycle 1 turn 5 and verify no `???` slot has leaked a card identity.

3. **Risk: prompt got longer, inference got slower.** Mean call_seconds per turn could rise. Acceptance: a rise of less than 10 percent is tolerable; more than 10 percent triggers a review of whether we can compress the timeline render.

4. **Risk: training corpus is on v1.1.** Even if v1.2 helps untuned models, the LoRA in arm B was trained on v1.1-shaped prompts. The B arms test how robust the LoRA is to a prompt-format change. If B_v1.2 regresses heavily while B_v1.1 holds, that is a sign the prompt is now distribution-shifted from the LoRA's training data. This is information, not a failure: it tells us future LoRA training must use v1.2 prompts.

5. **Risk: harvester ships its own v1.2 in the meantime.** Watch the c99da9 monitor and any new exports for a third template hash. If the harvester adopts a different design, our test result still informs the design conversation but we may not push our ask.

## 10. What we WILL NOT do in this pass

- Re-train any LoRA. We test inference-time only.
- Modify the static header beyond the one-word replacement (`SEEN IN WASTE` to `DRAW TIMELINE` in the format enumeration).
- Add the FACE-DOWN-by-elimination line (rejected by user; the timeline plus tableau lets the model derive it).
- File anything with the harvester team. That happens after Phase B at earliest.
- Touch any production ingest, bench, or HF pipeline. v1.2 is a runner-side test.

## 11. After-action update commitments

If Phase A passes:
- Update `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` with the local-v1.2 result.
- Draft a v1.2 harvester ask doc only after Phase B completes.

If Phase A fails:
- Document the failure here (this same doc) in a new section "Phase A result".
- Leave the runner on v1.1.
- Note for future work: "the draw-timeline / cycle-counter approach did not move the metrics on Gemma 4 E2B untuned; if a different intervention is tried, do not re-litigate this design without new evidence".

## 12. Pre-flight checklist for the test compute window

Before starting Phase A:

- [ ] v1.2 changes implemented in `play_deck_with_student.py`
- [ ] `--prompt-version` flag added; default = v1.1
- [ ] Renderer unit-tested on 3 fixture states (mid-cycle-1, end-of-cycle-1, mid-cycle-2)
- [ ] Cycle 1 `???` masking verified (no info leak)
- [ ] Audit script `/tmp/audit_prompt.py` or the moved version updated to recognize the new DRAW TIMELINE section header
- [ ] Pass criteria from section 6 re-read and confirmed; locked in this doc

## 13. Files this plan touches when executed

| Path | Change |
|---|---|
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_deck_with_student.py` | implement v1.2 changes + flag |
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_1/` | NEW run output |
| `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/play_runs/gemma4_untuned_seed2967897202_v1_2/` | NEW run output |
| `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_draw_timeline_local_test_plan.md` | this doc; append Phase A result section after the run |
| `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-1-shipped.md` | update if Phase A passes |
