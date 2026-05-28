# Prompt v1.3 Candidate Spec (P0/P1 only)

**Date**: 2026-05-28
**Status**: CANDIDATE SPEC. Implementation not started. Single-variable bench design included.
**Supersedes**: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260527_prompt_v1_2_candidate_spec.md` (v1.2 shipped as build `cef6291`; this doc proposes the next iteration).
**Audited against**: `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/SKILL.md` plus the research synthesis in `/Users/chayut/repos/solitaire-analytics/docs/Klondike Solitaire Strategies_ Common to Controversial.md`.
**Scope**: only fixes with high confidence backed by direct empirical evidence in the corpus, or that map to a canonical anti-pattern with a cited mechanism. Speculative changes and stylistic improvements are deferred to v1.4.

## 0. Why this exists

Three same-seed data points across builds show monotonic regression on the same physical deck:

| Build | Prompt | Seed 2967897202 outcome | finalProgress | moveCount | foundationCards |
|---|---|---|---|---|---|
| `7f01833` | (pre-hybrid-v1) | won | 100% | 194 | 52 |
| `20a825f` | hybrid-v1.1 | stalled_auto_terminated | 29% | 197 | 15 |
| `cef6291` | hybrid-v1.2 | incomplete | 17% | 199 | 9 |

And on the second known-winnable seed:

| Build | Prompt | Seed 3263196305 outcome | moveCount | Notable |
|---|---|---|---|---|
| `6dfc8a9` | (pre-hybrid-v1) | won | 174 | clean trajectory |
| `cef6291` | hybrid-v1.2 | won | 296 | 94-turn flat plateau on `(fc=14, fd=8)` mid-game with active `3H/4C col 3 ↔ col 4` oscillation; broke out only after recycling enough times |

The v1.2 prompt is doing worse on both deals where direct comparison exists. The audit at `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/SKILL.md` against the v1.2 prompt identified the proximal causes; this spec addresses only the high-confidence subset.

## 1. Hard constraint (unchanged from v1.2)

Every information item rendered must be obtainable by a human playing the same deck under standard Klondike rules. The principle from memory `prompt-closes-info-gap-not-logic` applies: the prompt renders state; the model brings the logic. v1.3 holds this line and explicitly does NOT add new strategy heuristics, even research-derived ones.

## 2. The v1.3 changes (P0 / P1 only)

### 2.1 [P0] Restore the deleted "drawing is reasonable" permission bullet

**Evidence**: this is the only material STRATEGY GUIDANCE diff between v1.1 and v1.2. v1.1 sample reasoning at turn 34 (c99da9): *"Since no tableau moves reveal hidden cards or advance the foundations, the priority is to draw from the stock... Drawing is the only productive action available."* Clean decision. v1.2 sample reasoning at turn 79 (6b491a) under identical board pressure: *"The priority is to reveal hidden cards. Move [2] moves the 9S-8D sequence from Column 3 to Column 1... Following this, the JS-TH sequence can be moved..."* Picks the shuffle, then reverses it next turn, 165 times across the session.

Mechanism: without explicit permission to draw, the model is forced to construct fabricated reveal-justifications for shuffles. Bullets #1 (reveal-as-main-obstacle) and #5 (prefer-exposing-over-shuffling) together with no escape valve produce the oscillation.

**Change**: re-insert the bullet exactly as it was in v1.0 and v1.1.

```
- Drawing from the stock is the correct action when no legal tableau or
  foundation move exposes a face-down card or advances a foundation.
```

(Stronger phrasing than v1.1's "is reasonable" because the audit found the soft form is interpreted away under pressure from #1's superlative.)

**Confidence**: HIGH. Single-variable cause identified, single-variable fix proposed, single-variable bench available (arm B below).

### 2.2 [P0] Replace soft anti-undo with a hard predicate

**Evidence**: v1.2 bullet text *"Avoid moves that simply undo a recent move or lead nowhere."* The model violated this 165 times in one session (`solitaire-ai-log-6b491a-1779877547228.json`, session-wide oscillation `8D col 1 ↔ col 3` 85x and `9S col 1 ↔ col 3` 80x). Empirical proof that the soft-modal form does not constrain Gemma 4 31B behavior. Falls under anti-pattern 5.1 (soft hedged heuristics) at `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/references/anti_patterns.md`.

**Change**: replace with an operational predicate the model can verify per turn.

```
- Do not move a card to a tableau column it occupied in the last 5 moves
  shown in RECENT MOVES.
```

The `5` is calibrated to the existing 10-move RECENT MOVES window, leaving room for a single forward-back cycle but not a sustained loop. The model can verify this against the visible RECENT MOVES block; no internal state required.

**Confidence**: HIGH. The empirical-violation count proves the soft form is non-binding; the predicate form is falsifiable per turn against state the prompt already renders.

### 2.3 [P0] Drop the PRIOR REASONING block (or cap to 1 entry with re-evaluate framing)

**Evidence**: PRIOR REASONING accumulates the last 5 turns of the model's own rationale and feeds them back unframed. When the session is in a doom-loop, those 5 entries all say "the priority is to reveal hidden cards / Move [X] exposes..." and the next-turn reasoning conditions on 5 in-context examples of the model committing the loop. This is the canonical Halawi et al. failure mode ("Overthinking the Truth", arxiv 2307.09476) reproduced verbatim. The anti-pattern catalog at `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert/references/anti_patterns.md` section 4.1 names "a production Klondike Solitaire prompt" as the canonical real-world example, citing this exact corpus. Also see "Understanding In-Context Learning from Repetitions" (arxiv 2310.00297) for the structural-repetition variant: even matching the SHAPE of prior outputs primes imitation.

**Change (preferred)**: drop PRIOR REASONING from the per-turn prompt entirely. The model has the current state and RECENT MOVES; it can reason from scratch.

**Change (fallback if recency cue is judged necessary)**: cap to 1 prior entry, relabel and add a one-line directive:

```
PREVIOUS DECISION (re-evaluate against the current state; if the board has
not advanced since, your previous reasoning was insufficient, pick a different
approach):
  move: <previous move>
  why: <previous rationale, truncated to 200 chars>
```

Single entry, explicit "re-evaluate" framing, hard cap on character count.

**Confidence**: HIGH. Cited mechanism, canonical anti-pattern, this corpus is literally the named example.

### 2.4 [P1] Rewrite STRATEGY GUIDANCE bullet #1 as a conditional predicate

**Evidence**: bullet #1 reads *"Prioritize moves that turn over (reveal) a face-down tableau card, hidden cards are the main obstacle."* Sampled reasoning traces show the model parrots the "primary objective is to reveal" phrasing in 4 of 4 mid-plateau v1.2 turns (and in c99da9 v1.1, and in the c98e59 v1.0 trace). The "main obstacle" superlative is being treated as a hard objective, which generates reveal-narrative fabrications when no real reveal is available (the turn-198 case: model justifies a draw with "drawing the 6H will reveal a hidden card in column 6", but 6H cannot land on tableau because both black 7s are face-down).

**Change**: rewrite as a conditional predicate with operational tie-break, drawing from the "Tableau Prioritization" section of `/Users/chayut/repos/solitaire-analytics/docs/Klondike Solitaire Strategies_ Common to Controversial.md`:

```
- If any legal move exposes a face-down tableau card, prefer such a move.
  When multiple legal moves expose a face-down card, prefer the one that
  exposes a card in the column with the most face-down cards remaining.
```

Drops the superlative ("main obstacle") that produces the parroting; keeps the substantive guidance; adds an operational tie-break the model can verify against the TABLEAU block.

**Confidence**: HIGH on the direction (audit finding plus parroting evidence). MEDIUM on the exact phrasing; alternate formulations exist.

### 2.5 [P1] Investigate auto-terminator regression on build `cef6291`

**Evidence**: c99da9 (v1.1, build `20a825f`) terminated cleanly via the stall auto-terminator at plateau ~24 turns. 6b491a (v1.2, build `cef6291`) reached a 90-turn flat plateau on foundation and ~36-turn plateau on the latest `(fc, fd)` pair and was exported with outcome `incomplete`, not `stalled_auto_terminated`. The plateau is well past the STALL_TURNS=25 threshold. Three explanations to distinguish:
1. Build `cef6291` silently changed the terminator threshold or logic.
2. Operator captured the export before the terminator fired (in which case a later snapshot would show the termination).
3. The terminator's plateau definition uses a metric that resets under certain v1.2 conditions (e.g., counts on the v1.2 text-format `(foundationCards, faceDownTotal)` extraction differently).

**Change**: this is not a prompt edit. The action is a separate harvester-team ask: confirm whether the stall auto-terminator fires correctly under build `cef6291`. If it does not, that explains why a v1.2 session can produce 199 moves of plateau without termination, and the harvester-side fix is independent of the prompt revision.

**Confidence**: HIGH that something is off (compare same-seed termination behavior across builds). MEDIUM on which of the three explanations is correct. The investigation itself is low-cost (one snapshot-pair comparison plus the harvester-team confirmation).

## 3. Items explicitly DEFERRED to v1.4 or later

Documented here so they are not silently forgotten and not silently bundled into the v1.3 bench (which would defeat single-variable attribution).

| Item | Reason for deferral |
|---|---|
| Schema collapse (drop `board_analysis`, keep one reasoning channel + decision) | Distillation pipeline needs the prose; tradeoff not yet quantified. Run after v1.3 lands. |
| DRAW TIMELINE direction flip (LEFT-of-NOW = NEXT is counterintuitive) | Cosmetic. Model interprets correctly. Save for a v1.4 grouped rewrite. |
| "Advisor" -> "player" role priming | Speculative. No direct evidence in traces. |
| Adding research-derived strategies (King parity, Rule of Three, etc.) | Memory `prompt-closes-info-gap-not-logic` says prompt renders state, not logic. Model already knows these from training. Adding them invites more parroting, not less. |
| Foundation-evacuation predicate (R-1 opposite-color must be accounted for) | Real, but not addressing the observed pathology (oscillation, not premature foundation play). Out of scope for v1.3. |
| Echo-back of chosen move (`describe` field) | Low-cost win for index-misalignment catching, but not addressing the regression. Bundle into the v1.4 schema rewrite. |
| Resign-trigger rewrite (operational predicate) | Depends on the auto-terminator investigation outcome. If the terminator works correctly once fixed, the resign output can stay as-is. |

## 4. Single-variable bench design

The whole point of constraining v1.3 to P0/P1 is to make the regression diagnosable. Use a four-arm bench, each arm holds all other variables constant.

### 4.1 Arms

| Arm | Prompt | Hypothesis |
|---|---|---|
| **A (control)** | v1.2 as-shipped (build `cef6291`) | baseline |
| **B** | v1.2 with change 2.1 only (restore drawing-permission bullet) | the regression is the missing bullet; B should recover most of v1.1's behavior on same seeds |
| **C** | v1.2 with change 2.1 + 2.3 (also drop PRIOR REASONING) | the PRIOR REASONING block adds material loop-reinforcement on top of the missing bullet |
| **D** | v1.2 with changes 2.1 + 2.2 + 2.3 + 2.4 (full v1.3) | the full P0/P1 fix bundle |

Arm B vs A isolates the bullet restore. Arm C vs B isolates the PRIOR REASONING drop. Arm D vs C isolates the bullets 1 and anti-undo rewrites combined.

### 4.2 Seed set (N=12)

Three confirmed-winnable anchor seeds (all have an empirical win on record, not just solver-estimated) plus 9 fresh seeds.

| Seed | Class | Reason |
|---|---|---|
| `2967897202` | confirmed-winnable (won under 7f01833), regressed under v1.1/v1.2 | direct same-seed comparison to the regression; does v1.3 recover it |
| `3263196305` | confirmed-winnable (won under 6dfc8a9 and cef6291/v1.2 in 296 moves) | check if D shortens the 296-move v1.2 trajectory back toward 174 |
| `2853966634` | confirmed-winnable (won under cef6291/v1.2 as session aca45a in 418 moves) | doom-loop-recovery deck; does v1.3 win it in fewer moves / shorter plateau than v1.2's 418 moves + two plateaus (86 + 122 turns) |
| 9 fresh seeds | unknown, sampled from harvester's standard distribution | population-level signal |

The three anchor seeds are the load-bearing same-seed evidence; the 9 fresh seeds give population-level statistics.

**Model constraint (load-bearing for the anchor comparisons)**: all v1.3 bench arms MUST run on `gemma-4-31b-it`, NOT `gemma-4-26b-a4b-it`. Every confirmed-win baseline above is a 31B result. Running a v1.3 arm on 26B-a4b reintroduces the model as a confound and destroys the same-seed attribution. The 26B-a4b cohort is a separate orthogonal axis (see `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-26b-cohort-cataloging.md`); do not mix it into these arms.

**Note on `2853966634` specifically**: the v1.2 win path is fully characterised (the KC chromatic-bottleneck analysis). The bottleneck was an early FORWARD move (building `QH-JC-TH` on top of the only exposed black King `KC`), which the v1.3 anti-undo predicate (change 2.2) cannot catch because it is neither an undo nor an oscillation. So watch two distinct outcomes on this seed: (a) does v1.3 AVOID the bottleneck (requires planning depth; predicted no), versus (b) does v1.3 ESCAPE the bottleneck faster once in it (predicted yes, modest). If v1.3 shows the same 200-turn oscillation before recovery, that confirms the bottleneck is a planning-depth problem beyond the reach of STRATEGY GUIDANCE edits, and points the next investigation at planning mechanisms (the INTENT hypothesis) rather than more rules.

### 4.3 Metrics (per session)

- `moveCount` at terminal export
- `finalProgress` at terminal export
- max-plateau-length on `(foundationCards, faceDownTotal)` (from `load_export.py` briefing)
- session-wide top-3 oscillation pair counts (from `load_export.py` briefing)
- `outcome` (`won` / `stalled_auto_terminated` / `incomplete` / `lost`)

### 4.4 Decision rules

- **Arm B recovers v1.1-level behavior on seed 2967897202** (foundationCards approaches 15, plateau approaches 24 turns, not 90+): change 2.1 is confirmed; ship B as v1.3.0 immediately, run C-vs-B and D-vs-B separately.
- **Arm B does NOT recover**: the deleted bullet is not the proximal cause; pivot to C (PRIOR REASONING is the cause).
- **All arms fail to recover on 2967897202**: the regression is not in STRATEGY GUIDANCE at all; investigate the inference-config delta between builds (sampling temperature, thinking-budget, etc.) before further prompt edits.

### 4.5 Cost estimate

- 4 arms x 12 seeds x ~200 moves x ~2s per turn x ~50% error rate x ~2 retries = roughly 19 hours of wall-clock if serial. Note aca45a's 418-move recovery shows winnable sessions can run 2x the ~200-move assumption, so budget for tail-heavy sessions.
- With the harvester web UI's existing concurrency (assume 2 in parallel) and overnight execution: one to two calendar days.

## 5. Open questions for the harvester team

1. Does the stall auto-terminator under build `cef6291` fire on `(foundationCards, faceDownTotal)` plateau or on a different metric? Reference session `6b491a` for the canonical "should have fired but did not" case.
2. Can the harvester run the four-arm bench under controlled prompt diffs without a full build rev each time? If yes, the bench can ship inside one harvester deploy; if no, four separate builds are needed and the bench's cost rises proportionally.
3. Is the `move_index: -1` resign output reachable in the current harvester schema under v1.2? It has never fired across any session. Want to confirm the wiring works before the resign-trigger rewrite is even worth designing.

## 6. What ships

If the bench validates the hypothesis:
- v1.3 prompt with changes 2.1, 2.2, 2.3, 2.4 applied.
- Auto-terminator confirmation from the harvester team (action 2.5).
- A short addendum to `/Users/chayut/repos/solitaire-analytics/data/DATASET_NOTES.md` summarizing the bench result.

If the bench does not validate:
- The negative result is also a publishable finding. Document the failed hypothesis at `/Users/chayut/repos/solitaire-analytics/docs/reports/20260530_v1_3_bench_result.md` (or whatever date), update `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/harvester-prompt-v1-2-shipped.md` with the disconfirmation, and pivot to the inference-config investigation in the v1.4 spec.
