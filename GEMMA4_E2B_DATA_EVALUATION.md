# Dataset Evaluation — Solitaire Advisor Distillation Logs

> Sample evaluated: `data/solitaire-ai-log-1779012106344.json` (200 interactions, exported 2026-05-17)
> Purpose: assess fitness as a distillation set to train Gemma 4 **E2B** to match the **gemma-4-31b-it** teacher
> Status: **NOT TRAINABLE YET — one blocking defect.**
> Update 2026-05-17: collection team has accepted the feedback; **P0 fixes are in progress.**
> Pilot will re-collect at **draw-3 Klondike** (`draw_count=3`, "normal difficulty 3").

---

## 1. Verdict

The collection *plumbing* works and the response *schema* is clean — but this sample
**cannot train a usable advisor**. One defect is fatal on its own:

> **Every single one of the 84 usable examples has the same answer: `move_index: 0` (draw from stock).**

A distillation set whose label never varies teaches the student exactly one behavior —
"always draw." The reasoning text varies, so E2B would learn to emit confident,
plausible-sounding analysis and then *always* pick draw. That is worse than useless:
it is a confidently wrong player.

This is fixable, but it is a **collection-design** fix, not a training fix. Details
and recommendations below.

---

## 2. What's good (keep doing this)

- **Response schema is 100% clean.** All 84 success rows parse as JSON with exactly
  the 3 required keys (`board_analysis`, `strategic_plan`, `final_decision`).
- `move_index` is always within the legal range (0–N). `rawResponse` and the parsed
  `decision` object agree perfectly (0 mismatches).
- `board_analysis` (307–678 chars) and `strategic_plan` (299–585 chars) are
  substantive, on-topic, and well-written — good supervision *text* if the decisions
  were sound.
- The prompt design (rules block + game JSON + numbered `legalMoves`) is solid.
- Rich token accounting (`promptTokens`, `thoughtTokens`, `outputTokens`) is captured.

---

## 3. BLOCKING DEFECT — Zero decision-label variance

| Measurement | Result |
|---|---|
| Usable examples (success + clean JSON) | 84 |
| Distinct `move_index` values chosen | **1** — always `0` |
| `legalMoves[0]` content | **always** the draw/recycle move (80× `draw_card`, 4× `recycle_stock`) |
| Real alternatives that existed but were never chosen | 168 `tableau_to_tableau`, 19 `discard_to_tableau` |
| Avg legal moves per board | 3–4 (so the teacher had genuine choices every time) |

The teacher faced 3–4 options on every board and chose index 0 — the draw — **84/84
times**, declining 187 tableau/discard moves in total. The strategist prompt's own #1
heuristic is *"prioritize moves that reveal a face-down card"*, yet some of those 187
declined moves do exactly that.

### Root cause — two hypotheses, and the test that distinguishes them

1. **Positional bias.** `legalMoves[0]` is *always* the draw move (the move generator
   emits draw first). The teacher may be defaulting to index 0 regardless of content.
2. **Pathological sampling.** The logged boards may all come from draw-favorable
   states (e.g. a game trajectory that stalled into a draw loop).

**Diagnostic the collection team must run:** shuffle the `legalMoves` array order
before sending it to the teacher (draw move at a random index, not always 0). Re-collect
a sample.
- If the teacher now picks a spread of indices → it was reasoning; the old sample was
  just unlucky/biased sampling.
- If it *still* concentrates on index 0 → confirmed positional bias; the prompt and/or
  teacher need rework before any collection continues.

Either way, **legalMoves order should be randomized per call from now on** — it both
removes the bias and makes the dataset robust.

---

## 4. MAJOR ISSUE — Collection reliability (58% failure rate)

| Measurement | Result |
|---|---|
| Outcomes | 84 success / **116 error** (58% error) |
| Retry chains (`requestId` reused) | 52 — attempts go up to 4 deep |
| Error `durationMs` | 664 ms – **1,042,980 ms** (one call hung ~17 min) |
| Error diagnostics captured | **none** — empty `rawResponse`, all token fields `null` |

Two distinct failure modes are visible: fast failures (<1.5 s — likely rate-limit,
safety block, or malformed request) and a 17-minute hang (timeout). **But the errors
log nothing** — no HTTP status, no error message, no `finishReason` — so the team
cannot diagnose or fix the 58%.

---

## 5. ISSUE — Low game diversity

82/84 boards have distinct `(faceDownCount, moveCount)` signatures, but `moveCount`
runs in **consecutive runs** (34, 35, 36, 37, 38…). These are sequential turns of a
small number of games, not 84 independent positions. Consecutive turns are highly
correlated — adjacent boards differ by one move — so the *effective* diversity is far
below 84. No game seed / game ID is logged, so game boundaries had to be inferred.

---

## 6. ISSUE — Teacher reasoning trace is discarded

The teacher used 755–6002 `thoughtTokens` of internal reasoning per call, but only the
final JSON (`rawResponse`) is stored. The success criterion is *"E2B matches the 31B
teacher."* If "match" means matching the *reasoning*, the actual reasoning is not in
the data — only the post-hoc `board_analysis` / `strategic_plan` summary. The team must
decide: capture the thinking trace, or accept that E2B distills final answers only.

---

## 7. MINOR label-quality notes

- `alternative_move_index` is **out of range in 4/84** rows — a label bug; validate it
  against `legalMoves` length at collection time.
- `confidence` only ever takes 3 values (0.9 / 0.95 / 1.0) — effectively constant,
  carries no usable signal. Either elicit a real distribution or drop the field.

---

## 8. Recommendations to the Collection Team (prioritized)

**P0 — blockers, fix before collecting more — 🔧 IN PROGRESS (collection team)**
1. 🔧 **Randomize `legalMoves` order per call.** Removes positional bias; makes index a
   real label. Run the §3 diagnostic to classify the teacher's behavior.
2. 🔧 **Sample boards from many independent games** (distinct shuffle seeds), not
   consecutive turns. Log a `gameSeed` / `gameId` and `turnIndex` on every interaction.
3. 🔧 **Stratify board selection** so the final dataset spans the decision space —
   deliberately include positions where revealing a face-down card, playing to a
   foundation, or moving a King is clearly correct. Target a balanced mix of decision
   types, not ~100% draw.

> **Pilot config:** the re-collection pilot runs **draw-3 Klondike** (`draw_count=3`).
> Note for the §3 diagnostic: draw-3 changes the legal-move mix (one `draw_card` still
> appears per turn) but does **not** affect the randomization fix — index 0 must still
> stop being "always draw."

**P1 — reliability & diagnosability**
4. **Log error details** on every failed interaction: HTTP status, error type/message,
   `finishReason`, and whether a safety filter triggered. Empty errors are undebuggable.
5. **Set a sane per-call timeout** (e.g. 120 s) with exponential backoff; no call
   should hang 17 minutes.
6. Investigate the sub-1.5 s failures — almost certainly rate limiting or safety
   blocks; throttle request rate accordingly.

**P2 — label quality & supervision richness**
7. Fix the `alternative_move_index` range bug (validate at write time).
8. Decide on `confidence`: make it meaningful or remove it.
9. Decide whether to **capture the teacher's reasoning trace**. Recommended: yes —
   it materially improves a "match the teacher" distillation and lets E2B learn
   *why*, not just *what*.
10. Optionally log the post-move ground truth (did the chosen move help? did the game
    eventually win?) so labels can later be quality-weighted or filtered.

---

## 9. Volume & composition target

For distillation that genuinely matches the 31B teacher's decisions across the game:

| Property | This sample | Recommended target |
|---|---|---|
| Usable examples | 84 | **2,000–5,000** |
| Independent games | ~few | **≥ 300**, varied seeds |
| Decision-type balance | 100% draw | roughly proportional to real legal-move mix |
| Error rate | 58% | < 10% |
| Reasoning trace | not captured | captured (recommended) |
| Per-row metadata | partial | + `gameSeed`, `gameId`, `turnIndex`, error detail |

84 clean examples is a fine *plumbing test* — it proves the pipeline emits valid JSON.
It is **not** a training set. Re-collect with the P0 fixes before any fine-tuning run.

---

## 10. Bottom line for the training plan

- **Do not train on this sample.** A model fit to it will always answer "draw."
- The blocker is in **data collection**, not the training stack — the MLX/E2B plan in
  `GEMMA4_E2B_FINETUNING_RESEARCH.md` remains valid and waits on a real dataset.
- Fastest path to unblock: apply P0 (#1–#3) — **in progress** — collect a ~500-example
  draw-3 pilot, re-evaluate label variance, then scale to the §9 target.

### Pilot acceptance gate (re-run this evaluation on the pilot)

The pilot passes and we proceed to training only if:
- [ ] `move_index` distribution is **spread**, not concentrated on one value
      (the §3 defect is gone)
- [ ] Error rate < 10% (P1) — or, if not yet, errors carry diagnostic detail
- [ ] Each row has `gameSeed` / `gameId` / `turnIndex`; ≥ ~100 independent games
- [ ] Decision-type mix roughly tracks the legal-move mix (not ~100% draw)
- [ ] Response JSON still parses 100% with the 3 required keys
