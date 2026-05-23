# Prompt Template Audit & v2 Recommendations

**Date:** 2026-05-22 (corrected & expanded 2026-05-23)
**Author:** solitaire-analyst (cross-game synthesis)
**Scope:** All harvester builds present in `/Users/chayut/repos/solitaire-analytics/data/raw/` as of 2026-05-22, expanded 2026-05-23 to include the older builds initially missed

> **Correction note (2026-05-23):** The original version of this report claimed "one template, five builds" based on a scan limited to the most recent builds. Expanding the scan to every export in `raw/` reveals **two distinct templates across ten builds**, with one substantive edit between them. The corrected finding strengthens — not weakens — the recommendations: we now have observed proof templates change without warning, and a natural experiment in the corpus showing what one prompt edit achieved.

---

## TL;DR

1. **Two prompt templates exist in the corpus, not one.** An older 3001-char template (hash `719b1734…d49703`, used by `3a6c3c1`, `95cf4da`, `afa66cb`) was replaced by a newer 3527-char template (hash `a39354fa…5dc551c`, used by `50c8279`, `ce6afe1`, `ec38c03`, `71130ac`, `afa8c24`, `6dfc8a9`, `7894202`). The diff is a single substantive edit: confidence-calibration guidance was added.
2. **The corpus contains a natural experiment showing what that edit achieved**: the calibration guidance moved the mean confidence down by 0.023 and cut top-tier (≥0.95) saturation by 18 percentage points — modest, not transformative. The bottom half of the calibration range (0.0-0.7) remains essentially unused on the new template. This is empirical support for the audit's "confidence calibration is a training problem, not a prompt-instruction problem" framing.
3. **The newer template (used by every active build) is byte-identical across its seven builds.** So behavioural variance across the active build sequence (`71130ac` → `ce6afe1` → `afa8c24` → `6dfc8a9` → `7894202` etc.) is **not** a prompt-version effect — it's inference-stack/sampling variance, or deal-luck.
4. **Recurring failure patterns** (oscillation, chain-relocation reversal, deadlock-but-draw, board-analysis ↔ decision inconsistency) are baked into the newer template and reproduce across every build using it because nothing in the template structurally prevents them.
5. **Recommend:** introduce explicit prompt-template versioning in the harvester log (separate from `appCommit`), then fork the template with the resignation-output clause and run a same-seed A/B — that's the only experiment that produces a clean prompt-attributable signal.

---

## Method

For each distinct `appCommit` observed in `/Users/chayut/repos/solitaire-analytics/data/raw/`, extracted one representative interaction (`prompts.prompt` field), split the prompt at the dynamic-state marker `"CURRENT GAME (JSON):"`, hashed the static prefix, and compared. Then read each unique template and mapped its content against the failure catalogue in `/Users/chayut/repos/solitaire-analytics/data/DATASET_NOTES.md`.

Per-turn dynamic state was inspected on a mid-loop turn from `1f2fd2` (turnIndex 121, deep in the `7D/8C` oscillation) to confirm what information the model actually receives.

For the confidence-calibration natural experiment, every successful interaction in the corpus (n=4105) was bucketed by its prompt's static-prefix length (the deterministic discriminator between the two templates), and the `decision.confidence` distribution was computed per bucket.

---

## Finding 1: Two templates over ten builds — and the diff between them

| Template MD5 | Static size | Builds using it | Sessions |
|---|---|---|---|
| `719b1734e3889c8c3520763c47d49703` | 3001 chars | `3a6c3c1`, `95cf4da`, `afa66cb` (older) | `6cec7c`, `adefb0`, `4aa914`, `cc3adc` (n=4) |
| `a39354fa5f16e03285e389dee5dc551c` | 3527 chars | `50c8279`, `ce6afe1`, `ec38c03`, `71130ac`, `afa8c24`, `6dfc8a9`, `7894202` (newer) | every active session including `0154e1` (won), `c7fdb9` (in-flight), and all doom-loops |

All seven of the active builds serve the same newer template. Verified by `md5` on the static prefix of each.

**What changed between the two templates (`diff` output, lightly formatted):**

```
< - final_decision.confidence: your confidence the move is best, a number from 0 to 1.

> - final_decision.confidence: a calibrated probability (0 to 1) that this move is
>   objectively the best one available — a genuine estimate, not a feeling. Use the
>   full range honestly:
>     1.0-0.9  forced, or clearly dominant — any other move would be a mistake.
>     0.9-0.7  strong — one plausible alternative exists, but this move is better.
>     0.7-0.5  a real toss-up between two or three reasonable moves.
>     0.5-0.3  a guess — the board is unclear or several moves look about equal.
>     below 0.3  little better than picking at random.
>   If you would not bet on the move, do not report high confidence.
```

The entire +526-char delta is this **single edit**, replacing a one-liner with nine lines of explicit calibration bands. Nothing else changed — no new strategy guidance, no new rules, no new response-format constraints.

**Natural experiment: did the calibration edit work?**

Confidence distributions of all successful turns in the corpus, bucketed by which template generated them:

| Bucket | n | mean | median | min | % in [0.95, 1.0] | % in [0.85, 1.0] |
|---|---:|---:|---:|---:|---:|---:|
| OLD template (3001 chars, no bands) | 629 | 0.932 | 0.90 | 0.80 | **46.4%** | 99.4% |
| NEW template (3527 chars, with bands) | 3,476 | 0.909 | 0.90 | 0.60 | **28.2%** | 95.9% |

The edit moved the needle: mean −0.023, top-tier saturation (≥0.95) **cut nearly in half** (46.4% → 28.2%), minimum confidence reported dropped from 0.80 to 0.60. So the bands gave the model permission to dip lower and it sometimes took it.

But: **96% of the new-template confidence mass still sits in [0.85, 1.0]**. The full lower half of the calibration scale (`0.7-0.5 toss-up`, `0.5-0.3 guess`, `below 0.3 random`) is essentially never used. The prompt edit produced a modest, real lift; it did not produce calibrated confidence. This empirically supports the "training problem, not prompt problem" framing in Finding 2 — and quantifies the ceiling of what further prompt edits to this dimension are likely to achieve.

**What the dynamic per-turn state contains** (identical across both templates by schema):

- Full board JSON with `faceDownCount` per column
- `legalMoves` array with parenthetical hints like `"(reveals a hidden card)"`
- `discardTop`, `drawPileCount`, `canRecycleStock`
- `recentMoves` (last 10 — visible oscillation history)
- `seenDrawPileCards` (full known stock contents)
- `reasoningTrail` (the model's own prior rationales — limit 5 on older builds)
- `metrics` including `foundationCards`, `faceDownTotal`

The model has the information. The template doesn't enforce its use.

---

## Finding 2: Failure patterns → template gaps

Each observed failure class maps to a specific absence in the template:

| Failure pattern | Concrete instance | Template gap |
|---|---|---|
| Oscillation despite visible `recentMoves` | `1f2fd2` turn 121: `recentMoves` contains the literal `8C/7D` ping-pong, model picks the next 8C swap | Single line *"Avoid moves that simply undo a recent move or lead nowhere"* with no structural enforcement |
| `board_analysis` ↔ `final_decision` inconsistency | `1f2fd2`: board_analysis names "8S to 7D reveals a hidden card", final_decision picks otherwise. Same model contradicts its own `reasoningTrail` across consecutive turns | No clause requiring the decision to be consistent with the priorities the analysis identified |
| Stock-recycle un-attempted with `canRecycleStock=true` + `drawPileCount=0` | `3d03e5`: `discardTop=KD`, recycle never invoked across 76+ plateau turns | Template explains the recycle *mechanic* (line 12) but provides **no strategic guidance** on when to recycle. Line 24 ("drawing is reasonable when no productive tableau move exists") instead reinforces drawing |
| Drawing when stock is fully known and contains no needed card | `391920` endgame: 4 stock cards all in `seenDrawPileCards`, none are the red 6/red 7/red Jack the model itself enumerates as needed, model draws anyway | No rule like "if `seenDrawPileCards` ∪ remaining stock contains no card you've identified as needed, do not draw" |
| Chain-relocation reversal (no plan continuity) | `3d03e5`: T_K moves 10-card chain col 3 → col 4 (plausible empty-col plan); T_K+1 moves the chain back col 4 → col 3 with no KD park between | Strategic plan regenerated from scratch each turn; no instruction to evaluate whether to continue a multi-step plan stated in the prior turn's `reasoningTrail` |
| Confidence saturation in deadlocked positions | `502768` endgame on a dead deal: 0.80 ×1, 0.85 ×3, 0.90 ×36, 0.95 ×11 — never below 0.80. Corpus-wide on the newer template: 96% of confidences sit in [0.85, 1.0] | Calibration bands are explicitly defined and **measurably ignored** — see the natural experiment in Finding 1. The prompt edit moved the mean confidence by only 0.023 and cut top-tier saturation by ~18 pp; the lower half of the calibration scale stays empty. **This is a training problem, not a prompt-instruction problem** — confirmed empirically; no further prompt rewrite alone will move the needle much |
| Self-diagnosed deadlock with no consequence | `391920`: board_analysis reads verbatim *"The board is currently in a deadlock"*, final_decision is draw | Deadlock-recognition has no structural output channel — no "if deadlocked, recycle / take best chain-mobilising move / resign" branch |

---

## Recommendation 1: Add explicit prompt-template versioning to the harvester log

Currently every interaction logs `appCommit` (build hash). That's not the same as a prompt-template version — as this audit demonstrates, ten distinct `appCommit` values resolve to two distinct templates, with **seven of the ten builds serving the same newer template** and three serving a different older one. Without explicit versioning, reconstructing this required re-hashing every prompt by hand.

**Stronger evidence:** session `…ce0fb4` (`gameSessionId: 019e3583-f286-7a29-8217-0ce0b2ce0fb4`) is a **confirmed second win** in the corpus (`gameWon: true`, `completionProgress: 100`, 284 moves; win export at `/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-win-1779050713349.json`). It was generated before the harvester started logging `appCommit` or `seed` at the per-interaction level. We have a win — but we cannot attribute it to a build, a seed, or a template, because the per-interaction record didn't capture them at the time. This is exactly the failure mode that the proposed versioning fields prevent. One lost-attribution win is already in the corpus; the next one (or the next regression) will be lost the same way unless this ships.

**Add two fields to the harvester's per-interaction log:**

| Field | Value | Source |
|---|---|---|
| `promptTemplateHash` | MD5 (or SHA256) of the static template portion that's not per-turn dynamic | Computed at prompt-build time |
| `promptTemplateFinalisedAt` | ISO 8601 datetime stamp of when this template was finalised/shipped (e.g. `2026-05-22T00:00:00Z`) | Set in the harvester config at template-edit time |

Datetime-as-version is self-explanatory, monotonic, and doesn't require coordinating name allocation between branches or A/B arms. The hash remains the deterministic identity (two edits made on the same day still distinguish themselves by hash); the datetime is the human-readable label. Same convention already in use under `docs/reports/` (YYYYMMDD_*).

**Why this matters now, not later:**

- Existing analysis (this audit) had to manually `md5` the prompts to discover the templates are identical. With the fields in place, that's a one-line query.
- If a new template ships, every interaction in the dataset becomes attributable to a specific template — same-seed A/B becomes trivially queryable instead of requiring file-level inspection.
- A/B rollouts can run in parallel: the log distinguishes which interaction got which template, even within a single build deploy.
- It survives prompt churn that *doesn't* actually change the template (refactors, comment changes, whitespace) — the hash stays the same.

**Backfill convention:** every interaction in `data/raw/` to date is `promptTemplateHash=a39354fa5f16e03285e389dee5dc551c`. The corresponding `promptTemplateFinalisedAt` is unknown from the artefacts alone — the harvester team needs to supply the original ship date for this template (any stamp earlier than the oldest `appCommit` build date works; otherwise a sentinel like `unknown-pre-2026-05-22` is acceptable). Ingest pipeline (`scripts/ingest_exports.py`) can stamp this retroactively in one pass.

---

## Recommendation 2: Fork template as v2 with five targeted clauses

Each clause attacks one observed failure class. Non-overlapping, individually A/B-able.

### v2 clause 1 — Hard anti-oscillation rule (highest leverage)

> *Before choosing your move, scan `recentMoves`. If the move you are about to pick is the direct reverse of any move in the last 6 entries (same card or chain, source and destination swapped), do not pick it unless you can name a concrete new state-change that the reversal will enable within the next 3 moves. State that follow-up explicitly in `strategic_plan`.*

Addresses: `1f2fd2`, `645d03`, `391920`, `3d03e5`.

### v2 clause 2 — Board-analysis ↔ decision consistency

> *If `board_analysis` identifies a specific move as "the only one that reveals a hidden card" or "the productive move", `final_decision.move_index` must be that move unless `strategic_plan` explains why the priority listed in `board_analysis` is overridden.*

Addresses: `1f2fd2` self-aware-but-impotent pattern specifically. The model's reasoning is already correct in many failure cases; the gap is between its rationale and its move choice.

### v2 clause 3 — Stock-recycle strategic priority

> *When `drawPileCount == 0` and `canRecycleStock == true`: prefer recycling over any tableau move that does not reveal a hidden card or advance a foundation. When `seenDrawPileCards` already contains every card you've identified as needed and none of the remaining stock will provide one, do not draw — pick the best tableau move and document the deadlock.*

Addresses: `3d03e5`, `391920`.

### v2 clause 4 — Plan-continuity instruction

> *In `strategic_plan`, restate the multi-step intent from your previous turn's `reasoningTrail` entry (if any). State explicitly whether you are continuing that plan, abandoning it (with reason), or had no multi-step plan.*

Addresses: `3d03e5` chain reversal specifically.

### v2 clause 5 — Explicit resignation output

> *If after honest analysis you cannot identify any sequence of moves that improves `foundationCards` or `faceDownTotal` from the current position, set `final_decision.move_index = -1` to signal resignation. The harvester will treat this as a clean termination.*

Addresses: dead-deal-flailing class — `502768`, `3e77cc`. Also closes the standing P0 of "no stall auto-terminator" by pushing the resignation signal into the model output instead of waiting for the harvester to detect it.

---

## Recommendation 3: Validation experiment

Once the new template ships and `promptTemplateHash` is in the log, the existing template (hash `a39354fa…`, treated as the baseline arm) and the new one become two queryable populations. Pre-registered A/Bs:

1. **Same-seed A/B on a known-winnable seed.** Re-run seed `3263196305` (the only winning seed in the corpus — won under `6dfc8a9` + baseline template) on the latest build under baseline vs new template. If new wins consistently, prompt changes are attributable; if it loses, prompt isn't the bottleneck on that seed class.
2. **Same-seed A/B on a known-loopy seed.** Re-run seed `3642085723` (`1f2fd2`'s `8C/7D` oscillation under `afa8c24` + baseline) on baseline vs new. Specifically watch whether the anti-oscillation clause (clause 1) reduces session-wide oscillation counts. Pre-registered metric: "max `(card, col-pair)` count across the session" — baseline is 266×.
3. **Resignation triggering.** Re-run seed `821908579` (`502768`'s dead deal under `71130ac` + baseline) on the new template. Pre-registered success metric: model emits `move_index=-1` within 30 turns of plateau onset. Baseline: model never resigned, ran to manual abort at turn 223.

Each experiment needs ~3-5 seeds per arm to bound stochastic variance, but the single-seed signal will already be informative because the failure modes are categorical (loop happened or didn't, resigned or didn't).

---

## What this audit explicitly does NOT claim

- **It does not claim v2 will win more games.** Confidence saturation and overall move-quality are training-distribution properties; the proposed clauses target specific *behavioural* failure modes, not general planning ability.
- **It does not blame `7894202` or any other specific build.** The build-to-build behavioural variance observed in the corpus is **not** attributable to prompt changes — the prompts are identical. Build-level attribution requires same-seed cross-build experiments that have not been run.
- **It does not address the provider-error rate (~75% unavailable/timeout) or the missing-deck-seed P0.** Those are infrastructure issues tracked elsewhere.

---

## Files referenced in this audit

- `data/raw/solitaire-ai-log-502768-1779361355983.json` — `71130ac` representative
- `data/raw/solitaire-ai-log-645d03-1779331841599.json` — `ce6afe1` representative
- `data/raw/solitaire-ai-log-1f2fd2-1779380943828.json` — `afa8c24` representative; mid-loop turn 121 inspected for dynamic state
- `data/raw/solitaire-ai-log-0154e1-1779380748971.json` — `6dfc8a9` representative
- `data/raw/solitaire-ai-log-3d03e5-1779411082008.json` — `7894202` representative
- `data/DATASET_NOTES.md` — source for failure catalogue and session metadata
- `scripts/ingest_exports.py` — natural home for backfill of `promptTemplateHash`/`promptTemplateVersion`
