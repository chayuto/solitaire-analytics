# Next change request to harvester team

**Date:** 2026-05-26
**From:** Chayut / dataset side
**Re:** This cycle, three changes bundled into one prompt template bump:
1. Ship the standing **resign output** (now with smoking-gun evidence)
2. Four small **prompt hygiene** edits to the static header
3. Adopt mandatory **semantic prompt versioning** going forward

**Companion documents:**
- `docs/reports/20260524_harvester_team_layout_ask.md` (last cycle, shipped as `hybrid-v1`)
- `data/DATASET_NOTES.md` "Cross-version teacher benchmarks" section (corpus evidence for what hybrid-v1 did and did not fix)
- `data/DATASET_NOTES.md` "Known doom-loop sessions" entry for session `adf71b` / seed `114946100` (the smoking gun for resign)

---

## TL;DR

1. **Ship resign.** Standing ask since 2026-05-23. Session `adf71b` (build `de7dc06`, hybrid-v1) ran 40 turns on a 3-card oscillation, with the AI explicitly writing *"The board is currently in a deadlock... the only productive action is to draw from the stock"* at turn 209 and drawing because it had no other valve. Pyksolve enumeration of all 720 face-down perms confirms the board was winnable; this is pure behavioural failure. Resign output (`move_index: -1`, `outcome: resigned`) is the action-space fix.
2. **Three prompt-hygiene edits to the static header.** Drop the confidence-calibration bands block (saturated 0.93 regardless), move the `NOTATION:` line to the static rules block (re-rendered every turn for no information), and add a one-line `NEXT NEEDED:` foundation summary (currently re-derived in every reasoning trace). Edit 4 (truncate `PRIOR REASONING` to move text only) was bench-tested 2026-05-26 and HELD this cycle: clean SHIP on the v1.1 deployed student but -10pp teacher-match on the Gemma 4 E2B untuned v2 ship target. Re-introduce after v2 LoRA is trained and shown robust. Net token budget: about 100 to 150 lighter per prompt this cycle.
3. **Adopt semantic prompt versioning.** Add a `promptTemplateVersion` field (e.g. `hybrid-v1.1`, `hybrid-v2`) that bumps with every visible-to-the-model change. The existing `promptTemplateHash` keeps fine-grained truth; the semantic version is what makes cross-version analysis tractable.

**Not asking this cycle:** state-repetition annotation on legal moves. Scoped and pre-validated on our side, but deferring one cycle so we can attribute any post-resign win-rate change cleanly.

---

## Status of standing asks

| | Ask | Status | Notes |
|---|---|---|---|
| 1 | Per-interaction `promptTemplateHash` + `promptTemplateFinalisedAt` | shipped | Working. Build attribution unblocked. |
| 2 | Resignation output (`move_index: -1` to `outcome: resigned`) | proposed (this cycle) | Now backed by adf71b smoking gun. Detailed below. |
| 3 | Same-seed cross-build experiment on seed `3263196305` | resolved | New session `010e01` on build `de7dc06` (hybrid-v1) won 52/52 in 170 moves, paired with the prior baseline-arm session `1abf26` on build `6dfc8a9` (json-format) that reached 51/52. Cross-version teacher behaviour now documented in `data/DATASET_NOTES.md`. |
| 4 | Hybrid layout for `CURRENT GAME` block | shipped (as `hybrid-v1`) | Working. Confirmed compact and parseable. |
| NEW | Resign output (re-raised this cycle) | proposed | See section below. |
| NEW | Prompt hygiene bundle (3 edits this cycle; edit 4 held) | proposed | See section below. Edit 4 deferred per local bench. |
| NEW | Semantic prompt versioning | proposed | See section below. |

---

## This cycle, ask 1: resign output

### Behaviour

The advisor may choose to return `final_decision.move_index = -1` instead of an index into the LEGAL MOVES list. The harness interprets `-1` as a resignation and records the interaction with `outcome: "resigned"` (new outcome value alongside `success` and `error`). No tableau move is applied; the session can be terminated cleanly by the operator or by an auto-terminator rule.

### Spec

**Prompt-side addition (one extra bullet under the existing `final_decision.move_index` description):**

```
- final_decision.move_index: the [index] of your chosen move from the LEGAL MOVES
  block, OR -1 to RESIGN. Resign only when no legal move can productively advance
  the game, drawing has been exhausted, and you would not bet on any of the
  available moves to recover. Resign is final and ends the session.
```

**Output-side handling (harness):**

- If `move_index == -1`: do not apply a move. Record interaction with `outcome: "resigned"`, set `decision.moveType: "resign"`, `decision.moveIndex: -1`. Leave `confidence` and the analysis text as the model emitted them.
- The session terminates after a resign. No retry, no follow-up call.
- Existing telemetry fields (durationMs, tokens, etc.) populated as normal.

**Schema addition:**

- `outcome` enum gains the value `"resigned"`.
- All other fields unchanged.

### Why now

Session `019e6136-cf0d-74aa-a3c4-993e6cadf71b` (seed `114946100`, build `de7dc06`) is the smoking gun:

- 40-turn plateau on `(foundationCards=13, faceDownTotal=3)`, turn 169 to 209.
- 3-card oscillation `6D-5C-4D` between col 2 and col 6, visible in `RECENT MOVES`.
- Pyksolve enumeration of all `6! = 720` permutations of the 6 unknown cards: 720/720 solvable. The board is winnable for every conceivable face-down assignment.
- Turn-209 `boardAnalysis` (the AI's own words): *"The board is currently in a deadlock. To reveal the hidden cards in column 6 (??, ??, 7C), a red 8 is needed. The available red 8s are 8D (buried in column 2) and 8H (previously seen in the waste)... Since no legal tableau move reveals a hidden card or breaks the red 8/red 9 deadlock, the only productive action is to draw from the stock."*
- Stock had 3 cards left and was non-recyclable. The agent had no out except picking one of the oscillating shuffles or burning one of the last 3 stock cards.

The agent was self-aware about the deadlock but had no action available that matched its analysis. Resign is the missing action.

### What success looks like

- Resign usage in the next harvest: non-zero on incomplete sessions where `recentMoves` shows oscillation and `faceDownTotal` is unchanged for tens of turns.
- Median stall length on doom-loop sessions drops (currently 30 to 85 turns; with resign, expect the first 5-10 turns of self-aware-stall to end with a resign rather than continuing to thrash).
- Win-rate on solvable seeds unchanged (resign should not fire on positions the agent can actually win from). If win-rate drops on seeds we have prior wins for (`3263196305`, `2967897202`), that's a regression signal that resign is firing too eagerly and we tighten the prompt language.

### Risk

Near zero. The change adds a new option but does not modify any existing option. The trained student model has never seen training data that uses `move_index: -1`, so it will not accidentally produce one. Risk path: the 31B teacher uses resign too eagerly. Mitigation: the prompt language above scopes resign to "no productive move AND drawing exhausted AND no recovery bet". Monitor first 10 sessions for false-positive resigns.

---

## This cycle, ask 2: four prompt hygiene edits

These are all in the static header (the part before `CURRENT GAME:`). All four are token-budget improvements with no behavioural risk; the savings buy back the ~30 tokens we add for resign and `NEXT NEEDED` and leave a net negative.

### Edit 1: drop the confidence-calibration bands block

**Current text:**

```
- final_decision.confidence: a calibrated probability (0 to 1) that this move is
  objectively the best one available, a genuine estimate, not a feeling. Use the
  full range honestly:
    1.0-0.9  forced, or clearly dominant, any other move would be a mistake.
    0.9-0.7  strong, one plausible alternative exists, but this move is better.
    0.7-0.5  a real toss-up between two or three reasonable moves.
    0.5-0.3  a guess, the board is unclear or several moves look about equal.
    below 0.3  little better than picking at random.
  If you would not bet on the move, do not report high confidence.
```

**Replace with:**

```
- final_decision.confidence: a probability estimate (0 to 1) that this is the
  best move. Use the full range honestly.
```

**Evidence:** mean confidence is `0.93` across every session in the corpus, winning and losing. Never below `0.8` in either prompt format. The detailed bands have not influenced calibration in any measurable way. They are dead static weight (~100 tokens per call, ~120 turns per session at the average, ~12K tokens per session wasted).

### Edit 2: move the `NOTATION:` line to the static rules block

**Currently:** the first line under `CURRENT GAME:` every turn:

```
NOTATION: rank+suit (A 2-9 T J Q K; H D C S). ?? = face-down. In each column the top of the stack is the rightmost card.
```

**Move it to:** the static header, as a final line of the `KLONDIKE SOLITAIRE RULES (this variant):` block, with the prefix `NOTATION:` retained. Drop it entirely from the per-turn `CURRENT GAME:` block.

**Evidence:** the line never changes. It carries zero per-turn information. Repeating it on every call costs roughly 35 tokens per turn for nothing.

### Edit 3: add a `NEXT NEEDED:` foundation summary line

**Add one line directly under FOUNDATIONS:**

```
FOUNDATIONS:   H: QH   D: 6D   C: 8C   S: 6S
NEXT NEEDED:   H: KH   D: 7D   C: 9C   S: 7S
```

For an empty suit, render `NEXT NEEDED` as the Ace (`H: AH`). For a full suit (King in foundation), render `done`.

**Evidence:** every reasoning trace I have read computes this manually. Sample from session 010e01 turn 80: *"The diamond foundation is at 6D so I need 7D next; the spades foundation is at 6S so I need 7S; both can advance from current column tops."* Pre-computing this one line saves ~50 tokens of derivation per turn and reduces error cases where the model picks the wrong "next needed" for one suit while reasoning about another.

### Edit 4: truncate `PRIOR REASONING` entries to move text only

**HELD this cycle, based on bench evidence below.**

**Original proposal:** drop the multi-paragraph `why:` text from each
`PRIOR REASONING` entry, keep only the `move:` header line. Reasoning:
~30% of prompt length, agent-echo, encourages over-commitment to a stale
prior plan.

**Bench results (20-state Phase 1.5, 2026-05-26):**

| arm | match% control to truncated | conf median delta | verdict |
|---|---:|---:|---|
| `gemma-3n + adapters_t5_at750` (v1.1 deployed) | 55 to 55 (no change) | 0.00 | SHIP |
| `Gemma 4 E2B untuned` (v2 ship target per `v2-distillation-teacher-doom-loop` memory) | 60 to 50 (-10pp) | 0.00 | REVISE |

JSON validity and move-parse rate both at 100% in every cell; confidence
median identical at 0.95. The split is on the teacher-match soft
guardrail (5 pp threshold).

**Interpretation:** the trained LoRA student has internalised the structure
and is robust to dropping the `why:` text. The untuned Gemma 4 base
genuinely uses the rationale to commit to multi-turn plans; when stripped,
4 mid-game / oscillation states flip from agree to disagree (2 also flip
the other way; net -2 of 20). -10pp on N=20 is at roughly 1 standard
deviation of binomial noise so borderline-significant, but the sign is
right and the mechanism is consistent.

**Phasing decision:** hold this edit until a v2 LoRA is trained and shown
robust to the truncation the same way the v1.1 LoRA is. Shipping the edit
now would regress the v2 ship target. The other three hygiene edits
(drop confidence-bands block, move NOTATION line, add NEXT NEEDED line)
are independent and ship in this cycle.

**Bench artefacts:** `gemma4_finetune/bench_prior_reasoning_truncation/v1_iter750.json` and
`gemma4_finetune/bench_prior_reasoning_truncation/gemma4_untuned.json`,
produced by `gemma4_finetune/bench_prior_reasoning_truncation.py`.

---

## This cycle, ask 3: semantic prompt versioning

### What we need

A short, human-readable version tag on every prompt template, alongside the existing `promptTemplateHash`. Carried per interaction in the export, just like the hash.

### Spec

Add one field to each interaction record:

```json
"promptTemplateVersion": "hybrid-v1.1"
```

Versioning convention:

- `hybrid-v1.0`: the current shipped template (the one introduced by build `de7dc06`).
- `hybrid-v1.1`, `hybrid-v1.2`, etc.: minor edits that preserve the same layout (token-budget tweaks, added small fields, reordered sections without restructure).
- `hybrid-v2.0`: major restructure that changes the rendering shape (new section, removed section, changed numbering convention).

Bump rule: any visible-to-the-model change bumps either the minor (small, local edits) or the major (layout-level changes). `promptTemplateHash` continues to capture all textual changes; the semantic version is the human-meaningful coarse identifier.

### Why

- The four hygiene edits in this cycle are about to push the template through one version step. The next likely change (state-repetition annotation, possibly in cycle N+1) is another step. Without a semantic version, we identify versions by an opaque sha256 in conversation, which slows every analysis cycle.
- The DATASET_NOTES cross-version comparison for seed `3263196305` already had to refer to two prompt formats by build commit (`6dfc8a9` versus `de7dc06`). That conflates the prompt version with the build version, which is wrong (multiple prompt-only edits could ship in a single build). A semantic version on the template itself decouples these.
- HF dataset consumers and any future external analysts will want this. The data card at `data/publish/README.md` documents the corpus and would gain a clean version table.

### What we will do on our side

- Add `promptTemplateVersion` to the corpus schema in `scripts/ingest_exports.py`.
- Surface it as a filter dimension in `data/SUMMARY.md` (rows per version).
- Use it as the primary key in any future cross-version benchmark table.
- Maintain a short changelog of versions and their meaningful differences, either in the harvester repo or in `data/DATASET_NOTES.md` (your preference).

### Backfill

Not required. Existing rows can carry `promptTemplateVersion: null` and we infer them from `appCommit` and `promptTemplateHash` for archival purposes. Going forward, the field is populated on every new export.

---

## What we are NOT asking this cycle

**State-repetition annotation on legal moves.** Scoped and ready: harness pre-computes a canonical signature of the OBSERVABLE board state (foundations, waste top, stock count, tableau face-up + face-down counts per column; deliberately no face-down identities) after each candidate legal move, compares against a rolling buffer of the last N board signatures (default N=10), and appends a parenthetical such as `(returns to turn 207 board)` to any legal-move line that would re-enter a recent state. Plain text only; the signature is internal data, never exposed to the model.

Deferring one cycle for two reasons:

1. Clean attribution. If we ship resign and state-repetition annotation together and the stall-rate moves, we cannot say which fix did the work. Resign alone is the cleaner experiment.
2. Honest expectation: the doom-loop in `adf71b` was not an awareness failure (the AI was self-aware about the deadlock in prose). It was an action-space failure. Resign addresses the binding constraint; state-repetition annotation addresses an information-side problem that may already be solved by RECENT MOVES being visible to the agent.

If post-resign harvest shows stall-rates dropping materially, state-repetition annotation becomes a "nice to have" rather than a "needed". If stall-rates do not drop, state-repetition annotation is the natural next cycle.

---

## Implementation checklist for harvester team

For this cycle, in one prompt template bump (`hybrid-v1.0` to `hybrid-v1.1`):

- [ ] Add `move_index: -1` resign path to prompt instruction block and harness handling.
- [ ] Drop the confidence-calibration bands block; replace with the one-line version.
- [ ] Move `NOTATION:` line from per-turn block to static rules block.
- [ ] Add `NEXT NEEDED:` line under FOUNDATIONS.
- [ ] Add `promptTemplateVersion: "hybrid-v1.1"` to every exported interaction.
- [ ] Bump `promptTemplateHash` (will happen automatically given the textual changes).

**Deferred:** truncate `PRIOR REASONING` to move text only. Bench held it
this cycle (see section above). Re-add to a future cycle bump once v2
LoRA is shown robust.

Schema additions:

- [ ] `outcome` enum gains `"resigned"`.
- [ ] Per-interaction `promptTemplateVersion: string`.

---

## What we will report back

After the new template has had one harvest cycle:

- Resign usage rate and the seed/session ids where it fired.
- Stall-length distribution on incomplete sessions, pre versus post.
- Win-rate on the two locked benchmark seeds (`3263196305`, `2967897202`) if they are re-run.
- 20-state bench results for the LoRA student on the new template (hard gate: JSON validity 100 percent; soft gate: match-to-teacher within 5 percentage points).
- Token budget comparison (mean per call) before and after.
