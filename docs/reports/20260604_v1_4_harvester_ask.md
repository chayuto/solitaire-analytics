# v1.4 Prompt Ask to the Harvester Team

**Date:** 2026-06-04
**From:** solitaire-analytics (corpus analysis side)
**To:** harvester team (prompt + state renderer)
**Scope:** prompting and state-rendering system only. Out of scope: model, inference config, training, auto-terminate harness.
**Target model:** `gemma-4-31b-it` only (the 26B MoE is a comparison cohort, not a fidelity target).
**Status:** v1.4 is two low-risk text edits to one block, shipped together as `hybrid-v1.4`. The renderer ideas (F1, F2) are tracked separately.

> **Versioning hygiene:** `promptLayoutVersion` is bumped in lockstep with `promptTemplateVersion` (test-enforced in `renderContext.test.ts`); the major part of the semver carries the original layout-shape meaning. v1.4 ships as the pair `hybrid-v1.4` on both stamps. Record it as one row, not two, and identify the variant by version rather than by reversing the templateHash.

## TL;DR

The v1.3 baseline is now large enough to act on: 23 sessions on `gemma-4-31b-it`, prompt `hybrid-v1.3` (templateHash `7d9ecda4`), every non-win adjudicated with the fixed engine solver. v1.4 keeps to the original bar: fix existing prompt errors, text only, test small.

**v1.4 (ship together, text only, low regression):** two edits to the STRATEGY GUIDANCE block.
1. **Anchor the reveal phrases to the tag.** Bind every "expose a face-down card" phrase to the `(reveals a hidden card)` tag you already render. Closes the documented fake-reveal override.
2. **Fix the anti-undo horizon.** It says "last 5 moves" but RECENT MOVES renders 10. Widen it to "any move shown." We do not write the productive-return carve-out into the prose (that injects decision logic); that exception is F1's job, as computed state.

**Renderer follow-ups (code, not text), both outside v1.4.** F1, an anti-undo move tag, is the eventual clean replacement of the anti-undo rule (it expresses the exception as computed state, not prose). F2, a temporal progress signal for resign, also stays parked: it can convert a slow win into a resigned loss, so it waits until v1.4 lands and validates.

We deliberately did not touch the rest of the block (see "What we are not changing"). v1.4 is two lines on one block, and the only item with regression risk is parked.

## Why now: the measured v1.3 baseline

23 sessions, `gemma-4-31b-it`, prompt `hybrid-v1.3` (templateHash `7d9ecda4`). Outcome: 14 won, 1 correct resign (#30e5e5), 8 incomplete, 0 terminal-loss records. All 9 non-wins were re-run through the repo engine solver (`.claude/skills/solitaire-analyst/scripts/check_winnability.py`, `--solver engine`, the sound backend; the old pyksolve verdicts in the corpus are unreliable). They split:

- **6 of 9 are structurally dead boards** where a stall or resign is correct. The model resigned exactly one of them (#30e5e5). The other five thrashed or froze.
- **3 of 9 are genuinely winnable boards** the model stalled on behaviourally: #3fd319 (winnable 10/10), #783780 (8/10), #a1d118 (7/10).

The v1.4 text edits target the winnable bucket (the reveal and anti-undo wording). The dead bucket is the parked temporal signal's job.

One thing that is already correct in v1.3 and must not be undone: RECENT MOVES renders actions, not the model's own past rationales. Do not add a reasoning trail or prior-rationale block. Feeding this model family its own prior reasoning is the worst in-context failure mode, and the current layout avoids it.

## v1.4: the two text edits

One block changes. Current, verbatim:

```
STRATEGY GUIDANCE (heuristics, not absolute rules):
- If any legal move exposes a face-down tableau card, prefer such a move. When multiple legal moves expose a face-down card, prefer the one that exposes a card in the column with the most face-down cards remaining.
- Play Aces and 2s to the foundations promptly; they are rarely useful in the tableau.
- Be cautious sending higher cards to the foundations too early — they are sometimes needed to receive tableau cards.
- Do not empty a column unless you have a King ready to occupy it.
- Prefer exposing new cards and creating useful sequences over shuffling cards between columns with no gain.
- Drawing from the stock is the correct action when no legal tableau or foundation move exposes a face-down card or advances a foundation.
- Do not move a card to a tableau column it occupied in the last 5 moves shown in RECENT MOVES.
```

Proposed (changes are on bullets 1, 5, 6, and 7; the header and bullets 2, 3, 4 are unchanged):

```
STRATEGY GUIDANCE (heuristics, not absolute rules):
- If any legal move is tagged "(reveals a hidden card)", prefer such a move. When several are tagged, prefer the one in the column with the most face-down (??) cards remaining.
- Play Aces and 2s to the foundations promptly; they are rarely useful in the tableau.
- Be cautious sending higher cards to the foundations too early — they are sometimes needed to receive tableau cards.
- Do not empty a column unless you have a King ready to occupy it.
- Prefer a move tagged "(reveals a hidden card)" over a tableau move that is not tagged and only shuffles cards between columns with no gain.
- Drawing from the stock is the correct action when no legal move is tagged "(reveals a hidden card)" and no legal move advances a foundation.
- Do not return a card to a tableau column it occupied in any move shown in RECENT MOVES.
```

### Edit 1: anchor the reveal phrases to the tag (bullets 1, 5, 6)

The word "expose" is model-judged in three bullets. On #9b1c4a the model toggled a run back and forth with the rationale "the priority is to expose face-down cards," while the run rested on an already face-up card and exposed nothing. You already compute and render an accurate `(reveals a hidden card)` tag in LEGAL MOVES. Binding all three references to that tag removes the model's room to call a non-revealing move a reveal. Anchoring only the first bullet would let the same narrative re-attach to bullet 5, so all three change together. Pure text, binds to an existing signal, so it cannot lose a genuine reveal (those moves are tagged).

### Edit 2: fix the anti-undo horizon (bullet 7)

The rule says "last 5 moves," but RECENT MOVES renders up to 10. At #9b1c4a turn 163 the trap move was the exact reverse of a move at position 2 of the 10 rendered lines, outside the 5-window, so the rule never fired. Change "last 5 moves" to "any move shown in RECENT MOVES": no count, and self-synchronizing if the render length ever changes. This corrects an existing rule's broken reference; it does not add new decision logic.

We deliberately do not add the "unless it reveals or plays to a foundation" carve-out as prose. That would write more of the model's decision procedure into the prompt, which is the injection the project principle refuses (state and notation in the prompt, logic in the model). It is also low value here: in the observed stalls the candidate return moves reveal nothing, so the carve-out would rarely fire. The unconditional rule is safe in the meantime because the block is soft ("heuristics, not absolute rules") and Edit 1 makes the reveal tag salient, so the model can still take a genuinely revealing return when one exists, and the regression set is the safety net. The proper exception lives in F1 below, where "a productive return is simply not tagged" is computed state, not prose.

### What we are not changing, and why

This is the scope check. We are leaving alone:
- **The header.** It stays "heuristics, not absolute rules." All 14 wins happened under that framing. Hardening it into a strict numbered priority order (the old Fix #1 idea) is an unvalidated behavior change that risks overtriggering and regressing wins, and the Edit 2 exception already dissolves the reveal-versus-undo conflict, so a precedence ranking is not needed. If you want to test enforcement versus heuristic framing, that is a separate experiment, not v1.4.
- **Bullets 2, 3, 4.** No measured failure points at them. The "be cautious" foundation bullet is a soft heuristic we could tighten, but tightening it is unevidenced churn, so it waits.

## Renderer follow-ups: F1 (anti-undo tag), F2 (temporal signal)

Both came out of the same analysis and are code changes, so they are tracked apart from the v1.4 text edits and are not part of v1.4. F1 is the principled end state of the anti-undo fix; F2 carries resign risk and stays parked until v1.4 validates. Listed so the work is not lost.

### F1: render the anti-undo condition as a move tag (renderer; the principled completion of Edit 2)

Instead of the model applying the Edit 2 rule by scanning RECENT MOVES, tag the move in LEGAL MOVES, the same way you tag reveals:

```
  [1] tableau_to_tableau   Move 7H plus 1 more from column 7 to column 6 (undo with no progress)
```

Tag a move when it returns a card to a column it left within RECENT MOVES and it is not also a reveal or a foundation play. The guidance rule then reduces to "do not pick a move tagged that." This deletes the prose rule entirely (no horizon, no carve-out) and expresses the productive-return exception as computed state: a return that reveals or advances is simply not tagged. That is the form that honors the no-injection principle. Keep the wording observable (a return that uncovers an already face-up card), not a judgment.

v1.4 ships the Edit 2 horizon fix as text. F1 is a later renderer cycle that replaces the anti-undo rule outright with the tag, at which point the prose rule is deleted. Both before and after, the carve-out stays out of the prompt prose.

### F2: render a temporal progress signal for resign (renderer; carries the only real regression risk)

Add state the model can see each turn, for example on the PROGRESS line: turns since the foundation count last increased, and recycles since it last increased. This is the largest lever on the loss set: 6 of 9 losses are dead boards the model could not perceive as stuck because it sees only the last 10 moves. The resign action already works (#30e5e5), so the gap is perception.

Why this is parked, not shipped with v1.4: the same signal that helps the model resign a dead board can make it resign a recoverable slow win, and resign is final. #2c84bac05ad4 won after a 94-turn plateau; #3ced34aca45a won after a roughly 290-move oscillation. Surfacing "turns since progress" to those sessions could convert a win into a resigned loss. So F2 must ship only after v1.4 validates, be tested on the full regression set with explicit attention to premature resigns, and be rendered as state only, never as a "resign if turns > N" threshold. This is the one item where more is not safer.

## Acceptance criteria and eval seeds

Run the candidate prompt on `gemma-4-31b-it` against the same-seed sets we have v1.3 ground truth for. Reproduce a board with `solitaire.chayuto.com/?seed=<seed>`; adjudicate with the repo engine solver; read oscillation and plateau metrics from the standard corpus tooling. Shipping the two edits together means we attribute v1.4 as a unit rather than edit by edit; that is the accepted tradeoff for one ship.

**Regression set (must still win), 14 seeds:** 1388178981, 2003817730, 2044240526, 3123337720, 3263196305 (the anchor), 350743738, 3590201206, 3841057237, 405489085, 4161700176, 4221577640, 549440324, 601852437, 839179948.
- Pass: all 14 still win. This is the floor.

**Loop-break set (should win or stop stalling), 3 winnable-stall seeds:** 1965004236, 1514988667, 1792828001.
- Pass: a win, or at minimum no 25-turn flat plateau.

**Throughput, measured on the messy wins:** fewer exact reversals on the ordered move stream, shorter foundation plateaus, fewer moves to win. This is where the v1.4 edits are most likely to show, since the model already takes offered reveals about 90 percent of the time, so the headroom is the messy and lost tail, not the clean wins.

Ship to the full harvest when the regression set holds and the loop-break or throughput numbers improve. Expect a modest throughput gain and little movement on the win rate; that is the honest prediction, because two-thirds of the losses are the dead boards that only the parked F2 reaches.

## Scope guardrails

Out of scope for v1.4: model swaps, inference-config changes, training, the auto-terminate harness, and both renderer follow-ups (F1 and F2). Specifically, please do not:
- add a reasoning trail or any prior-model-rationale block to the prompt (keep RECENT MOVES as actions only);
- harden anti-undo into an unconditional prohibition above reveal (the model already declines non-revealing undos correctly; a harder unconditional rule is net-negative);
- change the "heuristics" header into an enforced priority order as part of v1.4 (separate experiment if wanted);
- inject decision thresholds, "resign if X" predicates, or "unless Y" carve-outs into the guidance prose. Conditional move logic belongs in a computed tag (F1), not in rules text. Render state and let the model decide.

## Appendix: the 9-loss adjudication (measured 2026-06-04)

| Session | Seed | Solver verdict | Behaviour |
|---|---|---|---|
| #30e5e5 | 770499954 | DEAD | resigned correctly (only one) |
| #f75866 | 3925117923 | DEAD 40/40 | froze, operator kill, no resign |
| #7b6318 | 3161115466 | DEAD 12/12 | 173-turn 3-card shuffle, no resign |
| #b2d946 | 1152037935 | DEAD 10/10 | declined a move, drew into a dead stock |
| #8a5d12 | 3208238335 | DEAD 10/10 | stuck endgame, no resign |
| #4a9fe1 | 811891845 | DEAD 10/10 | false resign (said unwinnable, never emitted it) |
| #3fd319 | 1965004236 | WINNABLE 10/10 | stalled, ignored a known recycle path |
| #783780 | 1514988667 | WINNABLE 8/10 | shuffled into a corner, could not find the line |
| #a1d118 | 1792828001 | WINNABLE 7/10 | reveal pass-up 27 percent |

Two findings from reading the as-shipped prompt directly (the full text is in every export's `prompt` field):
- The `(reveals a hidden card)` tag is real, rendered in LEGAL MOVES, and accurate, so Edit 1 binds to an existing signal. Confirmed by the harvester team against `packages/app/src/ai/context/describeMove.ts:61` (the tag) and `rulesPrimer.ts:43-50` (the block, byte-matching the quote above).
- At #783780 turn 504 the model declines the only tableau move correctly (it reveals nothing) and draws. So on that board the anti-undo rule is firing correctly and the residual is a planning gap, not a wording bug. The v1.4 edits will not fix that planning gap; only deeper work, or the parked F2 letting it resign when genuinely stuck, will.

Background and prior thinking: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260530_v1_4_harvester_side_notes.md` (the parked notes this supersedes) and the per-session entries in `/Users/chayut/repos/solitaire-analytics/data/DATASET_NOTES.md`.
