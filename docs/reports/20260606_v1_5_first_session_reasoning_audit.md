# v1.5 first-session reasoning audit: did the prompt change the behaviour?

**Date:** 2026-06-06 | **Session:** `#6eb393` (full `019e99f9-05a8-7eb3-9d21-76995f6eb393`) | **Model:** `gemma-4-31b-it` | **Build:** `6810750` | **Prompt:** `hybrid-v1.5` (templateHash `8a46ca22…`) | **Seed:** `4250754298` | **Outcome:** WIN (progress 100, 254 moves, recycle 2)

**N=1 caveat, read first.** This is one session, one seed, a win, at temperature 0.3. It is suggestive, not causal. It cannot separate the prompt change from seed difficulty or sampling noise. The clean test is a same-seed v1.4-vs-v1.5 A/B plus at least one stuck/losing board. Everything below is "what the first v1.5 win's reasoning looks like", not "what v1.5 does in general".

## What v1.5 changed (the thing under test)

Per `docs/reports/20260606_v1_5_harvester_ask.md`, both changes are confirmed present in the rendered prompt:
1. **Deleted** the STRATEGY GUIDANCE draw-directive ("Drawing from the stock is the correct action when no legal move is tagged ... and no legal move advances a foundation").
2. **Added** two PROGRESS counts: `turns since foundation grew`, `turns since a card was revealed`.

The hypothesis the ask made: removing the unconditional draw rule hands the draw decision back to the model, which (on a winnable line) will reason about whether a draw is worth it instead of drawing mechanically; the counts give it the "am I stuck" signal a human feels.

## Method

From the 145 successful interactions in `data/raw/solitaire-ai-log-6eb393-1780743062479.json`, ordered by `turnIndex`: classified each chosen move by type (reading the legal-move verb at `decision.moveIndex` in the prompt), and read `decision.reasoning` / `thinkingText` on the draw turns. Counted draw-rationale flavour (mentions of discover/unknown vs naming a specific upcoming card near draw/stock language) and explicit references to the new counts.

## Findings

**1. Draw rate is healthy: 23% (34 / 145).** Contrast the v1.4 dead-deal `#136236`, whose late game was a silent ~56% draw loop. Deleting the draw-directive did not make this model under-draw and lose; it drew when it had a reason to, recycled the stock twice, and finished. This is the single most important result: the deletion did not backfire on this board.

**2. Draws are mostly deterministic, not speculative: 25 / 34 (74%) name a specific target card or a concrete multi-step path.** This is the won-run signature from the post-recycle-discovery-rationale finding (won runs draw toward a named card; lost runs draw to "discover"). Verbatim:
- turn 145: "draw the remaining stock to trigger a recycle. Once the waste is recycled, 2C will become available. After 2C is played, the 3C, then the 4C, then the 4D."
- turn 208: "the immediate next draw is KD, the subsequent draw is 7D ... draw to get 7D" (reads the DRAW TIMELINE order precisely).
- turn 220: "draw the 7D, play it to the foundation, then the 8D, thereby uncovering the two hidden cards in column 5."

**3. It decides to draw from the state, not from a directive.** At turns 25 and 52 it first evaluates the tableau alternatives and rejects them ("Move [1] is a reversal of recent moves", "provides no immediate benefit or revelation of hidden cards") and only then draws. That reasoning step is exactly what the deleted directive used to short-circuit. This is the intended behavioural change, visible in the chain.

**4. It barely uses the new numeric counts.** Across 145 turns there are only ~1-2 explicit mentions of "turns since ...". The model is not citing the counts to drive decisions. So this win does not demonstrate the counts' intended value at all; their payoff is "perceive stuck -> stop / resign", which a win never exercises. The counts remain unvalidated.

**5. Oscillation was not eliminated.** Mid-game the session still churned (`3C/4D col 2 ↔ col 4` by window count), and the anti-undo bullet (still present) did not prevent it. The win came despite the churn, by breaking out into a closing foundation cascade (51 to 52, last card `KC`). v1.5 did not make the model loop-proof.

## Verdict

On this one winnable board, the v1.5 draw-directive deletion produced the intended reasoning shape: the model evaluates alternatives and draws toward named cards rather than mechanically, and it did not under-draw. The stall counts were essentially ignored and are unproven. None of this is causal at N=1.

## What would actually validate it

- Same-seed v1.4-vs-v1.5 on the slow-win stress seeds the ask flagged (3263196305, 2853966634, 601852437): does v1.5 still win them, or does the deletion make it resign/stall? (the load-bearing risk of the deletion).
- At least one structurally-stuck board under v1.5: does the model now read the rising `turns since foundation grew` count and stop/resign, or keep drawing to the cap as before? That is the only test of the counts.
- N>1 per seed to separate prompt effect from temp=0.3 noise.
