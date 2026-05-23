# 502768 follow-up export — no, it's worse, and you already knew

**File:** `/Users/chayut/Downloads/solitaire-ai-log-502768-1779361355983.json` (2.36 MB, 201 interactions, exported 11:02:35Z on commit `71130ac`)

## TL;DR

This is **more of the same session** (`019e4853-…-502768`, seed `821908579`) — turns 170–223 of the same dead deal we flagged earlier. The good news: turn 223 outcome is `aborted` — you killed it, which is exactly the right call. The bad news: in the ~3 hours between the previous export and this one, the model ground through ~53 more turns to move the foundation by **a single card** (5H → 6H), then deadlocked again on the identical structural problem. Nothing about model behaviour or board state improved.

## Same session, same dead deal, +1 card

| | previous 502768 view | this export |
|---|---|---|
| Session | `…502768` | `…502768` (same) |
| Seed | `821908579` | same |
| Build | `71130ac` | `71130ac` (same — pre-`6dfc8a9` 0154e1 harvester) |
| Foundations at last seen turn | H:5, D:0, C:3, S:3 (11 cards) | **H:6**, D:0, C:3, S:3 (12 cards) |
| Face-down total | 6 (col 4:1, col 5:3, col 7:2) | **6** (same distribution) |
| Completion progress | 21 | **23** |
| Move count | ~170 | 223 |
| Final outcome | `incomplete`, looping | `incomplete` — final turn `aborted` |

Net delta: **+1 foundation card** (6H finally played to hearts) over ~53 retried turns. Zero new face-down cards revealed. The structural obstruction is unchanged.

## Why it's still stuck (structural)

From the final `success` turn (turnIndex 222) board JSON:

- Foundations: `H:6H, D:null, C:3C, S:3S`. Ace of Diamonds **still missing**.
- Col 4 (1 face-down): top `6C` — needs a **red 7** to move. Only red 7s are 7H (buried 6 deep in col 2) and 7D (seen in waste cycle — but no black 8 available to chain to it).
- Col 5 (3 face-down): top `7S` — needs a **red 8** (8H or 8D) to move. Neither has ever appeared in the waste cycle.
- Col 7 (2 face-down): top `7C, 6D, 5C` — also needs a **red 8**.
- `seenDrawPileCards`: `["2D","7D","KS","JH","9C","9H"]` — the full known stock contents. **Neither red 8 is in the stock.** 8H and 8D are both among the 6 face-down cards.
- `recentMoves` shows the closed loop: `draw QD, draw TD, draw TS, draw 8S, draw 8S, draw TS, draw TD, draw QD` — same waste cards reappearing across recycles.

That confirms what we suspected on the previous look: this is a **dead deal**, not a behavioural failure. Both red 8s sit face-down beneath cards the AI can't legally move without a red 8. There is no recovery path the model could find — even a perfect oracle couldn't escape from this configuration without seeing the hidden cards.

## Label-quality readout (the bit that matters for distillation)

Of 201 interactions, ~74% are provider errors (skipping detail per your standing instruction). The 51 `success` decisions look like this:

- **Move-type distribution (chosen + applied):** draw_card 38, tableau_to_tableau 13, tableau_to_foundation 1, discard_to_tableau 0 (rounded — `movesApplied` is the ground truth). That's **75% draws** — the model has effectively collapsed to a single action on this board. No variance to learn from.
- **Confidence:** 0.80 ×1, 0.85 ×3, 0.90 ×36, 0.95 ×11. Mean ≈ 0.91, **never below 0.80** — the same saturation pattern your earlier MEMORY note already called out, on a board where the model is objectively stuck.
- **Decision quality:** the chain-of-thought on the final turns is actually clean and self-aware ("the board is in deadlock," "tableau moves are counter-productive," correctly enumerates which red 7s/8s are missing). The reasoning isn't bad — it's just that there's nothing to do. So these decisions are **defensibly correct labels on a doomed board**, but they're 38 near-duplicate "draw" exemplars that will skew a student model toward "when in doubt, draw."

## What this means for the dataset

The ingest pipeline's stall filter (`b08d746`) should already exclude this session — `finalProgress: 23`, 223 moves with a long flat tail and a manually `aborted` terminal turn matches the dead-deal signature. **Verify it actually got filtered** before promoting any of these 51 success decisions into `data/dataset/training.jsonl`. If they slip through, you'd be teaching E2B to draw at 0.9 confidence regardless of board state — the exact miscalibration the earlier evaluation flagged.

## Is it better than the last one?

- **As gameplay:** no — same dead deal, same model behaviour, +1 foundation card across ~53 turns of grinding.
- **As a session to kill:** yes — you already killed it (turn 223 = `aborted`). That's the correct outcome.
- **As training data:** no — it adds 38 more "draw at 0.9" exemplars from a structurally hopeless position. Confirm the stall filter rejects this session; if it doesn't, the filter rule needs tightening.

## Recommendation

1. Don't re-harvest seed `821908579` — it's dead. Move on to a fresh deal.
2. Spot-check that `scripts/ingest_exports.py` drops this session (check `data/store/interactions.jsonl` for any `…-502768` rows landing in training, and `data/SUMMARY.md` for the stall-rejection count).
3. The structural lesson from this session reinforces the open P0: **the harvester needs a stall auto-terminator** so you don't have to manually intervene. ~150 retried calls were spent here on a board no model could solve.
