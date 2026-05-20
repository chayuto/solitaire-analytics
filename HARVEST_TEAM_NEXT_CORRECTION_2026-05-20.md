# Harvest Team — Next Correction Recommendation

**Date:** 2026-05-20
**Author:** analytics-side, follow-up to `HARVEST_TEAM_HANDOVER_2026-05-19.md`
**Question:** of everything open, what should the harvest team do next?
**Answer (TL;DR):** **Ship the stall auto-terminator (P1.4).** It is the single highest-ROI correction we can make right now, and it must precede any prompt-engineering A/B test or the test is uninterpretable.

## Where we are

### Resolved or downgraded since 2026-05-19

| Item | Status |
|---|---|
| P0 — no `gameId` / seed in collection | **Solved for new collections via the multi-session MCP server** (merged to main as PR #17, `7e39811`). The external harness P0 still applies if you keep collecting through it. |
| P0 — auto-played moves untagged | **N/A on the MCP server** (every move is an explicit `play_move` call). Still open on the external harness. |
| P0 — `aiConfig` / `seeHiddenCards` missing | **Solved for the MCP server** (`infoLevel` is stamped on every harvest record). Still open on the external harness. |
| P1.4 — no stall auto-terminator | **OPEN.** No change. |
| P1.5 — no `progressScore` / components | **OPEN.** No change. |
| P1.6 — 64% unavailable+timeout errors on 31B | **OPEN.** Operational, not a label-quality issue. |
| Confidence saturation / miscalibration | **OPEN.** Newly documented in `645d03` (model reports 0.9 confidence on the 5th repeat of a doom-loop move). |
| Doom-loop pathology | **Newly diagnosed.** See `PROMPT_ENGINEERING_DOOM_LOOP_2026-05-20.md` and `KLONDIKE_PROMPT_STRATEGY_DEEP_DIVE_2026-05-20.md` for prompt-side mitigations. |

### Open work items, ranked by ROI

ROI = (impact on training-data quality and yield) ÷ (engineering cost). All numbers are rough estimates.

| Rank | Item | Cost | Impact | ROI |
|---|---|---|---|---|
| 1 | **Stall auto-terminator** (P1.4) | low | high — reclaims ~50% of teacher calls currently wasted on stalled games | **highest** |
| 2 | Prompt-engineering pilot (doom-loop doc, Phase 1) | low | unknown — could be 0–50% stall reduction | high (but needs #1 first) |
| 3 | Migrate collection to the multi-session MCP server | medium | high — solves all 3 P0s at once, gives ingest-ready records | high |
| 4 | Patch the external harness for the 3 P0s | medium | high — keeps the current pipeline working | medium |
| 5 | `progressScore` metric (P1.5) | low | medium — better dataset filtering | medium |
| 6 | Confidence recalibration / decision to drop | low | medium — affects label quality only | medium |
| 7 | Investigate the 64% timeout (P1.6) | medium-high | medium — improves call yield, not label quality | low–medium |
| 8 | Rebuild prompt as priority cascade (deep-dive doc) | high | high if Phase 1 patches fail | medium (conditional) |

## Recommendation

**Do #1: implement the stall auto-terminator in the collection harness.** Concretely:

> After each AI decision is applied, check whether `(foundationCards, faceDownTotal)` has been unchanged for the last 25 successful turns. If yes, mark the session `outcome: "stalled_auto_terminated"` and stop calling the teacher. Reserve the rest of the session's call budget for new games.

This is the same threshold the ingest pipeline already uses (`STALL_TURNS = 25` in `scripts/ingest_exports.py`). Using the same value end-to-end means the harness terminates a game *exactly when* the ingest filter would have dropped its remaining decisions from training. Every call we save was guaranteed to land in the discard pile.

### Why this is the right correction, not any of the others

**Why not "migrate to MCP server first" (#3):**

- The MCP server doesn't fix the *teacher's* doom-looping; it just records loops better. The harvest team's pain right now is wasted call budget, not record quality.
- Migration is medium effort. Stall terminator is small effort. Sequence: ship the cheap win first, then evaluate the migration on its own merits.

**Why not "patch external harness P0s first" (#4):**

- Same reason. The P0s are about reproducibility and tagging cleanliness. They do not move the yield needle today. Stalled games still waste 50%+ of calls.
- The P0s become urgent only when we re-pilot. Stall terminator is urgent now.

**Why not "run the prompt-engineering pilot first" (#2):**

- This is the trap. Running Phase 1 against the *current* baseline (where ~50% of teacher calls land on stalled-game continuations) means our before/after comparison is dominated by stall noise. We will not be able to attribute a stall-ratio change to the prompt patch with any confidence.
- **Stall terminator is a prerequisite for a clean prompt-engineering A/B.** Get the noise floor down first; *then* measure the prompt patch against it.

**Why not "fix confidence calibration first":**

- Confidence is a labelling concern; it does not affect call budget. Out of scope for the "next correction" decision. Treat it as "decide whether to drop the label" in a separate pass; doesn't block anything.

## Implementation sketch

The terminator lives in the collection harness, not the ingest pipeline (the ingest pipeline already drops stalled decisions; the harness keeps making the calls regardless).

Pseudo-code, ~30 lines in the harness's per-turn loop:

```python
STALL_TURNS = 25         # match scripts/ingest_exports.py
SHUFFLE_FRACTION = 0.6   # see "Refinement" below

class StallTracker:
    def __init__(self):
        self.last_foundation = None
        self.last_face_down  = None
        self.unchanged_count = 0
        self.recent_move_types = []  # rolling window over the unchanged streak

    def update(
        self,
        foundation_cards: int,
        face_down_total: int,
        move_type: str,
    ) -> bool:
        """Return True if the game should auto-terminate.

        Two-gate rule: terminate only if (a) progress has been flat for at
        least STALL_TURNS and (b) the AI has been *shuffling* during the
        plateau rather than drawing/recycling. See "Refinement" for why.
        """
        if (foundation_cards == self.last_foundation
            and face_down_total == self.last_face_down):
            self.unchanged_count += 1
            self.recent_move_types.append(move_type)
        else:
            self.unchanged_count = 0
            self.recent_move_types.clear()
            self.last_foundation = foundation_cards
            self.last_face_down  = face_down_total

        if self.unchanged_count < STALL_TURNS:
            return False

        # Plateau is long enough. Check whether the AI is shuffling or hunting.
        window = self.recent_move_types[-STALL_TURNS:]
        shuffle_count = sum(
            1 for mt in window
            if mt in ("tableau_to_tableau", "discard_to_tableau")
        )
        return (shuffle_count / len(window)) >= SHUFFLE_FRACTION

# In the per-turn loop:
tracker = StallTracker()
while not game_over():
    decision = call_teacher(state)
    move_type = apply(decision)  # return the type string from the move applied
    if tracker.update(state.foundation_cards, state.face_down_total, move_type):
        mark_session_terminated(reason="stall_auto_terminated",
                                last_turn=state.move_count)
        break
```

Two harness-side changes beyond the loop:

1. **Export field.** Add `outcome: "stalled_auto_terminated"` as a new value alongside `outcome: "incomplete"`. The ingest pipeline already keeps everything; this is purely a tag for downstream analysis.
2. **Telemetry counter.** Track the number of terminations per harvest run, plus the per-termination `shuffle_fraction` at the moment of trigger — useful for tuning the threshold.

### Refinement — plateau length alone is not enough

Real-world evidence (2026-05-21, two more game exports analysed):

| Session | Turn @ export | Plateau length | Last-20 move mix | Verdict |
|---|---|---|---|---|
| `…b71279a3` (seed `2439067361`) | 34 | 24 turns | **18 draws + 1 recycle + 0 shuffles** | **honest hunt — keep playing** |
| `…22c73fd85` (seed `191155745`) | 128 | 15 turns | **14 shuffles** + 1 draw + 5 foundation/discard | **doom-loop — terminate** |

A plain `STALL_TURNS = 25` plateau gate would flag `1279a3` as terminate-worthy even though the AI is correctly drawing because two Aces are still unaccounted for and there is no productive tableau move. Terminating it would lose a legitimate hunt-for-Aces session.

The discriminator is **move type during the plateau:**

- *Drawing* and *recycling* are information-gathering moves; the AI is rational to repeat them while critical cards remain unseen. Plateau without shuffling = honest hunt, give it grace.
- *Tableau-to-tableau* shuffles during a plateau are doom-loop signal. Plateau with > 60% shuffles in the window = terminate.

The `SHUFFLE_FRACTION = 0.6` threshold is a starting point: in the two-sample contrast above, `1279a3`'s shuffle fraction is 0% and `73fd85`'s is ~70%. The threshold has wide headroom on both sides; tune after a few sessions if needed.

**Effect on `645d03` (the original case):** the late game showed 5C/4D oscillation in `recentMoves` — shuffle fraction near 100% in the plateau window — so the refined rule fires exactly as before. The refinement does not weaken termination of the original failure mode; it only protects honest hunts.

### What the harness should NOT do

- **Do not** end the game on the player side as "stuck" — `stuck` already has a defined meaning (no legal moves). A doom-loop game has legal moves; it just isn't using them productively. Use `stalled_auto_terminated` as a distinct outcome.
- **Do not** call the teacher one extra time to ask "should I terminate?" That's a wasted call. The harness decides; the model is not consulted.
- **Do not** change `STALL_TURNS` from 25 without also updating `scripts/ingest_exports.py` and `data/DATASET_NOTES.md`. The harness and ingest must stay in lockstep, otherwise the harness terminates games whose tail would have been kept (or vice versa).

## Acceptance criteria

Run one batch (3–5 sessions) after the change ships and measure:

| Metric | Source | Expected |
|---|---|---|
| Sessions with `outcome: "stalled_auto_terminated"` | new export field | > 0, ideally 30–60% of sessions match historical stall rate |
| Average call count per terminated session | harness telemetry | should approximate `(turn at first stall) + STALL_TURNS` |
| Wasted-call ratio | `(calls in stalled tail) / total calls` | should drop from ~50% to ~10–15% |
| Per-termination shuffle fraction | harness telemetry | should be ≥ `SHUFFLE_FRACTION` (0.6) by construction. If many terminations sit right at 0.6, the threshold is borderline — examine those games manually before tuning. |
| Ingest-pipeline behaviour | re-run `scripts/ingest_exports.py` | local training set size should be approximately **unchanged** (same number of training-eligible decisions; we are not adding or removing eligible rows, only stopping the bleed). If it shrinks meaningfully, the terminator is firing too eagerly. |
| Honest-hunt false-positive rate | manual review of any `1279a3`-shaped session (long plateau, draw-dominated) | should be **zero** — those sessions must not terminate. |

If the local training set shrinks by > 5% relative to the same seeds without the terminator, either `STALL_TURNS = 25` is too tight (raise to 30) **or** `SHUFFLE_FRACTION = 0.6` is too loose (raise to 0.7). Diagnose by inspecting the terminated sessions' shuffle fractions: tight cluster near 1.0 → raise `STALL_TURNS`; spread between 0.6 and 0.8 → raise `SHUFFLE_FRACTION`.

## What comes after this ships

In order:

1. **Re-run the existing 12-session corpus** with the same seeds, terminator enabled. This gives a clean before/after on call yield with the new noise floor.
2. **Then** run the doom-loop doc's Phase 1 prompt patches against that new baseline. The stall-ratio reduction (or lack thereof) is now interpretable.
3. **Then** decide on the rebuild (deep-dive doc) vs. status quo, based on Phase 1 results.
4. **In parallel** (does not block the above): decide whether to migrate collection to the MCP server. If yes, the stall terminator concept ports directly — register it as a `stuck_turns` watchdog parameter on `new_game` and have `play_move` raise / end the session when the threshold is hit.

The MCP-server migration is the right *medium*-term move; the stall terminator is the right *next* move regardless of that decision.

## Open questions for the harvest team

1. Where exactly in the harness does per-turn state land? (Decides whether the tracker is per-call middleware or in the game-loop driver.)
2. Is there any current code path that already counts unchanged turns? If yes, reuse it.
3. Does the export schema have room for a new `outcome` value, or do we need to coordinate a schema bump? (Affects ingest-pipeline tolerance — `scripts/ingest_exports.py` currently keys on `outcome == "success"`, so a new failure value is transparent. Confirm.)
