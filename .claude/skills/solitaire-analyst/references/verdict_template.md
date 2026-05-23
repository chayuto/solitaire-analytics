# Kill-or-continue verdict template

The user wants a verdict, then evidence. Always lead with the call. Quote
exact card/column/move strings — paraphrasing loses fidelity.

## Template

```
**Verdict: KILL** (or CONTINUE / WATCH)

**Class:** behavioural-doom-loop | dead-deal-flailing | honest-hunt | actively-progressing

**Why (one line):** the specific evidence that drove the call.

**Evidence**
- Plateau: <N> turns on (foundationCards=X, faceDownTotal=Y)
- Doom-loop signature: <card> col <a> ↔ col <b>, <N>× across plateau
  (or: no dominant oscillation pattern)
- Dead-deal signature: <which one, with the specific blocked card/column>
  (or: none — board still has reveal paths)
- Last reasoning excerpt: "<one quoted clause from the agent's boardAnalysis
  that shows it sees the problem and what it picked anyway>"

**Recommend:** kill the session now / let it run / re-check in <N> turns.
```

## Calibration

* **KILL** when the failure class is dead-deal-flailing OR when behavioural
  doom-loop has run ≥30 turns under the current prompt without breaking out.
* **CONTINUE** when actively-progressing (foundation or face-down moved in
  the last ≤15 turns AND recentMoves show varied moves).
* **WATCH** is the honest-hunt case (see 5061b71279a3) where plateau ≥25
  but the agent's reasoning genuinely names a specific missing card it's
  drawing for, and a shuffle-fraction check on recentMoves shows mostly
  draws rather than tableau re-shuffles. Re-check after ~20 more turns.

## Format conventions

* No emoji, no headings beyond level-3.
* Card names with the harvester's notation: `4D` not "four of diamonds".
* Columns are 1-indexed (matches the export and the user's screen).
* Quote agent reasoning sparingly — one clause max. Long quotes go in
  `data/DATASET_NOTES.md`, not the verdict.
* Don't restate the provider error rate as a concern. The user has acknowledged
  it as a known background issue.

## Worked example — KILL on 29a7f5

```
**Verdict: KILL**

**Class:** behavioural-doom-loop

**Why (one line):** 85-turn plateau with 59× 4D and 53× 5C oscillation between
cols 3 and 4 — identical pathology to the 645d03 baseline this session was
trying to fix.

**Evidence**
- Plateau: 85 turns on (foundationCards=6, faceDownTotal=16)
- Doom-loop signature: 4D col 3 ↔ col 4 (59×), 5C col 3 ↔ col 4 (53×) across plateau
- Dead-deal signature: none — 16 face-down cards but reveal paths exist
  (the model itself named black-7 needed for col 4, which is a known-but-buried card)
- Last reasoning: "The board is currently stalled... no black 8 available
  on the tableau to receive [the waste 7D]"

**Recommend:** kill now. The comparison arm against 645d03 baseline is
sufficient evidence that prompt v1 didn't address this class.
```

## Worked example — CONTINUE

```
**Verdict: CONTINUE**

**Class:** actively-progressing

**Why (one line):** foundation climbed 4 → 6 in the last 10 turns and the
agent is making varied moves with correct reasoning.

**Evidence**
- Plateau: 0 turns at current state (still moving)
- Doom-loop signature: none (longest pair-repetition is 3 across plateau)
- Dead-deal signature: none
- Last reasoning: "moving 7S to col 3 reveals the face-down card in col 5"

**Recommend:** let it run. Next routine check in ~30 turns.
```
