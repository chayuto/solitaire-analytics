---
name: solitaire-analyst
description: Analyze raw Klondike Solitaire AI export logs from this repo (data/raw/solitaire-ai-log-*.json or files dropped in /Users/chayut/Downloads/) and deliver a kill-or-continue verdict. Use this whenever the user shares a solitaire-ai-log path, asks "should I kill this session?", asks whether a board is solvable, asks to ingest a new export, or wants to update data/DATASET_NOTES.md with findings from a new harvest. The user is running an active harvest pipeline and needs the early-termination signal — not a postmortem after the session stalls further.
---

# solitaire-analyst

You are analyzing exports from the Klondike Solitaire harvester that feeds the Gemma-4 E2B distillation training set. Each export is a real session played by a live LLM agent under imperfect information. The user is the operator: they decide whether to keep a session running or kill it. Your job is to give them that decision quickly and accurately.

## The user's mental model (read this first)

The user has built a harvest pipeline. Sessions are expensive — each turn can require up to ~17 provider retries under the current error rate. Stalled sessions burn budget AND produce rows the stall filter excludes from training. The user wants to know **as soon as possible** when a session is dead so they can terminate.

Two distinct failure classes — different fixes, **always classify which one**:

1. **Behavioural doom-loop on a winnable board.** The deal is solvable, but the model is oscillating (e.g. session 645d03 looped `5C/4D col3↔col4` for 75 turns; 73fd85 looped `TS/9D` for 15 turns). Fix is prompt-side (anti-oscillation rules, reveal-priority).
2. **Dead-deal flailing.** The board is structurally lost — no sequence of legal moves wins. The model is doing what little it can. Fix is harness-side (stall auto-terminator) or prompt-side (resign rule). No prompt rephrasing solves an unsolvable board.

The exact diagnosis matters because the user has same-seed validation experiments in flight where the comparison arm's failure mode tells them whether the prompt fix worked.

## When this skill triggers

- User pastes/mentions a `/Users/chayut/Downloads/solitaire-ai-log-*.json` path
- User pastes/mentions a path under `data/raw/`
- User asks "should I kill this session?" / "is this stalled?" / "is this winnable?"
- User asks to ingest a new export (the verdict lands in `data/DATASET_NOTES.md`)
- User asks for the state of a particular sessionId

## Workflow

### 1. Load and parse the export

Use the bundled helper — it handles every shape the harvester emits.

```bash
.venv/bin/python .claude/skills/solitaire-analyst/scripts/load_export.py <path-to-export.json>
```

Or, in a Python context:

```python
from pathlib import Path
import sys
sys.path.insert(0, ".claude/skills/solitaire-analyst/scripts")
from load_export import load_export, latest_board

doc = load_export("data/raw/solitaire-ai-log-29a7f5-1779361593611.json")
# doc.session_id, doc.seed, doc.model, doc.final_progress, doc.move_count, doc.outcome
# doc.interactions  -> sorted by turnIndex, successes first
# doc.successes     -> just the successful ones
board = latest_board(doc)
# board is the parsed CURRENT GAME (JSON) from the latest successful interaction
```

The script also prints a one-screen briefing — session id, seed, model, build, success/error counts, final state, plateau length, last 5 successful turns' reasoning. Read that briefing first.

### 2. Read the latest board state

The export's `prompt` field embeds a `CURRENT GAME (JSON):` block. The parser extracts it. The fields you care about:

| Field | What it tells you |
|---|---|
| `foundations` | `{hearts, diamonds, clubs, spades}` — top card per suit, or null if empty. Sum of ranks = `foundationCards`. |
| `tableau[i].faceUp` | The visible cards in column i+1, bottom→top. |
| `tableau[i].faceDownCount` | Hidden cards below the face-up stack. Total = `faceDownTotal`. |
| `discardTop` | Top of waste pile (the playable waste card). |
| `drawPileCount` | Cards still in the stock. |
| `canRecycleStock` | True if the waste can be flipped back into the stock. |
| `legalMoves` | Numbered moves with `describe` text. |
| `metrics.completionProgress` | Harvester's progress %. |
| `recentMoves` | Last 10 moves — **the doom-loop fingerprint.** |
| `seenDrawPileCards` | Cards the agent has seen surface from the stock. |
| `reasoningTrail` | Last 5 LLM rationales. Read these — they often expose self-aware-but-impotent loops. |

### 3. Classify the failure mode

Run all checks; the patterns layer.

**Doom-loop fingerprint** — count pair-repetitions in `recentMoves`. A 2-move oscillation like `move X col A -> col B` / `move X col B -> col A` repeating ≥4 times in the last 10 moves is the canonical pattern. So is a 3-card cycle.

**Dead-deal signatures** — check the latest board for:

- **Missing low foundation card likely face-down with no reveal path.** Foundations show e.g. hearts=null (no AH yet). Search `seenDrawPileCards` and every `faceUp` column for `AH` — if absent, AH is face-down. Then check: is there *any* face-up card in the column above the face-down stack that can chain off (alternating-color, rank-1 lower) to expose what's beneath? If no column's face-down stack can be unburied, the missing Ace is permanently locked.
- **Small `faceDownTotal` concentrated in 1–2 columns with no reveal path.** When `faceDownTotal` is low (≤8) but stuck unchanged for many turns, it's almost always a structural lock — the remaining hidden cards are pinned under sequences that can't be moved coherently.
- **Stock fully known after recycles.** When `canRecycleStock: false` (mid-cycle through stock) AND `drawPileCount + len(seenDrawPileCards) ≈ 24 - foundationCards_from_stock_path`, the agent has seen everything the stock can give. No draw will produce a new card.
- **Foundation suit blocked by buried chain.** If next-needed foundation card (e.g. `5D` when `diamonds=4D`) is face-up but buried under a sequence longer than your available empty-column slack can disassemble, that suit can't progress.

**Honest-hunt vs flailing.** If `foundationCards` and `faceDownTotal` were flat for ≥25 turns AND `recentMoves` show repetitive same-card shuffling, that's flailing. If they're flat but `recentMoves` show varied moves and the agent is genuinely hunting for a specific named card, that may be an honest hunt — see the 5061b71279a3 entry in `data/DATASET_NOTES.md` for the canonical example.

### 4. If uncertain, ask the solver

When heuristics are split (e.g. the board *might* be unwinnable but you're not sure, and the user has high stakes — live session, big retry budget already spent), invoke:

```bash
.venv/bin/python .claude/skills/solitaire-analyst/scripts/check_winnability.py <path-to-export.json>
```

The script:
- Takes the latest CURRENT GAME (JSON) from the export.
- Determinises the unknown cards: face-down tableau slots + remaining stock slots get random assignments from the cards-not-yet-accounted-for set. The seed cannot be replayed (harvester uses non-Python PRNG), so this is the right approach — the deal is one specific assignment from the consistent-with-observation distribution.
- Runs `solitaire_analytics.ParallelSolver` on each sample.
- Reports `samples=N, solved=K, solve_rate=K/N` with a verdict band.

Defaults: 10 samples, beam_width=2000, timeout=30s each. Tune via `--samples` and `--timeout` if a large face-down count makes the sample space rough. Interpret with care: **success on any sample proves the board class is sometimes solvable; failure on every sample is suggestive but not proof** (beam search is one-sided).

Don't run the solver routinely. The heuristics are right on the obvious cases. Solver is for the borderline ones where being wrong costs the user real money.

### 5. Deliver the verdict

Always use the template in `references/verdict_template.md`. The key shape:

- **TL;DR line**: KILL / CONTINUE / WATCH with one-clause reason
- **Failure class**: behavioural-doom-loop / dead-deal-flailing / honest-hunt / actively-progressing
- **Evidence**: the specific `recentMoves` cycle, the specific missing foundation card, the specific blocked column — quote from the export, don't paraphrase
- **Recommendation**: one sentence

Keep it tight. The user wants the verdict, not a literary postmortem. Save the long-form for `data/DATASET_NOTES.md` when ingesting.

### 6. If ingesting, update DATASET_NOTES.md

When the user asks to ingest the export:

1. Move the file from `/Users/chayut/Downloads/` to `data/raw/` (use `mv`, not `cp` — the user is explicit about this).
2. Run `.venv/bin/python scripts/ingest_exports.py` (the ingest pipeline is idempotent and adds the file's hash to the manifest).
3. Append or update the session entry in `data/DATASET_NOTES.md` under "Known doom-loop sessions" — see `references/dataset_notes_format.md` for the canonical entry shape.
4. The "Same-seed validation experiments" section is for comparison arms (when the harvest team re-ran a known-failing seed under a different build/prompt). If the export is for an existing baseline seed, add it there instead.

## What NOT to do

- **Don't open with a postmortem.** The user has told you twice to be the early-warning signal, not the after-action reporter. Lead with the verdict; the evidence supports it.
- **Don't flag the ~75% provider error rate** as a concern in routine analyses. It's a known background issue the user has acknowledged. Only mention if it's materially different from baseline (new error kind, large rate jump). See memory `skip-redundant-provider-error-callouts`.
- **Don't claim a board is "unwinnable" from heuristics alone** when the user is going to act on it for the first time on this session. Hedge to "structural lock signature, likely dead" unless the solver confirms.
- **Don't run the solver routinely.** It's a 30s × N samples × import overhead operation; reserve for borderline cases.
- **Don't write `DATASET_NOTES.md` updates** unless the user has explicitly asked to ingest. Analysis ≠ ingestion.

## Reference files

- `references/verdict_template.md` — the kill-or-continue output format
- `references/dead_deal_signatures.md` — detailed pattern catalogue, with examples from real sessions
- `references/dataset_notes_format.md` — the long-form entry shape for `data/DATASET_NOTES.md`
- `scripts/load_export.py` — export loader + briefing printer
- `scripts/check_winnability.py` — Monte Carlo solver wrapper
