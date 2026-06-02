# `data/DATASET_NOTES.md` entry format

`data/DATASET_NOTES.md` is hand-maintained context for the corpus — the
long-form record of which sessions failed how. The auto-generated
`data/SUMMARY.md` covers stats; this file covers *interpretation*. When
the user asks to ingest an export, append (or update) an entry here.

## Where to add an entry

* **Known doom-loop sessions (kept; flagged by stall filter)** — for first-time
  sessions that ended in a stall/loss. This is the default section for new
  problem sessions.
* **Same-seed validation experiments** — for comparison arms: when the harvest
  team re-ran a known-failing seed under a different build/prompt. The
  original session becomes the locked baseline; the new session is documented
  under the seed's experiment block.
* **Same-seed baseline pair** — for the perfect-info vs imperfect-info
  comparison pair on seed 4153653383. Only add here if a new pair appears.

## Canonical entry shape (doom-loop section)

Match the prose style of existing entries — short paragraphs, no bullets
inside an entry. Quote specific cards/columns; don't paraphrase.

Lead each entry with the short game id `#<6-char id>` (the filename token); keep
the full UUID once in parentheses for the record. The user identifies sessions
by the short id, not the long hash.

```
- Session `#<6-char id>` (full `<sessionId>`), seed `<seed>`, model `<model>`, app
  build `<appCommit>`. <Files in raw/, naming the latest as canonical and
  noting any archived predecessors with row counts>. Final stored state:
  <total interactions>, max successful turn `<turn>`, `moveCount: <N>`,
  `finalProgress: <P>%`. <One-line failure summary — class + signature.
  Quote the specific oscillation pattern or dead-deal lock.>. <Optional:
  what the agent's reasoning said that was self-aware-but-impotent.>
  <Optional: role in the corpus — what this session is the canonical example
  of, e.g. "shuffle-fraction gate fires at 70% during plateau".>
```

## Canonical entry shape (same-seed experiments)

```
- Seed `<seed>` — baseline session `…<baseline-last-12>` (build `<commit>`,
  documented above; <one-line baseline pathology>). Comparison arm: session
  `…<comparison-last-12>`, model `<model>`, ingested <YYYY-MM-DD> via
  `<filename>.json` (<rows> rows, <success> success / <error> errors).
  **Result: <prompt vN did/didn't address the pathology>.** Final state
  `moveCount: <N>`, `finalProgress: <P>%`, <how it ended>. <Quote the
  specific evidence — exact recentMoves counts, plateau length comparison
  vs baseline, oscillation pattern match.>. <Quote the agent's
  self-aware-but-impotent reasoning.>. **Conclusions:** (1) <method-level>;
  (2) <prompt-level>; (3) <next-action-level>.
```

## What to include

* Session id (last 12 hex chars only — full uuid bloats the file)
* Seed, model, appCommit
* All raw files for this session (active and archived). Note which is
  canonical (most recent / largest non-subset).
* `moveCount` and `finalProgress` from `session` block — these are the
  terminal stats.
* The exact failure signature: card + columns + count for doom-loops; the
  specific locked card and reveal-path absence for dead deals.
* The agent's last-turn reasoning if it self-diagnoses the deadlock — these
  are gold for understanding the model's failure mode.
* Quantitative evidence: plateau length, confidence saturation level,
  shuffle-fraction.

## What NOT to include

* The full agent prompt — too long, lives in the raw export.
* The full reasoning trail — one quoted clause is enough.
* Per-turn breakdowns — those are derivable from the store.
* **Provider error rate as a "concern" section.** Known background issue;
  the user has asked not to re-flag it. (See memory
  `skip-redundant-provider-error-callouts`.) The exception is if error
  rate is materially different from baseline.
* Speculation about what fix would work. Stick to observed behaviour.

## Workflow

1. `mv` the export from `/Users/chayut/Downloads/` to `data/raw/` (the
   user is explicit: `mv`, not `cp`).
2. `.venv/bin/python scripts/ingest_exports.py` — adds the file's hash
   to the manifest and updates `SUMMARY.md`.
3. Read the existing entry for this session in `DATASET_NOTES.md` (if any)
   and either update it (if this export extends or supersedes a previous
   one) or append a fresh entry.
4. Don't reformat surrounding entries. Leave them alone.
