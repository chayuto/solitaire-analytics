# Handover to harvester team

**Date:** 2026-05-22
**From:** Chayut / dataset side
**Re:** Prompt template audit findings + asks for the harvester pipeline
**Companion document:** `docs/reports/20260522_prompt_template_audit.md` (full evidence and proposals)

---

## TL;DR — three asks, ranked

1. **Add two fields to the per-interaction log** (small change, unblocks all future analysis).
2. **Add one clause to the prompt template** (resignation output — fixes a P0 from inside the model).
3. **Run one same-seed cross-build experiment** (the only experiment that gives prompt-independent build attribution; everything else we conclude about build quality is single-coin-flip evidence until then).

The other recommendations in the audit (anti-oscillation / consistency / plan-continuity clauses) are worth eventually testing but are speculative. The three above are not — please prioritise them.

---

## Context you should know before deciding priorities

When I audited the prompts across every distinct `appCommit` in the corpus, **all five builds (`71130ac`, `ce6afe1`, `afa8c24`, `6dfc8a9`, `7894202`) are serving a byte-identical static template** — same 3,527-character prefix, MD5 `a39354fa5f16e03285e389dee5dc551c`. Verified by hashing each.

This is important because:

- Every behavioural difference we've been attributing to "the new build" so far is **not a prompt change**. It must be sampling, retry, temperature, post-processing, or backend infra.
- The recurring failure patterns (oscillation, deadlock-but-draws, chain reversal, model-says-deadlock-then-draws) are baked into a single shared template — they're not someone's regression on the latest build, they're load-bearing weaknesses in the original.
- Build hashes are advancing without anyone being able to point at *what visible thing* changed. From the dataset side that looks like noise — we can't attribute lift or regression to any specific decision.

This is fixable with one small log change (ask 1).

The other context worth knowing: the corpus now has **one win** (session `…0154e1` under `6dfc8a9`, seed `3263196305`). It is the only escape from the doom-loop asymptote we've seen. Could be the build, could be the deal — without same-seed cross-build replay we can't say which. That's ask 3.

---

## Ask 1 — Log two fields per interaction

Add these to each interaction record:

| Field | Type | Value |
|---|---|---|
| `promptTemplateHash` | string | MD5 (or SHA256) of the static template portion of the prompt (the part that doesn't change per turn) |
| `promptTemplateFinalisedAt` | string (ISO 8601 datetime) | When this template was finalised / shipped — e.g. `2026-05-22T00:00:00Z` |

**Why both:** the hash is the deterministic identity (catches accidental changes); the datetime is the human-readable companion (same convention as `docs/reports/YYYYMMDD_*`). Datetime-as-version avoids needing to coordinate semantic labels like `v1`/`v2` between branches or A/B arms.

**Backfill:** every interaction in `data/raw/` to date should be stamped `promptTemplateHash=a39354fa5f16e03285e389dee5dc551c`. The corresponding `promptTemplateFinalisedAt` is unknown from our artefacts — please supply the original ship date of this template (or use sentinel `unknown-pre-2026-05-22`). The ingest pipeline (`scripts/ingest_exports.py`) can stamp retroactively in one pass once you confirm the date.

**Effort estimate (from the outside, may be wrong):** one config read + one hash computation at prompt-build time. Should be a half-day at most.

---

## Ask 2 — Add resignation output to the prompt template

Append to the static template (between the `RESPONSE FORMAT` block and the trailing newline):

```
RESIGNATION:
If after honest analysis you cannot identify any sequence of moves that
improves foundationCards or faceDownTotal from the current position,
set final_decision.move_index = -1 to signal resignation. The harvester
will treat this as a clean termination.
```

**On the harvester side**, treat `move_index == -1` as:
- A terminal turn (don't retry, don't re-prompt)
- Mark the session `outcome: resigned` (new value alongside `won` / `incomplete` / `aborted`)
- Stop submitting further turns for that session

**Why this and not the other prompt clauses I sketched in the audit:** of the five clauses I proposed (anti-oscillation, board-analysis ↔ decision consistency, stock-recycle priority, plan continuity, resignation), this is the **only one with a structural payoff**. It directly closes the standing P0 of "harvester has no stall auto-terminator" by pushing the resignation signal into the model output — the harvester then doesn't need to detect stalls heuristically; the model says "I'm done" and you stop.

It also gives the distillation side clean training labels for "this board is unwinnable" — which we don't have today and can't get from inference rules.

The other four clauses are mostly cosmetic — Gemma-31B already sees the oscillation in `recentMoves` and the contradiction with its own reasoning in `reasoningTrail`, and picks the bad move anyway. That's a decode-level incoherence issue that a prompt clause is unlikely to fix. I'd hold off on those until you have data from the same-seed experiment (ask 3).

---

## Ask 3 — Run one same-seed cross-build experiment

The only experiment that disambiguates build quality from deal luck:

**Seed `3263196305`** (the one winning seed in the corpus, won under `6dfc8a9` + the shared template).

Run it under each of the following builds **until each produces one completed game** (retry through provider errors — given the documented ~75% provider error rate, expect 3–5 attempts per build before one lands):
- `71130ac` (oldest build in corpus)
- `7894202` (newest build in corpus)
- ideally also a fresh build, if you have one shipping

Same template (the shared `a39354fa…` one). Same model (`gemma-4-31b-it`). Same harvester config (same `temperature`, etc.).

**What this tells us:**
- If only `6dfc8a9` wins: build matters, `6dfc8a9` is the keeper, regression on `7894202`.
- If all three builds win: it's the deal, the `6dfc8a9` "win attribution" was deal-luck, build hashes are mostly cosmetic.
- If some lose and some win on this winnable deal: build effect is real but non-monotonic — needs more seeds.

We've been waiting on this experiment since the `645d03` baseline was locked. It's referenced as "needed" in three separate `data/DATASET_NOTES.md` entries now. Until it runs, every build-attribution claim in the dataset notes (including my own write-ups) is built on a single coin flip.

**Cost:** one game per build, maybe 10-20 min wall-clock each given the provider error rate. Maybe 1 hour total. The signal it provides is qualitatively different from anything else in the corpus.

---

## Open questions for you

These I can't answer from the artefacts; please reply:

1. **Why are build hashes advancing if the prompt template isn't?** What's actually changing between `71130ac` → `ce6afe1` → `afa8c24` → `6dfc8a9` → `7894202`? Sampling? Retry policy? Backend? Prompt-builder code that doesn't affect the static template? It would help dataset-side to know what build hashes mean.
2. **What's the original ship date of the current template** (`a39354fa…`)? Needed for backfill in ask 1.
3. **Is there a Gemma temperature / top_p / top_k currently in use, and has it changed across builds?** Best candidate for explaining build-to-build variance under an identical prompt.
4. **Status of the four standing P0s in `data/DATASET_NOTES.md`** — deck seed, stall auto-terminator, ~37% missing-move logging, ~75% provider error rate. Where is each? Ask 2 (resignation output) closes the stall-terminator P0 if shipped; the others remain open and continue to bite the dataset quality.

---

## What the dataset side will do once these ship

- Once ask 1 ships: backfill all of `data/raw/` with the existing hash, then every new ingest auto-stamps. Future analyses like the one in `20260522_prompt_template_audit.md` become a one-line query instead of a manual file inspection.
- Once ask 2 ships: ingest pipeline will route `outcome: resigned` rows into a new dataset slice (training-quality "model knew when to stop" examples) instead of the doom-loop tail-filter rejection bucket.
- Once ask 3 ships: lock in the prompt-vs-build vs deal-luck attribution, then revisit DATASET_NOTES.md entries that claim build-level effects and downgrade them to "deal-attributable" where the experiment shows that.

---

## What's NOT being asked

- I am **not** asking you to ship the other four proposed prompt clauses (anti-oscillation, consistency, recycle priority, plan continuity). Hold those until same-seed data exists.
- I am **not** raising the provider-error rate as a new concern in routine analyses — it's tracked, you're aware, my dataset notes skip re-flagging it.
- I am **not** asking you to retroactively re-run any of the existing failed sessions. The data we have is what we have.

---

## Where to find the evidence

- `docs/reports/20260522_prompt_template_audit.md` — full audit with hashes, per-failure-pattern attribution table, all five proposed clauses (resignation is clause 5), and validation experiment specs.
- `data/DATASET_NOTES.md` — session-by-session failure catalogue with builds, seeds, plateau counts, oscillation patterns.
- `.claude/skills/solitaire-analyst/` — the analyst tooling used to produce the briefings each ingest was based on (load_export.py + winnability check).
