# The "success rate by model" table measures transport, not play

**Date:** 2026-09-11
**Status:** finding confirmed on existing data; new harvest built but not yet run (billing)
**Code:** `scripts/assess_harvest.py`, `scripts/harvest_minimax.py`, `tests/test_harvest_minimax.py` (commit `716e3af`)

## Summary

`data/SUMMARY.md` carries a "By model" table with a success rate per teacher model.
It reads as a quality ranking. It is not one.

| model | headline success | infrastructure loss | **usable once a reply came back** |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 43.7 % | 56.3 % | **100.0 %** |
| `gemma-4-31b-it` | 40.7 % | 59.3 % | **100.0 %** |
| `gemma-4-26b-a4b-it` | 22.1 % | 77.8 % | **100.0 %** |
| `MiniMax-M3` | 14.9 % | 85.1 % | **99.9 %** |

Every model answers usably essentially every time a reply arrives. **The entire
14.9-to-43.7 spread is transport loss.** Not one point of it is a statement about
Klondike.

This matters beyond bookkeeping: `gemma-4-31b-it` was selected as *the* teacher for
all 12 021 rows of `training.jsonl` (`data/SUMMARY.md`, "Training-set selection
funnel"), and the selection metric was measuring its provider's uptime.

## How the losses break down

Reproduce with `.venv/bin/python scripts/assess_harvest.py data/raw --model <name>`.

**`MiniMax-M3`** — 6 893 model turns, 5 864 failures:

```
5715  client timeout          "The model did not respond within 240s."
 126  transport unreachable   "Could not reach TokenRouter."
  15  empty reply             "TokenRouter returned an empty answer / no message content."
   4  cancelled
   3  auth rejected           "TokenRouter rejected the API key."
   1  server error
```

97.5 % of the failures are one client timeout. The median error duration is
**240 997 ms** — the wall itself, not a model that gave up.

**`gemma-4-31b-it`** — 65 477 turns: 18 559 server errors, 16 688 client timeouts,
1 723 transport unreachable, 1 781 Gemini rate limits.

**`gemini-3.1-flash-lite`** — 6 056 turns: 2 023 server errors, 864 rate limits,
290 timeouts.

Different providers, same story: the number in the table is dominated by whichever
way that provider happened to fail.

## Why M3 in particular hit the wall

M3 is a reasoning model, and this prompt is not cheap for it. Over its 1 028
successful calls:

| | value |
|---|---|
| latency p50 | **182 s** |
| latency p90 | **251 s** |
| thinking tokens p50 | **7 323** |
| total tokens per move, mean | **13 603** |

The harvester's client timeout was **240 s**. That sits *below* the model's own p90
and barely above its p50, so most calls were cut off mid-thought and recorded as
model failures. It is not that M3 answered badly; it mostly never got to answer.

A note on what this does **not** show. "99.9 % usable once a reply came back" means
the reply parsed and named a legal move. It is not a claim that M3 plays well — no
M3 session in the corpus got past 31 % progress, and none of them got a fair run
either, with 85 % of their turns erroring. Play quality is still unmeasured, and
that is what the new harvester is for.

## Two parsing defects found on the way

**1. `strategies/llm.py` silently substitutes a heuristic for the model.** It calls
`json.loads(content)` on the reply body and catches every exception into
`_fallback_move`, which returns a priority-ordered heuristic move. The caller cannot
tell that apart from a real model decision. Measured against the 1 028 real M3
replies in `data/raw`, naive `json.loads` succeeds on **613** — so **415 (40 %)** of
genuine model answers would be thrown away and replaced by the heuristic, with no
signal that it happened:

```
411  markdown code fence
  4  prose preamble ("Looking at this position, I ...")
```

Calling the API directly adds a second shape the corpus does not show, because the
old harvester split it out into its own field: **M3 returns `<think>...</think>`
inline in `content`**. Confirmed live against `api.minimax.io`.

`harvest_minimax.extract_json` handles fences, think blocks, preambles, trailing
prose and mid-thought truncation, and reads **1 028 / 1 028**. Covered by
`tests/test_harvest_minimax.py`.

**2. The prompt offered the same move four times.** This engine's four foundation
piles are interchangeable slots rather than one-per-suit, so `generate_moves` returns
an Ace on an empty board as four distinct, equally legal `Move`s that render
identically. Four identical lines make the model's choice arbitrary and inflate the
branching factor. `offered_moves()` collapses them; every surviving index is still a
legal move.

## The new harvester

`scripts/harvest_minimax.py` collects samples with the transport fixed, changing only
what has to change so new rows stay comparable with the corpus:

- **Direct to the provider.** No TokenRouter in the path; 144 of M3's failures were
  that proxy being unreachable, empty, or rejecting the key.
- **Streams, and budgets on a stall plus a hard deadline** rather than a flat
  wall-clock cap. A model still emitting tokens is working, not hung. This is the
  specific fix for the 5 715.
- **Classifies every failure** (`network`, `http_5xx`, `http_429`, `http_4xx`,
  `stall`, `deadline`, `empty_content`, `unparseable`, `bad_index`, `illegal_move`)
  and retries only the retryable, with exponential backoff and jitter. A 4xx is the
  caller's fault and is not retried.
- **Reproduces the `hybrid-v1.6` prompt and the export envelope** that
  `ingest_exports.py` already reads — verified field-for-field against a real export.
- `--mock` exercises the whole loop offline with a canned reply in M3's real wire
  shape, so the plumbing is testable without spending a request.

## What is not done

The live harvest has not run. The MiniMax key is out of credit:

```
HTTP 402  insufficient_balance_error
"The Token Plan usage limit has been reached. (2067)"
```

Account-level, not per-model — all of `MiniMax-M3`, `M2.7-highspeed`,
`M2.5-highspeed`, `M2.1` and `M2` return the same. `/v1/models` still returns 200, so
the key is valid and simply unfunded.

The error handling was at least exercised by it: 402 was classified `http_4xx`,
correctly treated as non-retryable, and all four games failed in 2 s having spent
zero retries. The four zero-move envelopes were kept out of `data/raw` so they cannot
skew `SUMMARY.md`.

**Budget for when a key is available**, measured from the 1 028 successful calls:

| run | tokens |
|---|---|
| 4 games x 40 turns | ~2.2 M |
| 25-deck sweep at 40 turns | ~13.6 M |
| one full game to completion (~300 turns) | ~4.1 M |

The harvester is provider-agnostic — `--base-url` and `--model` are flags and the
contract is OpenAI-compatible — so the same run against `gemma-4-31b-it` would answer
the obvious follow-up: how much of *its* 40.7 % was also the pipe.

## Recommended follow-ups

1. **Re-caption the "By model" table in `data/SUMMARY.md`** so the success rate is not
   read as a quality ranking, or split it into the two columns above. It is currently
   load-bearing for teacher selection.
2. **Fix `strategies/llm.py`** — at minimum stop reporting a heuristic move as an LLM
   decision. Reuse `extract_json`, and surface the failure instead of swallowing it.
3. **Re-examine the teacher choice** for `training.jsonl` once per-model quality is
   measured on replies that actually arrived.
4. **Run the harvest** on a funded key and compare against these baselines.
