# Schema Audit: Code-Review LLM Response Format

## Verdict

The schema is workable but has several design problems that will hurt quality, parseability, and downstream usefulness. None are fatal — but for a code-review use case with Sonnet 4.6 + adaptive thinking, you can do much better. Below is a prioritized critique followed by a revised schema.

---

## Issues With the Current Draft

### 1. `analysis` and `suggestions` are unstructured prose blobs
You're asking for **200–500 words** of free text per field. Two problems:

- **Word-count constraints are notoriously unreliable** for LLMs. Claude (and every other model) approximates length poorly; you'll get 150 or 700 with regularity. If you genuinely need length control, specify *structure* (e.g., "3–7 bullets") instead of raw word counts. The model can count list items reliably; it cannot count words reliably.
- **Prose blobs are not actionable.** A code-review consumer (CI bot, PR comment renderer, dashboard) wants to attach findings to **specific files / line ranges / categories**. A 400-word essay collapses many distinct findings into one un-routable string. You lose the ability to dedupe, prioritize, link to source, or roll up metrics over time.

**Fix:** decompose into an array of structured `findings`, each with `file`, `line_range`, `category`, `severity`, `description`, `suggested_fix`. Keep one short top-level `summary` if humans need a TL;DR.

### 2. `severity` at the top level is the wrong granularity
A single severity for the whole review forces the model to collapse "one critical SQL injection + four nits" into one label. Either the nits get overweighted or the critical bug gets buried. Severity belongs **per finding**, and the top-level `final_decision.action` is what should reflect the aggregate.

### 3. `confidence: 0.0–1.0` is a known anti-pattern
LLM-emitted continuous confidence scores are **poorly calibrated** and tend to cluster around a few values (0.7, 0.8, 0.9, 0.95). They look quantitative but aren't. Two better options:

- **Discrete buckets** (`low | medium | high`) — the model picks these more consistently and they're easier to threshold on.
- **Drop confidence entirely** and rely on the thinking trace (with adaptive thinking enabled, Sonnet 4.6 already exposes reasoning quality through `thinking` blocks; a redundant numeric score adds noise).

If you really want a number, force *justification*: `confidence: {level: "low|medium|high", reason: "<one sentence>"}`. The reason field disciplines the model into not picking 0.9 reflexively.

### 4. The enum `action: approve|request_changes|comment` is fine but underspecified
What does the model do when it finds **one critical bug and five passing checks**? Today, the answer depends entirely on the prompt. Bake the decision rule into the schema description:

- `approve` — no findings of severity ≥ medium
- `request_changes` — any finding of severity = high, OR ≥ 3 findings of severity = medium
- `comment` — only nits / questions / suggestions; no blocking issues

Models follow explicit thresholds much better than implicit ones.

### 5. No place for "questions to the author" or "out-of-scope observations"
Real reviews routinely include "I wasn't sure why you did X — can you explain?" These aren't bugs, suggestions, or approvals. Without a slot, the model either crams them into `suggestions` (polluting actionable items) or drops them. Add an optional `questions` array.

### 6. No diff/scope acknowledgement
The schema doesn't ask the model to state **what it actually reviewed**. For multi-file diffs this matters: if the model silently skips a file (context-window pressure, distraction), you have no signal. Add `reviewed_files: string[]` and ideally `skipped_files: {path, reason}[]`.

### 7. JSON-in-prose risks for code snippets
When the model embeds code in `suggested_fix`, you'll hit escaping bugs (backslashes, quotes, newlines) — common with strict JSON output. Two mitigations:

- Use **tool_use / structured outputs** rather than raw JSON-in-text. With Sonnet 4.6 you should be using the structured output API (or a tool schema), not asking the model to print JSON in a code fence.
- For code, prefer `suggested_diff` as a unified-diff string with explicit newline handling, and validate it parses before accepting.

### 8. Adaptive thinking interaction
With adaptive thinking enabled, the model's reasoning lives in `thinking` blocks separate from the final JSON. Don't try to also force a "reasoning" field inside the schema — you'll duplicate work and the in-schema version will be a worse summary of the real trace. If you want to *expose* reasoning to consumers, surface the thinking blocks alongside the JSON, don't bake them in.

---

## Revised Schema

```json
{
  "summary": "<1-3 sentence TL;DR for humans>",
  "reviewed_files": ["src/foo.py", "src/bar.py"],
  "skipped_files": [
    {"path": "vendor/big.js", "reason": "generated/minified"}
  ],
  "findings": [
    {
      "id": "F1",
      "file": "src/foo.py",
      "line_range": [42, 58],
      "category": "correctness | security | performance | style | testing | docs | maintainability",
      "severity": "high | medium | low",
      "title": "<short, ≤80 chars>",
      "description": "<2-5 sentences explaining the issue>",
      "suggested_fix": "<concrete change, prose>",
      "suggested_diff": "<optional unified diff>"
    }
  ],
  "questions": [
    {"file": "src/foo.py", "line_range": [12, 12], "question": "..."}
  ],
  "final_decision": {
    "action": "approve | request_changes | comment",
    "rationale": "<1-2 sentences tying action to findings>",
    "confidence": "low | medium | high"
  }
}
```

### Why this is better
- Every finding is independently routable, dedupable, and renderable as a PR comment.
- Severity lives where it belongs (per finding) and aggregates into `final_decision.action` via an explicit rule you put in the prompt.
- `confidence` is bucketed, not a fake float.
- `questions` keeps non-actionable observations out of `findings`.
- `reviewed_files` / `skipped_files` give you an audit trail and lets you detect silent skips.
- `category` enables long-term metrics (e.g., "what % of our reviews surface security issues?").
- The thinking trace handles the "show your work" need natively.

---

## Implementation Notes

1. **Use structured outputs / tool_use, not JSON-in-text.** Define this as a tool schema; Sonnet 4.6's structured output adherence is far better than free-form JSON emission, especially for nested arrays.
2. **Provide a one-shot exemplar in the prompt** showing a realistic review of a small diff. Models follow schemas dramatically better with one concrete example than with descriptions alone.
3. **Validate before accepting.** Reject and retry once if (a) JSON doesn't parse, (b) any `findings[i].line_range` is outside the diff, (c) `action` violates the severity-aggregation rule. Pair this with a deterministic post-processor.
4. **Cap `findings` length** (e.g., max 25) to prevent the model from padding a clean diff with nits. State the cap in the schema description.
5. **Skip a `confidence` field entirely on the first iteration.** Measure whether you actually use it downstream before adding it back. Most teams discover they never threshold on it.

---

## TL;DR

The current schema treats a code review as two essays plus a verdict. A code review is really **a list of localized findings plus a verdict** — model the data that way. Replace the prose blobs with a `findings[]` array, move severity per-finding, bucket confidence (or drop it), add `questions` and `reviewed_files`, and emit via structured outputs rather than raw JSON.
