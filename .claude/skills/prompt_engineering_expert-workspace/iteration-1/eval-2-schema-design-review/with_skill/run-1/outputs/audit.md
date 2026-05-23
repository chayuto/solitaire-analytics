# Prompt audit: code-review response schema for Claude Sonnet 4.6 (adaptive thinking enabled)

The schema is small, which is good, but two of its four fields work against the target model. Sonnet 4.6 with adaptive thinking does CoT internally; the two long prose fields (`analysis` + `suggestions`) duplicate that work in the output, compete for the same response budget, push the parseable decision to the end where truncation risk lives, and overlap in scope enough that the model will say similar things in both.

## Top 1–2 highest-leverage fixes

1. **Delete `analysis`. Keep `suggestions` only if downstream actually uses prose feedback** — otherwise drop both prose fields and let the model's internal thinking carry the reasoning. With adaptive thinking on, you do not need (or want) visible CoT in the JSON.
2. **Enforce the remaining schema via Anthropic Structured Outputs**, not by pasting a JSON example into the prompt. Define `severity` and `action` as enums and `confidence` with `minimum`/`maximum` so the values cannot drift.

## HIGH severity

**1. Visible reasoning prose required from a thinking model (`analysis`, `suggestions`)** — The schema asks Sonnet 4.6 to emit 400–1000 words of prose in two separate string fields before the structured decision. Sonnet 4.6 with adaptive thinking already produces a private reasoning channel; demanding visible reasoning in the output duplicates that work, eats the output budget, and incentivizes the model to spread the same content across both fields.
- Why it fails: Anthropic's guidance for 4.6+ is to trust adaptive thinking and tune it via API parameters (`thinking.budget_tokens`), not to ask for visible CoT in the response [1]. Practitioner guidance for reasoning models is the same — visible CoT in the output competes with internal CoT for budget and adds noise [8]. Two overlapping prose fields then trigger the "redundant prose fields" failure: the model duplicates content across `analysis` and `suggestions`, biasing the final decision toward whichever framing dominated [3].
- Fix: If you need an audit trail for humans, capture it from the thinking channel out-of-band, not in the schema. If downstream tools actually consume prose suggestions, keep ONE field and name it for its consumer:
  ```json
  {
    "severity": "low|medium|high",
    "action": "approve|request_changes|comment",
    "confidence": 0.0,
    "review_comment": "<<=150 words, addressed to the PR author, actionable changes only>>"
  }
  ```
  No `analysis`. The model's internal thinking is the analysis.

**2. Schema described in prose instead of enforced via Structured Outputs** — The draft is shown as a literal JSON example with placeholder values (`"<200-500 word string of …>"`, `"low|medium|high"`, `0.0-1.0`). On Claude 4.6+, that is the deprecated pattern; Anthropic moved JSON enforcement to Structured Outputs.
- Why it fails: Pasted-schema-as-text wastes tokens re-describing the contract, drifts under load (keys get renamed or dropped on long outputs), and silently corrupts when prose strings contain colons/braces [13][6]. With Structured Outputs you get 100% schema-valid output and the model spends its budget on content, not on remembering field names.
- Fix: Pass the schema via `response_format` (Anthropic Structured Outputs) with `action` and `severity` as enums and `confidence` as a bounded number. Stop describing it in the prompt body.
  ```json
  {
    "type": "json_schema",
    "json_schema": {
      "name": "code_review_decision",
      "strict": true,
      "schema": {
        "type": "object",
        "additionalProperties": false,
        "required": ["severity", "action", "confidence"],
        "properties": {
          "severity":   { "type": "string", "enum": ["low", "medium", "high"] },
          "action":     { "type": "string", "enum": ["approve", "request_changes", "comment"] },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "review_comment": { "type": "string", "maxLength": 1200 }
        }
      }
    }
  }
  ```

## MEDIUM severity

**3. `severity` and `action` are unconstrained: their relationship is implicit** — `severity: "high"` paired with `action: "approve"` is almost certainly a bug, but the schema permits it. Same for `severity: "low"` + `action: "request_changes"`. Sonnet 4.6 will mostly get this right, but "mostly" is the wrong target when the field controls merge gating.
- Why it fails: Implicit conventions in the schema mean each call re-derives the policy; behavior drifts across temperature settings and across prompt revisions. Operational predicates outperform implicit norms [3].
- Fix: Either collapse the two fields into one composite enum, or state the mapping as a rule in the prompt and have the model self-check it (see #6). Composite form:
  ```json
  "decision": {
    "type": "string",
    "enum": [
      "approve_clean",
      "approve_with_nits",
      "comment_only",
      "request_changes_minor",
      "request_changes_major",
      "request_changes_blocking"
    ]
  }
  ```
  Whoever calls the API can derive `severity` and `action` from this without the model having to keep them consistent.

**4. `final_decision` nesting buys nothing** — Wrapping `action` and `confidence` inside a `final_decision` object adds two characters of nesting and no value. Flat schemas are easier to validate, log, and diff.
- Why it fails: Information-density anti-pattern — structure that doesn't carry signal still costs tokens and adds a layer to every downstream consumer.
- Fix: Flatten. `action` and `confidence` at the top level (shown above).

**5. `confidence: 0.0-1.0` is a vague calibration target** — Without anchors, the model's confidences will cluster around 0.7–0.9 regardless of actual certainty. Useful as a downstream filter only if calibrated.
- Why it fails: Confidence asks help only when the model knows what each value means. Otherwise the field becomes noise [7.3 in patterns.md].
- Fix: Anchor the scale in the system prompt:
  ```
  confidence reflects how likely a senior reviewer would reach the same decision:
    0.9-1.0  near-certain (clear bug or clear approve)
    0.7-0.89 confident but a second opinion could differ on tone/severity
    0.5-0.69 genuinely uncertain — flag for human review
    <0.5     do not auto-action; treat as comment_only regardless of `action`
  ```
  Then route on the threshold downstream.

**6. No self-check / verification step** — For an action that gates merging, a cheap end-of-prompt verification reliably cuts errors.
- Why it fails: Without a verification ask, the model commits the first action that "felt right" after reasoning. Self-check is one of the cheapest known accuracy wins on coding/decision tasks [1].
- Fix: Add to the prompt (not the schema):
  ```
  Before finalizing, verify:
  1. severity reflects the worst issue you found, not the average
  2. action is consistent with severity (high severity => request_changes; low severity => approve or comment)
  3. confidence is below 0.7 if you are guessing about runtime behavior you cannot see
  ```

## LOW severity

**7. `comment` as an action value is ambiguous** — Does it mean "I have nothing blocking, just notes" or "I declined to decide"? Two consumers will read it two ways.
- Fix: Rename to `comment_only` (or use the composite enum from #3 which removes the ambiguity).

**8. No field for which files/lines the review addresses** — Optional, but if this output drives a PR comment, the reviewer almost always wants location anchors. Worth considering an array of `{file, line, message}` instead of (or alongside) a single prose field. Out of scope of a "good schema?" question, mentioned for awareness.

## What's already good

1. **The schema is small.** Four top-level fields, no nested arrays of objects, no per-file repetition. That is the right baseline for a decision endpoint.
2. **`severity` and `action` are enumerated rather than free-text.** This is the correct instinct — keep going further with it (enforce via Structured Outputs).
3. **A `confidence` field exists at all.** Many production review-bot schemas omit this and then have no way to gate auto-merge. Keep it; just anchor the scale (#5).
4. **No `reasoning_trail` / `history` / `prior_attempts` field.** This is a common code-review-bot footgun (accumulating prior bot rationales across PR iterations). The schema as drafted avoids it — do not add one later [4][14].

## Sources

- [1] Anthropic, Claude prompting best practices — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
- [3] Maxim AI, A Practitioner's Guide to Prompt Engineering in 2026 — https://www.getmaxim.ai/articles/a-practitioners-guide-to-prompt-engineering-in-2025/
- [4] Halawi et al., Overthinking the Truth — https://arxiv.org/pdf/2307.09476
- [6] Pockit Tools, LLM Structured Output in 2026 — https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk
- [8] Helicone, How to Prompt Thinking Models — https://www.helicone.ai/blog/prompt-thinking-models
- [13] Anthropic, Structured Outputs documentation — https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- [14] Emergent Misalignment via In-Context Learning (2025) — https://arxiv.org/pdf/2510.11288
