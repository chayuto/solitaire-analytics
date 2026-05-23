# Per-model calibration notes

Best-practice recommendations differ meaningfully across models. The
biggest split is **reasoning models** (those with native internal CoT)
vs **non-reasoning models**. Within reasoning models, each provider has
quirks worth knowing.

Citations refer to `references/citations.md` numbered list.

## The reasoning-vs-non-reasoning split

| | Reasoning models | Non-reasoning models |
|---|---|---|
| Examples (2026) | Gemma 4 IT (with `<\|think\|>`), Claude 4.6+ with adaptive thinking, GPT-5, o-series, DeepSeek R1 | GPT-3.5/4 base, smaller open-weights without thinking, Gemma 4 IT with thinking disabled |
| Visible CoT in output | **Anti-pattern.** Duplicates internal work, competes for output budget. | **Recommended.** Use `<thinking>` scratchpad before the answer. |
| "Think step by step" in prompt | Largely vestigial. Set thinking budget via API parameters instead. | Still useful. |
| `<thinking>` tags around scratchpad | Skip — the model does this internally and may strip the tags. | Use them. |
| Effort/thinking controls | `effort`, `reasoning_effort`, `thinking_budget` — set at API level | N/A |

If you don't know which family the target model is in, ask the user.
For unfamiliar models, default to the non-reasoning advice (safer; works
either way).

---

## Gemma 4 IT (the production model behind this skill's motivating audit)

Documentation: [7] `https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4`

### Critical specifics

- **Use the official chat template.** Hand-rolled formats degrade quality. The template uses `<|turn>system / <turn|>` markers.
- **Enable thinking with `<|think|>` in the system block** if you want internal CoT.
- **Strip the model's `<|channel|>thought ... <channel|>` from history before the next turn.** Feeding the thought channel back interferes with the next turn's reasoning. This is the single most common Gemma 4 deployment bug.
- **Multimodal content goes BEFORE text.** The chat template expects images first, then text questions.

### Common Gemma 4 31B failure modes

- **Long-prompt instability** — at ~7KB+ prompts, HTTP 500 rates climb to ~75% on the AI Studio API. The model itself is fine when it responds; the gateway is the issue. Build in retries.
- **Free-tier rate limit** — 20 requests/minute on AI Studio. Add 4-second sleeps between calls or parse the `retry-in Ns` from 429 responses.
- **Anchoring on in-context examples** — Gemma is more susceptible to imitating prior model outputs in the context window than Claude or GPT-5. This is the failure mode behind the motivating Klondike audit.

### Gemma-specific audit checks

When auditing a prompt destined for Gemma 4 IT, prioritize:
1. Are model thought tokens being stripped from multi-turn history?
2. Is the chat template official, not hand-rolled?
3. Multimodal content placed before text?
4. In-context prior-output fields (history, reasoningTrail, etc.) — especially watch for these on Gemma.

---

## Claude (Anthropic, 4.6 / 4.7+)

Documentation: [1] `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices`

### Critical specifics

- **Structured Outputs replaces JSON-mode prefill.** Anthropic deprecated prefill-for-JSON in 4.6+. Migrate.
- **Adaptive thinking is the lever, not "think step by step".** Tune `effort` at the API level.
- **Aggressive language causes overtriggering on 4.5+/4.6+.** Use normal prose, not "CRITICAL: YOU MUST."
- **Prompt caching** is automatic for stable system prompts. Don't put dynamic state in the system prompt.

### Claude-specific audit checks

1. Is the response schema using Structured Outputs, or pasting a schema as text?
2. Any aggressive CAPS emphasis (overtriggering risk)?
3. Is dynamic data in the system prompt (caching defeat)?
4. For thinking-enabled use: is the schema asking for visible reasoning prose? (Anti-pattern — see 3.1.)

---

## GPT-5 / o-series (OpenAI)

Documentation: [10] `https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide`

### Critical specifics

- **`reasoning_effort` + `verbosity` are the new levers.** Don't burn prompt tokens on "think very carefully" — set the parameter instead.
- **Remove output schema definitions from the prompt where possible** — use Structured Outputs instead. [10]
- **Conflicting rules are expensive.** GPT-5 burns visible reasoning tokens reconciling contradictions. Numbered priority lists matter more here than on older models.
- **Tool-use prompts converge with Anthropic patterns.** Tool budgets ("max 2 calls"), explicit persistence vs. ask-for-help, per-tool when-to-use criteria.

### GPT-5-specific audit checks

1. Is `reasoning_effort` set appropriately, or is the prompt asking for reasoning via text?
2. Is the schema enforced via Structured Outputs?
3. Are any rules unranked (priority list missing)?

---

## DeepSeek R1 / o-series-class reasoning models

Documentation: [8] `https://www.helicone.ai/blog/prompt-thinking-models`

### Critical specifics

- **Reasoning happens before the answer in a `<think>` channel.** Don't ask for it again in the output.
- **Prompts work best when they're focused on the GOAL, not the METHOD.** Outcome-first specification (see `patterns.md` 5.5) matters more here than on non-reasoning models.
- **Shorter prompts often work better.** The model's reasoning chain is long; over-specified instructions add noise.

### Reasoning-model-specific audit checks

1. Is the output schema asking for `reasoning: string`? (Anti-pattern.)
2. Is the prompt over-prescribing the approach? (Outcome-first is better.)
3. Is the prompt unnecessarily long? Try the same prompt at half the length.

---

## Generic non-reasoning models (GPT-4 base, smaller open-weights)

### Critical specifics

- **`<thinking>` scratchpad is essential** for non-trivial reasoning tasks. The model has no internal CoT to fall back on.
- **Few-shot examples matter more.** With no internal reasoning, the model leans hard on in-context examples.
- **Schema descriptions in prompt are often the only option** if the provider doesn't support constrained decoding.

### Non-reasoning model audit checks

1. Is there an explicit `<thinking>` block in the response format?
2. Are few-shot examples present, diverse, and labeled?
3. If the provider supports constrained decoding, is it being used?

---

## When the user hasn't named the model

If the user shares a prompt without naming the model:
1. Ask once: "Which model is this going to?" — the audit depends on it.
2. If they don't know or refuse, default to the **non-reasoning** advice (safer; works on both classes).
3. Note in the audit which findings would change if the model is in the other class. E.g.: "If this is going to a reasoning model (Claude 4.6+ adaptive thinking, GPT-5, Gemma 4 IT with thinking enabled), then the `reasoning: string` field in your schema is an anti-pattern. If non-reasoning, it's fine."
