# Citations

Numbered reference list for the skill's anti_patterns.md, patterns.md,
and model_notes.md files. When citing in an audit, use these IDs
(e.g. "[1][4]") and surface the URL in the audit's "Sources" section.

## Primary sources (provider documentation)

[1] **Anthropic — Claude prompting best practices** (Opus 4.7 / 4.6 / Sonnet 4.6 / Haiku 4.5)
`https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices`
The most-cited authoritative source for prompting in this skill.
Covers: XML tags, positive vs negative framing, explaining the why,
adaptive thinking, structured outputs migration, examples diversity,
self-check pattern, aggressive-language warning.

[2] **Anthropic — Long context tips**
`https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips`
Source for the "data first, instructions last" pattern and the ~30%
quality measurement for long-context layouts.

[7] **Google — Gemma 4 prompt formatting**
`https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4`
The official Gemma 4 chat template. Source for the thought-channel
stripping requirement, multimodal-before-text ordering, and the
`<|think|>` activation flag.

[10] **OpenAI — GPT-5 prompting guide (cookbook)**
`https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide`
Source for `reasoning_effort`/`verbosity` parameters, "remove schema
from prompt" guidance, conflict-resolution cost, tool-use patterns.

[13] **Anthropic — Structured Outputs documentation**
`https://platform.claude.com/docs/en/build-with-claude/structured-outputs`
Reference for the migration from prefill-for-JSON to Structured
Outputs in Claude 4.6+.

## Research papers (2023–2025)

[4] **Halawi et al., "Overthinking the Truth: Understanding how Language Models Process False Demonstrations"** (arXiv 2307.09476)
`https://arxiv.org/pdf/2307.09476`
Empirical evidence that models faithfully imitate in-context examples
even when they "know" the examples are wrong. Foundational citation
for the reasoning-trail anti-pattern.

[5] **"Understanding In-Context Learning from Repetitions"** (arXiv 2310.00297)
`https://arxiv.org/pdf/2310.00297`
Shows that even structural repetition (not just semantic content)
primes imitation. Supports the case for capping/labeling repetitive
history fields.

[14] **"Emergent Misalignment via In-Context Learning"** (arXiv 2510.11288, 2025)
`https://arxiv.org/pdf/2510.11288`
Recent (2025) reinforcement of the in-context-bad-example anchoring
problem. Demonstrates that misalignment can emerge purely from
unlabeled examples in the context window.

## Practitioner guides

[3] **Maxim AI — A Practitioner's Guide to Prompt Engineering in 2026**
`https://www.getmaxim.ai/articles/a-practitioners-guide-to-prompt-engineering-in-2025/`
Production-team perspective. Source for "operational predicates
instead of soft heuristics" framing and several modern-models
specifics.

[6] **Pockit Tools — LLM Structured Output in 2026: Stop Parsing JSON with Regex**
`https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk`
Modern survey of constrained-decoding approaches across providers
and open-weights. Useful for the "use grammars for local Gemma"
recommendation.

[8] **Helicone — How to Prompt Thinking Models like DeepSeek R1 and OpenAI o3**
`https://www.helicone.ai/blog/prompt-thinking-models`
Practical guide for reasoning-model-specific prompting. Source for
"outcome-first specification" and "shorter prompts often work better"
on reasoning models.

[9] **IntuitionLabs — LLM Position Bias: Primacy and Recency Effects in Prompts**
`https://intuitionlabs.ai/articles/llm-position-bias-primacy-recency-effects`
Survey of position bias research. Source for the U-shaped attention
curve and lost-in-the-middle framing.

[12] **CodeConductor — Structured Prompting Techniques: XML & JSON**
`https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/`
Cross-provider convergence evidence for XML-tagged structural blocks.

## Use guidance

When writing an audit:
- Cite at least one source for every HIGH-severity finding.
- Prefer [1] (Anthropic prompting docs) and [10] (GPT-5 guide) for
  general best practices — they're the most authoritative.
- Use [4] / [5] / [14] specifically for the in-context-anchoring
  finding family (reasoningTrail, history fields, prior_thoughts).
- Use [7] for any Gemma-specific finding.
- Use [13] for any Structured Outputs / JSON-mode finding.
- If you find yourself citing a source that's not in this list,
  add it here (numbered next, e.g. [15]) and reference it from the
  audit. Update the catalog files if the source supports a pattern
  worth permanent inclusion.
