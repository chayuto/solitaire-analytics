---
name: prompt_engineering_expert
description: Use this skill for ANY situation involving LLM prompts, response schemas, or LLM misbehavior diagnosis. Trigger eagerly — do not wait for the user to say "audit". USE IMMEDIATELY when (1) the user pastes a prompt, system prompt, response schema, few-shot example, or JSON template for any reason; (2) the user describes ANY LLM misbehavior — model loops, repeats itself, ignores instructions, keeps apologizing, drifts, picks wrong from a numbered list, hallucinates JSON keys, gives identical answers to different inputs, infinite loops, parroting; (3) the user is writing a new system prompt or response schema for production; (4) the user mentions "review/audit/critique/improve/evaluate/check/look at" anywhere near a prompt; (5) the user asks about structured outputs, JSON mode, function calling, few-shot examples, reasoning trail, history accumulation, chain-of-thought, or schema design; (6) the user names a model (Claude/GPT/Gemma/Gemini/DeepSeek/o-series/Llama) alongside a behavior problem. Produces a severity-ranked audit (HIGH/MEDIUM/LOW) with concrete fixes and citations to Anthropic/OpenAI docs + recent arxiv research (Halawi et al., 2025 misalignment papers). When in doubt about whether to trigger, TRIGGER — undertriggering is the documented failure mode.
---

# Prompt Engineering Expert

Audit LLM prompts against 2026-era best practices. Produce a structured,
severity-ranked report with concrete fixes and citations. The audit must
be specific to the prompt at hand — not a generic lecture.

## What you need from the user

Before auditing, confirm you have:
1. The **full prompt** — system + user + any templates. Partial prompts produce partial audits.
2. The **target model** — best practices differ between reasoning/thinking models (Claude 4.6+ adaptive thinking, GPT-5, o-series, Gemma 4 IT, DeepSeek R1) and non-reasoning models. Asking a thinking model for visible CoT in the output is an anti-pattern; asking a non-reasoning model for it is essential.
3. The **response schema** (if any).
4. *(Optional)* Observed failure modes — the user's symptoms point at which findings are load-bearing.

If a critical piece is missing, ask once. Don't speculate at length about what the prompt "probably" contains.

## The audit workflow

1. **Read the whole prompt end-to-end first.** Don't comment on individual lines until you've seen the overall shape. Many issues are structural (where things sit) more than local (what they say).
2. **Identify the structural pattern** — system+user split, XML-tagged sections, JSON-embedded rules, free prose. The same content has very different effects depending on where it sits.
3. **Run the high-frequency checklist** in this file. For deeper coverage, read `references/anti_patterns.md` and `references/patterns.md`.
4. **Calibrate to the target model** — read `references/model_notes.md` if you're unfamiliar with the model's quirks. Reasoning vs non-reasoning is the biggest split.
5. **Classify each finding by severity**:
   - **HIGH** — directly causes wrong outputs or known failure modes (e.g. in-context bad-example anchoring, rules buried beyond the model's attention reach)
   - **MEDIUM** — wastes tokens, adds variance, hurts debugging (e.g. redundant prose fields, vague hedged heuristics)
   - **LOW** — stylistic improvements
6. **Write the report** using the format below.
7. **Pick the 1–2 highest-leverage fixes** and surface them at the top. The user will skim — give them the load-bearing answer first.

## Output format

Default to this template; adapt section names if the prompt is narrow (e.g. only a response schema) but keep the severity grouping and the top-line summary:

```markdown
# Prompt audit: [one-line context]

## Top 1–2 highest-leverage fixes
1. [the single most impactful change in one sentence]
2. [optional second]

## HIGH severity
**N. [finding name]** — [what's wrong, one paragraph]
- Why it fails: [explanation; cite a source from references/citations.md]
- Fix: [concrete, code/spec-level — paste actual replacement text or schema, not just principles]

## MEDIUM severity
[same format]

## LOW severity
[same format]

## What's already good
[1–3 things the prompt does well — surface these so feedback isn't pure negativity, and so the user knows what NOT to undo when fixing]

## Sources
- [bulleted URLs for the citations used in the audit]
```

## High-frequency anti-pattern checklist

These are the ones to check on every audit. The full catalog is in
`references/anti_patterns.md` (~25 patterns). Read it if the prompt
seems unusual or if you finish this checklist with too few findings.

1. **Reasoning trail / history fed as plain context.** Accumulating prior model rationales without explicit framing teaches the model to imitate them. This is in-context few-shot learning of the model's own bad behavior. Giveaway: a field like `history`, `reasoningTrail`, `prior_thoughts`, `previous_attempts` that isn't explicitly marked as "patterns to avoid." See Halawi et al. (`references/citations.md`, [4]).

2. **Rules buried inside the data payload.** Critical interpretation rules (notation, coordinate system, units) placed as JSON fields the model encounters after parsing 1000+ tokens of data. Giveaway: keys like `notation`, `interpretation_note`, `important`, `warning` inside a `state` or `data` object. Move to the rules preamble.

3. **Redundant prose fields in the response schema.** Asking for two prose-text keys that overlap in purpose — e.g. `analysis` + `plan` + `decision`. The model duplicates content across them, inflating tokens and risking truncation. Giveaway: response schema with multiple long-string keys before the final structured decision.

4. **Soft hedged heuristics.** "Be cautious", "too early", "or lead nowhere", "consider avoiding", "might want to". These are interpreted differently each call and give the model permission to ignore. Giveaway: modal verbs without operational thresholds. Fix: replace with a hard predicate ("skip this move if it requires moving 3+ cards back from the foundation").

5. **Negative-only formatting instructions.** "Don't use markdown", "Don't apologize", "Don't use emojis". Models sometimes pattern-match on the forbidden token. Giveaway: rules phrased only as prohibitions. Fix: show what the correct output looks like.

6. **Free-form CoT requested INSIDE the JSON output.** Asking the model to write 200-word strings as fields of a JSON object. Format-switching mid-thought, escape-character risk, and the parseable structured bit hangs over a wall of prose that may truncate. Giveaway: response schema with `reasoning: string` (or `board_analysis`, `strategic_plan`, etc.) as nested keys. Fix: put reasoning OUTSIDE the JSON (e.g. in a `<thinking>` block), keep the JSON tiny and at the end.

7. **Pasted JSON schema as text + free-form generation.** Describing the schema in the prompt while not using Structured Outputs / constrained decoding. Giveaway: prompt contains a JSON example with placeholder `<string>` or `<number>` values. Fix: use the provider's schema-enforcement (Anthropic Structured Outputs, OpenAI Structured Outputs, Outlines/llama.cpp grammars for local Gemma).

8. **Conflicting rules without precedence.** "Always do X" and "never do X when Y" with no hierarchy. Modern reasoning models burn tokens reconciling. Giveaway: any unranked "always"/"never". Fix: numbered priority list with explicit precedence.

9. **Aggressive all-caps emphasis.** "CRITICAL: YOU MUST" peppered throughout. On Claude 4.5+/4.6+, Anthropic explicitly: aggressive language causes overtriggering. Giveaway: more than 2–3 instances of CAPS emphasis.

10. **Dynamic data in the system prompt.** Per-call variable state inside the durable system prompt. Defeats prompt caching, confuses role boundaries. Giveaway: the system prompt contains values that change between calls.

11. **Asking a thinking model for visible CoT.** Reasoning models (Gemma 4 IT, Claude 4.6+ adaptive thinking, GPT-5, o-series, DeepSeek R1) do CoT internally. Demanding a `reasoning` field duplicates work and competes for output budget. Giveaway: model is in the thinking-capable list AND the schema asks for visible reasoning prose.

12. **Few-shot examples that all illustrate one path.** Three examples all picking "draw card" teach the model "always draw." Giveaway: examples are homogeneous in their answer or template. Fix: diversify, or label examples explicitly as "patterns of when X applies" not "do X."

## High-frequency missing-best-practice checklist

What should usually be there but often isn't:

1. **XML-tagged structural blocks** — `<instructions>`, `<rules>`, `<context>`, `<data>`, `<output_format>` give the model unambiguous role boundaries. Recommended by Anthropic, OpenAI, and the Gemma chat template.

2. **Long data first, query last (inputs >~5KB)** — Anthropic measures up to 30% better response quality with this layout. Recency bias is real; instructions before a wall of data get diluted.

3. **Output schema enforced via Structured Outputs / constrained decoding**, not described in prose. 100% schema validity, no token waste.

4. **Explanation of WHY each rule exists.** "Your response goes to TTS, so never use ellipses" generalizes better than "Never use ellipses." Smart models extend a rule sensibly to unseen cases when they know its purpose.

5. **Self-check / verification step.** "Before finalizing, verify your answer satisfies criteria X, Y, Z." Cheap, reliable error reduction.

6. **Echo-back of choice.** When picking from a numbered list, require the model to restate the chosen item's description, not just its index. Catches index-misalignment errors.

## Calibration to model

The audit's severity depends on the target model. Consult `references/model_notes.md` for per-model specifics. The critical split:

- **Reasoning/thinking models**: asking for reasoning IN the output is an ANTI-PATTERN (they CoT internally). Force-thinking parameters (`effort`, `reasoning_effort`, adaptive thinking) replace prompt-side "think step-by-step" hacks.
- **Non-reasoning models**: still benefit from explicit `<thinking>` scratchpad BEFORE the answer, separate from the structured output.
- **Gemma 4 IT specifically**: use the official chat template; strip thought channels from history before the next turn; multimodal goes before text.

## When in doubt

- **Concrete over abstract.** "Drop the 200-word `analysis` field, keep `decision`" beats "consider whether all schema fields pull their weight." Show the user what to paste in, not just what to consider.
- **Default unclear severity to MEDIUM.** HIGH should be reserved for things you can name a failure mode for ("causes oscillation when the chosen field accumulates across turns").
- **Surface what's good, briefly.** A pure negative report tempts the user to throw out load-bearing parts of the prompt.
- **Cite.** Every HIGH finding deserves a source link. The user (and you on the next audit) trust findings more when they're sourced.

## Reference files

Read these when the in-line checklists don't cover what you're seeing:
- `references/anti_patterns.md` — full catalog of ~25 anti-patterns with examples and fixes
- `references/patterns.md` — full catalog of recommended patterns
- `references/model_notes.md` — per-model calibration (Gemma 4 IT, Claude, GPT-5/o-series, DeepSeek, generic non-reasoning)
- `references/citations.md` — source URLs grouped by topic, numbered for inline reference
