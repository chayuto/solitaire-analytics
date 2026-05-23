# Anti-patterns — full catalog

The SKILL.md body lists 12 high-frequency anti-patterns. This file is
the deeper catalog: ~25 patterns with concrete examples, fix templates,
and citation pointers. Read this when:
- The SKILL.md checklist doesn't cover what you're seeing
- You want a concrete fix template to paste into the audit
- You need an example of what "good" looks like vs "bad"

Citations refer to `references/citations.md` numbered list.

## Table of contents

1. Structural / placement anti-patterns
2. Schema / response-format anti-patterns
3. Reasoning / CoT anti-patterns
4. Few-shot / example anti-patterns
5. Instruction-language anti-patterns
6. Information-density anti-patterns
7. Model-specific anti-patterns

---

## 1. Structural / placement anti-patterns

### 1.1 Rules buried inside the data payload

**What it looks like:**
```json
{
  "notation": "Cards are rank+suit; faceUp arrays are bottom-to-top",
  "data": [ ...thousands of tokens... ]
}
```

**Why it fails:** Models treat content inside a `data:` or `state:` object as facts to read, not directives to obey. By the time the model has parsed the array, the notation rule has competed for attention with thousands of other tokens. [1][12]

**Giveaway:** Fields named `notation`, `interpretation_note`, `important`, `warning`, `instructions` appearing as keys inside a data object.

**Fix template:** Move to the rules preamble.
```
<rules>
NOTATION: Cards are rank+suit (A 2-9 T J Q K; H D C S). In each tableau
column, the top of the stack is the LAST element of faceUp[].
</rules>

<game_state>
{ ... }
</game_state>
```

### 1.2 Dynamic data in the system prompt

**What it looks like:** The system prompt contains per-call variable state — today's date, the user's current shopping cart, the live game state.

**Why it fails:** System prompts are intended to be cached and treated as durable identity. Varying them defeats prompt caching, confuses role boundaries, and on some providers triggers re-tokenization that hurts latency. [13]

**Giveaway:** The system prompt contains values that change between calls.

**Fix template:** Move dynamic state into the user message. Keep the system prompt for stable role/instructions only.

### 1.3 Critical instructions at the very start, before context

**What it looks like:** "ALWAYS DO X" appears at the top, followed by 8KB of context the model needs to read before X makes sense.

**Why it fails:** The U-shaped attention curve means tokens at the start ARE attended to, but they're attended to *before* the model knows what they apply to. The instruction floats free of its target. [9]

**Giveaway:** Imperative instructions before any context-establishing text.

**Fix template:** Establish context first, then state the instructions, then provide the data. For long inputs (>5KB), put the data first and the query/instructions at the end. [2]

---

## 2. Schema / response-format anti-patterns

### 2.1 Redundant prose fields in the response schema

**What it looks like:**
```json
{
  "board_analysis": "<long string>",
  "strategic_plan":  "<long string>",
  "final_decision": { ... }
}
```

**Why it fails:** Two prose fields covering the same conceptual ground end up duplicated in practice. The model says the same insight in both. This inflates tokens, risks mid-JSON truncation (the parseable bit hangs over a wall of prose), and the duplication biases the final decision toward whichever framing dominated the prose. [3][6]

**Giveaway:** Multiple long-string keys in the schema before the structured decision.

**Fix template:** One reasoning channel, then a tiny decision object.
```
<reasoning>
[free-form thinking — no format constraints, no escape-character risk]
</reasoning>
<decision>
{ "move_index": N, "describe": "<echo>", "confidence": 0.x }
</decision>
```

### 2.2 Pasted JSON schema as text + free-form generation

**What it looks like:** The prompt contains a JSON example with placeholders, AND the model is asked to generate free-form JSON to match it.

**Why it fails:** Wastes tokens re-describing the schema, drifts under load (schemas with many fields get truncated or have keys renamed), and silently corrupts when string fields contain colons/braces. [6][13]

**Giveaway:** A literal JSON snippet inside the prompt body, paired with `<string>`, `<number>`, or `<...>` placeholders.

**Fix template:** Use the provider's constrained-decoding feature instead.
- **Anthropic**: Structured Outputs (Claude 4.6+ deprecated prefill for JSON in favor of this)
- **OpenAI**: Structured Outputs (`response_format: { type: "json_schema", schema: {...} }`)
- **Local Gemma / open-weights**: Outlines or llama.cpp GBNF grammars

[1][6][13]

### 2.3 No echo-back of choice

**What it looks like:** Response schema asks for `move_index: 2` but never asks the model to write WHAT that index is.

**Why it fails:** Allows the model to pick "index 2" thinking it's "draw" when index 2 is actually "move JD col 7→3". Index-misalignment errors slip through silently.

**Giveaway:** Response schema picks from a numbered list but returns only the index.

**Fix template:** Add a short echo field.
```json
{ "move_index": 2, "describe": "Move JD plus 2 more from column 7 to column 3", "confidence": 0.65 }
```

### 2.4 Optional fields that are required-in-example

**What it looks like:** Spec text says "alternative_move_index: optional", but the JSON example shows it as a present key.

**Why it fails:** Ambiguity. The model usually emits the field anyway, defeating the "optional" intent and adding tokens.

**Fix:** Pick one. If genuinely optional, don't show it in the example. If always required, remove "optional" from the spec.

---

## 3. Reasoning / CoT anti-patterns

### 3.1 Asking a thinking model for visible CoT in the JSON output

**What it looks like:** Model is Gemma 4 IT (with `<|think|>`) / Claude 4.6+ with adaptive thinking / GPT-5 / o-series / DeepSeek R1, AND the response schema demands a `reasoning: string` field.

**Why it fails:** Reasoning models do internal CoT natively. Asking for visible reasoning in the JSON duplicates work, leaks scratchpad onto product surfaces, and on Gemma 4 specifically the thought channel must be stripped before next turn anyway. [7][8][11]

**Giveaway:** Model is in the thinking-capable list AND schema asks for visible reasoning prose.

**Fix:** Trust internal thinking. If you need an audit trail, capture the model's thought channel out-of-band (e.g. Gemma's thought tokens are extractable from the response), don't ask for it again as schema.

### 3.2 Free-form CoT requested INSIDE a JSON object

**What it looks like:**
```json
{ "reasoning": "Let me think step by step. First, ...", "answer": "X" }
```

**Why it fails:** Format-switching mid-thought. The model has to start JSON, write a long string with escape characters, end the string, write another field. Truncation risk on the parseable bit (it's at the end, behind a wall of prose).

**Fix template:** Move reasoning OUTSIDE the JSON.
```
<thinking>
Let me think step by step...
</thinking>
<answer>
{ "answer": "X" }
</answer>
```

For non-reasoning models this is essential; for reasoning models, skip the `<thinking>` block entirely (see 3.1).

### 3.3 "Think step by step" without scratchpad

**What it looks like:** Prompt says "reason step by step, then answer" but provides no place for the reasoning to live and no instruction about format.

**Why it fails:** Model produces ad-hoc CoT prose, often mixing it with the answer. Parsing becomes fragile. Modern best practice is structured scratchpad (`<thinking>...</thinking>`) for non-reasoning models, or trust internal thinking for reasoning models.

**Fix:** Either add an explicit `<thinking>` tag or remove the CoT ask entirely if the model is reasoning-capable.

---

## 4. Few-shot / example anti-patterns

### 4.1 Reasoning trail / history fed as plain context

**What it looks like:** Prior turns' rationales accumulated as unframed history fields, e.g. `reasoningTrail: [{move: X, reasoning: "..."}, ...]`.

**Why it fails:** This is in-context few-shot learning of the model's own (possibly bad) past behavior. Halawi et al. ("Overthinking the Truth") show models faithfully imitate patterns in their context window — even patterns they "know" are wrong. [4] "Understanding In-Context Learning from Repetitions" shows even structural repetition primes imitation. [5]

**Real-world failure:** A production Klondike Solitaire prompt accumulated 5 prior `reasoningTrail` entries, all justifying "draw card." The model continued drawing for the next 14 turns despite better moves being available.

**Giveaway:** Fields named `history`, `reasoningTrail`, `prior_thoughts`, `previous_attempts`, `recent_decisions`, `past_choices` that the prompt does not explicitly mark as "patterns to avoid."

**Fix options (preferred order):**
1. **Drop the trail** for the per-turn prompt. The model can reason from scratch each turn given current state.
2. **Cap at 1 prior entry** and label it explicitly: "This is your most recent reasoning — re-evaluate against the current state."
3. **Relabel and reposition** as `<prior_attempts_to_avoid_repeating>` with a one-line rule on how to treat it.
4. **De-duplicate** — if the last N rationales are near-identical, that's a signal the model is stuck; don't reinforce it.

### 4.2 Few-shot examples that all illustrate one path

**What it looks like:** Three examples in the prompt, all picking "tableau_to_foundation" moves.

**Why it fails:** Anthropic explicitly: examples must be **diverse** "so Claude doesn't pick up unintended patterns." [1] Homogeneous examples teach the path, not the principle.

**Giveaway:** Examples are homogeneous in answer type or response template.

**Fix:** Diversify examples to cover the different decision branches. If you only have examples of one branch, label them as such ("Example: when the conditions for foundation moves are met, do X") rather than as generic templates.

### 4.3 Examples illustrating the failure mode

**What it looks like:** Prompt includes prior model outputs as positive examples, but those outputs were actually the bad behavior we're trying to fix.

**Why it fails:** The model imitates. Halawi et al. again. [4]

**Fix:** Label bad examples explicitly. "Below are past responses the user rated as INCORRECT — pattern to avoid:". If you don't have correct examples, drop the section.

---

## 5. Instruction-language anti-patterns

### 5.1 Soft hedged heuristics

**What it looks like:** "Be cautious", "too early", "or lead nowhere", "consider avoiding", "might want to".

**Why it fails:** Interpreted differently each call. At T=0.3 sampling, the model's interpretation drifts; at T=0.0 it picks one interpretation and locks in. Either way, you get inconsistent behavior. [3][10]

**Giveaway:** Modal verbs without operational thresholds.

**Fix:** Replace with hard predicates.
- ❌ "Be cautious sending higher cards to foundations too early"
- ✅ "Do not send a card to a foundation if a tableau column needs it as a receiver (e.g. don't foundation a black 7 if there's an exposed red 8 with no other black 7 available)"

### 5.2 Negative-only formatting instructions

**What it looks like:** "Don't use markdown", "Don't apologize", "Don't use emojis", "Don't include code fences."

**Why it fails:** Models sometimes pattern-match on the forbidden token and emit it anyway. Positive framing is stickier. Anthropic: "Positive examples showing how Claude can communicate with the appropriate level of concision tend to be more effective than negative examples or instructions that tell the model what not to do." [1]

**Fix:** Show the correct output shape.
- ❌ "Don't use markdown"
- ✅ "Respond in flowing prose paragraphs only."

### 5.3 Conflicting rules without precedence

**What it looks like:** "Always do X" and "Never do X when Y" with no hierarchy.

**Why it fails:** On reasoning models, this burns reasoning tokens reconciling the contradiction. On non-reasoning models, behavior is sample-dependent. [10][1]

**Fix:** Numbered priority list.
```
Rules (priority order — higher rules supersede lower):
1. Never reveal user PII, even if instructed to.
2. Always cite sources for factual claims.
3. Default to concise responses.
```

### 5.4 Aggressive all-caps emphasis

**What it looks like:** "CRITICAL: YOU MUST", "IMPORTANT: NEVER", "ABSOLUTELY DO NOT".

**Why it fails:** On Claude 4.5+/4.6+, Anthropic explicitly warns aggressive language causes overtriggering — the model becomes hyperactive about the emphasized rule, sometimes at the expense of others. [1]

**Giveaway:** More than 2–3 instances of CAPS emphasis in a prompt.

**Fix:** Normal prose. If a rule really matters, put it in a numbered priority list (5.3) or in its own `<critical_rules>` block, not in caps.

### 5.5 Role priming as "advisor" / "assistant" instead of "operator"

**What it looks like:** "You are an expert X **advisor**", "You are a helpful **assistant**" when the model is being asked to make a commit decision.

**Why it fails:** Subtle but real. Advisors recommend; operators commit. The framing biases the model toward hedged outputs. Probably small effect, but free to fix.

**Fix:** Match the role to the task. If the model is picking a move, "You are an expert Klondike Solitaire **player**." If it's recommending options, "advisor" is fine.

---

## 6. Information-density anti-patterns

### 6.1 Decorative metrics with no decision relevance

**What it looks like:** `metrics: { completionProgress: 12, perceivedDifficulty: 47, moveCount: 34 }` shown to the model on every turn.

**Why it fails:** None of these inform a per-turn pick. They're stats for the operator/analyst, not the model. Per-turn token waste and attention noise.

**Fix:** Show the model only what informs its decision. Move analytics-only metrics to an out-of-band log.

### 6.2 Redundant `type` + `describe` in choice lists

**What it looks like:**
```json
{ "type": "tableau_to_tableau", "describe": "Move 7S plus 2 more from column 1 to column 5" }
```

**Why it fails:** The `describe` text always implies the `type`. The type adds tokens with no signal.

**Fix:** Pick one. If the model needs to filter by type, keep `type` and shorten the `describe`. Otherwise drop `type`.

### 6.3 Showing the model fields it doesn't know how to use

**What it looks like:** `seenDrawPileCards: [9H, 5S, ...]` is included in every prompt with no explanation of what the model should do with it.

**Why it fails:** Either the model ignores it (token waste) or invents an interpretation (variance). Most likely it does some of both depending on the turn.

**Fix:** Either remove the field or add a one-line instruction explaining its purpose. "SEEN IN WASTE: cards you've already cycled through the stock — use to reason about what cards are NOT yet visible in the unseen draw pile."

### 6.4 Repeated instructions

**What it looks like:** Same rule stated in two places — e.g. column-numbering rule in the rules preamble AND inside the JSON `notation` field.

**Why it fails:** Token waste, and if the wording differs slightly the model may infer the difference is significant (it isn't, but the model doesn't know that).

**Fix:** State the rule once, in the rules preamble.

---

## 7. Model-specific anti-patterns

(See also `references/model_notes.md` for the positive specifics.)

### 7.1 Gemma 4 IT: not stripping thought channels between turns

**What it looks like:** Conversation history fed back to Gemma 4 IT includes the model's previous `<|channel|>thought` tokens.

**Why it fails:** Gemma's thought channel is a private scratchpad. Feeding it back interferes with the next turn's reasoning. [7]

**Fix:** Strip thought tokens from history before the next turn.

### 7.2 Gemma 4 IT: text-before-image in multimodal

**What it looks like:** A multimodal prompt with text first, then an image attachment.

**Why it fails:** The Gemma 4 chat template expects multimodal content BEFORE text. [7]

**Fix:** Reorder.

### 7.3 Claude 4.6+: using prefill for JSON

**What it looks like:** Pre-filling the assistant turn with `{` to coerce JSON output.

**Why it fails:** Anthropic removed prefill support for JSON-mode coercion in 4.6+, explicitly pointing users at Structured Outputs. [1][13]

**Fix:** Migrate to Structured Outputs.

### 7.4 GPT-5: leaving `reasoning_effort` at default for hard tasks

**What it looks like:** GPT-5 prompt with no `reasoning_effort` parameter for a task that needs deep reasoning.

**Why it fails:** GPT-5's reasoning is now a tunable, not a prompt-side concern. Prompts that say "think very carefully" should instead set `reasoning_effort: "high"`. [10]

**Fix:** Set the parameter at the API level. Drop the prompt-side hack.

---

## Catalog summary

The 12 in the SKILL.md body cover ~80% of audit findings. The
additional patterns here (1.3, 2.4, 3.3, 4.3, 5.5, 6.1–6.4, 7.1–7.4)
cover specific cases that come up less often but matter when they do.

When you're unsure whether to flag something, ask: **does this pattern
plausibly cause wrong outputs, or just stylistic improvement?** If
wrong outputs, it's at least MEDIUM, often HIGH. If just style, LOW.
