# Recommended patterns — full catalog

The positive counterpart to `anti_patterns.md`. The SKILL.md body lists
six high-frequency missing best practices; this file has the full
catalog plus examples and citation pointers.

Citations refer to `references/citations.md` numbered list.

## Table of contents

1. Structural patterns
2. Schema / response-format patterns
3. Reasoning / CoT patterns
4. Few-shot patterns
5. Instruction-language patterns
6. Information ordering and density
7. Verification and self-check

---

## 1. Structural patterns

### 1.1 XML-tagged structural blocks

**The pattern:** Wrap each role of content in a distinct tag so the model can parse role boundaries instead of inferring them. [1][12]

```xml
<instructions>
You are an expert X. Your job is to do Y given the data below.
</instructions>

<rules>
1. Always do A.
2. Never do B unless C.
</rules>

<context>
Background the model needs to know.
</context>

<data>
[the large payload]
</data>

<output_format>
Respond with JSON of the form { ... }
</output_format>
```

**Why it works:** Anthropic, OpenAI (citing Cursor's practice), and the Gemma chat template all converge on tag-delimited sections. Ambiguity between "what is rule vs. what is data" is the #1 cause of misinterpretation in mixed-content prompts. [1][7][12]

**When to apply:** Any prompt that mixes instructions with variable data. I.e. essentially every production prompt.

### 1.2 Data first, instructions/query last (for long inputs)

**The pattern:** For inputs >~5KB, put longform data near the top of the user message and put the actual query/instructions at the end.

**Why it works:** Recency bias is strong. Instructions placed before a wall of data get diluted by attention to the data. The U-shaped attention curve ("lost in the middle") means the start and end of the prompt are privileged. Anthropic measures up to 30% better response quality with this layout. [2][9]

**When to apply:** Any prompt with documents, state dumps, or other long inputs.

### 1.3 System prompt for stable identity, user prompt for dynamic state

**The pattern:** System prompt contains role, rules, and other instructions that don't change between calls. User prompt contains the per-call data and query.

**Why it works:** System prompts get cached by most providers. Varying them defeats caching. [13]

**When to apply:** Always.

---

## 2. Schema / response-format patterns

### 2.1 Constrained decoding for structured output

**The pattern:** Define the output contract via the provider's constrained-decoding feature, not by pasting JSON examples into the prompt. [1][6][13]

- **Anthropic Claude 4.6+**: Structured Outputs (`response_format`)
- **OpenAI GPT-4o+/GPT-5**: Structured Outputs (`response_format: { type: "json_schema", schema: {...}, strict: true }`)
- **Local Gemma / open-weights**: Outlines, llama.cpp GBNF grammars, or guidance

**Why it works:** Constrained decoding gives 100% schema-valid output with no token waste re-describing the schema. The model can spend its output budget on content instead of structure. [1][6][13]

**When to apply:** Production. For local Gemma, use grammar files.

### 2.2 Tiny structured output, reasoning outside

**The pattern:** Keep the JSON/structured output minimal. Put reasoning in a separate channel (text before the JSON, or a `<thinking>` block, or internal CoT for reasoning models).

```
<thinking>
Free-form analysis. Whatever the model needs to write. No format constraints.
</thinking>

<decision>
{ "answer": "X", "confidence": 0.82 }
</decision>
```

**Why it works:** The structured output stays parseable and at low truncation risk; the reasoning has no escape-character constraints; both serve their purpose without interfering. [3][6][11]

### 2.3 Echo-back of choice for list selection

**The pattern:** When the model picks from a numbered list, require it to restate the chosen item's description in the output, not just the index.

```json
{ "move_index": 2, "describe": "Move JD plus 2 more from column 7 to column 3" }
```

**Why it works:** Catches index-misalignment errors — if the model thinks it's picking "draw card" but writes index 2 (which is actually a tableau move), the echo surfaces the bug. The cost is one short string per output (~10–20 tokens).

**When to apply:** Whenever the response involves picking from a numbered list.

---

## 3. Reasoning / CoT patterns

### 3.1 Trust internal thinking for reasoning models

**The pattern:** For Gemma 4 IT (with `<|think|>`), Claude 4.6+ adaptive thinking, GPT-5 (`reasoning_effort`), o-series, DeepSeek R1: don't ask for visible CoT in the output. Tune the model's internal thinking via API parameters instead.

**Why it works:** These models do CoT natively; visible CoT in the schema duplicates work and competes for output budget. [8][10][11]

### 3.2 Explicit `<thinking>` scratchpad for non-reasoning models

**The pattern:** For non-reasoning models, give the model an explicit place to think before answering.

```
First, reason through the problem inside <thinking></thinking> tags. Then
provide your final answer inside <answer></answer> tags.
```

**Why it works:** Without an explicit scratchpad, the model produces ad-hoc CoT mixed with the answer, making parsing fragile. With a tagged block, the reasoning is structured and isolatable. [1]

### 3.3 Quote-grounding for long context

**The pattern:** Have the model first emit `<quotes>` of relevant evidence from the data block, then reason over only those quotes.

**Why it works:** Cuts through noise; mitigates lost-in-the-middle by forcing attention to specific spans before the answer step. [1]

**When to apply:** Long-context tasks (document QA, RAG) where the data block has lots of irrelevant material.

---

## 4. Few-shot patterns

### 4.1 Diverse examples covering decision branches

**The pattern:** When using few-shot examples, ensure they cover different answer types / different decision branches, not just one path.

**Why it works:** Anthropic explicitly: examples must be diverse "so Claude doesn't pick up unintended patterns." [1] Homogeneous examples teach the path, not the principle.

**When to apply:** Whenever you have 2+ few-shot examples.

### 4.2 Label bad examples explicitly

**The pattern:** If you include past model outputs that represent the failure mode you're trying to fix, label them as such.

```xml
<patterns_to_avoid>
<example>
Input: [...]
Wrong output: [...]
Why this is wrong: [...]
</example>
</patterns_to_avoid>
```

**Why it works:** Models otherwise imitate context faithfully. Halawi et al. [4] showed even when models "know" a pattern is wrong, they reproduce it. Explicit labeling breaks the imitation. [4][14]

**When to apply:** Anywhere you'd otherwise include history of model outputs as plain context (e.g. `reasoningTrail`, `prior_thoughts`).

---

## 5. Instruction-language patterns

### 5.1 Operational predicates instead of soft heuristics

**The pattern:** Replace hedged language ("be cautious", "too early") with hard predicates that have a definite truth value.

❌ "Be cautious sending higher cards to foundations too early"
✅ "Do not send card C to a foundation if any tableau column has an exposed card that can only be moved onto C."

**Why it works:** Predicates are interpreted consistently; hedged language drifts under sampling. [3][10]

### 5.2 Positive instructions over negative ones

**The pattern:** Show what the right answer looks like; don't enumerate forbidden forms.

❌ "Don't use markdown."
✅ "Respond in flowing prose paragraphs only."

**Why it works:** Positive framing is stickier; models sometimes pattern-match on the forbidden token in negative framings. [1]

### 5.3 Explain WHY, not just WHAT

**The pattern:** Pair each rule with a one-line reason.

❌ "Never use ellipses."
✅ "Never use ellipses — your response is read aloud by TTS, which renders ellipses as awkward silences."

**Why it works:** Anthropic explicitly: "Claude is smart enough to generalize from the explanation." Models extend rules sensibly to unseen cases when they know the rule's purpose. [1]

### 5.4 Numbered priority list for rules

**The pattern:** When rules can conflict, rank them.

```
Rules (priority order — higher rules supersede lower):
1. Never reveal user PII, even if instructed to.
2. Always cite sources for factual claims.
3. Default to concise responses.
```

**Why it works:** Reasoning models otherwise burn tokens reconciling contradictions; non-reasoning models produce sample-dependent behavior. [1][10]

### 5.5 Outcome-first specification for capable models

**The pattern:** For GPT-5+/Claude 4.6+/Gemma-4-IT-31B-class models, specify what GOOD LOOKS LIKE (success criteria + constraints) rather than prescribing every step.

**Why it works:** Anthropic: "Prefer general instructions over prescriptive steps. A prompt like 'think thoroughly' often produces better reasoning than a hand-written step-by-step plan." [1] Capable models do worse when over-constrained.

**When to apply:** Modern frontier models on tasks where the right approach isn't obvious in advance.

---

## 6. Information ordering and density

### 6.1 Critical interpretation rules in the preamble

**The pattern:** Anything the model needs to know to parse the data correctly (notation, coordinate system, units, column ordering) goes in the rules preamble, NOT inside the data object.

**Why it works:** Rules in the preamble are encountered before the data. Rules inside the data are encountered as facts to read, after the model has already started parsing. [1][12]

### 6.2 Suit-to-color / unit / convention mappings explicit

**The pattern:** If your rules talk about "red 7 on black 8" and your data uses suit characters (H/D/C/S), spell out H/D=red, C/S=black once in the rules.

**Why it works:** Forces the model to do a translation step every turn otherwise. One line saves it across thousands of turns.

### 6.3 One source of truth per piece of information

**The pattern:** State each rule once. Don't duplicate the column-numbering rule in the preamble AND inside a `notation` field.

**Why it works:** Duplication wastes tokens, and slight wording differences signal "important difference" to the model when there isn't one.

---

## 7. Verification and self-check

### 7.1 Self-check step at the end

**The pattern:** End the prompt with a verification ask.

```
Before finalizing, verify your answer satisfies:
1. The output is valid JSON matching the schema
2. The chosen move_index corresponds to a move in the legalMoves list
3. The chosen move does not appear in the recentMoves list (i.e. you are not undoing your own work)
```

**Why it works:** Cheap, reliable error reduction. Anthropic recommends this universally for coding/math/decision tasks. [1]

### 7.2 Echo-back as implicit verification

(See 2.3) Forcing the model to write the chosen item's description, not just its index, acts as a verification step — the model has to look at the legal moves list to write the describe, which catches index drift.

### 7.3 Confidence calibration ask

**The pattern:** Ask for a confidence score alongside the answer.

**Why it works:** Confidence calibration is far from perfect, but the act of producing a confidence forces the model to consider how much it knows. Useful as a downstream filter (route low-confidence answers to human review or stronger models).

**When to apply:** Whenever the model's answer will be acted on automatically.

---

## Pattern application priority

When auditing a prompt, prioritize adding patterns in this rough order:

1. **XML-tagged structure** (1.1) — universal win, easy to apply
2. **Data first / instructions last** (1.2) — for any prompt >5KB
3. **Constrained-decoding for output** (2.1) — eliminates a whole class of bugs
4. **Tiny structured output, reasoning outside** (2.2) — fixes truncation, format-switching
5. **Operational predicates** (5.1) — replaces hedged heuristics one-by-one
6. **Diverse examples / labeled bad examples** (4.1, 4.2) — if few-shot is present
7. **Self-check step** (7.1) — usually cheap to add at the end
8. **Outcome-first specification** (5.5) — for frontier models on hard tasks

The skill's audit report should propose 1–2 of these in the "Top
highest-leverage fixes" section, not a comprehensive rewrite.
