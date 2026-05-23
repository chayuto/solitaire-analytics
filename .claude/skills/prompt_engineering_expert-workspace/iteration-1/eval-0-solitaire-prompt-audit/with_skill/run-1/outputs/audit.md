# Prompt audit: Klondike Solitaire advisor for Gemma 4 31B IT (oscillation failure mode)

## Top 1–2 highest-leverage fixes

1. **Delete the `reasoningTrail` field, or relabel it as `<prior_attempts_to_avoid_repeating>` capped at 1 entry.** This is the load-bearing fix. The field is unframed history of the model's own prior rationales — Gemma 4 is particularly prone to imitating context, and you're showing it a multi-entry trail of "this is the move I picked and here is the case for picking it." That is in-context few-shot learning of past behavior, which produces exactly the symptom you describe (oscillation when a recent move can be reversed, perseveration when it can't). Without this field the next two highest-leverage fixes can stop being defensive workarounds and just become hygiene.
2. **Replace the soft heuristic `"Avoid moves that simply undo a recent move or lead nowhere"` with a hard predicate that the model is forced to evaluate against `recentMoves`.** Combine with an echo-back of the chosen move's `describe` and a self-check step that explicitly compares the chosen move to the last move. The current wording is the kind of hedged language Gemma will paraphrase back at you but not actually act on.

## HIGH severity

**1. `reasoningTrail` is in-context few-shot learning of prior model output.** The CURRENT GAME JSON includes a `reasoningTrail` array with five entries — each one is a past `{move, reasoning}` pair where the model justified its choice. There is no framing telling Gemma what to DO with this field, so it treats it the way the literature predicts: as authoritative demonstrations to imitate. In your sample all five entries justify "draw card" with overlapping phrasing ("the primary goal is to reveal hidden cards", "no available tableau moves reveal face-down cards", "the only way to introduce new cards"), which trains the model in-context to keep drawing. On turns where the trail instead shows a recent tableau move and that move is reversible, the same imitation mechanism produces the oscillation you're measuring at ~33%.
- Why it fails: Halawi et al. show models faithfully imitate in-context demonstrations even when they "know" the demonstration is wrong [4]; "Understanding In-Context Learning from Repetitions" shows that even structural repetition (5 entries with the same shape) primes imitation [5]; "Emergent Misalignment via In-Context Learning" reinforces this in 2025 specifically for unlabeled examples [14]. Gemma 4 is explicitly more susceptible to in-context anchoring than Claude or GPT-5 [7]. This is the single most likely cause of the oscillation symptom.
- Fix: Drop the field entirely for the per-turn prompt. The current `tableau`, `foundations`, `discardTop`, and `legalMoves` give the model everything it needs to decide from scratch. If you need an audit trail downstream, capture it out-of-band (your `recentMoves` already covers "what happened"; the model does not need the prior rationales). If you must keep history in-prompt, cap at 1 entry, relabel, and reframe:

```
<prior_decision_under_review>
Your most recent move was: "Draw the next card from the stock onto the waste".
Re-evaluate the board AS IT NOW STANDS. Your prior reasoning is NOT
evidence that the same move is correct again. If the same move is still
optimal, justify it from the current state, not from past intent.
</prior_decision_under_review>
```

**2. The anti-oscillation rule is a soft heuristic with no operational handle.** Line 25 says `"Avoid moves that simply undo a recent move or lead nowhere."` This is exactly the rule you need to enforce — and it's the weakest sentence in the rules block. "Avoid", "simply", "or lead nowhere" are all hedged; the model has no way to compute whether a candidate move satisfies them. Worse, this single soft sentence is fighting an entire `reasoningTrail` whose structural signal pulls the other way (finding #1).
- Why it fails: Hedged modal verbs without operational thresholds drift between samples; at non-zero temperature the model reinterprets them every call [3][10]. Anthropic's guidance is explicit: replace heuristics with predicates that have a definite truth value [1].
- Fix: Turn the rule into a checkable predicate and reference the data field that makes it checkable.

```
ANTI-OSCILLATION RULE (hard constraint, not a heuristic):
A candidate move is OSCILLATING if it moves the same card (or a superset of
the same run) back to the column that card came from in the last entry of
recentMoves. Before choosing your final move, list each candidate move and
mark it OSCILLATING or OK against recentMoves[-1]. Do not pick an
OSCILLATING move unless every non-oscillating legal move is strictly worse
(state the reason).
```

**3. Response schema demands two long prose fields BEFORE the structured decision.** `board_analysis: <string>` and `strategic_plan: <string>` are both free-form reasoning channels inside the JSON object that ends in `final_decision`. This is two anti-patterns at once: (a) free-form CoT inside the JSON output (format-switching mid-thought, escape-character risk, the parseable decision hangs over a wall of prose and is at maximum truncation risk on long outputs), and (b) two prose fields whose purposes overlap — "assess the board including the opportunities each legal move opens or closes" and "explain your plan and why the chosen move is best" will get answered with substantially duplicated content. The duplication then biases `final_decision` toward whichever framing dominated the prose.
- Why it fails: see anti_patterns 2.1 (redundant prose fields) and 3.2 (free-form CoT inside JSON). Critically, Gemma 4 IT is a reasoning-capable model with `<|think|>` — it does CoT internally already, so asking for visible CoT prose in the output schema duplicates work that already happened in the thought channel and competes for output budget [7][8][11].
- Fix: Move reasoning OUT of the JSON. Either trust Gemma's internal thinking (preferred — enable `<|think|>` in the system block and drop both prose fields), or use an outside-the-JSON `<thinking>` block. Either way, shrink the JSON to a tiny decision object with an echo-back:

```
<thinking>
Free-form analysis. No format constraints. (Omit entirely if <|think|> is enabled.)
</thinking>
<decision>
{
  "move_index": 2,
  "describe": "Move TD from column 2 to column 5",
  "oscillation_check": "OK — TD was not moved in recentMoves[-1] (which was 'draw 5D')",
  "confidence": 0.7
}
</decision>
```

The `describe` echo is non-negotiable: it catches index-misalignment errors silently, and the `oscillation_check` field weaponizes finding #2 into a single token-cheap forcing function the model has to write before it can emit the index.

## MEDIUM severity

**4. The notation rule is buried inside the data payload.** The string `"notation":"Cards: rank then suit ... Tableau columns are numbered 1 to 7 by their \"column\" field; faceUp arrays are bottom-to-top. Always refer to columns by that 1-based number in your reasoning."` sits as a key inside the CURRENT GAME JSON. Two problems: (a) interpretation rules embedded in a data object are encountered as facts to read, not directives to obey [1][12]; (b) it duplicates the column-numbering rule already stated in line 5 of the rules preamble, and the wording differs slightly which signals "important difference" to the model when there isn't one.
- Why it fails: see anti_patterns 1.1 (rules buried in data) and 6.4 (repeated instructions).
- Fix: Delete the `notation` key from the JSON. State the notation once, in an `<notation>` block above the data, including the H/D=red and C/S=black mapping (which is currently nowhere — the model has to do that translation every turn even though "red on black" appears in the rules):

```xml
<notation>
- Cards are rank then suit: A 2 3 4 5 6 7 8 9 T J Q K, suits H D C S.
- Suit color: H and D are red; C and S are black.
- Tableau columns use the 1-based "column" field.
- Each column's faceUp array is bottom-to-top; the top card is faceUp[-1].
</notation>
```

**5. No XML-tagged structural blocks; everything is free prose.** The prompt has labeled sections ("KLONDIKE SOLITAIRE RULES", "STRATEGY GUIDANCE", "RESPONSE FORMAT", "CURRENT GAME") but they're prose headers, not delimited blocks. Gemma 4's official chat template, Anthropic's docs, and OpenAI's converge on XML-tagged sections [1][7][12]. With the current shape the model has no unambiguous boundary between "this is a rule" and "this is data" — most visibly, the `notation` key gets confused with both (finding #4).
- Fix: Wrap each section in tags: `<role>`, `<rules>`, `<strategy>`, `<anti_oscillation>`, `<game_state>`, `<legal_moves>`, `<output_format>`.

**6. Multiple soft hedged heuristics in the strategy block.** Beyond the oscillation rule (finding #2), the strategy guidance is mostly hedged: "Prioritize", "Be cautious", "too early", "Do not empty a column unless you have a King ready", "Prefer", "reasonable when no productive ... move exists". The "Do not empty a column unless ..." is actually the only operational predicate in the list; the rest will be paraphrased and forgotten under sampling pressure [3][10].
- Fix: Convert each to a predicate when possible. E.g.: "Do not move an Ace or 2 to a tableau column" (positive form of "play Aces and 2s to foundations"); "Do not send a card C to a foundation if some legal tableau move would place a card onto an exposed C, unless C is an Ace or 2". The ones you cannot reduce to predicates (the genuinely heuristic ones) should be moved to a `<preferences>` block separated from `<hard_rules>`, so the model knows which is which.

**7. Asking a reasoning-capable model for visible CoT in the schema.** Gemma 4 IT with `<|think|>` enabled produces internal CoT in a thought channel that you can extract from the response out-of-band [7]. The current schema's `board_analysis` and `strategic_plan` fields ask Gemma to do the same work twice: once in the thought channel (silently, if enabled) and once in the JSON (visibly, where it competes for the output budget that is supposed to carry `final_decision`). This is also what makes the long-prompt truncation risk in finding #3 actually bite.
- Fix: Decide deliberately whether you want the model thinking via `<|think|>` (recommended for this task — there is real reasoning to do per turn) or via a single visible `<thinking>` block. Pick one. Don't do both, and don't ask for prose reasoning inside the JSON in either case. (This is the same finding #3 from the model-calibration angle; calling it out separately because the fix is "configure `<|think|>` in the system block" not "edit the schema".)

**8. No self-check / verification step.** The prompt does not ask the model to verify its output before emitting. For a JSON-emitting decision prompt with a known oscillation failure mode, this is the cheapest possible mitigation [1].
- Fix: End the prompt with an explicit verification block:

```
Before emitting the JSON, verify:
1. move_index appears in legalMoves.
2. describe matches the legalMoves entry at move_index exactly.
3. The move is NOT the inverse of recentMoves[-1] (oscillation check).
4. If you cannot satisfy (3), state why no non-oscillating move is acceptable.
```

## LOW severity

**9. Decorative metrics with no decision relevance.** `metrics: { completionProgress: 2, perceivedDifficulty: 47, moveCount: 34, difficulty: 3 }` is shown to the model every turn. None of these inform a per-turn pick — they are analyst stats. Token waste plus attention noise.
- Fix: Move analytics-only metrics to an out-of-band log; do not include in the per-turn prompt.

**10. Redundant `type` + `describe` in legalMoves.** Every legal move carries both `"type":"tableau_to_tableau"` and a natural-language `describe` that implies the type. Pick one. The model needs `describe` for the echo-back (finding #3); `type` is redundant unless you have downstream code that needs it (in which case it does not need to be sent to the model).
- Fix: Drop `type` from the model-facing payload.

**11. Role primes as "advisor" but the model is actually picking the move.** Line 1: "You are an expert Klondike Solitaire strategist acting as an advisor." The model is not advising — it is committing to a single move every turn. Advisor framing biases toward hedged outputs.
- Fix: "You are an expert Klondike Solitaire player. Each turn you choose exactly one move from the legal moves provided." Small effect, free to fix.

**12. `recentMoves` is 10 consecutive `draw X` entries — also a low-grade in-context signal.** Less severe than finding #1 because there is no rationale attached, but Gemma will still notice the structural repetition [5]. Cap at the last 3–5 moves to give the oscillation check enough lookback without amplifying the imitation prime.

**13. `alternative_move_index` is described as "optional" but shown as a required key in the example.** Pick one. If genuinely optional, omit from the example; if always required, drop "optional" from the spec.

**14. Negative formatting instruction.** "no prose or markdown fences outside the object" is the kind of negative phrasing models sometimes pattern-match on and emit anyway. Replace with positive form: "Emit a single JSON object as your entire response, beginning with `{` and ending with `}`." (Or better: switch to constrained decoding via Outlines or llama.cpp GBNF grammar for local Gemma [6], which makes the instruction unnecessary.)

## What's already good

- **Rules block is concrete and accurate about Klondike specifics** — face-up-only runs move as a unit, Kings on empty columns, recycling stock — none of the common rules wrong-by-one-step bugs. Keep this block; just split "hard rules" from "preferences" (finding #6).
- **`legalMoves` is pre-computed and given as a numbered list.** This is the right pattern — it removes the legality-checking burden from the model and turns the task into a selection problem, which Gemma is much better at than generation. Keep, just trim `type` (finding #10) and add the echo-back contract (finding #3).
- **The model is explicitly told the order of keys to produce in the JSON ("Produce the keys in the order above").** Order-stability is a real win for downstream parsing. The keys themselves are wrong (findings #3, #7) but the ordering discipline is right and should be preserved in the rewritten schema.

## Sources

- [1] Anthropic — Claude prompting best practices: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
- [2] Anthropic — Long context tips: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
- [3] Maxim AI — A Practitioner's Guide to Prompt Engineering in 2026: https://www.getmaxim.ai/articles/a-practitioners-guide-to-prompt-engineering-in-2025/
- [4] Halawi et al., "Overthinking the Truth: Understanding how Language Models Process False Demonstrations" (arXiv 2307.09476): https://arxiv.org/pdf/2307.09476
- [5] "Understanding In-Context Learning from Repetitions" (arXiv 2310.00297): https://arxiv.org/pdf/2310.00297
- [6] Pockit Tools — LLM Structured Output in 2026: https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk
- [7] Google — Gemma 4 prompt formatting: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- [8] Helicone — How to Prompt Thinking Models like DeepSeek R1 and OpenAI o3: https://www.helicone.ai/blog/prompt-thinking-models
- [10] OpenAI — GPT-5 prompting guide (cookbook): https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide
- [11] (covered via [7] and [8]) — reasoning models do internal CoT; visible-CoT-in-schema is duplicative
- [12] CodeConductor — Structured Prompting Techniques: XML & JSON: https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/
- [14] "Emergent Misalignment via In-Context Learning" (arXiv 2510.11288, 2025): https://arxiv.org/pdf/2510.11288
