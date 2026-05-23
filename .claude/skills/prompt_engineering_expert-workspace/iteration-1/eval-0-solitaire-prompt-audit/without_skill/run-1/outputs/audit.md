# Klondike Advisor Prompt — Oscillation Audit

**File audited:** `/Users/chayut/repos/solitaire-analytics/.claude/skills/prompt_engineering_expert-workspace/iteration-1/klondike_prompt_sample.txt`
**Model:** Gemma 4 31B
**Reported failure:** When a legal move would undo a recent move, the model picks it ~33% of the time.

## TL;DR — root cause in one line

The prompt **tells the model not to undo previous moves, but never actually shows it which moves it made.** The "anti-undo" rule is a negative instruction with no observable referent in the state, so the model can't enforce it even when it tries. Several secondary issues (reasoning trapped inside grammar‑constrained JSON, a `reasoningTrail` that reinforces stale plans, ambiguous `recentMoves` semantics) compound the problem.

---

## 1. The headline bug: anti-undo has no data to bind to

Line 25:

> Avoid moves that simply undo a recent move or lead nowhere.

This is the only place oscillation is addressed, and it fails for three independent reasons:

### 1a. The model can't see its own moves

The state JSON exposes:

- `recentMoves`: an array of strings like `"draw 8H", "draw 3H", ...` — **all 10 entries in the sample are `draw X`**. These look like *cards revealed by draws*, not *actions chosen by the agent*. A tableau move (e.g. "moved TD col 2 → col 5") apparently never appears here. So the agent has no record of the tableau or foundation moves it actually made — exactly the moves that get oscillated.
- `reasoningTrail`: 5 prior reasoning blobs. In the sample, the trailing `"move"` field is always `"Draw the next card..."` — again only draws are recorded, and only the *most recent* 5.
- No `previousState`, no diff, no per-card "last touched on turn N" annotation.

So when the agent on turn T moves `JD+TS+9D` from col 1 → col 6, then on turn T+1 sees a legal move `"Move JD plus 2 more from column 6 to column 1"`, **nothing in its context window flags that move as the inverse of what it just did.** The "avoid undo" rule asks the model to detect something it has no input for. Gemma 4 31B is not powerful enough to silently reconstruct prior turns from card positions and a noisy draw log; even GPT‑class models do this poorly without explicit state.

### 1b. The rule is a negative instruction

Negative instructions ("don't do X", "avoid X") are well-documented to underperform positive instructions and to scale *worse* with model size in some studies — the "pink elephant" effect ([Eval blog](https://eval.16x.engineer/blog/the-pink-elephant-negative-instructions-llms-effectiveness-analysis), [Gadlet](https://gadlet.com/posts/negative-prompting/)). Mentioning "undo a recent move" in the prompt actually keeps the *concept of undoing* in the activation pattern without giving any concrete escape hatch.

### 1c. The rule is buried in a list of soft heuristics

Line 18 frames the whole block as "heuristics, not absolute rules." That gives the model permission to override the anti-undo rule with self-justification ("this isn't really an undo, I'm setting up a new line"). For something that breaks the run ~33% of the time, this needs to be a **hard rule with a deterministic check**, not bullet #7 in a soft list.

---

## 2. `reasoningTrail` is a hallucination-loop amplifier

The prompt feeds back the agent's last 5 reasonings verbatim. In the sample, all 5 say variations of *"the only productive action is to draw from the stock... looking for a black 7 for Col 3, a black 6 for Col 4, a black 3 for Col 7."*

This is the exact hallucination-loop / context-reinforcement failure documented in the LLM-game-agent literature ([Towards Data Science](https://towardsdatascience.com/how-i-built-an-llm-based-game-from-scratch-86ac55ec7a10/), [Dev.to: Stop the Loop](https://dev.to/alessandro_pignati/stop-the-loop-how-to-prevent-infinite-conversations-in-your-ai-agents-ekj)): the agent's own prior justifications become the dominant signal in the next turn, and it doubles down on whatever it just decided. Two ways this drives oscillation:

- **Action confirmation bias.** If the trail says "I moved TD to col 5 because Y", next turn the model is anchored to "TD belongs on col 5" — so if the board now offers "move TD from col 5 to col 2", the model rejects it as obviously wrong... and then on turn T+2 the trail says "I rejected moving TD back", so on T+3 with a fresh context window collision the rationale flips. Oscillation.
- **Stale plan reinforcement.** The reasoning is from a stale board. If the model 3 turns ago decided "draw because we need a black 7", it'll keep deciding "draw because we need a black 7" even after one was found, because the trail dominates the per-turn analysis. The same dynamic applies to "this card is best on col X" reasoning, producing back-and-forth shuffles.

This is closely related to the [arxiv finding](https://arxiv.org/pdf/2510.15974) that LLM agents in deterministic games "fall into looping behavior characterized by sequences which transition return to previously visited states... incapable of learning from past mistakes" — and the [Markov state paper](https://arxiv.org/pdf/2603.19987) explicitly argues that piling history into the prompt instead of distilling it into a Markovian state hurts reasoning.

**Fix:** stop sending raw `reasoningTrail`. If you want continuity, send a single compact *plan* field ("current objective: empty col 7") that the model updates each turn, plus a **structured `myRecentActions` array** of the model's last N *chosen* moves (not draws revealed).

---

## 3. `recentMoves` semantics are ambiguous and misleading

`"recentMoves": ["draw 8H","draw 3H","draw 2S","draw 9H","draw 3C","draw 7H","draw TH","draw 6H","draw JH","draw 5D"]`

Problems:

- Are these (a) the *agent's chosen actions*, (b) the *cards that flipped onto the waste*, or (c) all state changes? It looks like (b), since they're all draws and the agent has made other moves (move count is 34 but only draws are listed).
- The name `recentMoves` strongly implies (a) to both human and model readers. The model will treat them as its own action log, which is wrong.
- Tableau and foundation moves are never represented. So even if the model tries to obey "don't undo", it literally cannot see the moves it could undo.
- The "draw X" entries don't even include *which* prior turn drew them, just the order.

**Fix:** rename to `revealedCards` (or split into `myActions` vs `cardsRevealed`), use unambiguous descriptors, and include a `myActions` array of the form:

```json
"myActions": [
  {"turn": 33, "type": "tableau_to_tableau", "move": "JD+TS+9D col1 -> col6"},
  {"turn": 32, "type": "draw", "revealed": "5D"},
  ...
]
```

This is the *direct* fix for oscillation — once the model can see "I moved JD+TS+9D from 1 → 6 last turn", a legal move of "JD+TS+9D from 6 → 1" is identifiable as an undo without any heuristic reasoning.

---

## 4. Reasoning is trapped inside grammar-constrained JSON strings

Lines 27–35:

> Reason step by step, then respond with ONLY a single JSON object containing exactly these three keys, in this order... no prose or markdown fences outside the object

The "step by step reasoning" must happen *inside* `board_analysis` and `strategic_plan` — both string fields inside a constrained JSON envelope. For Gemma 4 specifically, this is a known footgun:

- [ollama/ollama#15502](https://github.com/ollama/ollama/issues/15502) — "gemma4:31b repetition loop during constrained JSON generation with free-text string fields"
- [vllm-project/vllm#40080](https://github.com/vllm-project/vllm/issues/40080) — "Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema)"
- [ollama/ollama#15260](https://github.com/ollama/ollama/issues/15260) — `format` parameter constraint silently breaks chain-of-thought

The relevant mechanic: when the decoder is grammar‑constrained to JSON tokens, Gemma 4's mild repetition bias becomes a strong bias because the grammar prevents EOS until the field "looks done." This produces under-reasoned (or repetitive) `board_analysis` text, which then drives a mediocre `final_decision`. Mediocre decisions in repeated similar boards are exactly how oscillation manifests — the model isn't really analyzing the board, it's pattern-matching to its own prior strings.

**Fixes (any one helps; combine for best effect):**

- Let the model emit a `<reasoning>...</reasoning>` block in plain text **before** the JSON, and only parse the trailing JSON. This is the pattern that's been shown to work with Gemma 4 + structured output ([Dev.to Gemma 4 production patterns](https://dev.to/jpablortiz96/5-production-patterns-for-running-gemma-4-in-the-browser-what-the-docs-dont-tell-you-2ai1)).
- Or, if grammar constraints are mandatory, add an explicit `scratchpad` field *before* `board_analysis` and instruct the model to write all chain-of-thought there. Length-bound the analysis fields (e.g. "≤ 80 words") to discourage repetition loops inside the strings.
- If using ollama, double-check whether `think=true/false` is set — both settings have documented format/output interactions for Gemma 4.

---

## 5. The output schema actively encourages wobble

```
"final_decision": { "move_index": <number>, "confidence": <number>, "alternative_move_index": <number> }
```

- `confidence` and `alternative_move_index` are emitted **at the very end** of generation, *after* the entire analysis is written. By the time it's choosing, the model has filled its working context with its own (possibly looped) prose. With no explicit "if your top move would reverse a recent action, pick the alternative" rule, the alternative slot is decorative.
- Asking for an alternative move *every* turn nudges the model to find one even when none exists, which builds a habit of "yes there's another reasonable move" — making it easier to flip to that alternative on a similar future turn.
- There's no `"this_undoes_action_id"` field forcing the model to *explicitly* check. Structured output works best when the schema *makes the model commit to the check*, not when it just leaves a field.

**Fix:** add a required pre-decision check field, e.g.

```json
"undo_check": {
  "would_reverse_recent_action": <bool>,
  "reversed_action_turn": <number|null>,
  "justification_if_true": <string>
}
```

The act of producing this field forces the model to *look* at the recent-actions list (which #3's fix now contains) before committing. This pattern — forcing an explicit self-check field in structured output — is a standard trick for getting smaller models to actually obey a rule.

---

## 6. Ordering / placement issues (smaller impact, but worth fixing)

- **Critical rules at top, state at bottom, decision at very bottom.** With ~2000 tokens of game state in between, the "avoid undo" instruction is far from where decoding happens. Gemma 4 31B's instruction-following degrades with distance. Consider repeating the anti-undo rule *inline* in the JSON state, e.g. an `instructions` key just above `legalMoves`.
- **`legalMoves` lacks per-move undo annotation.** The cheapest possible fix: the *engine* (which knows the move history) should annotate each legal move with `"undoesRecentAction": true|false`. Then the prompt rule becomes "never pick a move where `undoesRecentAction == true` unless your `board_analysis` explicitly explains a forced reason." That converts a model-side reasoning problem into a model-side selection problem, which small models do much better.
- **"Reason step by step, then respond with ONLY a single JSON object"** is mildly contradictory. Most Gemma 4 users get better behavior with either pure JSON or pure CoT+JSON-at-end, not "reason inside the JSON."

---

## 7. Sample-specific sanity check

The sample shown is actually a *correct* situation for `draw_card` — neither of the two tableau moves reveals a face-down card. So this particular sample isn't itself an oscillation; it's a window into how the prompt presents history. The key tell is in the `reasoningTrail`: 5 nearly identical justifications. That feedback loop is exactly what produces oscillation a few turns *later*, when a tableau move appears and the model can no longer rely on its parroted "draw because we need black 7" rationale — at which point it picks something, justifies it, then next turn justifies the inverse.

---

## Prioritized fix list

In order of expected impact on the 33% oscillation rate:

1. **Have the engine compute & expose `undoesRecentAction: bool` per legal move.** Hard signal, no model reasoning required. Expected to cut oscillation by most of the 33%.
2. **Add a structured `myActions` array of the agent's last N *chosen* moves** (not draws revealed). Replace the misleading `recentMoves` name.
3. **Drop or radically compress `reasoningTrail`.** Replace with a single `currentPlan: <string>` that the model rewrites each turn (forces distillation, kills parroting).
4. **Add a required `undo_check` field at the start of `final_decision`.** Force the model to commit to a yes/no before picking.
5. **Move reasoning outside the JSON envelope** (or add an explicit `scratchpad` field) to dodge Gemma 4's documented constrained-JSON repetition bug.
6. **Convert anti-undo from a soft heuristic to a hard rule with an explicit escape clause** ("Never pick a move flagged `undoesRecentAction=true` unless `undo_check.justification_if_true` cites a concrete new opportunity that did not exist when the original move was made.")
7. **Inline the anti-undo rule next to `legalMoves`** so it's close to the decoding point.
8. Length-cap `board_analysis` and `strategic_plan` (e.g. ≤ 80 words each) to suppress in-string repetition loops on Gemma 4.

Fixes 1–3 alone should resolve most of the issue; 4–8 harden it.

---

## Sources

- [Pink Elephant: Why "Don't Do That" Fails with LLMs](https://eval.16x.engineer/blog/the-pink-elephant-negative-instructions-llms-effectiveness-analysis)
- [Why Positive Prompts Outperform Negative Ones with LLMs (Gadlet)](https://gadlet.com/posts/negative-prompting/)
- [LLMs Don't Understand Negation (HackerNoon)](https://hackernoon.com/llms-dont-understand-negation)
- [Limits of Emergent Reasoning of LLMs in Agentic Frameworks for Deterministic Games (arxiv 2510.15974)](https://arxiv.org/pdf/2510.15974)
- [Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States (arxiv 2603.19987)](https://arxiv.org/pdf/2603.19987)
- [ollama issue #15502 — gemma4:31b repetition loop during constrained JSON generation](https://github.com/ollama/ollama/issues/15502)
- [vllm issue #40080 — Gemma 4 infinite repetition loops with structured output](https://github.com/vllm-project/vllm/issues/40080)
- [ollama issue #15260 — `think=false` breaks `format` for gemma4](https://github.com/ollama/ollama/issues/15260)
- [5 production patterns for running Gemma 4 (Dev.to)](https://dev.to/jpablortiz96/5-production-patterns-for-running-gemma-4-in-the-browser-what-the-docs-dont-tell-you-2ai1)
- [Stop the Loop! How to Prevent Infinite Conversations in Your AI Agents (Dev.to)](https://dev.to/alessandro_pignati/stop-the-loop-how-to-prevent-infinite-conversations-in-your-ai-agents-ekj)
- [How I Built an LLM-Based Game from Scratch (Towards Data Science)](https://towardsdatascience.com/how-i-built-an-llm-based-game-from-scratch-86ac55ec7a10/)
- [Best practices for LLM prompt engineering (Palantir)](https://www.palantir.com/docs/foundry/aip/best-practices-prompt-engineering)
