# Prompt audit: Claude Haiku 4.5 customer-service agent that keeps apologizing despite "NEVER apologize"

## Top 1–2 highest-leverage fixes

1. **Replace every "NEVER / Don't / NEVER use the word 'sorry'" with a positive description of what the response SHOULD look like** (with one or two short example replies). Negative-only framing is the single load-bearing reason the model keeps apologizing — it pattern-matches on the very tokens you're forbidding, and on Claude 4.5+ aggressive CAPS prohibitions actively cause overtriggering on the wrong axis.
2. **Drop the all-caps "NEVER … NEVER … NEVER … Always" stack and rewrite as a short numbered priority list in normal prose**, with one line per rule explaining WHY. Anthropic explicitly warns that aggressive emphasis on Claude 4.5+/Haiku 4.5 causes the model to over-attend to the wrong rules at the expense of others (here: it focuses on "I must help solve this" while losing the "no apology" register).

## HIGH severity

**1. Negative-only formatting instructions — the direct cause of the apologies you're seeing.** Your prompt is a stack of prohibitions: "NEVER apologize. NEVER use the word 'sorry'. NEVER admit fault. … Don't use excessive empathy. Don't use markdown." There is no positive example of what a "confident, no-apology" reply actually reads like.
- Why it fails: Models pattern-match on the forbidden tokens in negative framings, and the more emphatically the token is forbidden the more salient it becomes in the model's attention. Anthropic's own prompting guide says directly: "Positive examples showing how Claude can communicate with the appropriate level of concision tend to be more effective than negative examples or instructions that tell the model what not to do." [1] On top of that, post-trained customer-service register is a deeply baked default in Haiku 4.5 — every RLHF turn the model has seen for "user has a problem" started with some softening phrase. You have to give it a replacement register, not just forbid the default one.
- Fix: Specify what good looks like, and show a short before/after.

```
<style>
Respond in direct, confident prose. Open with the answer or the next action, not with an acknowledgement of the user's feelings.

Examples of the desired register:

User: "Your billing page is broken, I've been trying for an hour."
Good: "The billing page is loading now on our end — can you try a hard refresh (Cmd+Shift+R / Ctrl+F5)? If it still fails, paste the URL and I'll pull the request ID."
Avoid: "I'm so sorry you've had trouble! That sounds frustrating. Let me help…"

User: "I was charged twice this month."
Good: "I can see two charges on your account dated [date]. I'm refunding the duplicate now; you'll see it back on your card in 3–5 business days."
Avoid: "I apologize for the inconvenience. I completely understand how upsetting that must be…"
</style>
```

Two short examples will outperform ten "NEVER" lines.

**2. Aggressive CAPS emphasis — overtriggers on Claude 4.5/Haiku 4.5.** Your prompt has four "NEVER"s and one "Always" in caps in a single sentence-stream.
- Why it fails: Anthropic's Claude prompting best practices page explicitly flags that on Claude 4.5+ models (including Haiku 4.5) aggressive language causes overtriggering — the model becomes hyperactive about the emphasized rule, sometimes at the expense of the others, and can also interpret CAPS as "this is a high-stakes context, soften and reassure the user" — which is the *opposite* of what you want here. [1] Combined with finding #1, the CAPS makes "sorry" the most attended token in the prompt; the model then has to actively suppress its strongest associated continuation, and it sometimes leaks.
- Fix: Drop the caps. Use a numbered priority list in normal prose.

```
<rules>
Rules (priority order — higher rules supersede lower):
1. Solve the user's stated problem. The user is here for an outcome, not for emotional support.
2. Lead with the next concrete action or the answer. Save context for after.
3. Use a confident, peer-to-peer register — the way a senior engineer talks to another engineer, not the way a hotel concierge talks to a guest. No apologies, no "I understand how frustrating…", no "I'd be happy to…".
4. If we made a factual mistake, state the correction plainly and what we're doing about it. Don't editorialize about it being unfortunate.
5. Plain text only — no markdown headings, bullets, bold, or code fences. Inline `code` is fine for commands or identifiers.
</rules>
```

Each rule has an implicit positive action. The model can now reach for "what should I do" instead of "what must I avoid."

**3. No explanation of WHY — the model can't generalize the rules to cases you didn't list.** Your rules are bare prohibitions. The model doesn't know whether "NEVER apologize" applies to "Apologies for the delay" (it should), "I'd be happy to help" (it should — same register), "That's a great question!" (it should), or "Unfortunately, that's not supported" (you probably want it to). With no purpose attached, every borderline case is a coin-flip.
- Why it fails: Anthropic explicitly: "Claude is smart enough to generalize from the explanation." [1] Rules paired with a reason extend sensibly to unseen cases; bare rules don't.
- Fix: Pair each rule with a one-line reason (as in the rules block above). For your specific case, the underlying reason is something like "Users on the support channel are technical, time-pressed, and find concierge-style empathy condescending — they want a fix and an ETA." When the model knows that, it will avoid not just "sorry" but the whole family of softening phrases.

## MEDIUM severity

**4. Conflicting rules without precedence: "Don't use excessive empathy" vs "Be helpful" vs "NEVER admit fault".** What happens when admitting fault IS the helpful, problem-solving thing to do (the user reports a real bug we caused)? The current prompt gives no precedence, so the model picks per-sample.
- Why it fails: Unranked rules force the model to reconcile contradictions on the fly, producing inconsistent behavior across calls. On reasoning-capable models this burns tokens; on Haiku 4.5 it produces sample-dependent output. [1][10]
- Fix: The numbered list in finding #2 already handles this — rule 4 ("state the correction plainly, don't editorialize") explicitly says the right thing to do when we *did* cause the problem. Without that, the model defaults to apology because that's the safest interpretation of two rules pulling in opposite directions.

**5. No role / structure — the whole thing is one undifferentiated sentence stream.** "You are a customer service agent for a SaaS company. NEVER apologize. NEVER use the word 'sorry'. …" Identity, rules, formatting, and audience are all run-on in one paragraph.
- Why it fails: XML-tagged blocks (or any clear structural separation between identity, rules, and style) measurably improve adherence by giving the model unambiguous role boundaries. Anthropic, OpenAI, and the Gemma chat template all converge on this. [1][12]
- Fix:

```
<identity>
You are a senior support engineer at a SaaS company. Your users are
technical (developers, ops, IT admins). They contact support to get
unblocked, not to vent.
</identity>

<rules>
[numbered priority list from finding #2]
</rules>

<style>
[positive examples block from finding #1]
</style>
```

**6. "Don't use markdown" is itself an instance of finding #1 — restate as positive.** Same anti-pattern, lower severity because format leakage is more visible/recoverable than tone leakage.
- Fix: "Plain text only — no markdown headings, bullets, bold, or code fences. Inline `code` formatting (single backticks) is fine for commands or identifiers." (Already folded into the rules block in #2.)

**7. "The user just wants their problem solved" is a hint, not a directive — and it's the only positive guidance in the whole prompt.** It's good intent, but a single line of vague aspiration can't counterbalance five lines of prohibitions. Promote it to the lead instruction and operationalize it.
- Fix: This is the spirit of identity + rule 1 + rule 2 in the rewrite above.

## LOW severity

**8. "Be confident" / "Be helpful" are non-operational adjectives.** They have no truth value the model can check against. Replaced by the concrete register described in the style examples (finding #1), which are operational.

**9. "Customer service agent" is generic role-priming.** "Senior support engineer at a SaaS company speaking with developers/ops" is more specific and will pull the right register without you having to negate the wrong one. (Folded into the identity block in finding #5.)

## What's already good

- **You named the model.** Haiku 4.5 has specific quirks (overtriggering on aggressive language is the load-bearing one here) that the audit can target. Most users don't share this.
- **You correctly identified that the prompt is the problem**, not that the model is "broken" — the misbehavior IS prompt-induced and IS fixable in the prompt.
- **The underlying intent is sound**: a direct, problem-solving register for a technical SaaS audience is a real and defensible product choice. Don't water it down when you rewrite — keep the spine, just stop fighting the model with prohibitions and start showing it the target register.

## Putting it all together — drop-in replacement system prompt

```
<identity>
You are a senior support engineer at a SaaS company. Your users are
technical (developers, ops, IT admins) who contact support to get
unblocked, not to vent. Speak to them as a peer.
</identity>

<rules>
Rules (priority order — higher rules supersede lower):
1. Solve the user's stated problem. Lead with the answer or the next concrete action.
2. Use a confident, peer-to-peer register. Skip openers like "I'm sorry to hear that", "I understand how frustrating", "I'd be happy to help", "Great question". Go straight to substance.
3. If we made a factual mistake, state the correction plainly and what we're doing about it. Don't editorialize about it being unfortunate.
4. Plain text only — no markdown headings, bullets, bold, or code fences. Inline `code` (single backticks) is fine for commands, paths, and identifiers.
</rules>

<style_examples>
User: "Your billing page is broken, I've been trying for an hour."
Good: "The billing page is loading on our end now. Try a hard refresh (Cmd+Shift+R / Ctrl+F5). If it still fails, paste the URL you're hitting and I'll pull the request ID."

User: "I was charged twice this month."
Good: "I see two charges on your account dated [date]. I'm refunding the duplicate now — it'll be back on your card in 3–5 business days."

User: "Your last update broke our integration."
Good: "Confirmed — the 4.2.1 release changed the `X-Tenant-Id` header casing. Patch is going out tonight. Workaround until then: send the header as `x-tenant-id` lowercase."
</style_examples>
```

That replacement is roughly the same length as your current prompt, removes every "NEVER" / "Don't", and gives the model a concrete register to imitate rather than a register to suppress. Empirically, "show the target" beats "forbid the alternative" on Haiku-class models by a wide margin.

## Sources

- [1] Anthropic — Claude prompting best practices (positive vs negative framing, aggressive-language overtriggering on 4.5+, XML tags, explain-the-why, numbered priority lists): https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
- [10] OpenAI — GPT-5 prompting guide (conflicting-rules reconciliation cost, numbered priority lists): https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide
- [12] CodeConductor — Structured Prompting Techniques: XML & JSON (cross-provider convergence on tagged structural blocks): https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/
