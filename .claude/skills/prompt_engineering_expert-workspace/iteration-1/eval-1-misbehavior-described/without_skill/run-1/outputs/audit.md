# Audit: Customer-Service Chatbot Still Apologizes

## TL;DR

Your prompt is fighting the model in three ways at once:

1. **It's almost entirely negative ("NEVER X, NEVER Y, Don't Z").** LLMs — Claude included — are much better at following *positive* instructions ("do X this way") than *prohibitions*. Negations also re-activate the very tokens you're trying to suppress (telling the model "don't say sorry" makes "sorry" highly salient in its working context).
2. **You're overriding a deeply trained behavior with a single sentence.** Claude's post-training reinforces empathetic, apologetic phrasing in customer-service-shaped contexts. One-shot "NEVER apologize" instructions rarely beat that prior, especially on Haiku (smaller models lean harder on training defaults).
3. **The prompt has no structure, no rationale, no examples, and no recovery behavior.** The model knows what *not* to do but has no concrete replacement pattern, so when it hits an apology-shaped moment it falls back to the trained default.

You will get dramatically better adherence by rewriting it as a positive, structured prompt with rationale and a few before/after examples. Sketch below.

---

## Why your current prompt under-performs

### 1. Negation is a weak control signal

The prompt has 6 "NEVER/Don't" clauses and 2 weak positives ("Be helpful", "Always be confident"). Problems with this:

- **Salience flip.** "NEVER use the word 'sorry'" puts "sorry" into the active context. The model now has to actively suppress a token that's been primed. Suppression is probabilistic, not deterministic — every generation step is a fresh dice roll, and the longer the conversation, the more chances to slip.
- **No replacement behavior.** Telling someone "don't apologize" without telling them what to *do* when a user is frustrated leaves a vacuum. The model fills the vacuum with its strongest prior, which in customer-service training data is… apologizing.
- **Lossy instruction stacking.** Six prohibitions compete for attention. The model partially follows several rather than fully following any.

### 2. You're overriding a strong post-training prior with weak prompt-time pressure

Claude is trained to be helpful, harmless, and to acknowledge user frustration. In customer-service contexts, the training distribution is saturated with apology phrasing ("I'm sorry to hear that…", "I apologize for the inconvenience…"). A one-line "NEVER apologize" instruction is a featherweight counterweight to that.

This is worse on Haiku 4.5 than on Sonnet/Opus:

- Smaller models have less "instruction-following headroom" — they lean harder on default behaviors when instructions conflict with priors.
- Smaller models also do worse with abstract/global constraints ("be confident") vs. concrete patterns ("respond in this format").

### 3. No rationale, no examples, no escape hatch

The prompt doesn't explain *why* not to apologize. Without rationale, the model can't generalize — it doesn't know whether "I understand this is frustrating" counts, whether "unfortunately" counts, whether acknowledging a bug counts as "admitting fault", etc. So it errs on the side of its training prior.

There are also no few-shot examples, which are the single most reliable way to override a default response pattern.

### 4. "Don't admit fault" is a legal/brand instruction masquerading as a tone instruction

Worth separating: "don't apologize" (tone) and "don't admit fault" (liability) are different goals and need different handling. Conflating them makes both harder to follow.

---

## What to change

### Rewrite as a positive, structured prompt

Here's a concrete rewrite you can adapt:

```
You are a customer service agent for <Company>, a SaaS product that does <one-line description>.

# Voice
Your voice is confident, direct, and solution-focused. You write like a senior engineer
who respects the user's time: lead with the answer or the next step, then add context only
if needed. Plain text only — no markdown, no headers, no bullet lists unless the user
explicitly asks for steps.

# Response pattern
Every response follows this shape:
1. Acknowledge the situation in one short, neutral sentence (state the facts, not feelings).
2. Give the answer, fix, or next action.
3. Stop. Do not pad with reassurance or closing pleasantries.

# Language guidance
Instead of apologizing or expressing regret, name the issue and move to the fix.

  Avoid: "I'm so sorry you're experiencing this issue!"
  Use:   "That error happens when the API key has expired. Regenerate it at Settings > API."

  Avoid: "I apologize for the inconvenience — let me look into that."
  Use:   "Looking into it now. Can you share the request ID from the failed call?"

  Avoid: "Unfortunately, that feature isn't available."
  Use:   "That feature isn't available today. The closest workaround is <X>."

If the user is upset, acknowledge the impact factually ("That blocked your deploy — let's
unblock it now.") rather than emoting at them.

# Fault and liability
Do not speculate about the cause of an outage or attribute blame to <Company>, the user,
or a third party until you have evidence. If the user asks who is at fault and you do not
know, say: "I don't have the root cause yet. I'll confirm and follow up."

# Why this voice
Users contacting support want their problem solved, not emotional labor. Apologetic
phrasing reads as scripted and delays the fix. Confidence — backed by a concrete next
step — is what reassures them.
```

Why this works better:

- **Positive framing** ("lead with the answer", "name the issue and move to the fix") gives the model a target behavior rather than a forbidden one.
- **Few-shot before/after examples** are the single highest-leverage change. They show the model exactly what to substitute for apology phrasing — this is what actually overrides the trained default.
- **Rationale** ("users want their problem solved, not emotional labor") lets the model generalize to cases your examples don't cover.
- **Separates tone from liability.** "Don't admit fault" becomes a specific rule about not speculating on cause, which is what you actually want.
- **Specifies an escape hatch.** "I don't have the root cause yet" gives the model a non-apologetic way to handle uncertainty — without this, it falls back to "I'm sorry, I'm not sure…".
- **Structure** (sections with headers in the *prompt*, not in the output) makes each constraint easier to follow than a run-on sentence.

### Other levers, in order of impact

1. **Add 3–5 few-shot examples** of real-ish support exchanges in your domain. This alone typically moves adherence from ~60% to >90%.
2. **Use prefill.** If you control the API call, prefill the assistant turn with the first few characters of a non-apologetic opener (e.g. `Here's what's happening:` or `That's caused by`). Prefilling is the most reliable way to suppress a specific opening phrase across an entire conversation. Note this is more effective than any system-prompt instruction for first-token control.
3. **Consider upgrading to Sonnet 4.5** for the agent persona if adherence still matters after the rewrite. Haiku 4.5 is fast and cheap but trades instruction-following nuance for both. If cost is a constraint, keep Haiku but invest more in examples and prefill.
4. **Add a post-generation check** (cheap regex or a tiny Haiku call) that flags responses containing "sorry|apologize|apologies|regret|unfortunately" and regenerates with a stricter prefill. Belt-and-suspenders, but useful if this is customer-facing and a single slip is expensive.
5. **Lower temperature** (0.2–0.3) for support responses. Higher temperatures make the model more likely to drift back toward the trained default phrasing.

### What not to do

- Don't add more "NEVER" clauses. Each one you add makes the others less effective.
- Don't ALL-CAPS the rules. It doesn't help and often makes outputs read as anxious.
- Don't tell the model "you are not allowed to apologize under any circumstances or you will be penalized." Threats and stakes-inflation don't improve adherence on modern Claude and sometimes make the model hedge more.

---

## Quick diagnostic checklist

If, after rewriting, the model still apologizes, check in this order:

1. Is the apology in the *first sentence* of the response? → Use prefill.
2. Is it triggered by user expressions of frustration? → Add an example pair that handles a frustrated user without apologizing.
3. Is it triggered by the model not knowing the answer? → Add the "I don't have the root cause yet" escape hatch and an example using it.
4. Is it triggered by tool/API failures? → Add a rule for that specific case ("If a tool call fails, state what failed and the retry/next step — do not apologize for the tool.").
5. Still happening? → Run the same prompt on Sonnet 4.5 to confirm it's a Haiku capacity issue vs. a prompt issue.

---

## Summary

The prompt isn't being ignored — it's being partially followed against a strong trained prior, with no positive replacement behavior to fall back on. Rewrite it as positive instructions + before/after examples + a stated rationale + a non-apologetic escape hatch for uncertainty. Add prefill if you need first-token guarantees. Upgrade the model only if the rewrite plus prefill still isn't enough.
