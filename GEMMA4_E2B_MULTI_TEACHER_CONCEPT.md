# Multi-Teacher Distillation — Concept Note

> A concept-level design for how Gemma 4 E2B could be distilled from **multiple
> teachers** rather than a single LLM. This note covers the *idea, rationale, and
> trade-offs only* — no implementation detail.
>
> Companion to: `GEMMA4_E2B_IMPLEMENTATION_PLAN.md`, `GEMMA4_E2B_DATA_EVALUATION.md`.
> Status: **proposal — not yet adopted.** Date: 2026-05-17

---

## 1. Why consider this

The current plan distils E2B from one teacher (`gemma-4-31b-it`). The data audit
exposed three weaknesses inherent to a single-LLM teacher:

- **The teacher is a weak Solitaire player.** It chose "draw" on every sampled board.
  Imitating it faithfully produces an equally weak student.
- **Single-source bias.** A positional bias in one teacher propagates straight into
  the student with nothing to counteract it.
- **Imitation has a hard ceiling.** Pure imitation of one teacher cannot, by
  definition, exceed that teacher.

Multi-teacher distillation is the standard remedy: supervise the student with
agreement across several independent sources instead of one fallible source.

---

## 2. Two flavors of "multi-teacher"

**Flavor A — multiple LLM teachers.** Several models (e.g. Gemma 4 31B, Gemma 4
26B-A4B, and a non-Gemma model) each act as advisor on the same boards. The student
learns from their *consensus*. This denoises individual mistakes and lets divergent
biases cancel.

**Flavor B — hybrid: LLM + the repo's classical solver.** `solitaire-analytics`
already contains deterministic, near-authoritative move sources — the parallel
solver, the lookahead strategy, the weighted strategy. These are not noisy LLMs;
for Klondike they approximate an oracle. They become teachers for the *decision*,
while an LLM remains the teacher for the *explanation*.

Flavor B is the more powerful idea for this project and is the recommended direction.

---

## 3. Core concept — decompose the label

A training example has two very different kinds of content, and they need not come
from the same teacher:

| Part of the label | Best teacher | Rationale |
|---|---|---|
| The **decision** (`move_index`) | a verified source — the solver / lookahead | categorical, checkable, an oracle exists |
| The **rationale** (`board_analysis`, `strategic_plan`) | a single LLM | natural-language quality; needs a consistent "voice" |

Decoupling these is the central concept: distil *what to do* from the most reliable
source available, and *how to explain it* from a fluent one.

---

## 4. Concept — how teachers are combined

The decision is **categorical**, so teachers are **voted, not averaged** (averaging
move indices is meaningless). Three combination concepts, in increasing strength:

1. **Consensus filtering** — keep an example only where teachers agree; discard or
   flag disagreements. Simple, yields a clean but smaller set.
2. **Majority vote** — the label is the most-supported move; ties dropped.
3. **Oracle arbitration** — among the moves the teachers proposed, the verified
   solver decides which genuinely advances the game. This is the strongest: it turns
   the panel from "imitation of opinions" into "imitation of a checked answer."

Text rationale should come from **one** LLM, not blended — mixing prose styles gives
the student a noisier, less learnable target.

---

## 5. Why this raises the ceiling

With a single LLM teacher, the student's quality ceiling is that LLM. With the
classical solver supplying (or arbitrating) the decision label, the ceiling becomes
the **solver's** quality — far above a weak LLM advisor. Under this concept,
"the student beats the 31B LLM at Solitaire" is the expected outcome, not an
aspiration, because the student is no longer imitating a weak player — it is
imitating a checked, near-optimal decision with a fluent explanation attached.

This also reframes the success metric. "Match the teacher" made sense for one LLM
teacher. Under multi-teacher, the meaningful metrics become **agreement with the
verified decision** and, ultimately, **actual win rate over played games**.

---

## 6. Trade-offs and costs

| Benefit | Cost / risk |
|---|---|
| Denoised, higher-quality decision labels | N× collection cost and orchestration complexity |
| Single-teacher bias cancels out | Needs an explicit disagreement-handling policy |
| Student can surpass any single LLM teacher | Each LLM must reliably emit the required schema |
| Solver gives a checkable ground truth | Solver coverage/【cost on hard boards must be acceptable |
| Rationale stays fluent (one voice) | Decision and rationale can disagree — must be reconciled |

---

## 7. Recommended phasing (concept level)

This is a direction, not a schedule, and deliberately avoids implementation detail:

1. **Single teacher first.** Let the collection team finish the P0 fixes with one LLM
   teacher. Do not expand scope mid-fix.
2. **Introduce the solver as a verifier — cheap, in-repo, high leverage.** Even using
   it only to *score* the teacher's chosen move (helpful / neutral / harmful) enables
   quality-filtering of the dataset.
3. **Promote the solver to decision teacher** once verification proves valuable —
   adopt the §3 label decomposition.
4. **Add a second LLM teacher** for consensus only if agreement metrics justify the
   extra collection cost.

---

## 8. Open questions for the collection team and project owner

- Which models are realistically available as LLM teachers (cost, rate limits)?
- Is the classical solver fast enough to label/arbitrate at the data volumes in the
  §9 target of the data-evaluation note?
- Should the student be trained to reason (rationale + decision) or to answer
  decision-only? This interacts with whether teacher reasoning traces are captured.
- Does the success metric move from "teacher agreement" to "verified-move agreement"
  and/or "win rate"? If so, the evaluation design changes accordingly.

---

## 9. Bottom line

Multi-teacher distillation — specifically the **LLM + classical-solver hybrid** — is
the cleanest path to a student that is not merely a smaller copy of a weak advisor but
a genuinely stronger one. It is proposed as a **v2 direction**: single-teacher
collection and the P0 fixes come first; the solver-as-verifier step is low-cost and
can begin as soon as single-teacher data is flowing.
