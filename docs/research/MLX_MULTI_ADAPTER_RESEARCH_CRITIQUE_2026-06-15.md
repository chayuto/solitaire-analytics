# Critique: the MLX Gemma 4 E2B multi-adapter research doc

Date: 2026-06-15
Subject: `docs/research/MLX Gemma 4 E2B Multi-Adapter Research.md`
Author of the subject: an AI "deep research" tool (Gemini-style, numbered web
citations). Filed as a reference; this is the analyst read on it.
Verification basis: claims checked against our actual stack (mlx_lm 0.31.3,
mx 0.31.2, `lora_config_volume.yaml`, `gemma4_text_patch.py`, the won-only /
volume SFT results) on 2026-06-15.

## 0. One-paragraph verdict

This is a competent, generic blueprint for a THREE-adapter hot-swap architecture
(Theory / Heuristics / JSON-Execution) on Gemma 4 E2B + MLX. It is not specific
to our project and it solves a problem we do not have. Several of its concrete
prescriptions are wrong for our setup, its flagship deliverable (the section 6.3
hot-swap script) is broken as written, and a recurring block of "facts" (MXFP4,
`<|channel|>thought` leakage, the fuse bug) is bled in from gpt-oss and
mis-attributed to Gemma, traceable in its own citation [35]. Keep it as an idea
source. Do not adopt the architecture now. There are exactly two small, real
takeaways for us (sections 4 and 5 below).

## 1. What the document actually is

It answers a brief that was never ours: "train three isolated cognitive-domain
adapters (declarative theory, sequential heuristics, strict-JSON execution) and
serve them by dynamic hot-swap." That is a sensible general pattern for an agent
with three genuinely distinct data domains. Our project has ONE data domain:
teacher decision logs (the doc's "Adapter C"). We have no textbook-theory corpus
and no separate heuristics-checklist corpus. So the central architectural thesis
does not map onto our data. Read it as a literature survey of MLX LoRA mechanics,
not as a plan.

## 2. Verification scorecard (checked against our stack)

VERIFIED correct (I confirmed these in our tree, correcting my own first-pass
skepticism on two of them):

- `--grad-checkpoint`, `--mask-prompt`, and `--grad-accumulation-steps` are all
  real flags in our `mlx_lm` 0.31.3 (`lora.py` lines 70, 75, 147). The
  grad-accumulation flag in particular I expected to be hallucinated; it is not.
- `mx.gather_mm` exists in our `mx` 0.31.2 (`hasattr` is True). The primitive the
  doc leans on for batched multi-adapter decode is real.
- 4-bit E2B footprint "~1.5 to 2.0 GB": consistent with our measured ~2.7 GB peak
  and ~3.3 GB inference. Right order of magnitude.
- Loss-masking rationale (section 5.2) and the TIES/SLERP descriptions (4.1) are
  textbook-accurate.
- The detached `chat_template.jinja` quirk for Gemma 4 (template not in
  `tokenizer_config.json`) matches what our family-architecture reference already
  records. Plausible and consistent.

WRONG or MISALIGNED for us (concrete, checkable):

- LoRA hyperparameters are internally inconsistent and wrong-valued. The doc's
  `--lora-parameters '{"rank": 16, "alpha": 32, ..., "scale": 10.0}'` lists BOTH
  `alpha` and `scale`. Our `mlx_lm` tuner has NO `alpha` key (grep of
  `tuner/` returns nothing); it uses `scale`. And our working `scale` is 2.0,
  not 10.0. A scale of 10 is ~5x our adapter influence and is a recipe for the
  exact malformed-JSON / repetition-loop degradation the doc itself warns about
  in 3.2.3. Do not copy these numbers.
- Target modules: the doc says target attention only (q/k/v/o, section 3.1). Our
  configs target attention AND MLP (q/k/v/o + gate/up/down, 7 keys). Not
  universally "wrong," but narrower than what we actually run and have validated.
- Base model id mismatch with consequences: the doc assumes
  `mlx-community/gemma-4-e2b-it-4bit` (the multimodal IT model). We use
  `mlx-community/Gemma4-E2B-IT-Text-int4`, a TEXT-only int4. So the entire
  multimodal / image-token / "multimodal-before-text" apparatus is moot for our
  path, and so is most of the channel-token discussion.
- "Thought-channel leakage" is the wrong diagnosis for our Gemma 4 E2B pain. Our
  actual blocker was a LOADER bug (the mlx quants ship redundant KV-shared
  projection weights, producing `ValueError: Received 140 parameters not in
  model`), fixed by our 6-line `gemma4_text_patch.py` sanitize extension. That is
  not chat-template thought leakage. The doc's `<|channel|>thought` /
  `<channel|>` delimiters are GPT-OSS "harmony" channel syntax, not what our
  `gemma4_text` path emits.

SUSPECT (likely hallucination / cross-model bleed; treat as unverified):

- "MXFP4" used repeatedly for the 4-bit base. mlx-community Gemma quants are
  affine/symmetric int4 (our model literally says `int4`). MXFP4 is the
  gpt-oss / Blackwell micro-scaled FP4 format. The bleed is provable from the
  doc's own citation [35], which is an `mlx_lm fuse` bug filed for
  **gpt-oss-20b MXFP4-Q8** and then silently re-applied to Gemma. So the "static
  merging is broken on Apple Silicon" warning rests on a gpt-oss bug, not a
  Gemma one. The caution may still be prudent, but it is not evidence about our
  model.
- "MOLA framework (Modular Optimization for Local Adaptation), developed by the
  MLX community." Citations [37][38][40] point at ml-explore GitHub discussion
  threads and a Reddit post titled "multi-LoRA inference server for MLX." A
  discussion thread is not a named, adoptable framework. The hot-swap TECHNIQUE
  is real and easy; the branded "MOLA framework" is overstated. Note also the
  name collides with the academic "MoLA" (mixture-of-LoRA-experts), a different
  thing.
- The performance numbers ("~24% throughput overhead," "~28% time / 60% memory
  for gradient checkpointing") are oddly precise and not tied to a primary
  measurement we can see. Treat as illustrative, not measured.

BROKEN (do not run):

- The section 6.3 hot-swap script. `preload_adapter` does `mx.load(weight_files)`
  then `route_to` does `self.model.update(self.adapter_cache[name])`. Calling
  `model.update(...)` with raw LoRA weight arrays against a base whose Linear
  layers were never wrapped as LoRA layers does NOT apply any delta. mlx_lm
  applies LoRA by converting layers (`linear_to_lora_layers`) and loading adapter
  weights into those wrappers. As written the script would error or silently
  no-op back to base behavior, which is ironically the very failure mode the doc
  attributes to `fuse`. This is the headline deliverable and it does not work.

## 3. Strategic fit: should we do any of this now?

No. A three-adapter hot-swap architecture is premature complexity for where we
are. As of today we have just established (faithful harness, held-out decks) that
SINGLE-adapter volume SFT beats the untuned base and generalizes moderately to
fresh decks. Our live bottlenecks are: (a) the JSON-discipline regression from
SFT, (b) the teacher ceiling (~31%), and (c) logging yield (~78% of attempts
wasted). Multi-adapter routing addresses none of these. It would add a routing
surface, a second/third corpus we do not have, and a serving path whose reference
implementation here is broken, in exchange for solving a problem (catastrophic
interference across distinct domains) that our single-domain setup does not have.

## 4. Takeaway worth keeping #1: the format-vs-policy framing

The one genuinely useful idea, almost accidental: the doc treats "strict JSON /
tool-calling syntax" as its OWN competence (Adapter C), separate from reasoning.
We independently measured that our decision-SFT DEGRADES JSON discipline ~6x (the
gate adapter triggered 163 temp-retry turns vs base ~12). The doc's framing
reinforces a real hypothesis: format competence and policy competence are in
tension inside a single adapter, and our adapter is paying for better policy with
worse formatting.

But the doc's REMEDY (a dedicated high-rank, scale-10 JSON adapter) is backwards
for us and unvalidated. Our cheaper, already-planned fix is constrained decoding
/ a JSON grammar at inference time, which removes the format burden from the
weights entirely and sidesteps the tension without a second adapter or a routing
layer. Net: the doc validates the DIAGNOSIS, not its remedy. Keep the framing,
keep our remedy.

## 5. Takeaway worth keeping #2: grad-accumulation is a free, untried knob

`--grad-accumulation-steps` is real in our `mlx_lm` and we are not using it (our
configs run batch_size 1, no accumulation, so effective batch is 1). Accumulating
over, say, 8-16 steps gives a smoother effective batch at near-zero extra memory.
This is a cheap one-line A/B if we ever suspect gradient noise is hurting the
small-batch SFT. Minor, but concrete and correct, and it came from this doc.

## 6. Bottom line

- File the external doc as a reference (done). Do not treat its claims as facts.
- Do NOT adopt the multi-adapter architecture now. Premature; wrong problem;
  reference code broken.
- Do NOT copy its hyperparameters (scale 10, alpha key, attention-only targeting)
  or its hot-swap script.
- DO keep two things: the format-vs-policy framing for the JSON regression
  (supports our planned constrained-decoding fix), and grad-accumulation as an
  available untried knob.
- If we ever revisit multi-adapter serving, re-verify every named external (MOLA,
  the channel tokens, MXFP4, the perf numbers) against primary MLX sources first.
  The doc has a demonstrable gpt-oss-to-Gemma bleed, so its specifics are not
  trustworthy without independent confirmation.
