# Lab notebook: Gemma 4 E2B as v2 Solitaire-advisor student

**Investigator**: Chayut Orapinpatipat (with Claude Opus 4.7 as instrument operator)
**Dates of active work**: 2026-05-25 through 2026-05-26
**Apparatus**: Apple M5 16 GB unified memory, macOS Darwin 25.3.0, Python 3.12 venv
**Repo state at start**: branch `data-ingestion-pipeline` at b08d746, v1.1 LoRA shipped to HF
**Repo state at end of log**: same branch, four new local files staged for commit, no remote pushes

---

## 1. Background and motivation

A v1.1 LoRA adapter distilling a 31B Gemma 4 Klondike Solitaire advisor into a 2.3B-active student was published earlier this week at `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`. v1.1 was the previous-generation Gemma 3n architecture, chosen as a workaround: every attempt to load the architecturally-correct Gemma 4 E2B base via `mlx_lm.load` had failed with `Received 140 parameters not in model`, a hard architecture-mismatch error against the published `mlx-community/gemma-4-E2B-it-*` quants. The project memory `gemma4-to-3n-pivot-mlxlm-blocker.md` recorded this as an upstream library bug, indefinitely blocking the intended target.

The arrival of a third-party research document (`/Users/chayut/Downloads/MLX Gemma 4 Custom Training.md`) referencing `mlx-vlm` and `mlx-tune` as Gemma 4 capable, plus the user's explicit ask to find a working long-term Gemma 4 path, prompted re-opening the question. The motivating user prompt was: *"good to explore anyway, we will need a way forward with Gemma 4."*

### 1.1 Pre-registered hypothesis

> **H1**: A Gemma 4 E2B student, distilled from the 31B teacher using the same procedure that worked for v1.1, will produce a v2 LoRA adapter that strictly beats v1.1 on the 20-state evaluation bench (mean tier strictly above 3.15, gap-to-teacher strictly below 0.27). The same-series student-teacher pairing is expected to distill more cleanly than the cross-series Gemma-3n student.

### 1.2 Success criteria pre-registered

- **Primary**: best v2 checkpoint mean tier > v1.1's 3.15 on the 20-state bench
- **Secondary**: foundation recovery >= 6/7 (matching v1.1)
- **Memory budget**: peak training memory < 16 GB unified memory envelope
- **Ship gate**: if primary criterion met, publish at `chayuto/gemma-4-e2b-it-solitaire-advisor-lora`; if not met, document negative result and hold

### 1.3 Equipment and dependencies

| component | version | role |
|---|---|---|
| mlx | 0.31.2 | Apple Silicon array framework |
| mlx-lm | 0.31.3 | LLM training / inference on mlx |
| mlx-vlm | 0.5.0 | Vision-language sibling library (gemma4 model class lives here) |
| mlx-metal | 0.31.2 | Metal kernel backend |
| huggingface-hub | 1.16.1 | Model + dataset download |
| Python | 3.12.13 | Required by mlx wheels (no 3.14 support) |

Existing v1 virtual environment at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/venv/` was reused; an isolated sibling venv at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/venv_vlm/` was created on 2026-05-25 to test mlx-vlm without polluting the proven v1 stack.

### 1.4 Evaluation bench

Twenty hand-curated game states from `experiments/a4_phase1.5_2026_05_24/prompts/C0/`, classified into three categories:
- `early-*` (5 states): early-game positions where multiple productive moves usually exist
- `midgame-*` (8 states): mid-game positions including the difficult `midgame-4ab5735a4f20` (a "stubborn unrecovered foundation state" already known from v1)
- `oscillation-*` (7 states): doom-loop-prone positions where the teacher chose `draw_card` because no productive tableau move was available

Each prompt has a known teacher pick recorded in `gemma4_finetune/teacher_picks_n20.json`. Quality is scored on a six-level tier scale via `scripts/ab_test_prompt_formats.py::classify_pick` (foundation > reveal > waste_play > shuffle > draw > illegal, with `TIER_RANK` 6/5/4/2/1/0 respectively). Mean-tier deltas of plus or minus 0.40 on this 20-state bench were established as the noise floor in the Phase 1.5 study.

---

## 2. Experiment 1: does `mlx-vlm` load Gemma 4 E2B at all?

### 2.1 Hypothesis

H1a: The research doc's claim that `mlx-vlm` registers a `gemma4` model class is correct, and the 4-bit multimodal quant will load and generate coherent text.

### 2.2 Method

Installed mlx-vlm 0.5.0 into the sibling venv. Inventoried the model registry under `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/venv_vlm/lib/python3.12/site-packages/mlx_vlm/models/`. Wrote a probe script that imports `mlx_vlm.load` + `mlx_vlm.generate` + `mlx_vlm.prompt_utils.apply_chat_template`, loads `mlx-community/gemma-4-E2B-it-4bit`, and runs a single 16-token generation against the prompt "What is 2+2? Answer with one digit."

### 2.3 Observations

- `gemma4` model class confirmed present in mlx-vlm 0.5.0 (alongside `gemma3`, `gemma3n`, `paligemma`).
- Load: 3.5 s wall, 3.65 GB peak MLX memory.
- Prompt templating produced a 73-character wrapped prompt from a 21-token input.
- Generation: 5.4 s wall, returned "Four" (correct, coherent).
- Throughput: 152.5 tokens/sec (generation_tps from the GenerationResult struct).
- One spurious warning emitted at load: `transformers/audio_utils.py:555: At least one mel filter has all zero values.` This corresponds to the audio Conformer's mel-filter bank initialising. Conformer is therefore resident in memory even for text-only inference.

### 2.4 Interpretation

H1a confirmed. The "Gemma 4 cannot load on mlx" framing was wrong; it should have been "Gemma 4 cannot load *via mlx-lm*." The architecture is registered in the sibling vision-language package because all Gemma 4 dense variants ship with audio + vision conformer towers baked into the same model graph as the language head.

### 2.5 Status

Rung 1 of the exploration plan: PASS. Branch alive. Proceeded to E2.

---

## 3. Experiment 2: does untuned Gemma 4 E2B beat untuned Gemma 3n on the 20-state bench?

### 3.1 Hypothesis

H2: Untuned Gemma 4 E2B (5.1B total, 2.3B active) will outperform untuned Gemma 3n E2B (text-only 4bit DWQ) on mean tier, on the basis of architectural improvements and same-series alignment with the teacher.

### 3.2 Method

Wrote `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_gemma4_runner.py` mirroring the v1 baseline runner exactly, swapping mlx-lm for mlx-vlm and using temperature 0.0 + max_tokens 512. Ran against the 20 C0 prompts. Post-hoc tier scoring via `score_n20_gemma4.py`, applying the shared `classify_pick` from `/Users/chayut/repos/solitaire-analytics/scripts/ab_test_prompt_formats.py`.

### 3.3 Observations

| metric | untuned gemma-3n (v1 baseline) | untuned gemma-4 (mlx-vlm) | delta |
|---|---:|---:|---:|
| JSON validity | 20/20 | 20/20 | 0 |
| Illegal moves | 1/20 | 0/20 | -1 |
| Teacher-pick agreement | 11/20 | 12/20 | +1 |
| Mean tier | 2.10 | 2.75 | +0.65 |
| Gap vs teacher (3.42) | -1.32 | -0.67 | halved |
| Foundation recovery (of 7) | 2 | 4 | +2 |
| Peak MLX memory | 6.26 GB | 5.30 GB | -1.0 GB |
| Mean call time | 13.9 s | 5.1 s | 2.7x faster |

### 3.4 Interpretation

H2 confirmed with strong signal. The +0.65 mean-tier delta clears the 0.40 single-run noise band established by the Phase 1.5 study. The unexpected secondary result was the memory and throughput advantage: despite carrying an audio Conformer the multimodal Gemma 4 4-bit quant is leaner and faster than the text-only Gemma 3n 4-bit DWQ. The intuition that "Gemma 3n is leaner because text-only" turned out to be wrong at the measured 4-bit quantisation levels.

### 3.5 Status

Rung 2: PASS with strong signal. Branch genuinely worth pursuing.

---

## 4. Experiment 3: can we route through `mlx-lm` instead of `mlx-vlm` for cleaner pipeline integration?

### 4.1 Motivation

Adopting mlx-vlm as the v2 toolchain would mean maintaining a parallel virtual environment, parallel runners, and a parallel trainer (`mlx_vlm.lora`) alongside the proven v1 mlx-lm pipeline. The user explicitly requested optimising the student for text-only use, noting "we never ever need audio or visual." If a text-only Gemma 4 quant exists and loads in mlx-lm, the v2 pipeline collapses into the existing v1 infrastructure with the model name swapped.

### 4.2 Hypothesis

H3: A text-only Gemma 4 quant published by `mlx-community` will load via mlx-lm 0.31.3, since mlx-lm ships a `gemma4_text.py` model class.

### 4.3 Method

Searched the `mlx-community` HF namespace for any repo containing the substring "gemma-4" or "Gemma4". Identified two text-only candidates:
- `mlx-community/Gemma4-E2B-IT-Text-int4` (apache-2.0, 2026-05-19, audio + vision encoder weights physically absent, `model_type: gemma4_text`)
- `mlx-community/gemma-4-text-26b-a4b-it-8bit` (too large for our envelope)

Attempted to load `Gemma4-E2B-IT-Text-int4` via `mlx_lm.load`.

### 4.4 Observations

Hard failure with the same error signature as the original blocker:

```
ValueError: Received 140 parameters not in model.
  model.layers.15.self_attn.k_proj.biases,
  model.layers.15.self_attn.k_proj.scales,
  ...
  model.layers.34.self_attn.v_proj.weight
```

H3 falsified for the naive approach. Repeated with mlx-vlm 0.5.0 (which does register `gemma4` but not `gemma4_text`):

```
ValueError: Model type gemma4_text not supported.
Error: No module named 'mlx_vlm.speculative.drafters.gemma4_text'
```

Both library paths refuse to load the text-only quant.

### 4.5 Method (continued): source inspection

Read `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/venv/lib/python3.12/site-packages/mlx_lm/models/gemma4_text.py` (676 lines). Examined the `Attention.__init__` constructor at line 176-229 and the `Model.sanitize` method at line 610-639.

### 4.6 Observations (source)

At line 183, the attention module decides per-layer whether to allocate KV projection matrices:

```python
self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers
```

For Gemma 4 E2B (`num_hidden_layers=35`, `num_kv_shared_layers=20`), `has_kv` evaluates True for layers 0-14 and False for layers 15-34. The KV-shared layers (15-34) reuse projections from earlier layers via the `previous_kvs` mapping constructed in `Gemma4TextModel.__init__` at line 433-442. Accordingly, lines 206-217 skip allocating `k_proj`, `v_proj`, `k_norm`, and `v_norm` modules when `has_kv` is False.

The existing `sanitize()` at line 610-639 normalises `experts.gate_up_proj` splitting and strips a handful of rotary-embedding artifacts, but does *not* strip the redundant k/v weights for KV-shared layers. The published quants ship these weights (140 total: 20 layers x 7 weights per layer = 140 matches the error count exactly).

### 4.7 Interpretation

The bug is a six-line oversight in `sanitize()`: the loader has all the information it needs to identify and drop the redundant weights, but currently does not. The model class is otherwise correct; the runtime KV-sharing logic works fine when the matrices are simply absent.

### 4.8 Method (continued): monkey-patch and re-test

Wrote `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/gemma4_text_patch.py` (15 lines) that imports `mlx_lm.models.gemma4_text` and replaces `Model.sanitize` with a wrapper that drops KV-shared-layer k/v weights:

```python
import re
import mlx_lm.models.gemma4_text as _g4t

_PAT = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)(\.|$)")
_orig_sanitize = _g4t.Model.sanitize

def _patched(self, weights):
    sanitized = _orig_sanitize(self, weights)
    n_shared = self.args.num_kv_shared_layers
    if n_shared <= 0:
        return sanitized
    first_shared = self.args.num_hidden_layers - n_shared
    return {
        k: v for k, v in sanitized.items()
        if not ((m := _PAT.match(k)) is not None and int(m.group(1)) >= first_shared)
    }

_g4t.Model.sanitize = _patched
```

Re-ran the load probe with this module imported before `mlx_lm.load`.

### 4.9 Observations (patched)

| measurement | value |
|---|---|
| weights dropped by patch | 140 (matches the original error count exactly) |
| load wall time | 1.9 s |
| peak MLX memory at load | 2.62 GB |
| generation wall (16 tokens) | 2.0 s |
| peak MLX memory at generation | 2.70 GB |
| output | `<|channel>thought\nThinking Process:\n1. **Analyze the Request:**...` |

The output ran into the max_tokens cap mid-reasoning-chain but was structurally coherent. Repeated with max_tokens=200 produced a complete thinking-mode response ending with "4." (correct).

### 4.10 Status

H3 confirmed conditionally: the text-only quant loads via mlx-lm with the patch applied. The patch is non-invasive (no edits to the installed package), reversible (just don't import the module), and a candidate for upstream submission. v2 pipeline can therefore stay in the v1 venv.

---

## 5. Experiment 4: full bench against the patched text-only base

### 5.1 Hypothesis

H4: The patched text-only Gemma 4 E2B will produce bench results within noise of the multimodal Gemma 4 (both use the same underlying language layers; only the audio + vision towers differ, which are unused for text input).

### 5.2 Method

Wrote `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_gemma4_text_runner.py` and `score_n20_gemma4_text.py`, both importing `gemma4_text_patch` before any `mlx_lm.load` call. Initial run at max_tokens=512 produced 0/20 JSON validity because the text-only model generates substantially longer thinking-mode chains than the multimodal model on the same prompts (a confound worth flagging, see Section 9). Re-ran at max_tokens=2048.

### 5.3 Observations

Three-way comparison, all untuned:

| metric | gemma-3n (v1 base) | gemma-4 multimodal | gemma-4 text-only (patched) |
|---|---:|---:|---:|
| JSON validity | 20/20 | 20/20 | 20/20 |
| Illegal moves | 1/20 | 0/20 | 0/20 |
| Teacher agreement | 11/20 | 12/20 | 12/20 |
| Mean tier | 2.10 | 2.75 | 2.55 |
| Gap vs teacher (3.42) | -1.32 | -0.67 | -0.87 |
| Foundation recovery (of 7) | 2 | 4 | 4 |
| Peak memory | 6.26 GB | 5.30 GB | 3.35 GB |
| Mean call time | 13.9 s | 5.1 s | 15.7 s (longer chains) |
| max_tokens cap | 512 | 512 | 2048 |
| Chat template SHA | n/a | 2f1b4d75 | 2f1b4d75 (byte-identical) |

### 5.4 Interpretation

H4 nearly confirmed: 0.20 mean-tier delta between text-only and multimodal is within the 0.40 noise band but is not statistically zero. Agreement and foundation recovery match exactly. The delta is almost certainly the quantisation scheme: the multimodal quant is "4bit", the text-only is "int4", which are not identical algorithms despite the similar names. Same prompts, byte-identical chat template, deterministic sampling all rule out other obvious confounds.

The +1.95 GB memory headroom over multimodal, the apache-2.0 licensing, and the v1-toolchain-unchanged property all favour the text-only path. Selected as the v2 base.

### 5.5 Status

Rung 2b: PASS. v2 base finalised as `mlx-community/Gemma4-E2B-IT-Text-int4` plus `gemma4_text_patch.py`.

---

## 6. Experiment 5: distillation smoke test

### 6.1 Hypothesis

H5: The 15-line monkey-patch will survive a real `mlx_lm.lora` invocation (not just `mlx_lm.load`), and loss will decrease over a short training loop.

### 6.2 Method

Regenerated game-level train/valid/test splits from the 1635-row clean post-stall-filter training set at `/Users/chayut/repos/solitaire-analytics/data/dataset/training.jsonl`. The `prepare_dataset.py` script dropped 152 rows for malformed completion JSON, leaving 1483 examples across 27 distinct games. Split fractions 0.8 / 0.1 / 0.1 produced 1168 train / 126 valid / 189 test rows.

Wrote `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/train_v2.py` as a thin wrapper that imports `gemma4_text_patch` before delegating to `mlx_lm.lora.main()`. Wrote `lora_config_v2_smoke.yaml` with iters=30, save_every=30, steps_per_eval=15, otherwise identical hyperparameters to v1.

### 6.3 Observations

| iter | val loss | train loss | peak mem | wall (cumulative) |
|---:|---:|---:|---:|---:|
| 1 | 3.155 | n/a | 8.36 GB | 7 s |
| 5 | n/a | 2.251 | 8.36 GB | 27 s |
| 10 | n/a | 0.905 | 8.36 GB | 52 s |
| 15 | 0.581 | 0.670 | 8.36 GB | 80 s |
| 20 | n/a | 0.591 | 8.36 GB | 105 s |
| 25 | n/a | 0.497 | 8.36 GB | 130 s |
| 30 | 0.390 | 0.480 | 8.36 GB | 158 s |

Adapter saved to `adapters_v2_smoke/adapters.safetensors`. Trainable parameters: 0.275% of base (12.7M of 4.63B). Throughput: 415 tok/s, 0.20 it/sec.

### 6.4 Interpretation

H5 confirmed. Loss curve has the expected shape (steep drop in the first few iterations followed by gradual descent). Peak memory at 8.36 GB leaves 7.6 GB of envelope headroom, lower than v1's T5 peak of 11.49 GB. Projected wall time for 1000 iters at this throughput: 83-90 minutes.

### 6.5 Status

Rung 3: PASS. No surprises. Proceed to full training.

---

## 7. Experiment 6: full 1000-iter v2 distillation and checkpoint sweep

### 7.1 Pre-registered prediction (most important)

Per H1: best v2 checkpoint will achieve mean tier > 3.15. Specifically: based on v1.1's +1.05 mean-tier lift from its untuned baseline of 2.10 to 3.15, an equivalent absolute lift on v2's untuned baseline of 2.55 predicts a v2 endpoint near 3.60 (above the teacher). A more conservative prediction based on fractional gap closure (v1.1 closed 80% of its -1.32 gap) predicts v2 closes 80% of its -0.87 gap, landing at 3.27 (just below the teacher, slightly above v1.1).

### 7.2 Method

Ran `train_v2.py --config lora_config_v2.yaml` in the background. Configuration identical to v1's T5 except for `model` (text-only Gemma 4) and `data` (dataset_v2). Hyperparameters: rank 16, scale 2.0, dropout 0.05, learning rate 2e-4, max_seq_length 2048, batch_size 1, num_layers 16, grad_checkpoint true, target modules `self_attn.{q,k,v,o}_proj` plus `mlp.{gate,up,down}_proj`, save_every 250.

Post-training, ran `sweep_v2_checkpoints.sh` which:
1. Stages each `0000XXX_adapters.safetensors` checkpoint into a temp directory with the matching `adapter_config.json`
2. Runs `posttune_n20_gemma4_text_runner.py` against each (with the patch applied), max_tokens=2048
3. Scores all checkpoints with `score_v2_learning_curve.py` and compares against v1 baselines and v1.1 iter-750

### 7.3 Observations

Training (single run, 1000 iters, ~83 min wall):

| iter | val loss | train loss | peak mem |
|---:|---:|---:|---:|
| 1 | 3.16 | n/a | 8.41 GB |
| (during) | (final) | 0.226 | 8.41 GB |
| 1000 | 0.358 | 0.226 | 8.41 GB |

Loss trajectory closely matches v1's T5 run. Trainable parameters: 12.7 M (0.275%). Throughput: 427 tok/s. Total trained tokens: 2,026,733.

Checkpoint sweep (each run is N=20 single-shot at temperature 0.0, max_tokens 2048):

| config | json | illegal | agree | tier | gap | foundations | peak | mean sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2 untuned** | 20/20 | 0/20 | 12/20 | **2.55** | -0.87 | **4/7** | 3.35 GB | 15.7 |
| v2 iter 250 | 20/20 | 0/20 | 10/20 | 2.30 | -1.12 | 3/7 | 3.41 GB | 5.7 |
| v2 iter 500 | 20/20 | 2/20 | 10/20 | 2.45 | -0.97 | 4/7 | 3.41 GB | 5.4 |
| v2 iter 750 | 20/20 | 2/20 | 9/20 | 2.20 | -1.22 | 4/7 | 3.41 GB | 5.0 |
| v2 iter 1000 | 20/20 | 2/20 | 8/20 | 2.05 | -1.37 | 3/7 | 3.41 GB | 5.2 |
| v1 untuned (3n) | 20/20 | 1/20 | 11/20 | 2.10 | -1.32 | 2/7 | 6.26 GB | 13.9 |
| **v1.1 iter 750 (3n, shipped)** | 20/20 | 2/20 | 11/20 | **3.15** | **-0.27** | **6/7** | 6.37 GB | 14.8 |

### 7.4 Interpretation

**H1 falsified**. The pre-registered prediction was that best v2 mean tier would clear 3.15. Observed best v2 mean tier was 2.45 (iter 500). The gap to teacher *widened* from the untuned baseline (-0.87 became -0.97 at the best checkpoint, and -1.37 at iter 1000). This is the opposite of the v1.1 behaviour, which closed its gap by 80%.

The internal training metrics (val loss 3.16 to 0.36, matching v1's curve) suggested the run was healthy. The discrepancy between cleanly-decreasing val loss and degrading downstream task quality is a known distillation failure pattern: the model is learning to match output *format* very closely while losing the underlying decision *quality*.

### 7.5 Status

Rung 4: HOLD. v2 LoRA strictly underperforms v1.1 across mean tier (2.45 vs 3.15), agreement (10/20 vs 11/20), and foundation recovery (4/7 vs 6/7). Surface area of the regression is investigated in the next section.

---

## 8. Post-hoc analysis: locating the regression

### 8.1 Method

Wrote an inline diff over the v2-untuned and v2-iter-1000 response JSONs:

```python
for sid in base_by:
    bu, iu, tu = base_by[sid]["e2b_move_index"], iter1k_by[sid]["e2b_move_index"], base_by[sid]["teacher_move_index"]
    if base["agreement"] and not iter1k["agreement"]: tag = "REGRESSED"
    ...
```

### 8.2 Observations

20 per-state outcomes:
- 12 unchanged (same pick, both right or both wrong identically)
- 1 improved (`early-e6291973dd07`: wrong before, correct after)
- 2 shifted (both wrong; pick changed)
- **5 regressed (correct untuned, wrong post-tuning)**: every single one is an `oscillation-*` state

The five regressed states:
- `oscillation-21cc5243e1d8`: 0 -> 2 (teacher 0)
- `oscillation-30700e2ca639`: 0 -> 1 (teacher 0)
- `oscillation-a774c0d22f24`: 2 -> 0 (teacher 2)
- `oscillation-d0ff552ed744`: 0 -> 1 (teacher 0)
- `oscillation-d729a3bd2b7a`: 2 -> 3 (teacher 2)

One additional oscillation state (`oscillation-026f3139d6f2`) shifted from one wrong answer to a different wrong answer (3 -> 1, teacher 0).

The regression is fully localised to oscillation states. Early-game and midgame categories are essentially unchanged.

### 8.3 Method: response inspection

Pulled the full text of one regressed pair: `oscillation-21cc5243e1d8`, untuned (correct, picked 0) vs iter-1000 (wrong, picked 2).

### 8.4 Observations (qualitative)

Untuned response, abbreviated:
> *strategic_plan*: "Since no productive tableau moves (revealing hidden cards or creating sequences) or foundation moves are available, the optimal strategy is to draw from the stock. This introduces new cards into the game, which increases the probability of finding a card that can be played... This is the standard procedure when the board is static and no tactical progress can be made."
> *final_decision*: move_index 0, confidence 0.95

Iter-1000 response, in full:
> *board_analysis*: "...The waste card is 4H, which cannot be played on any current tableau card (requires a black 5) or the foundations (requires a red 5)."
> *strategic_plan*: "The primary goal is to reveal hidden cards and advance the foundations. Move [1] (6S from waste to Col 3) is the most productive move because it utilizes the 6S to receive a red 5, thereby creating a new sequence (7H-6S) and exposing the face-down card in Column 5..."
> *final_decision*: move_index 2, confidence 0.9

The iter-1000 model:
1. Correctly identified in `board_analysis` that the waste card 4H is useless
2. *Fabricated* a non-existent move ("Move [1] (6S from waste to Col 3)") in `strategic_plan`. The actual move [1] is something else; the model invented this description to justify picking move 2
3. Committed to move_index 2 at confidence 0.9

### 8.5 Interpretation

This is the **doom-loop confabulation anti-pattern** documented in the project memory `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/flag-unsolvable-boards-early.md`. The memory describes the 31B teacher's tendency to "rationalise a two-card shuffle as a 'setup move' even when `recentMoves` clearly shows the exact reversal was just played. Confidence stays saturated at 0.9+ throughout." The published dataset card additionally notes: "Confidence is miscalibrated. Reported `confidence` spans 0.60-1.00 (mean 0.91); the teacher signals near-certainty regardless of board state."

The training corpus contains the teacher's doom-loop responses (we know ~21% of post-cutover games stall; the per-turn stall filter does not catch individual rationalising turns before the plateau reaches 25 turns). LoRA training transferred those response patterns into the student. The trained student now exhibits the same confabulation behaviour at the same confidence levels.

### 8.6 Reframing: why v1.1 did not show this failure mode

v1.1's untuned Gemma 3n baseline had mean tier 2.10 with 2/7 foundation recovery. It was already weak at the same axis where the corpus has problems. The teacher's average behaviour was a strict improvement on the untuned baseline, so training showed monotonic gains regardless of the specific direction of those gains.

v2's untuned Gemma 4 baseline had mean tier 2.55 with 4/7 foundation recovery and zero illegal moves and clean draw choices on all seven oscillation states. The teacher's average behaviour was a strict improvement on *some* states (foundation moves) but a strict *regression* on others (oscillation states where untuned beat the teacher). The net effect was negative.

### 8.7 Reformulated finding

> **Observation O1**: A 2.3B-active student locally beats a 31B teacher on a specific failure mode (doom-loop oscillation recognition). Naive distillation of the teacher's average behaviour erodes this strength.

This is not a "smaller is smarter" result in general. It is a "task-specific advantage on a specific failure pattern can exist independent of model size, and distillation can be net-negative when it does."

---

## 9. Confounds and caveats

### 9.1 Sample size

The strength claim ("E2B untuned outperforms teacher on oscillation states") rests on 5 of 20 bench states. The Phase 1.5 study established noise of plus or minus 0.40 mean tier for single-run measurements. The observed +0.65 untuned mean-tier delta over v1's baseline clears that noise band. The observed -0.70 delta of trained vs untuned v2 also clears it. But individual state-level claims (e.g. "v2 untuned beats teacher on `oscillation-21cc5243e1d8`") are noisier and not multiply-seeded. A larger oscillation-only bench (N=40 or more) with multiple seeds per state would tighten this.

### 9.2 Inference temperature asymmetry

All v2 measurements were taken at temperature 0.0 (deterministic). The 31B teacher in the harvester runs at the harvester's configured temperature, which we did not measure for this experiment. If the teacher runs at temperature 0.3-0.7 in production, some fraction of its doom-loop responses are sampling artefacts rather than deterministic model behaviour. This does not change the corpus-poisoning argument (the corpus still contains those samples), but it weakens the "untuned beats teacher" framing somewhat.

### 9.3 Thinking-mode deliberation asymmetry

The untuned Gemma 4 E2B emits the `<|channel>thought` chain-of-thought prefix by default and produces ~2000-token reasoning chains. The teacher in production almost certainly emits direct JSON (no thinking mode) to keep per-turn latency tolerable. So the smaller student had more inference-time compute per decision than the larger teacher, routed through deliberation rather than weights. This is a real asymmetry not controlled for; it likely accounts for some fraction of the untuned advantage.

Post-distillation, mean call time dropped from 15.7 s to 5.0-5.7 s. The trained model appears to have learned to skip the thinking-mode prefix and emit JSON directly (matching the training data's `{prompt, completion}` shape where the completion is raw JSON with no thinking-mode wrapper). This is consistent with the "compressed deliberation" hypothesis: the trained model lost not just oscillation accuracy but also its deliberation chain.

### 9.4 Quantisation asymmetry between bench measurements

The text-only Gemma 4 base is "int4" quantised. The multimodal Gemma 4 base is "4bit" quantised. These produce a 0.20 mean-tier gap on the same prompts with the same chat template (verified byte-identical). The downstream training measurements all use the text-only int4 base; the comparison to v1.1 uses Gemma 3n 4bit DWQ. Cross-base comparisons inherit unknown quantisation-induced biases.

### 9.5 Single-run training

The v2 1000-iter training was run once. v1.1's training was also run once, with the iter-750-promotion decision based on a four-checkpoint sweep. The shape of the v2 learning curve (monotonically downward after iter 500) is unlike v1.1's peaked curve. Whether this is reproducible across seeds is unknown.

### 9.6 Possible benign explanation for the regression

It is possible that the trained model is genuinely *trying* the same strategy as the teacher on these oscillation states (a tableau swap rationalised as setup), and the bench-state design happens to be unfair to that strategy. In games with full lookahead, a setup move that looks pointless one turn might pay off three turns later. Single-turn evaluation cannot distinguish "wrongly chose tableau swap" from "correctly set up for a sequence the bench cannot see." This caveat does not rescue the v2 LoRA from the HOLD decision (the original goal was to match teacher choice, and it does not), but it leaves open the possibility that the trained model is in fact closer to optimal play than this bench can measure.

---

## 10. Conclusions

1. The mlx-lm 0.31.3 `gemma4_text.py` loader bug is real, diagnosable, and fixable with a 15-line `sanitize()` extension. The patch is staged for upstream submission as a PR.
2. The patched text-only Gemma 4 E2B (`mlx-community/Gemma4-E2B-IT-Text-int4`) loads cleanly via the v1 mlx-lm toolchain at 2.7 GB peak inference memory, half the v1 base. Apache-2.0 licensed.
3. Untuned Gemma 4 E2B on the 20-state bench: mean tier 2.55, 12/20 agreement, 4/7 foundations, zero illegal moves, clean oscillation-state choices. Strictly better than v1's untuned Gemma 3n baseline (2.10, 11/20, 2/7, 1 illegal).
4. **Naive distillation of the same v1.1 recipe does not transfer to Gemma 4 E2B**. Best v2 checkpoint scored mean tier 2.45, below the v2 untuned baseline of 2.55 and far below v1.1's 3.15.
5. The regression is fully localised to `oscillation-*` states. The failure mode is doom-loop confabulation transferred from the teacher's responses in the training corpus.
6. **A 2.3B-active untuned student locally beats a 31B teacher on a specific failure mode** that the project memory had already flagged independently. Distillation erodes this strength.
7. Pre-registered ship gate (best v2 mean tier > 3.15) not met. Decision: HOLD on shipping v2 LoRA.

---

## 11. Recommended next experiments

In order of expected information per unit compute.

### 11.1 Ship v2 untuned as an apache-2.0 companion to v1.1 (zero compute)

The untuned text-only Gemma 4 base, with no LoRA, is already a publishable artifact. Different value proposition than v1.1 (lower foundation recovery but oscillation-resistant, half the memory, cleaner licensing). Use case: 8 GB Macs, commercial scope, oscillation-heavy game patterns. The publish_hf staging directory at `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/publish_hf_v2/` is already prepared; the README needs to be rewritten to drop the LoRA framing.

### 11.2 Re-train with per-turn shuffle-fraction filter (~2 hours)

The current `scripts/ingest_exports.py` stall filter catches plateaus >= 25 turns at the session level. A per-turn filter would mark individual training rows whose chosen move is a tableau-to-tableau move matching a reversal in `recentMoves`, and drop them from the training set. This addresses the root cause (corpus contains the failure pattern) rather than working around it. Estimated row drop: probably 5-15% of the 1483-row training set.

Pre-registered prediction for this experiment: if corpus poisoning is the dominant cause of the regression, retraining on a shuffle-filtered subset should produce a v2 LoRA with oscillation behaviour closer to untuned (4/7 or higher foundations, 12/20 or higher agreement on oscillation states) while preserving foundation-move gains. If the regression is dominated by other factors (capacity mismatch, learning rate, thinking-mode loss), this experiment will not help.

### 11.3 Re-train with thinking-mode preserved in training labels (~4 hours including data prep)

The current training data is `{prompt, completion}` where the completion is raw JSON. The trained model collapsed its inference from 15.7 s of thinking to 5.0 s of direct JSON output. If we re-rendered completions as `<|channel>thought\n<derived reasoning>\n<|channel>final\n{JSON}` (the derivation could be the existing `board_analysis` and `strategic_plan` fields, since they already are the reasoning trail), the student might retain its deliberation advantage. Cost includes a one-time corpus regeneration step.

### 11.4 Re-train with lower learning rate (~85 min)

Cheapest experiment. Re-run with learning_rate 5e-5 instead of 2e-4. If v2's untuned baseline is genuinely close enough to the teacher that the high LR is overshooting, a gentler training run should at minimum not regress below untuned. If it still regresses, the corpus-poisoning hypothesis is strengthened and the LR is not the bottleneck.

### 11.5 Submit the mlx-lm upstream PR (one afternoon, no compute)

Independent of any v2 outcome. The patch fixes a real bug affecting every Gemma 4 text-only quant user on mlx-lm. Draft staged at `/Users/chayut/repos/solitaire-analytics/docs/internal/mlx_lm_gemma4_text_pr_draft.md`. First step: clone mlx-lm main and confirm the bug is not already fixed there.

---

## 12. Files of record

Created or substantively modified during this experiment:

- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/gemma4_text_patch.py` (the loader patch)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_gemma4_runner.py` (untuned multimodal bench)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_gemma4_text_runner.py` (untuned text-only bench)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/score_n20_gemma4.py`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/score_n20_gemma4_text.py`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/train_v2.py` (patched trainer entry)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/lora_config_v2.yaml`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/lora_config_v2_smoke.yaml`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/posttune_n20_gemma4_text_runner.py`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/sweep_v2_checkpoints.sh`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/score_v2_learning_curve.py`
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/dataset_v2/{train,valid,test}.jsonl` (1168 / 126 / 189 rows, game-level split, gitignored)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/adapters_v2/` (4 checkpoint adapters plus final, ~46 MB each, gitignored)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/baseline_n20_gemma4_text/` (response files, scored JSONs, learning_curve.json)
- `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/publish_hf_v2/` (staging tree drafted before the HOLD decision; needs rework for untuned-only publication)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260525_gemma4_e2b_v2_exploration_plan.md` (plan + rung-by-rung)
- `/Users/chayut/repos/solitaire-analytics/docs/reports/20260525_session_close_v2_gemma4_text_TBD.md` (skeleton; needs renaming and HOLD-section fill)
- `/Users/chayut/repos/solitaire-analytics/docs/internal/mlx_lm_gemma4_text_pr_draft.md` (upstream PR text)
- `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/gemma4-to-3n-pivot-mlxlm-blocker.md` (updated with patched-loader resolution)
- `/Users/chayut/.claude/projects/-Users-chayut-repos-solitaire-analytics/memory/v2-distillation-teacher-doom-loop.md` (new memory documenting the negative result)

Resource budget consumed across the whole experiment: approximately 3 hours of M5 compute, of which 83 minutes was the v2 training itself and the remainder was eval runs, sanity checks, and the four `baseline_n20*` benches.
