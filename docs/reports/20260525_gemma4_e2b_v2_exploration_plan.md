# Gemma 4 E2B v2 exploration plan

**Date**: 2026-05-25 (rungs 1-2b); 2026-05-26 (rungs 3-4)
**Status**: rungs 1-4 complete. **Decision: HOLD on v2 LoRA**; v2 untuned has a documented strength the 31B teacher lacks, and distillation harms it.
**Triggered by**: external research doc on MLX + Gemma 4 (saved at `/Users/chayut/Downloads/MLX Gemma 4 Custom Training.md`) which surfaced that the Gemma 4 architecture is registered in `mlx-vlm`, not `mlx-lm`. Our earlier block at `mlx-lm 0.31.3` failing to load Gemma 4 E2B may have been a wrong-library issue rather than a missing-architecture-support issue.

## What stays untouched

The shipped v1.1 LoRA on `chayuto/gemma-3n-e2b-it-solitaire-advisor-lora` (iter-750, mean tier 3.15 vs teacher 3.42, gap -0.27) remains canonical. The pipeline under `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/` is the proven path. None of it gets edited or migrated during this exploration. Eval bench, training scripts, adapters, baseline results all stay in place.

If v2 ever ships, it goes to a sibling HF slot at `chayuto/gemma-4-e2b-it-solitaire-advisor-lora`. The decision to swap canonical models is contingent on v2 strictly beating v1.1 on the existing 20-state bench. Until then v1.1 is the recommended model and v2 is an experiment.

## Why explore now

Two signals from the research doc are load-bearing for whether this branch is worth pursuing:

1. The Gemma 4 dense family has a higher baseline ceiling than Gemma 3n (E2B is 5.1B total, 2.3B active vs 3n E2B's text-only architecture). If the architecture loads and untuned eval is meaningfully above 3n untuned (2.10 mean tier), fine-tuning has a higher upper bound.
2. The community tooling has matured around `mlx-vlm` (Blaizzy) and `mlx-tune` (ARahim3, Unsloth-compatible). If either works on Apple Silicon for Gemma 4 LoRA training, the runway is dramatically shorter than the handwritten path we built for v1.

What is **not** load-bearing and therefore not being investigated: TriAttention KV pruning, TurboQuant compression, K-eq-V attention as a feature, the throughput numbers in the doc. These don't gate whether v2 ships.

## Probe ladder

Each rung either unlocks the next or kills the branch.

### Rung 1: can `mlx-vlm` load Gemma 4 E2B at all?

**Cost**: ~5 minutes of install + ~2 GB of model download.
**Setup**: sibling venv at `gemma4_finetune/venv_vlm/` so the v1 pipeline stays reproducible. Do not touch `gemma4_finetune/venv/`.
**Command**: `mlx_vlm.generate --model mlx-community/gemma-4-E2B-it-4bit --prompt "test" --max-tokens 8`.
**Pass**: produces any coherent or even partly-coherent text without architecture-registration errors. The doc's core routing claim is validated.
**Fail**: architecture-missing or weight-load error. Capture the exact error, write a `failed` memory, branch dies cleanly here.

### Rung 2: untuned Gemma 4 E2B vs untuned Gemma 3n E2B on the 20-state bench

**Cost**: ~10 minutes, same harness as `gemma4_finetune/baseline_n20_runner.py`.
**Setup**: reuse the 20 game-state prompts already serialised under `gemma4_finetune/baseline_n20/`. Swap model path only.
**Pass**: Gemma 4 E2B untuned scores meaningfully above 2.10 mean tier. Fine-tuning has a higher ceiling than v1.1's distillation did, so the experiment is worth the compute.
**Fail**: Gemma 4 E2B untuned at or below 2.10. Architecture does not buy us anything pre-tune, hold the branch.

### Rung 3: validate `mlx-tune` actually trains on Apple Silicon

**Cost**: ~30 minutes to wire a smoke run.
**Setup**: tiny LoRA loop (10-20 iters) on the existing `data/dataset/training.jsonl` (1635 stall-filtered rows). Use `train_on_responses_only` masking per the doc's recommendation.
**Pass**: loss decreases, adapter weights save, no Apple-Silicon-specific crashes.
**Fail**: any blocker. Either roll back to a handwritten `mlx_vlm.lora` invocation, or kill the branch if even that path doesn't work.

### Rung 4: full T5-equivalent v2 training run

**Cost**: ~95 minutes (matched to v1's T5 budget) plus ~15 minutes of intermediate-checkpoint eval.
**Setup**: 1000-iter QLoRA on Gemma 4 E2B, same hyperparameters as v1.1's iter-750 winner unless rung 3 surfaces a reason to change them. Memory budget envelope is the same 16 GB; v1 peaked at 11.49 GB at iter 1000, Gemma 4 E2B active 2.3B should fit similarly but verify on a smoke loop first.
**Pass**: v2 beats v1.1 on the 20-state bench (mean tier strictly above 3.15, gap to teacher strictly below 0.27). Promote to `chayuto/gemma-4-e2b-it-solitaire-advisor-lora`.
**Fail**: v2 underperforms or matches v1.1. Keep v1.1 canonical, archive v2 results in `gemma4_finetune/adapters_t5_gemma4/` for reference, document negative result.

## Memory budget envelope

The v1 T5 run on the M5 16 GB peaked at 11.49 GB MLX RAM / 15.25 GB system. Gemma 4 E2B should fit, but it brings audio + vision conformer towers that the `gemma-3n-E2B-it-text-4bit` we shipped does not have. If those towers stay loaded even in text-mode inference, memory will be tighter. This is one of the unknowns that rung 1 should reveal: peak RSS during a single short generation tells us whether the full multimodal stack stays resident.

If the towers do stay resident and push us over 16 GB, two fallbacks: try the OptiQ mixed-precision quant (`mlx-community/gemma-4-e4b-it-OptiQ-4bit` exists, may have an E2B sibling) which compresses sensitive layers to 8-bit while aggressively quantising the rest, or accept that v2 needs to run on a borrowed larger Mac.

## What gets written when

- This plan: `docs/reports/20260525_gemma4_e2b_v2_exploration_plan.md` (committed to repo).
- Rung 1 result: appended to this plan as a section, plus a one-line update to the existing `gemma4-to-3n-pivot-mlxlm-blocker.md` memory if the load succeeds (the pivot stays valid as history but the blocker framing needs correction).
- Each subsequent rung: short addendum to this plan, no separate doc unless results warrant.
- Final v2 ship-or-hold decision: standalone session-close doc following the v1 template.

## Open questions (not blockers)

- Does `mlx-vlm`'s Gemma 4 implementation honour quantised weights for fine-tuning, or does it dequantise first and blow the memory envelope? Discoverable at rung 3.
- Is `mlx-tune`'s Unsloth-compatible API a thin wrapper over `mlx_vlm.lora` or a separate trainer? If the former, going directly to `mlx_vlm.lora` and skipping `mlx-tune` is fine.
- Does our prompt-template stability (the v1.1 model was trained against the pre-cutover legacy prompt schema) carry over to a v2 trained against the post-cutover hybrid-v1 schema? The 1635-row clean training set is a mix of both schemas, and a v2 trained on a post-cutover-only slice would be a more honest test of the post-cutover advisor behaviour.

## Rung 1 result (2026-05-25)

**Outcome**: pass.

| measurement | value |
|---|---|
| install of `mlx-vlm 0.5.0` into `gemma4_finetune/venv_vlm/` | clean, ~90 s |
| model registry contains `gemma4` | yes (alongside `gemma3`, `gemma3n`) |
| `mlx_vlm.load("mlx-community/gemma-4-E2B-it-4bit")` | succeeded in ~3 s after first-pull cache warmed |
| short generation ("what is 2+2?") | returned "Four" |
| generation throughput | 152 tok/s on M5 16 GB |
| peak MLX memory (single short generation) | 3.65 GB |
| peak process RSS | 1.19 GB |
| spurious warning | `audio_utils.py` mel-filter UserWarning at load time (Conformer init, not a fault) |

The audio Conformer initialises during load (per the mel-filter warning), confirming that the multimodal stack stays resident in memory even when only text generation is requested. The 3.65 GB peak on a 21-token prompt is the right reference point for scaling: a 2048-token training context should fit inside the 16 GB envelope but will be tighter than v1.1's `gemma-3n-E2B-it-text-4bit` (which has no audio tower).

**Implication for the memory `gemma4-to-3n-pivot-mlxlm-blocker.md`**: the original framing ("mlx-lm 0.31.3 can't load Gemma 4 E2B") was accurate but incomplete. The blocker was specifically `mlx-lm` lacking the `gemma4` model class, not Apple Silicon MLX lacking support for Gemma 4. `mlx-vlm 0.5.0` loads it cleanly. The pivot to Gemma 3n shipped v1.1 successfully and stays valid; v2 on Gemma 4 is now unblocked as a parallel experiment.

**Next**: rung 2 (untuned Gemma 4 E2B vs untuned Gemma 3n E2B on the 20-state bench) is unlocked but not yet scheduled. Decision point on whether to run it stays with the user.

## Rung 2 result (2026-05-25)

**Outcome**: pass with strong signal. Untuned Gemma 4 E2B beats untuned Gemma 3n E2B on every metric.

**Setup**:
- Runner: `gemma4_finetune/baseline_n20_gemma4_runner.py` (mirrors `baseline_n20_runner.py`, swapping mlx-lm for mlx-vlm)
- Scoring: `gemma4_finetune/score_n20_gemma4.py` (post-hoc, applies the shared `classify_pick` + `TIER_RANK` from `scripts/ab_test_prompt_formats.py`)
- Bench: same 20 C0 prompts from `experiments/a4_phase1.5_2026_05_24/prompts/C0/` used for the Gemma 3n baseline
- Temperature: 0.0 (deterministic), max_tokens 512, single run per state

**Side-by-side (untuned)**:

| metric | gemma-3n-E2B (v1 base) | gemma-4-E2B | delta |
|---|---:|---:|---:|
| JSON validity | 20/20 | 20/20 | tie |
| Illegal moves | 1/20 | 0/20 | -1 |
| Teacher agreement | 11/20 | 12/20 | +1 |
| Mean tier | 2.10 | **2.75** | **+0.65** |
| Gap vs teacher (3.42) | -1.32 | **-0.67** | **halved** |
| Foundation recovery (of 7) | 2 | 4 | +2 |
| Peak MLX memory | 6.26 GB | 5.30 GB | -1.0 GB |
| Mean call seconds | 13.9 s | 5.1 s | **2.7x faster** |

**Cross-comparison against shipped v1.1**:

| metric | shipped v1.1 (3n + iter-750 LoRA) | gemma-4-E2B untuned |
|---|---:|---:|
| Mean tier | 3.15 | 2.75 |
| Gap vs teacher | -0.27 | -0.67 |
| Foundation recovery | 6/7 | 4/7 |

v1.1 still beats Gemma 4 untuned (as expected, distillation works). The relevant question is how far a v2 distillation can lift Gemma 4 from its 2.75 starting point. Three reference points for the projection:

- If v2 gets the same +1.05 absolute lift v1 did (2.10 -> 3.15), v2 lands at **3.80**, *above* the teacher (3.42).
- If v2 gets the same fractional gap closure v1 did (80% of -1.32 = -0.27), v2 closes 80% of -0.67 = **-0.13**, landing at **3.29**.
- If v2 gets a more modest +0.40 lift, v2 lands at **3.15**, matching v1.1 but with the speed and memory advantage.

All three projections are favourable to v2 shipping. The conservative case matches v1.1 with a 2.7x speed-up; the optimistic case clears the teacher.

**Caveats**:
- N=20, single run per state, no seed variance. The Phase 1.5 study established that single-run deltas of around plus or minus 0.40 on this bench are within noise. The +0.65 delta is comfortably outside that band but is not statistically certified by a multi-seed run.
- Untuned baseline behaviour may not extrapolate to fine-tuned behaviour. v1's LoRA recipe was tuned for Gemma 3n's quirks (notably the doom-loop pattern); Gemma 4 may need a different rank or learning rate to hit the same lift.
- Memory headroom is real but tighter than v1: peak inference was 5.30 GB on a 21-to-2257 token prompt range. Training on 2048-token contexts will push significantly higher (v1 hit 11.49 GB peak in T5). The 16 GB envelope should hold but smoke-test before committing.

**Implications**:
- v2 is a clear bet. The starting point for Gemma 4 untuned is already where v1's iter-1000 checkpoint landed (mean tier 2.75 at iter 1000 vs 2.75 here). Distillation has room.
- The 2.7x throughput at ~1 GB less memory is a meaningful production win even before quality gains.
- Foundation recovery is the area where v1 worked hardest (2/7 untuned -> 6/7 trained). Gemma 4 starts at 4/7 untuned, giving v2 a higher ceiling on that specific failure mode.

**Next**: rung 3 (validate `mlx-tune` actually trains on Apple Silicon for Gemma 4) is unlocked. Decision to schedule it stays with the user.

## Text-only optimisation: investigated, deferred

The student is a Klondike Solitaire move advisor. It will never need audio or vision. The Gemma 4 E2B dense model ships with an audio Conformer tower (~12-layer, ~750-token-per-30s capacity per the published architecture) that initialises at load time even for text-only inference. Worth checking whether we can avoid carrying it.

**Findings**:

| variant | exists on HF | loads in mlx-lm 0.31.3 | loads in mlx-vlm 0.5.0 |
|---|---|---|---|
| `mlx-community/gemma-4-E2B-it-4bit` (multimodal) | yes | no, same alt-attention bug | **yes**, validated at rung 1 |
| `mlx-community/Gemma4-E2B-IT-Text-int4` (text-only) | yes (2026-05-19, apache-2.0) | no, same alt-attention bug | no, `gemma4_text` model_type is not registered |
| `mlx-community/gemma-3n-E2B-it-text-4bit-dwq` (v1 baseline) | yes | yes (the v1 path) | n/a |

A `gemma4_text.py` model class exists in mlx-lm 0.31.3 but inherits the same alternating-attention bug that blocks the multimodal `gemma4.py` class. PyPI's latest mlx-lm is 0.31.3; no newer version available. mlx-vlm 0.5.0 only knows the multimodal `gemma4` model class, not `gemma4_text`.

**Workarounds considered**:

1. **Force config.json model_type to `gemma4` and load the text-only weights via mlx-vlm**. Likely fails because the mlx-vlm gemma4 class expects audio_encoder + vision_encoder weights that the text-only repo doesn't ship. Would need a forked model class that tolerates missing multimodal weights. Effort: ~2-4 hours.
2. **Build mlx-lm from GitHub main**, hoping the alternating-attention bug has been fixed there but not yet released to PyPI. Effort: ~30 minutes to test. Risk: probably still broken; the bug is in the model class, not the runtime.
3. **Live with the multimodal model** via mlx-vlm. Already validated.

**The cost-benefit reframed**:

The intuition that "stripping audio + vision saves significant memory" turns out to overstate the gain. Rung 2 measured the multimodal Gemma 4 E2B 4-bit at 5.30 GB peak inference, **1 GB lower than v1's Gemma 3n text-only** (6.26 GB). The Conformer is small relative to the language model: maybe 200-500 MB of additional resident memory, comfortably absorbed by Gemma 4's lower base footprint. Disk download is ~1.7 GB vs an estimated ~1.5 GB for text-only — a 200 MB difference.

For training, the LoRA only updates the language layers we target (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). The frozen audio Conformer weights consume some forward-pass memory but no gradient memory. If the 2048-context training peak threatens the 16 GB envelope, that is the point to revisit stripping. Until then it is a premature optimisation.

**Decision (superseded below)**: ~~proceed with `mlx-community/gemma-4-E2B-it-4bit` via mlx-vlm for rungs 3 and 4. Capture text-only stripping as a post-ship optimisation only if...~~ See "Text-only unblocked" update below; the patched mlx-lm path is now the recommended baseline.

**Concrete cleanup that does land in the published v2** even without architectural stripping:

- Document explicitly in the v2 model card that the audio + vision capabilities of the base model are **not exercised** in our use case and the LoRA adapter does not modify them. Users who want a leaner artifact for production deployment can fuse the LoRA into a future text-only base variant when one becomes loadable.
- Set inference defaults that skip multimodal preprocessing paths at serve time (the mlx-vlm server already does this when no image/audio inputs are passed; just call it out in the README).
- Do not include the audio Conformer's preprocessing config in any custom inference wrapper we publish.

## Text-only unblocked: root-cause fix to mlx-lm's gemma4_text loader (2026-05-25)

After confirming `mlx-community/Gemma4-E2B-IT-Text-int4` exists upstream as a real text-only quant (apache-2.0, audio Conformer and vision encoder weights physically absent from the safetensors), I dug into *why* mlx-lm 0.31.3 still couldn't load it. The error from rung 0 was `Received 140 parameters not in model` on `model.layers.{15..34}.self_attn.{k_proj,v_proj}.{biases,scales,weight}`.

**Root cause**: in `mlx_lm/models/gemma4_text.py`, line 183:
```python
self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers
```
For Gemma 4 E2B (`num_hidden_layers=35`, `num_kv_shared_layers=20`), layers 15-34 are "KV-shared" and skip allocating `k_proj`/`v_proj`/`k_norm`/`v_norm`. At runtime they reuse KV projections from earlier layers via the `previous_kvs` map (lines 433-442). But the published quant *ships* those weight matrices for every layer anyway. The model's `sanitize()` method (line 610) doesn't strip them, so the loader sees 140 weights with no destination.

**Fix**: a 15-line `sanitize()` patch that drops `k_proj`/`v_proj`/`k_norm`/`v_norm` weights for any layer with `layer_idx >= num_hidden_layers - num_kv_shared_layers`:

```python
import re
import mlx_lm.models.gemma4_text as g4t
_orig = g4t.Model.sanitize
def patched(self, w):
    s = _orig(self, w)
    first = self.args.num_hidden_layers - self.args.num_kv_shared_layers
    pat = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)(\.|$)")
    return {k: v for k, v in s.items()
            if not (pat.match(k) and int(pat.match(k).group(1)) >= first)}
g4t.Model.sanitize = patched
```

**Result**: text-only Gemma 4 E2B loads cleanly via mlx-lm with the patch.

| metric | v1 baseline (3n text-only) | full multimodal (mlx-vlm) | patched text-only (mlx-lm) |
|---|---:|---:|---:|
| peak inference memory | 6.26 GB | 5.30 GB | **2.70 GB** |
| load time | 2.3 s | 3.5 s | **1.9 s** |
| short-prompt generation tps | (n/a, not measured) | 152 | (measured at next step) |
| training toolchain | `mlx_lm.lora` (v1) | `mlx_vlm.lora` (new) | `mlx_lm.lora` (v1) |
| audio Conformer in memory | no | yes (~200-500 MB) | no |
| dataset format / pipeline | v1 unchanged | needs new wrapper | **v1 unchanged** |

The patched path strictly dominates the multimodal path: half the memory, faster load, zero migration, audio/vision weights physically absent. It also dominates the v1 Gemma 3n baseline on memory while having the higher-capacity Gemma 4 architecture underneath.

**Implication for the freezing question**: with LoRA, audio/vision weights are *implicitly* frozen whenever they're not in `target_modules`. Explicit `finetune_audio_layers=False` flags are mostly a full-fine-tuning idiom; they save optimizer-state but not weight residency. Path A (stripping at load) saves the weight residency itself, which is what actually matters for the 16 GB envelope.

**Open follow-ups for the patch**:
1. Confirm the patch survives an actual `mlx_lm.lora` training invocation (sanitize() runs at load time, but the trainer may have its own weight-validation paths). To verify at rung 3.
2. Re-run the 20-state bench against the patched text-only base for clean v2 starting numbers. To verify at rung 2b (below).
3. Package the patch as a local override module (e.g. `gemma4_finetune/gemma4_text_patch.py`) that the runner imports before mlx_lm.load. Cleaner than inline monkey-patching.
4. Upstream the patch as a PR to https://github.com/ml-explore/mlx-lm. Low effort, fixes a real bug for everyone using the text-only Gemma 4 quants.

## Rung 2b result (2026-05-25): patched text-only Gemma 4 on the 20-state bench

**Outcome**: pass with the expected caveat. Patched text-only Gemma 4 E2B (apache-2.0) matches the multimodal variant on agreement and foundation recovery at slightly lower mean tier, while using just over half the memory.

**Setup**:
- Patch: `gemma4_finetune/gemma4_text_patch.py` (imported before mlx_lm.load, strips redundant KV-shared layer weights)
- Runner: `gemma4_finetune/baseline_n20_gemma4_text_runner.py` (`max_tokens=2048` because text-only's reasoning chains are longer than multimodal's at the same prompts; chat template is byte-identical between the two repos)
- Scoring: `gemma4_finetune/score_n20_gemma4_text.py`

**Three-way comparison (all untuned)**:

| metric | gemma-3n-E2B (v1 base) | gemma-4 multimodal | **gemma-4 text-only (patched)** |
|---|---:|---:|---:|
| JSON valid | 20/20 | 20/20 | 20/20 |
| illegal | 1/20 | 0/20 | 0/20 |
| agreement | 11/20 | 12/20 | 12/20 |
| mean tier | 2.10 | 2.75 | **2.55** |
| gap vs teacher (3.42) | -1.32 | -0.67 | -0.87 |
| foundation recovery (of 7) | 2 | 4 | 4 |
| peak memory | 6.26 GB | 5.30 GB | **3.35 GB** |
| mean call seconds | 13.9 | 5.1 | 15.7 |
| max_tokens cap | 512 | 512 | 2048 |
| training toolchain | `mlx_lm.lora` (v1) | `mlx_vlm.lora` | `mlx_lm.lora` (v1) |

**Observations**:

1. The text-only `int4` quant is 0.20 mean tier weaker than the multimodal `4bit` quant. Same prompts, same patched / unpatched logic-equivalent model class, byte-identical chat template. The delta is almost certainly the quantisation scheme (int4 likely uses a different group size or bit allocation than the 4bit multimodal quant). The Phase 1.5 study established plus or minus 0.40 single-run noise on this bench, so 0.20 is inside the noise band but not statistically zero.
2. Agreement (12/20) and foundation recovery (4/7) match the multimodal variant exactly. Where it differs from multimodal, it differs in *tier* (which move category was picked from the legal set), not in correctness.
3. Memory delta is real and material: 3.35 GB peak inference vs 5.30 GB multimodal vs 6.26 GB v1. Training at 2048-context will scale this up, but the 1.95 GB headroom over multimodal gives a meaningful cushion against the 16 GB envelope (v1's T5 peaked at 11.49 GB).
4. Inference time is 3x longer than multimodal at the same prompts because thinking-mode reasoning chains are 3-4x longer. This is a quirk of the model+quant combination, not the patch. v2 fine-tuning against our terse training data (1635 rows of short JSON responses) should compress generation length substantially.

**Final v2 base decision**:

| candidate | mean tier | memory | toolchain | license | decision |
|---|---:|---:|---|---|---|
| `mlx-community/gemma-4-E2B-it-4bit` (mm) | 2.75 | 5.30 GB | new (`mlx-vlm`) | Gemma TOS | rejected: heavier, requires migration |
| `mlx-community/Gemma4-E2B-IT-Text-int4` (text + patch) | **2.55** | **3.35 GB** | **v1 (`mlx-lm`)** | **apache-2.0** | **selected** |

The text-only patched path is chosen. The 0.20 mean tier dip is within noise, recoverable through distillation, and overwhelmingly outweighed by:
- 1.95 GB memory headroom for training
- Full v1 toolchain reuse (no migration cost, no learning curve, no second venv to maintain in production)
- Apache-2.0 vs Gemma TOS (cleaner publishing story; permits commercial reuse without Google's separate accept-license dance)
- Audio Conformer physically absent (no "useless model frozen" mental overhead, no warnings, no risk of unexpected memory growth from multimodal initialisation paths)
- 60% smaller download for end users of the published v2 fused model

**Action items locked in for rung 3+**:
- Use `mlx-community/Gemma4-E2B-IT-Text-int4` as the v2 base
- Import `gemma4_text_patch` before any `mlx_lm.load` call in the v2 pipeline
- Keep `gemma4_finetune/venv/` (the existing v1 venv with mlx-lm 0.31.3); the `venv_vlm` sibling can stay as exploration scratch or be removed
- File the upstream PR to mlx-lm with the sanitize() fix (6 lines of real code; high leverage for the whole community using Gemma 4 text-only quants)
- v2 model card must cite the apache-2.0 base and note that the LoRA was trained against the patched loader (so users who reproduce can apply the same patch if mlx-lm upstream hasn't merged the fix yet)

## Rung 3 result (2026-05-26): smoke training passes cleanly

30-iter LoRA against the patched text-only base. Val loss 3.16 -> 0.39, peak 8.36 GB, ~415 tok/s, adapter saves and reloads. The `gemma4_text_patch.py` monkey-patch survives the trainer's load path. No surprises.

## Rung 4 result (2026-05-26): full 1000-iter v2 training; HOLD recommendation

**Training itself ran cleanly**. 1000 iters in ~83 minutes, val loss 3.16 -> 0.36 (final), peak 8.41 GB, 4 checkpoints saved at iter 250 / 500 / 750 / 1000. Loss curve matches the shape of v1's successful T5 run. Memory and throughput budget well within envelope.

**But downstream quality regressed at every checkpoint**:

| config | json | illegal | agree | tier | gap | foundations | peak | mean sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 untuned | 20/20 | 0/20 | 12/20 | **2.55** | -0.87 | 4/7 | 3.35 GB | 15.7 |
| v2 iter 250 | 20/20 | 0/20 | 10/20 | 2.30 | -1.12 | 3/7 | 3.41 GB | 5.7 |
| v2 iter 500 | 20/20 | 2/20 | 10/20 | 2.45 | -0.97 | 4/7 | 3.41 GB | 5.4 |
| v2 iter 750 | 20/20 | 2/20 | 9/20 | 2.20 | -1.22 | 4/7 | 3.41 GB | 5.0 |
| v2 iter 1000 | 20/20 | 2/20 | 8/20 | 2.05 | -1.37 | 3/7 | 3.41 GB | 5.2 |
| (v1 untuned, 3n) | 20/20 | 1/20 | 11/20 | 2.10 | -1.32 | 2/7 | 6.26 GB | 13.9 |
| (v1.1 iter 750, 3n) | 20/20 | 2/20 | 11/20 | **3.15** | **-0.27** | **6/7** | 6.37 GB | 14.8 |

The trend goes the wrong direction across the full curve. v1.1 lifted Gemma 3n by +1.05 mean tier; v2 *lowers* Gemma 4 by 0.50.

**Per-state diff (v2 untuned vs v2 iter-1000)**:

- 12 states unchanged (same pick)
- 1 state improved (1 wrong answer became 1 different but correct answer)
- 2 states shifted (both wrong before and after, but to a different wrong answer)
- **5 states regressed** (was correct, became wrong): every single one is an `oscillation-*` state

**Diagnosed failure mode**: doom-loop confabulation, the exact anti-pattern documented in `[[flag-unsolvable-boards-early]]`. Worked example on `oscillation-21cc5243e1d8`:

- *Untuned response (correct)*: "no productive tableau moves available... optimal strategy is to draw from the stock... standard procedure when the board is static." Picks move_index=0 (draw). Matches teacher.
- *Iter-1000 response (wrong)*: invents a non-existent move ("Move [1] (6S from waste to Col 3)") to justify picking the King swap. Confidence 0.9. Picks move_index=2. Teacher pick was 0.

The model has lost the ability to recognise "no productive move available, draw is correct" and now confabulates productive-sounding reasoning to override that judgement. The harvester corpus's documented ~21% teacher-stall rate means doom-loop responses ARE in the training data, and the LoRA happily transferred the teacher's weakness into the student.

**Why v1.1 didn't have this problem**: Gemma 3n's untuned mean tier was 2.10 with 2/7 foundation recovery; it was *already weak* at oscillation recognition. The training corpus only had upside to offer. Gemma 4's untuned baseline was 2.55 with 4/7 foundation recovery and zero oscillation regressions; it had a *genuine strength* the teacher itself lacks, and the corpus contained signal that actively eroded that strength.

This validates the `flag-unsolvable-boards-early` memory more strongly than expected. The teacher genuinely is the bottleneck on a specific failure mode, and we now have a small local model that beats the teacher on it.

**Decision**: HOLD on shipping v2 LoRA. Three actionable paths forward:

1. **Ship v2 untuned as an apache-2.0 companion to v1.1**. No LoRA, just the base model with a README explaining the use case (oscillation-resistant, half the memory of v1, apache-2.0). Different tradeoffs than v1.1 on the same task. Distinct value proposition: when the user is on an 8 GB Mac, in commercial scope, or playing a game where doom-loops are common, prefer v2 untuned. When the user wants the highest foundation-recovery rate, stay on v1.1. Zero new compute required.
2. **Re-train with a stricter corpus filter**. The current stall-filter catches plateaus >= 25 turns but does not catch individual doom-loop *turns* before the plateau gets long enough. A per-turn shuffle-fraction gate, applied to drop the rationalising turns from the training set, may preserve Gemma 4's untuned oscillation-resistance while still teaching the foundation moves. Estimated cost: dataset-rebuild + retrain (~2 hours).
3. **Try a much lower learning rate** (5e-5 or 2e-5 vs current 2e-4), accepting that we are training a model that already nearly matches the teacher on average and may need a gentler touch. Quick to validate: re-run with `learning_rate: 5.0e-5` and check the val loss curve. The high training LR may simply be too aggressive for a base this close to the target.

Recommendation: do (1) first (zero compute, immediate apache-2.0 win for users), in parallel investigate (2) (the principled fix; addresses the actual root cause). Defer (3) unless (2) doesn't help.

**What this changes about the project**:

- v1.1 stays the canonical "highest foundation-recovery" model. Nothing changes about its HF repo.
- v2 untuned becomes the recommended "oscillation-resistant + cheap memory + apache-2.0" model. Publish at `chayuto/gemma-4-e2b-it-solitaire-advisor-untuned` (or fold into the existing v2 LoRA repo with a `(untuned)` directory and let the README pick which to recommend).
- The harvester team escalation note about doom-loop responses in the corpus moves from "would be nice" to "load-bearing for future v2 LoRA training." The current corpus actively trains models toward the failure mode we're trying to avoid.
- Memory `[[flag-unsolvable-boards-early]]` gets a confirmation update: a 2.3B-active student beats a 31B teacher on this specific failure mode.

## Pointer to related work

- v1 final report: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260525_session_close_v1_iter750_promoted.md`
- v1 untuned baseline report: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260524_gemma3n_e2b_untuned_n20_baseline.md`
- v1 first training run notes: `/Users/chayut/repos/solitaire-analytics/docs/reports/20260525_t5_first_distillation_run.md`
- Source research doc: `/Users/chayut/Downloads/MLX Gemma 4 Custom Training.md`
- Existing pipeline: `/Users/chayut/repos/solitaire-analytics/gemma4_finetune/`
- Published v1.1 model: `https://huggingface.co/chayuto/gemma-3n-e2b-it-solitaire-advisor-lora`
- Published dataset (driving both v1 and v2 training): `https://huggingface.co/datasets/chayuto/klondike-llm-decisions`
