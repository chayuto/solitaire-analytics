---
license: cc-by-4.0
pretty_name: Klondike Solitaire LLM Advisor Decisions
language:
- en
task_categories:
- text-generation
tags:
- solitaire
- klondike
- game-playing
- llm-decisions
- reasoning
- distillation
- failure-modes
size_categories:
- 10K<n<100K
configs:
- config_name: client_v1_full_corpus_raw
  data_files: client_v1_full_corpus_raw.jsonl
  default: true
- config_name: client_v1_teacher_clean_raw
  data_files: client_v1_teacher_clean_raw.jsonl
- config_name: client_v1_teacher_clean_lean
  data_files: client_v1_teacher_clean_lean.jsonl
- config_name: client_v1_26b_raw
  data_files: client_v1_26b_raw.jsonl
- config_name: client_v1_26b_lean
  data_files: client_v1_26b_lean.jsonl
---

# Klondike Solitaire LLM Advisor Decisions

Per-decision traces from large language models acting as advisors in Klondike Solitaire, collected to support **distillation research** and the study of **LLM failure modes in sequential decision tasks**. Every row records one advisor call against a reproducible game state.

## Configs at a glance

Several subsets under one dataset path. Pick the one that fits your use-case; researchers who want everything should use the default.

| Config | Rows | Schema | Best for |
|---|---:|---|---|
| `client_v1_full_corpus_raw` (default) | **15213** | full interaction (prompt + rawResponse + decision blob + call metadata) | failure-mode research, replay, end-to-end audit |
| `client_v1_teacher_clean_raw` | **6636** | full interaction | fine-tuning, honest training-quality subset (single teacher model, current schema, non-stalled) |
| `client_v1_teacher_clean_lean` | **6636** | derived per-decision (flat schema; see *Fields*) | quick analytics, lightweight loading, headline-statistics work |
| `client_v1_26b_raw` | **1243** | full interaction | comparison: the `gemma-4-26b-a4b-it` MoE cohort alone (current schema, stalled decisions kept, 1 winning session) |
| `client_v1_26b_lean` | **1243** | derived per-decision (flat schema) | comparison analytics: the MoE on matched game states |

```python
from datasets import load_dataset

# Default -- the full corpus, including failure modes
full = load_dataset("chayuto/klondike-llm-decisions")  # 15213 rows

# The training-friendly subset (filtered, single teacher)
clean_raw  = load_dataset("chayuto/klondike-llm-decisions", "client_v1_teacher_clean_raw")   # 6636 rows
clean_lean = load_dataset("chayuto/klondike-llm-decisions", "client_v1_teacher_clean_lean")  # 6636 rows, flat schema

# The 26B MoE cohort on its own, for comparison
moe_26b    = load_dataset("chayuto/klondike-llm-decisions", "client_v1_26b_raw")  # 1243 rows
```

The `client_v1_26b_*` configs are a **behavioural-comparison cohort**: all 1243 current-schema decisions from the `gemma-4-26b-a4b-it` MoE, with stalled/loop decisions deliberately kept (754 of 1243 sit inside a stall, where the `teacher_clean` configs drop such rows). The corpus now holds 1 winning session for the 26B MoE; the cohort still centres on failure-mode contrast (stalled/loop decisions kept), so treat it as a behavioural contrast set more than as additional training data.

### Filtering by model

Every row in `*_raw` configs carries a `model` field (e.g. `"gemma-4-31b-it"`, `"gemini-3.1-flash-lite"`). Use the standard HF `.filter()` to subset:

```python
ds = load_dataset("chayuto/klondike-llm-decisions")  # full corpus
teacher_only = ds["train"].filter(lambda r: r["model"] == "gemma-4-31b-it")
other_only   = ds["train"].filter(lambda r: r["model"] != "gemma-4-31b-it")
```

The `client_v1_teacher_clean_*` configs are already filtered to a single teacher model (currently `gemma-4-31b-it`); use them if you want a homogeneous training subset without writing a filter. The `gemma-4-26b-a4b-it` MoE subset is likewise available directly as the `client_v1_26b_*` configs, no filter needed.

## Collection method (`client_v1_*`)

Collected via an external client-side harness (closed-source) running the Klondike app and capturing every teacher-advisor call. Each game seeds a reproducible deal. Rows are deduplicated by their UUIDv7 `id` across re-exports; nothing is discarded.

- **Collection window**: 2026-05-17 to 2026-06-05
- **Sessions**: 83 distinct game sessions
- **Models**: `gemma-4-31b-it` (13904), `gemma-4-26b-a4b-it` (1243), `gemini-3.1-flash-lite` (66)
- **Schema tiers**: current (14842), legacy (371)

### Planned: `server_v1_*` configs

A second collection method is being prepared using the open-source MCP server in this project's parent codebase (`solitaire_analytics.mcp_server`). Server-collected rows will ship as `server_v1_*` configs under this same dataset path. They will carry agent identity (`agent_id`, `model`, `provider`, `app_commit`) stamped per decision, plus an `infoLevel` block per session so perfect-vs-imperfect-information runs are unambiguous. Not yet published.

## Fields

### `*_raw` configs

Verbatim interaction records as captured by the harness.

- `id`: globally unique UUIDv7 for the interaction
- `sessionId`, `turnIndex`: game session and move number (current-schema rows)
- `model`, `provider`: the advisor model
- `prompt`: full prompt: Klondike rules + board state JSON + legal-move list
- `rawResponse`: the model's raw text reply
- `decision`: parsed `moveIndex`, `confidence`, `alternativeMoveIndex`, `boardAnalysis`, `reasoning`
- `outcome`, token counts, timing: call metadata

### `*_lean` config

Derived per-decision rows, flattened. Built by joining each successful interaction against its parsed prompt + decision.

- `id`, `sessionId`, `turnIndex`, `timestamp`, `model`, `provider`, `appCommit`
- `chosenMoveType`, `chosenMoveDescribe`, `moveIndex`, `nLegalMoves`
- `confidence`, `alternativeMoveIndex`
- `completionProgress`, `moveCount`, `perceivedDifficulty`: from the prompt metrics block
- `foundationCards`, `faceDownTotal`, `progressScore`, `turnsSinceProgress`: computed by the ingest from board state
- `boardAnalysis`, `reasoning`, `thinkingText`: agent's natural-language fields

## Chosen-move distribution (full corpus)

| Move type | Count | Share |
|---|---:|---:|
| `draw_card` | 8831 | 58% |
| `tableau_to_tableau` | 3243 | 21% |
| `tableau_to_foundation` | 1164 | 8% |
| `recycle_stock` | 708 | 5% |
| `discard_to_tableau` | 695 | 5% |
| `discard_to_foundation` | 572 | 4% |

## Failure modes are a feature of `*_full_corpus_raw`, not a bug

The full corpus deliberately includes sessions where the teacher fails to make progress. These are research signal, not noise. The cleaned configs (`*_teacher_clean_*`) filter them out per a stall heuristic; the full corpus keeps them so you can study the failure modes directly.

Two documented pathologies recur in the corpus:

1. **Doom-loop / oscillation.** The teacher rationalises a two-card shuffle (e.g. moving `5C`/`4D` back and forth between two columns) as a 'setup move' even when `recentMoves` clearly shows the exact reversal was just played. Confidence stays saturated at 0.9+ throughout. Example: session with the longest plateau in the corpus carries 75 consecutive turns of foundation/face-down unchanged.

2. **Honest hunt that degrades.** Some sessions begin with draw-dominated card hunting (correct behaviour when needed cards are still hidden) and only descend into oscillation after extended no-progress windows. A plateau-only stall detector would over-fire on these; a *shuffle-fraction* gate is needed alongside the plateau gate to discriminate honest hunt from doom-loop.

Use the `progressScore` / `turnsSinceProgress` columns in the `*_lean` config to locate stalled stretches; use the `chosenMoveType` distribution within those stretches to classify them.

## Known limitations

- **Confidence is miscalibrated.** Reported `confidence` spans 0.50 to 1.00 (mean 0.89); the teacher signals near-certainty regardless of board state. Do not treat it as a calibrated probability; in our experience using it as a training-time signal teaches student models to be overconfident.
- **Mixed schema versions** in `client_v1_full_corpus_raw`. Older rows lack `sessionId` / `turnIndex` / `appCommit`. Filter on field presence if you need a homogeneous subset, or use the `client_v1_teacher_clean_*` configs which exclude legacy schema rows.
- **Outcome skew.** Most logged games were lost or stalled; winning play is under-represented. End-game (foundation_cards > ~10) is particularly sparse. Student models trained on this corpus will lack guidance for late-game transitions.
- **Mixed information modes.** A few early sessions had perfect-information game state exposed to the advisor; most run under imperfect information. The `client_v1_teacher_clean_*` configs select a single information mode.
- **Move-type skew toward `draw_card`.** Draws are ~50 to 66% of eligible rows in the cleaned configs, reflecting the teacher's tendency to keep drawing when no productive tableau move is obvious. Apply your own re-weighting if this matters for your task.

## Build on top of this

This corpus is intentionally public so others can study or build on Gemma 4 31B's Klondike behaviour without reproducing the harvest infrastructure from scratch. The license is permissive (CC-BY-4.0); attribution is the only ask. Some specific ways the data is set up to be useful:

### Replay any seed in your browser

Every row carries a `sessionId` (and most carry a `seed` derivable from the source repo's `data/index/manifest.jsonl`). The harvester web UI at `https://solitaire.chayuto.com/?seed=<seed>` deals the same board deterministically: you can load any seed from this corpus and play or feed it to your own model, then compare your model's decisions against the rows here turn for turn.

### Run your own kill-or-continue analysis

The source repo at [`chayuto/solitaire-analytics`](https://github.com/chayuto/solitaire-analytics) publishes the tooling used to produce this corpus, including:

- `scripts/ingest_exports.py`: the dedup + stall-filter pipeline that produced these configs from raw exports.
- `.claude/skills/solitaire-analyst/`: a Claude Code skill that reads any raw export and produces a kill-or-continue verdict with failure-mode classification. Includes a Monte Carlo solvability check via `pyksolve` (DFS with dominance pruning, ~10 ms per sample) at `.claude/skills/solitaire-analyst/scripts/check_winnability.py`.
- `data/DATASET_NOTES.md`: the long-form taxonomy of every documented session in the corpus. Each entry calls out the failure class (behavioural-doom-loop, dead-deal-flailing, honest-hunt, self-rescue-fails) with the specific evidence that drove the call. Useful if you want to know which sessions are which kind of failure before pulling them.

### Compare a model on the same boards

A 20-state Klondike-state benchmark used by this project's distillation evaluations lives in the source repo under `experiments/a4_phase1.5_2026_05_24/prompts/C0/`. Five early-game, eight midgame, seven oscillation-prone states; each state's reference answer is the teacher model's pick scored on a six-level tier (`foundation` > `reveal` > `waste_play` > `shuffle` > `draw` > `illegal`). If you want to bench your own Klondike-playing model on the same positions and compare apples-to-apples against `gemma-4-31b-it`, this is the fastest way.

### Cite if you publish

If this corpus shows up in a paper, blog post, or model card, please cite the dataset URL (`https://huggingface.co/datasets/chayuto/klondike-llm-decisions`) and link to the source repo (`https://github.com/chayuto/solitaire-analytics`) so readers can find the analysis context. The corpus continues to grow; pin a specific revision (`load_dataset(..., revision=...)`) if your work depends on a fixed snapshot.

### Talk to us

Issues, comparisons, replay videos, alternative analyses are all welcome. Open an issue on the source repo or comment on the dataset discussion tab.

## License

Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Free to use, share, and adapt with attribution.

_Card and data generated by `scripts/ingest_exports.py`._
