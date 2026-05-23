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
- 1K<n<10K
configs:
- config_name: client_v1_full_corpus_raw
  data_files: client_v1_full_corpus_raw.jsonl
  default: true
- config_name: client_v1_teacher_clean_raw
  data_files: client_v1_teacher_clean_raw.jsonl
- config_name: client_v1_teacher_clean_lean
  data_files: client_v1_teacher_clean_lean.jsonl
---

# Klondike Solitaire LLM Advisor Decisions

Per-decision traces from large language models acting as advisors in Klondike Solitaire, collected to support **distillation research** and the study of **LLM failure modes in sequential decision tasks**. Every row records one advisor call against a reproducible game state.

## Configs at a glance

Three subsets under one dataset path. Pick the one that fits your use-case; researchers who want everything should use the default.

| Config | Rows | Schema | Best for |
|---|---:|---|---|
| `client_v1_full_corpus_raw` (default) | **2959** | full interaction (prompt + rawResponse + decision blob + call metadata) | failure-mode research, replay, end-to-end audit |
| `client_v1_teacher_clean_raw` | **1034** | full interaction | fine-tuning, honest training-quality subset (single teacher model, current schema, non-stalled) |
| `client_v1_teacher_clean_lean` | **1034** | derived per-decision (flat schema; see *Fields*) | quick analytics, lightweight loading, headline-statistics work |

```python
from datasets import load_dataset

# Default -- the full corpus, including failure modes
full = load_dataset("YOUR_ORG/klondike-llm-decisions")  # 2959 rows

# The training-friendly subset (filtered, single teacher)
clean_raw  = load_dataset("YOUR_ORG/klondike-llm-decisions", "client_v1_teacher_clean_raw")   # 1034 rows
clean_lean = load_dataset("YOUR_ORG/klondike-llm-decisions", "client_v1_teacher_clean_lean")  # 1034 rows, flat schema
```

### Filtering by model

Every row in `*_raw` configs carries a `model` field (e.g. `"gemma-4-31b-it"`, `"gemini-3.1-flash-lite"`). Use the standard HF `.filter()` to subset:

```python
ds = load_dataset("YOUR_ORG/klondike-llm-decisions")  # full corpus
teacher_only = ds["train"].filter(lambda r: r["model"] == "gemma-4-31b-it")
other_only   = ds["train"].filter(lambda r: r["model"] != "gemma-4-31b-it")
```

The `client_v1_teacher_clean_*` configs are already filtered to a single teacher model (currently `gemma-4-31b-it`); use them if you want a homogeneous training subset without writing a filter.

## Collection method (`client_v1_*`)

Collected via an external client-side harness (closed-source) running the Klondike app and capturing every teacher-advisor call. Each game seeds a reproducible deal. Rows are deduplicated by their UUIDv7 `id` across re-exports; nothing is discarded.

- **Collection window**: 2026-05-17 to 2026-05-23
- **Sessions**: 25 distinct game sessions
- **Models**: `gemma-4-31b-it` (2893), `gemini-3.1-flash-lite` (66)
- **Schema tiers**: current (2588), legacy (371)

### Planned: `server_v1_*` configs

A second collection method is being prepared using the open-source MCP server in this project's parent codebase (`solitaire_analytics.mcp_server`). Server-collected rows will ship as `server_v1_*` configs under this same dataset path. They will carry agent identity (`agent_id`, `model`, `provider`, `app_commit`) stamped per decision, plus an `infoLevel` block per session so perfect-vs-imperfect-information runs are unambiguous. Not yet published.

## Fields

### `*_raw` configs

Verbatim interaction records as captured by the harness.

- `id` — globally unique UUIDv7 for the interaction
- `sessionId`, `turnIndex` — game session and move number (current-schema rows)
- `model`, `provider` — the advisor model
- `prompt` — full prompt: Klondike rules + board state JSON + legal-move list
- `rawResponse` — the model's raw text reply
- `decision` — parsed: `moveIndex`, `confidence`, `alternativeMoveIndex`, `boardAnalysis`, `reasoning`
- `outcome`, token counts, timing — call metadata

### `*_lean` config

Derived per-decision rows, flattened. Built by joining each successful interaction against its parsed prompt + decision.

- `id`, `sessionId`, `turnIndex`, `timestamp`, `model`, `provider`, `appCommit`
- `chosenMoveType`, `chosenMoveDescribe`, `moveIndex`, `nLegalMoves`
- `confidence`, `alternativeMoveIndex`
- `completionProgress`, `moveCount`, `perceivedDifficulty` — from the prompt metrics block
- `foundationCards`, `faceDownTotal`, `progressScore`, `turnsSinceProgress` — computed by the ingest from board state
- `boardAnalysis`, `reasoning`, `thinkingText` — agent's natural-language fields

## Chosen-move distribution (full corpus)

| Move type | Count | Share |
|---|---:|---:|
| `draw_card` | 1952 | 66% |
| `tableau_to_tableau` | 560 | 19% |
| `tableau_to_foundation` | 131 | 4% |
| `discard_to_tableau` | 119 | 4% |
| `recycle_stock` | 118 | 4% |
| `discard_to_foundation` | 79 | 3% |

## Failure modes — a feature of `*_full_corpus_raw`, not a bug

The full corpus deliberately includes sessions where the teacher fails to make progress. These are research signal, not noise. The cleaned configs (`*_teacher_clean_*`) filter them out per a stall heuristic; the full corpus keeps them so you can study the failure modes directly.

Two documented pathologies recur in the corpus:

1. **Doom-loop / oscillation.** The teacher rationalises a two-card shuffle (e.g. moving `5C`/`4D` back and forth between two columns) as a 'setup move' even when `recentMoves` clearly shows the exact reversal was just played. Confidence stays saturated at 0.9+ throughout. Example: session with the longest plateau in the corpus carries 75 consecutive turns of foundation/face-down unchanged.

2. **Honest hunt that degrades.** Some sessions begin with draw-dominated card hunting (correct behaviour when needed cards are still hidden) and only descend into oscillation after extended no-progress windows. A plateau-only stall detector would over-fire on these; a *shuffle-fraction* gate is needed alongside the plateau gate to discriminate honest hunt from doom-loop.

Use the `progressScore` / `turnsSinceProgress` columns in the `*_lean` config to locate stalled stretches; use the `chosenMoveType` distribution within those stretches to classify them.

## Known limitations

- **Confidence is miscalibrated.** Reported `confidence` spans 0.60–1.00 (mean 0.91); the teacher signals near-certainty regardless of board state. Do not treat it as a calibrated probability; in our experience using it as a training-time signal teaches student models to be overconfident.
- **Mixed schema versions** in `client_v1_full_corpus_raw`. Older rows lack `sessionId` / `turnIndex` / `appCommit`. Filter on field presence if you need a homogeneous subset, or use the `client_v1_teacher_clean_*` configs which exclude legacy schema rows.
- **Outcome skew.** Most logged games were lost or stalled; winning play is under-represented. End-game (foundation_cards > ~10) is particularly sparse — student models trained on this corpus will lack guidance for late-game transitions.
- **Mixed information modes.** A few early sessions had perfect-information game state exposed to the advisor; most run under imperfect information. The `client_v1_teacher_clean_*` configs select a single information mode.
- **Move-type skew toward `draw_card`.** Draws are ~50–66% of eligible rows in the cleaned configs, reflecting the teacher's tendency to keep drawing when no productive tableau move is obvious. Apply your own re-weighting if this matters for your task.

## License

Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Free to use, share, and adapt with attribution.

_Card and data generated by `scripts/ingest_exports.py`._
