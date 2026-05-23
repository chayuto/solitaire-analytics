# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python 3.12 analytics engine for Klondike Solitaire: core models, a rules engine, pluggable move-selection strategies, a parallel solver, analysis tools, and an MCP server for agentic AI play. Small codebase (~33 files, ~108 tests).

## Setup

Python 3.12 is required (homebrew: `/opt/homebrew/bin/python3.12`). The repo uses a local venv at `.venv/` — no system pytest is installed.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv      # one-time
source .venv/bin/activate                       # or use `.venv/bin/python` directly
pip install -r requirements.txt                 # torch is large; takes a few minutes
pip install -e .                                # REQUIRED — imports fail without the editable install
```

Re-run `pip install -e .` after editing `setup.py` or adding new modules.

## Testing

Always invoke pytest through the venv (`.venv/bin/python -m pytest …`) — the system `pytest` binary is not installed and `python3 -m pytest` from outside the venv will miss `torch` and friends.

```bash
.venv/bin/python -m pytest                                    # full suite (~3-4s) — config in pytest.ini
.venv/bin/python -m pytest -m unit                            # by marker: unit, integration, models, engine, solver, analysis, slow, gpu
.venv/bin/python -m pytest tests/test_engine.py::test_name    # single test
.venv/bin/python -m pytest --cov=solitaire_analytics --cov-report=term   # what CI runs
```

`pytest.ini` enables `--strict-markers` and coverage by default — new tests must use a declared marker. CI (`.github/workflows/ci.yml`) runs unit, then integration, then full coverage on Python 3.12. There is no linter config; follow PEP 8 manually.

## Architecture

The package is layered — each layer builds on the one below:

- **`models/`** — `Card` (rank/suit/face_up), `GameState` (7 tableau piles, 4 foundations, stock, waste, score), `Move`. `GameState` is mutable but provides `.copy()` for deep copies and `to_json`/`from_json` for serialization.
- **`engine/`** — pure rules layer. `generate_moves(state)` returns all legal moves; `validate_move(state, move)` checks legality; `apply_move(state, move)` returns a **new** `GameState` (immutable-style, never mutates input) or `None` if illegal. Everything above depends on this contract.
- **`strategies/`** — move selection. `Strategy` ABC + `StrategyConfig`. Use the registry factory `get_strategy(name, config)` rather than instantiating directly. Built-ins: `simple`, `weighted` (priority weights), `lookahead` (depth search), `llm` (OpenAI; needs `OPENAI_API_KEY`). `StrategyConfig.know_face_down_cards` controls perfect vs. imperfect information.
- **`solvers/`** — `ParallelSolver` runs beam search over the engine using joblib (`max_depth`, `beam_width`, `n_jobs`, `timeout`).
- **`analysis/`** — `MoveTreeBuilder` (state-space graph via networkx), `DeadEndDetector`, `move_analyzer` utilities (`compute_all_possible_moves`, `find_best_move_sequences`, `calculate_progression_score`).
- **`game/`** — interactive play layer. `deal_klondike` deals; `GameSession` wraps a game for turn-based play with `ObservationConfig` controlling the **information level** (`human()` = imperfect info, `perfect_information()`, `minimal()`). Sessions log seed + deck + every action.
- **`mcp_server.py`** — MCP server (`python -m solitaire_analytics.mcp_server`) exposing `GameSession` as agent tools, plus cross-game analytics (`server_analytics.py`). Set `SOLITAIRE_MCP_LOG_FILE` to persist an event stream as JSON Lines.
- **`play_logger.py`** — `PlayLogger` records initial state + timestamped moves for replay; disabled by default for zero overhead.

The public API is re-exported from `solitaire_analytics/__init__.py` — import top-level symbols from there.

Runnable examples for each subsystem live in `scripts/`; deeper docs in `docs/` (`mcp_server.md`, `STRATEGY_SYSTEM.md`, `play_logger.md`).
