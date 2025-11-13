# Copilot Instructions for Solitaire Analytics Engine

## Repository Overview
Python 3.12 analytics engine for Solitaire games. Small codebase (~2.5MB, 33 Python files, ~1000 LOC, 108 tests, 81% coverage). Core components: models, game engine, strategies, solvers, analysis tools. Tech: PyTorch (CPU), pytest, joblib, networkx.

## Installation & Setup (CRITICAL - Follow Order)
```bash
pip install -r requirements.txt  # Takes 2-3 minutes (torch is large)
pip install -e .                 # REQUIRED - package won't import without this
```
**ALWAYS run both commands.** Reinstall with `pip install -e .` if you modify setup.py or add modules.

## Testing (Fast: ~3-4 seconds)
```bash
# CI default - run this before committing
pytest --cov=solitaire_analytics --cov-report=term

# By category (use markers: unit, integration, models, engine, solver, analysis)
pytest -m unit --tb=short           # 87 unit tests
pytest -m integration --tb=short    # 2 integration tests
```
**Target:** 70%+ coverage. Tests MUST pass before PR. New features MUST include tests.

## Project Structure
```
solitaire_analytics/
├── models/          # Card, GameState, Move (data models)
├── engine/          # move_generator.py (generate_moves), move_validator.py (validate_move)
├── strategies/      # base.py, registry.py, simple/weighted/lookahead/llm strategies
├── solvers/         # parallel_solver.py (ParallelSolver with CPU+GPU)
├── analysis/        # move_tree_builder.py, dead_end_detector.py, move_analyzer.py
└── play_logger.py   # Game recording
tests/               # test_models.py, test_engine.py, test_solver.py, test_strategies.py, test_analysis.py, test_play_logger.py
scripts/             # example_analysis.py, example_strategies.py, example_play_logger.py
.github/workflows/   # ci.yml (runs on main/develop, Python 3.12, all test suites)
```

## Key Files
- **pytest.ini**: Test configuration, markers, coverage settings
- **requirements.txt**: All dependencies (torch, pytest, networkx, pandas, jupyter, etc.)
- **setup.py**: Package metadata (Python >=3.12 required)
- **README.md**: Full documentation, API examples, quick start guide
- **CONTRIBUTING.md**: Dev workflow, test markers, code style (PEP 8)

## CI/CD (GitHub Actions)
Runs on push/PR to main/develop branches:
1. Python 3.12 setup
2. Install deps: `pip install -r requirements.txt && pip install -e .`
3. Unit tests: `pytest -m unit --tb=short`
4. Integration tests: `pytest -m integration --tb=short`
5. Full coverage: `pytest --cov=solitaire_analytics --cov-report=xml`
6. Upload to Codecov

**Total CI time:** 1-2 minutes. **All must pass.**

## Architecture Quick Reference
**Models:** Card (rank/suit/face_up), GameState (tableau/foundation/stock/waste), Move (from/to/card/type)
**Engine:** generate_moves() returns all legal moves, validate_move() checks legality, apply_move() returns new state (immutable)
**Strategies:** Registry-based factory `get_strategy(name, config)`. Built-in: simple, weighted, lookahead, llm
**Solvers:** ParallelSolver with beam search (max_depth, beam_width, n_jobs, timeout)
**Analysis:** MoveTreeBuilder (state space), DeadEndDetector (risk), move_analyzer (utilities)

## Common Operations
```bash
# Run examples
python scripts/example_analysis.py
python scripts/example_strategies.py

# Import pattern
from solitaire_analytics import Card, GameState, Move, generate_moves, ParallelSolver
from solitaire_analytics.strategies import get_strategy
```

## Critical Notes
1. **No linting config** - Follow PEP 8 manually (no .flake8, .pylintrc, .black, pyproject.toml)
2. **Python 3.12 required** - Will not work on earlier versions
3. **Torch is CPU-only** - No CUDA/GPU dependencies
4. **Package installation is mandatory** - Cannot import without `pip install -e .`
5. **Tests are fast** - Run frequently (~3-4 seconds for full suite)
6. **Ignore pytest timeout warning** - Known issue, tests still work

## Troubleshooting
**Import fails:** `pip install -e .`
**Tests fail:** `pip install -r requirements.txt && pip install -e .`
**Coverage low:** Check test markers, ensure new code has tests

## Pre-PR Checklist
- [ ] `pytest` passes (all 108 tests)
- [ ] `pytest -m unit` passes
- [ ] `pytest -m integration` passes
- [ ] Coverage ≥70%
- [ ] PEP 8 compliance
- [ ] New features have tests
- [ ] Example scripts work (if core changed)

**Trust these instructions. Search only if you encounter undocumented errors or need specific implementation details.**
