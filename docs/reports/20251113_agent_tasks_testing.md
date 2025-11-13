# Coding Agent Tasks - Testing Improvements

**Date:** November 13, 2025  
**Category:** Testing  
**Difficulty Range:** Easy to Medium

## Overview

This document contains self-contained testing tasks suitable for coding agents. Each task is independent, well-defined, and includes acceptance criteria.

---

## Task 1: Add Property-Based Tests for Card Class

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `tests/test_models.py`

### Description
Add property-based tests using Hypothesis to verify Card class invariants.

### Context
The Card class needs comprehensive testing to ensure it handles all edge cases correctly. Property-based testing can automatically find edge cases we might miss.

### Requirements
1. Install hypothesis: `pip install hypothesis`
2. Add property-based tests for:
   - Card creation with valid/invalid ranks (1-13)
   - Card creation with all suit types
   - Card comparison operations
   - Card string representation

### Acceptance Criteria
- [ ] Hypothesis is added to requirements-dev.txt
- [ ] At least 4 property-based tests added
- [ ] Tests pass with default 100 examples
- [ ] Tests cover rank validation (1-13 valid, outside invalid)
- [ ] Tests cover all 4 suits
- [ ] Tests verify card equality and comparison

### Example Test
```python
from hypothesis import given, strategies as st
from solitaire_analytics.models import Card
from solitaire_analytics.models.card import Suit

@given(st.integers(min_value=1, max_value=13), st.sampled_from(list(Suit)))
def test_card_creation_valid_range(rank, suit):
    """Valid cards can always be created."""
    card = Card(rank=rank, suit=suit)
    assert card.rank == rank
    assert card.suit == suit

@given(st.integers().filter(lambda x: x < 1 or x > 13), st.sampled_from(list(Suit)))
def test_card_creation_invalid_rank(rank, suit):
    """Invalid ranks raise ValueError."""
    with pytest.raises(ValueError):
        Card(rank=rank, suit=suit)
```

### Related Files
- `solitaire_analytics/models/card.py`
- `tests/test_models.py`

### Testing
```bash
pytest tests/test_models.py::test_card_creation_valid_range -v
pytest tests/test_models.py -k property
```

---

## Task 2: Create Test Fixtures for Common Game States

**Difficulty:** Easy  
**Estimated Time:** 3-4 hours  
**Module:** `tests/conftest.py`

### Description
Create reusable pytest fixtures for common game states used across multiple tests.

### Context
Many tests need similar game state setups. Centralizing these as fixtures improves maintainability and reduces duplication.

### Requirements
Create fixtures for:
1. Empty game state (initial state)
2. Early game state (few moves made)
3. Mid-game state (half-way through)
4. Near-winning state (almost complete)
5. Dead-end state (no valid moves)
6. Solvable game state (known to be solvable)

### Acceptance Criteria
- [ ] `tests/conftest.py` created with fixtures
- [ ] At least 6 fixtures defined
- [ ] Each fixture has clear docstring
- [ ] Fixtures are used in existing tests
- [ ] All tests pass

### Example Implementation
```python
# tests/conftest.py

import pytest
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit

@pytest.fixture
def empty_state():
    """Empty game state with no cards placed."""
    return GameState()

@pytest.fixture
def early_game_state():
    """
    Early game state with a few cards in tableau.
    
    Setup:
    - 3 cards in first tableau pile
    - 2 cards in second tableau pile
    - 24 cards in stock
    """
    state = GameState()
    state.tableau[0] = [
        Card(rank=13, suit=Suit.HEARTS),
        Card(rank=12, suit=Suit.SPADES),
        Card(rank=11, suit=Suit.HEARTS)
    ]
    state.tableau[1] = [
        Card(rank=10, suit=Suit.DIAMONDS),
        Card(rank=9, suit=Suit.CLUBS)
    ]
    # Add remaining cards to stock
    for rank in range(1, 9):
        state.stock.append(Card(rank=rank, suit=Suit.HEARTS))
    return state

@pytest.fixture
def dead_end_state():
    """Game state with no valid moves (dead end)."""
    state = GameState()
    # ... create state with no valid moves ...
    return state
```

### Related Files
- `tests/test_models.py`
- `tests/test_engine.py`
- `tests/test_solver.py`

### Testing
```bash
pytest tests/ --fixtures  # List all fixtures
pytest tests/ -v
```

---

## Task 3: Add Test Coverage for Edge Cases in Move Generator

**Difficulty:** Medium  
**Estimated Time:** 4-5 hours  
**Module:** `tests/test_engine.py`

### Description
Add comprehensive tests for edge cases in the move generation logic.

### Context
The move generator is critical but may not cover all edge cases. We need tests for unusual game states.

### Edge Cases to Test
1. Empty tableau piles
2. All cards in foundations
3. Empty stock pile
4. Empty waste pile
5. Only Kings available to move
6. Multiple valid moves to same destination
7. Tableau piles with only face-down cards
8. Circular move possibilities

### Acceptance Criteria
- [ ] At least 8 new test cases added
- [ ] Each edge case has dedicated test
- [ ] Tests verify expected move count
- [ ] Tests verify move types are correct
- [ ] All tests pass
- [ ] Test coverage for move_generator.py increases

### Example Tests
```python
def test_generate_moves_empty_tableau_pile(empty_state):
    """Test move generation with empty tableau pile."""
    # Setup state with empty pile
    empty_state.tableau[0] = []
    empty_state.tableau[1] = [Card(rank=13, suit=Suit.HEARTS)]
    
    moves = generate_moves(empty_state)
    
    # Should have move to place King on empty pile
    king_moves = [m for m in moves if m.destination_pile_type == 'tableau' 
                  and m.destination_pile_index == 0]
    assert len(king_moves) > 0

def test_generate_moves_only_face_down_cards():
    """Test move generation when only face-down cards remain."""
    state = GameState()
    # Setup tableau with only face-down cards
    for pile in state.tableau:
        pile.append(Card(rank=1, suit=Suit.HEARTS, face_down=True))
    
    moves = generate_moves(state)
    
    # Only stock-to-waste move should be available
    assert len(moves) == 1
    assert moves[0].move_type == 'stock_to_waste'
```

### Related Files
- `solitaire_analytics/engine/move_generator.py`
- `tests/test_engine.py`

### Testing
```bash
pytest tests/test_engine.py -v
pytest tests/test_engine.py --cov=solitaire_analytics/engine
```

---

## Task 4: Add Integration Tests for Solver

**Difficulty:** Medium  
**Estimated Time:** 4-6 hours  
**Module:** `tests/test_solver.py`

### Description
Add end-to-end integration tests for the parallel solver with various game configurations.

### Context
Current tests may focus on unit testing. We need integration tests that verify the solver works correctly in realistic scenarios.

### Test Scenarios
1. Solvable game (should find solution)
2. Unsolvable game (should terminate without solution)
3. Timeout handling (should respect timeout)
4. Memory constraints (should not exceed limits)
5. Multi-job execution (verify parallel processing)
6. GPU vs CPU consistency (same results)

### Acceptance Criteria
- [ ] At least 6 integration tests added
- [ ] Tests marked with `@pytest.mark.integration`
- [ ] Tests verify solver results
- [ ] Tests check performance characteristics
- [ ] Timeout tests actually test timeout
- [ ] All tests pass

### Example Tests
```python
@pytest.mark.integration
def test_solver_finds_solution_solvable_game(solvable_game_state):
    """Test solver finds solution for known solvable game."""
    solver = ParallelSolver(max_depth=20, n_jobs=2, timeout=10.0)
    
    result = solver.solve(solvable_game_state)
    
    assert result.success, "Solver should find solution"
    assert len(result.moves) > 0, "Solution should have moves"
    assert result.time_elapsed < 10.0, "Should complete within timeout"
    assert result.states_explored > 0, "Should explore states"

@pytest.mark.integration
def test_solver_respects_timeout():
    """Test solver stops when timeout is reached."""
    # Create complex game state
    complex_state = create_complex_game_state()
    
    solver = ParallelSolver(max_depth=50, timeout=1.0)
    
    start_time = time.time()
    result = solver.solve(complex_state)
    elapsed = time.time() - start_time
    
    assert elapsed < 2.0, "Should respect timeout (with margin)"
    assert result.time_elapsed >= 1.0, "Should run for at least timeout duration"

@pytest.mark.integration
def test_solver_consistency_cpu_vs_gpu():
    """Test that CPU and GPU solvers give consistent results."""
    state = create_test_game_state()
    
    # CPU solver
    cpu_solver = ParallelSolver(max_depth=10, use_gpu=False)
    cpu_result = cpu_solver.solve(state)
    
    # GPU solver (if available)
    if torch.cuda.is_available():
        gpu_solver = ParallelSolver(max_depth=10, use_gpu=True)
        gpu_result = gpu_solver.solve(state)
        
        assert cpu_result.success == gpu_result.success
        # Results may differ slightly but should be comparable
```

### Related Files
- `solitaire_analytics/solvers/parallel_solver.py`
- `tests/test_solver.py`

### Testing
```bash
pytest tests/test_solver.py -m integration -v
pytest tests/test_solver.py -m integration --durations=10
```

---

## Task 5: Add Parametrized Tests for Move Validation

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `tests/test_engine.py`

### Description
Use pytest parametrization to test move validation with many input combinations.

### Context
Move validation has many rules. Parametrized tests can efficiently verify all combinations.

### Requirements
Create parametrized tests for:
1. Tableau-to-foundation moves (valid/invalid)
2. Tableau-to-tableau moves (valid/invalid)
3. Waste-to-foundation moves (valid/invalid)
4. Waste-to-tableau moves (valid/invalid)
5. Stock-to-waste moves

### Acceptance Criteria
- [ ] At least 3 parametrized test functions
- [ ] Tests cover all move types
- [ ] Each test has minimum 5 parameter sets
- [ ] Valid and invalid moves both tested
- [ ] Clear test names describe the scenario
- [ ] All tests pass

### Example Implementation
```python
@pytest.mark.parametrize("card_rank,card_suit,foundation_rank,should_be_valid", [
    (1, Suit.HEARTS, 0, True),   # Ace on empty foundation
    (2, Suit.HEARTS, 1, True),   # 2 on Ace
    (3, Suit.HEARTS, 1, False),  # 3 on Ace (skip 2)
    (2, Suit.SPADES, 1, False),  # Wrong suit
    (1, Suit.DIAMONDS, 1, False), # Ace on non-empty foundation
])
def test_validate_tableau_to_foundation(card_rank, card_suit, foundation_rank, should_be_valid):
    """Test validation of tableau-to-foundation moves."""
    state = GameState()
    
    # Setup state
    if foundation_rank > 0:
        state.foundations[0] = [Card(rank=foundation_rank, suit=Suit.HEARTS)]
    state.tableau[0] = [Card(rank=card_rank, suit=card_suit)]
    
    # Create move
    move = Move(
        source_pile_type='tableau',
        source_pile_index=0,
        destination_pile_type='foundation',
        destination_pile_index=0
    )
    
    # Validate
    if should_be_valid:
        assert validate_move(move, state) is True
    else:
        assert validate_move(move, state) is False
```

### Related Files
- `solitaire_analytics/engine/move_validator.py`
- `tests/test_engine.py`

### Testing
```bash
pytest tests/test_engine.py::test_validate_tableau_to_foundation -v
pytest tests/test_engine.py -k validate
```

---

## Task 6: Create Smoke Test Suite

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `tests/test_smoke.py`

### Description
Create a fast smoke test suite that catches major breakage quickly.

### Context
Smoke tests run before the full test suite to catch obvious problems fast. They should complete in < 30 seconds.

### Requirements
Create smoke tests for:
1. Package imports successfully
2. Basic game state creation
3. Simple move generation
4. Basic move validation
5. Solver initialization
6. Example script runs without errors

### Acceptance Criteria
- [ ] New file `tests/test_smoke.py` created
- [ ] Tests marked with `@pytest.mark.smoke`
- [ ] At least 6 smoke tests
- [ ] All tests complete in < 30 seconds total
- [ ] Tests catch import errors
- [ ] Tests catch basic API breakage
- [ ] Update pytest.ini with smoke marker

### Example Implementation
```python
# tests/test_smoke.py

import pytest

@pytest.mark.smoke
def test_imports():
    """Smoke test: All imports work."""
    from solitaire_analytics import Card, GameState, Move
    from solitaire_analytics import generate_moves, validate_move
    from solitaire_analytics import ParallelSolver
    from solitaire_analytics import MoveTreeBuilder, DeadEndDetector
    
    # If we get here, imports succeeded
    assert True

@pytest.mark.smoke
def test_create_game_state():
    """Smoke test: Can create game state."""
    from solitaire_analytics import GameState
    
    state = GameState()
    assert state is not None
    assert hasattr(state, 'tableau')
    assert hasattr(state, 'foundations')

@pytest.mark.smoke
def test_generate_moves_basic():
    """Smoke test: Move generation works."""
    from solitaire_analytics import GameState, generate_moves
    
    state = GameState()
    moves = generate_moves(state)
    
    assert isinstance(moves, list)
    # Empty state should have at least stock-to-waste move
    assert len(moves) >= 0

@pytest.mark.smoke
def test_solver_initialization():
    """Smoke test: Solver can be created."""
    from solitaire_analytics import ParallelSolver
    
    solver = ParallelSolver(max_depth=5, n_jobs=1)
    assert solver is not None

@pytest.mark.smoke
def test_example_script_runs():
    """Smoke test: Example script runs without errors."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'scripts/example_analysis.py'],
        capture_output=True,
        timeout=30
    )
    
    # Script should complete (may fail for other reasons)
    # Just checking it doesn't crash
    assert result.returncode in [0, 1]  # Allow some errors in example
```

### Configuration Update
```ini
# pytest.ini
[pytest]
markers =
    smoke: Quick smoke tests (< 30 seconds total)
    # ... other markers ...
```

### Related Files
- All main modules

### Testing
```bash
pytest -m smoke -v  # Run only smoke tests
pytest -m smoke --durations=10  # Check timing
```

---

## Task 7: Add Performance Regression Tests

**Difficulty:** Medium  
**Estimated Time:** 4-5 hours  
**Module:** `tests/test_performance.py`

### Description
Create performance benchmark tests that detect regressions.

### Context
Performance can degrade over time. We need tests that measure and alert on performance regressions.

### Requirements
Create benchmarks for:
1. Move generation speed
2. Move validation speed
3. Solver throughput
4. Tree building speed
5. State cloning performance

### Acceptance Criteria
- [ ] New file `tests/test_performance.py`
- [ ] Tests marked with `@pytest.mark.performance`
- [ ] At least 5 benchmark tests
- [ ] Tests measure execution time
- [ ] Tests have reasonable thresholds
- [ ] Tests use pytest-benchmark if available
- [ ] Document expected performance

### Example Implementation
```python
# tests/test_performance.py

import pytest
import time
from solitaire_analytics import GameState, generate_moves

@pytest.mark.performance
def test_move_generation_performance(benchmark):
    """Benchmark move generation speed."""
    state = create_typical_game_state()
    
    # Run benchmark
    result = benchmark(generate_moves, state)
    
    # Verify performance
    assert benchmark.stats['mean'] < 0.01, "Move generation should be < 10ms"

@pytest.mark.performance
def test_state_cloning_performance():
    """Test that state cloning is fast."""
    state = create_complex_game_state()
    
    start = time.perf_counter()
    for _ in range(1000):
        clone = state.copy()
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / 1000
    assert avg_time < 0.001, f"State cloning should be < 1ms, got {avg_time*1000:.2f}ms"

@pytest.mark.performance
def test_solver_throughput():
    """Test solver can handle multiple games efficiently."""
    games = [create_test_game_state() for _ in range(10)]
    solver = ParallelSolver(max_depth=5, n_jobs=-1)
    
    start = time.time()
    for game in games:
        solver.solve(game)
    elapsed = time.time() - start
    
    throughput = len(games) / elapsed
    assert throughput > 1.0, f"Solver should handle > 1 game/sec, got {throughput:.2f}"
```

### Related Files
- All performance-critical modules

### Testing
```bash
pytest -m performance -v
pip install pytest-benchmark
pytest tests/test_performance.py --benchmark-only
```

---

## Task 8: Add Test Data Files and Loaders

**Difficulty:** Easy  
**Estimated Time:** 3-4 hours  
**Module:** `tests/fixtures/`

### Description
Create JSON test data files and helper functions to load them.

### Context
Tests often need complex game states. Storing them as JSON makes tests clearer and more maintainable.

### Requirements
1. Create `tests/fixtures/game_states/` directory
2. Create JSON files for various game states
3. Add helper functions to load test data
4. Update tests to use fixtures

### Acceptance Criteria
- [ ] Directory structure created
- [ ] At least 5 JSON game state files
- [ ] Helper module `tests/test_utils.py` created
- [ ] Load functions handle errors gracefully
- [ ] Example tests updated to use fixtures
- [ ] Documentation of fixture format

### Example Structure
```
tests/fixtures/
├── game_states/
│   ├── initial_empty.json
│   ├── early_game.json
│   ├── mid_game.json
│   ├── near_win.json
│   └── dead_end.json
└── README.md
```

### Example Implementation
```python
# tests/test_utils.py

import json
from pathlib import Path
from solitaire_analytics.models import GameState

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GAME_STATES_DIR = FIXTURES_DIR / 'game_states'

def load_game_state(filename: str) -> GameState:
    """
    Load a game state from fixtures.
    
    Args:
        filename: Name of JSON file (e.g., 'early_game.json')
    
    Returns:
        GameState loaded from file
    
    Raises:
        FileNotFoundError: If fixture doesn't exist
    """
    filepath = GAME_STATES_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fixture not found: {filepath}")
    
    with open(filepath) as f:
        data = json.load(f)
    
    return GameState.from_json(data)

def list_game_state_fixtures() -> list:
    """List all available game state fixtures."""
    return [f.name for f in GAME_STATES_DIR.glob('*.json')]
```

### Example JSON File
```json
{
  "name": "early_game",
  "description": "Early game state with few moves made",
  "tableau": [
    [{"rank": 13, "suit": "hearts", "face_down": false}],
    [{"rank": 12, "suit": "spades", "face_down": false}]
  ],
  "foundations": [[], [], [], []],
  "stock": [],
  "waste": []
}
```

### Related Files
- All test files

### Testing
```bash
pytest tests/ -v
python -c "from tests.test_utils import list_game_state_fixtures; print(list_game_state_fixtures())"
```

---

## Summary

These testing tasks progressively improve test coverage, quality, and maintainability. Recommended order:

1. **Start with:** Task 6 (Smoke tests) - Quick wins
2. **Then:** Task 2 (Test fixtures) - Foundation
3. **Then:** Task 5 (Parametrized tests) - Good patterns
4. **Then:** Task 1 (Property-based) - Advanced testing
5. **Then:** Tasks 3, 4, 7, 8 - Comprehensive coverage

Each task is independent and can be completed separately.
