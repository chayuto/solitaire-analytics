# Solitaire Analytics Engine - Coding Agent Friendly Adjustments

**Date:** November 13, 2025  
**Version:** 0.1.0

## Executive Summary

This document outlines specific adjustments and best practices to make the Solitaire Analytics Engine more accessible and workable for coding agents (AI-assisted development tools). These recommendations focus on clear structure, comprehensive documentation, automated validation, and self-contained tasks.

---

## 1. Project Structure Improvements

### 1.1 Clear Module Boundaries

**Current State**: Good module separation already exists.

**Enhancements for Agents**:

1. **Explicit Module Purposes**
   - Add module-level docstrings explaining purpose
   - Document dependencies between modules
   - Clarify data flow

   ```python
   """
   solitaire_analytics.models
   
   This module provides core data structures for representing Solitaire games.
   
   Modules:
   - card: Card representation with rank and suit
   - game_state: Complete game state including tableau, foundations, stock, waste
   - move: Move representation with source and destination
   
   Dependencies: None (pure data structures)
   Used by: engine, solvers, analysis
   """
   ```

2. **Create Architecture Diagram**
   - ASCII art diagram in README
   - Mermaid diagram in docs
   - Shows module relationships

   ```mermaid
   graph TD
       A[models] --> B[engine]
       A --> C[solvers]
       A --> D[analysis]
       B --> C
       B --> D
   ```

**Priority**: High  
**Effort**: 1-2 days

---

### 1.2 Configuration Files

**Recommendations**:

1. **Centralized Configuration**
   ```python
   # solitaire_analytics/config.py
   
   from dataclasses import dataclass
   from typing import Optional
   
   @dataclass
   class SolverConfig:
       """Configuration for parallel solver."""
       max_depth: int = 10
       n_jobs: int = -1
       beam_width: int = 100
       timeout: Optional[float] = None
       use_gpu: bool = False
   
   @dataclass
   class AnalysisConfig:
       """Configuration for analysis tools."""
       max_tree_nodes: int = 1000
       max_tree_depth: int = 5
   ```

2. **YAML Configuration Support**
   ```yaml
   # config/default.yaml
   solver:
     max_depth: 10
     n_jobs: -1
     beam_width: 100
   
   analysis:
     max_tree_nodes: 1000
     max_tree_depth: 5
   ```

3. **Environment Variable Override**
   ```python
   import os
   
   max_depth = int(os.getenv('SOLITAIRE_MAX_DEPTH', '10'))
   ```

**Priority**: Medium  
**Effort**: 2-3 days

---

### 1.3 Example Patterns

**Recommendations**:

1. **Create Examples Directory**
   ```
   examples/
   ├── README.md                    # Index of examples
   ├── basic/
   │   ├── 01_create_game_state.py
   │   ├── 02_generate_moves.py
   │   └── 03_validate_moves.py
   ├── intermediate/
   │   ├── 01_solve_game.py
   │   ├── 02_build_move_tree.py
   │   └── 03_analyze_dead_ends.py
   └── advanced/
       ├── 01_custom_heuristics.py
       ├── 02_parallel_processing.py
       └── 03_gpu_acceleration.py
   ```

2. **Self-Contained Examples**
   - Each example runs independently
   - Clear comments explaining each step
   - Expected output documented

**Priority**: High  
**Effort**: 3-5 days

---

## 2. Documentation Standards

### 2.1 Docstring Completeness

**Current State**: Some docstrings present.

**Enhancements**:

1. **Complete Google-Style Docstrings**
   ```python
   def generate_moves(state: GameState) -> List[Move]:
       """Generate all valid moves from a game state.
       
       This function examines the current game state and returns all legal
       moves according to Klondike Solitaire rules.
       
       Args:
           state: The current game state to analyze.
       
       Returns:
           A list of Move objects representing all valid moves.
           Empty list if no moves are possible (dead end).
       
       Raises:
           ValueError: If state is invalid or corrupted.
       
       Example:
           >>> state = GameState()
           >>> moves = generate_moves(state)
           >>> print(f"Found {len(moves)} possible moves")
       
       See Also:
           validate_move: Check if a specific move is valid
           apply_move: Apply a move to create new state
       """
   ```

2. **Type Hints Everywhere**
   - All function parameters
   - All return types
   - Class attributes
   - Use typing.Optional, typing.List, etc.

3. **Inline Comments for Complex Logic**
   ```python
   # Calculate tableau-to-tableau moves
   # We can move a King to an empty tableau pile
   for src_idx, src_pile in enumerate(state.tableau):
       if not src_pile:
           continue
       
       # Only consider the revealed portion of the pile
       for card_idx, card in enumerate(src_pile):
           if card.face_down:
               continue  # Skip face-down cards
   ```

**Priority**: High  
**Effort**: 2 weeks (gradual addition)

---

### 2.2 Decision Documentation

**Recommendations**:

1. **Architecture Decision Records (ADRs)**
   ```markdown
   # ADR 001: Use Beam Search for Game Solving
   
   ## Status
   Accepted
   
   ## Context
   Need efficient algorithm for exploring large state space.
   
   ## Decision
   Implement beam search with configurable beam width.
   
   ## Consequences
   - Positive: Fast, memory-efficient
   - Negative: Not guaranteed optimal
   - Positive: Configurable trade-off
   ```

2. **Design Rationale Comments**
   ```python
   # We use immutable game states to enable:
   # 1. Safe parallel processing without locks
   # 2. Easy state rollback for tree search
   # 3. Clear data flow (no hidden state changes)
   class GameState:
       ...
   ```

**Priority**: Medium  
**Effort**: 1 week

---

### 2.3 Error Message Quality

**Recommendations**:

1. **Descriptive Error Messages**
   ```python
   # Bad
   raise ValueError("Invalid move")
   
   # Good
   raise ValueError(
       f"Cannot move {card} from {source} to {destination}: "
       f"Destination pile requires {expected_color} {expected_rank}, "
       f"but card is {card.color} {card.rank}"
   )
   ```

2. **Error Message Guidelines**
   - What went wrong
   - Why it's wrong
   - How to fix it (if applicable)

3. **Custom Exception Types**
   ```python
   class InvalidMoveError(ValueError):
       """Raised when a move violates game rules."""
       
       def __init__(self, move: Move, reason: str):
           self.move = move
           self.reason = reason
           super().__init__(f"Invalid move {move}: {reason}")
   ```

**Priority**: High  
**Effort**: 1 week

---

## 3. Testing Infrastructure

### 3.1 Clear Test Organization

**Recommendations**:

1. **Naming Convention**
   ```python
   # test_models.py
   
   class TestCard:
       """Tests for Card class."""
       
       def test_card_creation_valid(self):
           """Test creating valid cards."""
       
       def test_card_creation_invalid_rank(self):
           """Test that invalid ranks raise ValueError."""
       
       def test_card_comparison(self):
           """Test card equality and comparison operations."""
   ```

2. **Test Fixtures**
   ```python
   @pytest.fixture
   def empty_state():
       """Empty game state for testing."""
       return GameState()
   
   @pytest.fixture
   def sample_state():
       """Typical mid-game state for testing."""
       state = GameState()
       # ... setup state ...
       return state
   
   @pytest.fixture
   def winning_state():
       """Near-winning game state for testing."""
       state = GameState()
       # ... setup state ...
       return state
   ```

3. **Parameterized Tests**
   ```python
   @pytest.mark.parametrize("rank,suit,valid", [
       (1, Suit.HEARTS, True),
       (13, Suit.SPADES, True),
       (0, Suit.HEARTS, False),
       (14, Suit.DIAMONDS, False),
   ])
   def test_card_validation(rank, suit, valid):
       """Test card validation for various inputs."""
       if valid:
           card = Card(rank=rank, suit=suit)
           assert card.rank == rank
       else:
           with pytest.raises(ValueError):
               Card(rank=rank, suit=suit)
   ```

**Priority**: High  
**Effort**: 1 week

---

### 3.2 Test Documentation

**Recommendations**:

1. **Test Purpose Documentation**
   ```python
   def test_solver_finds_solution():
       """
       Test that solver can find solution to solvable game.
       
       This test verifies:
       1. Solver completes without errors
       2. Solution is found for solvable game
       3. Solution is valid (all moves legal)
       4. Final state is winning state
       
       Test data: fixtures/solvable_game_01.json
       Expected outcome: Solution with 25-30 moves
       """
   ```

2. **Regression Test Markers**
   ```python
   @pytest.mark.regression
   @pytest.mark.issue(123)
   def test_solver_handles_specific_dead_end():
       """
       Regression test for issue #123.
       
       Previously, solver would hang on certain dead-end states.
       This test ensures it returns promptly with failure.
       """
   ```

**Priority**: Medium  
**Effort**: 3-5 days

---

### 3.3 Automated Validation

**Recommendations**:

1. **Input Validation Tests**
   ```python
   def test_generate_moves_validates_input():
       """Test that generate_moves validates its input."""
       with pytest.raises(TypeError):
           generate_moves(None)
       
       with pytest.raises(ValueError):
           generate_moves("not a game state")
   ```

2. **Invariant Tests**
   ```python
   def test_game_state_invariants(sample_state):
       """Test that game state maintains invariants."""
       # Total cards should be 52
       total = sum(len(pile) for pile in sample_state.tableau)
       total += sum(len(pile) for pile in sample_state.foundations)
       total += len(sample_state.stock)
       total += len(sample_state.waste)
       assert total == 52
       
       # No duplicate cards
       all_cards = []
       # ... collect all cards ...
       assert len(all_cards) == len(set(all_cards))
   ```

**Priority**: High  
**Effort**: 1 week

---

## 4. Build and Development Tools

### 4.1 Development Scripts

**Recommendations**:

1. **Create Makefile or scripts/dev.sh**
   ```makefile
   # Makefile
   
   .PHONY: install test lint format check clean
   
   install:
   	pip install -e .
   	pip install -r requirements-dev.txt
   
   test:
   	pytest
   
   test-fast:
   	pytest -m "not slow"
   
   lint:
   	flake8 solitaire_analytics
   	mypy solitaire_analytics
   
   format:
   	black solitaire_analytics tests
   	isort solitaire_analytics tests
   
   check: lint test
   
   clean:
   	find . -type d -name __pycache__ -exec rm -rf {} +
   	find . -type f -name "*.pyc" -delete
   	rm -rf .pytest_cache .coverage htmlcov
   ```

2. **Development Setup Script**
   ```bash
   # scripts/setup_dev.sh
   #!/bin/bash
   
   echo "Setting up development environment..."
   
   # Create virtual environment
   python3.12 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -e .
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   
   # Run tests to verify
   pytest -v
   
   echo "Setup complete!"
   ```

**Priority**: High  
**Effort**: 2-3 days

---

### 4.2 Quick Validation Commands

**Recommendations**:

1. **Single Command Check**
   ```bash
   # scripts/quick_check.sh
   #!/bin/bash
   set -e
   
   echo "Running quick validation..."
   
   # Fast tests only
   pytest -m "not slow" -x
   
   # Lint
   flake8 solitaire_analytics --select=E,W,F
   
   # Type check critical modules
   mypy solitaire_analytics/models
   
   echo "✓ All checks passed!"
   ```

2. **Full Validation**
   ```bash
   # scripts/full_check.sh
   #!/bin/bash
   set -e
   
   echo "Running full validation..."
   
   # Format check
   black --check solitaire_analytics
   isort --check solitaire_analytics
   
   # Lint
   flake8 solitaire_analytics
   
   # Type check
   mypy solitaire_analytics
   
   # All tests with coverage
   pytest --cov=solitaire_analytics --cov-fail-under=80
   
   # Security check
   safety check
   
   echo "✓ All checks passed!"
   ```

**Priority**: High  
**Effort**: 1 day

---

## 5. Code Patterns and Conventions

### 5.1 Consistent Patterns

**Recommendations**:

1. **Factory Functions**
   ```python
   # Consistent pattern for creating objects
   
   @classmethod
   def from_json(cls, json_data: dict) -> 'GameState':
       """Create GameState from JSON data."""
       # ... implementation ...
   
   @classmethod
   def from_file(cls, filepath: str) -> 'GameState':
       """Create GameState from JSON file."""
       with open(filepath) as f:
           return cls.from_json(json.load(f))
   
   def to_json(self) -> dict:
       """Convert GameState to JSON-serializable dict."""
       # ... implementation ...
   
   def to_file(self, filepath: str) -> None:
       """Save GameState to JSON file."""
       with open(filepath, 'w') as f:
           json.dump(self.to_json(), f, indent=2)
   ```

2. **Builder Pattern for Complex Objects**
   ```python
   class GameStateBuilder:
       """Builder for constructing complex game states."""
       
       def __init__(self):
           self.state = GameState()
       
       def with_tableau(self, tableau: List[List[Card]]) -> 'GameStateBuilder':
           """Set tableau cards."""
           self.state.tableau = tableau
           return self
       
       def with_foundations(self, foundations: List[List[Card]]) -> 'GameStateBuilder':
           """Set foundation cards."""
           self.state.foundations = foundations
           return self
       
       def build(self) -> GameState:
           """Build and validate the game state."""
           self.state.validate()
           return self.state
   ```

3. **Context Managers for Resources**
   ```python
   class SolverSession:
       """Context manager for solver with resource cleanup."""
       
       def __init__(self, config: SolverConfig):
           self.solver = ParallelSolver(**config.__dict__)
       
       def __enter__(self):
           self.solver.initialize()
           return self.solver
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           self.solver.cleanup()
   ```

**Priority**: Medium  
**Effort**: 1 week

---

### 5.2 Error Handling Patterns

**Recommendations**:

1. **Consistent Validation**
   ```python
   def validate_move(move: Move, state: GameState) -> None:
       """
       Validate move against current state.
       
       Raises:
           InvalidMoveError: If move is not legal.
           ValueError: If inputs are malformed.
       """
       if move is None:
           raise ValueError("Move cannot be None")
       
       if state is None:
           raise ValueError("State cannot be None")
       
       if not _is_legal_move(move, state):
           raise InvalidMoveError(
               move=move,
               reason=_get_move_error_reason(move, state)
           )
   ```

2. **Graceful Degradation**
   ```python
   def solve_with_fallback(state: GameState) -> SolverResult:
       """Solve game with GPU, fallback to CPU if unavailable."""
       try:
           return solve_with_gpu(state)
       except RuntimeError as e:
           logger.warning(f"GPU unavailable: {e}. Falling back to CPU.")
           return solve_with_cpu(state)
   ```

**Priority**: Medium  
**Effort**: 1 week

---

## 6. Agent-Specific Features

### 6.1 Validation Hooks

**Recommendations**:

1. **Pre/Post Condition Checks**
   ```python
   def require_valid_state(func):
       """Decorator to validate game state before operation."""
       @functools.wraps(func)
       def wrapper(state: GameState, *args, **kwargs):
           if not state.is_valid():
               raise ValueError(f"Invalid game state: {state.get_errors()}")
           return func(state, *args, **kwargs)
       return wrapper
   
   @require_valid_state
   def generate_moves(state: GameState) -> List[Move]:
       ...
   ```

2. **Runtime Assertions**
   ```python
   def apply_move(state: GameState, move: Move) -> GameState:
       """Apply move to create new state."""
       new_state = state.copy()
       # ... apply move ...
       
       # Verify invariants
       assert new_state.card_count() == 52, "Card count changed!"
       assert new_state.is_valid(), "State became invalid!"
       
       return new_state
   ```

**Priority**: Medium  
**Effort**: 3-5 days

---

### 6.2 Self-Diagnostic Tools

**Recommendations**:

1. **Health Check Function**
   ```python
   def health_check() -> dict:
       """
       Run system health checks.
       
       Returns dict with:
       - dependencies_ok: bool
       - gpu_available: bool
       - test_status: str
       - warnings: List[str]
       """
       health = {
           'dependencies_ok': True,
           'gpu_available': False,
           'test_status': 'unknown',
           'warnings': []
       }
       
       # Check dependencies
       try:
           import torch
           import networkx
           import joblib
       except ImportError as e:
           health['dependencies_ok'] = False
           health['warnings'].append(f"Missing dependency: {e}")
       
       # Check GPU
       try:
           import torch
           health['gpu_available'] = torch.cuda.is_available()
       except Exception:
           pass
       
       # Run smoke test
       try:
           state = GameState()
           moves = generate_moves(state)
           health['test_status'] = 'pass'
       except Exception as e:
           health['test_status'] = 'fail'
           health['warnings'].append(f"Smoke test failed: {e}")
       
       return health
   ```

2. **Diagnostics Command**
   ```bash
   # scripts/diagnose.sh
   #!/bin/bash
   
   echo "=== Solitaire Analytics Diagnostics ==="
   echo
   
   echo "Python version:"
   python --version
   echo
   
   echo "Installed packages:"
   pip list | grep -E "torch|joblib|networkx|numpy|pandas"
   echo
   
   echo "GPU availability:"
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   echo
   
   echo "Running health check:"
   python -c "from solitaire_analytics import health_check; import json; print(json.dumps(health_check(), indent=2))"
   ```

**Priority**: High  
**Effort**: 2-3 days

---

## 7. Onboarding for Agents

### 7.1 Quick Start Guide

**Recommendations**:

1. **Create QUICKSTART.md**
   ```markdown
   # Quick Start for Coding Agents
   
   ## Setup (< 5 minutes)
   ```bash
   git clone https://github.com/chayuto/solitaire-analytics.git
   cd solitaire-analytics
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -e .
   pytest -v  # Verify installation
   ```
   
   ## Run Example
   ```bash
   python scripts/example_analysis.py
   ```
   
   ## Common Tasks
   - Add feature: See docs/reports/20251113_agent_tasks_features.md
   - Fix bug: See CONTRIBUTING.md
   - Add tests: See tests/README.md
   ```

2. **Agent-Specific README**
   ```markdown
   # README_FOR_AGENTS.md
   
   This project is designed to be agent-friendly!
   
   ## Key Files
   - `/docs/reports/20251113_project_state.md` - Understand the project
   - `/docs/reports/20251113_agent_tasks_*.md` - Find tasks to work on
   - `/examples/` - Learn by example
   - `/tests/` - Understand expected behavior
   
   ## Before Making Changes
   1. Run `make test` to verify current state
   2. Read relevant test files
   3. Check similar code for patterns
   
   ## After Making Changes
   1. Add/update tests
   2. Run `make check`
   3. Update docs if needed
   ```

**Priority**: High  
**Effort**: 1 day

---

### 7.2 Task Templates

**Recommendations**:

1. **Issue Template**
   ```markdown
   # .github/ISSUE_TEMPLATE/agent_task.md
   
   ---
   name: Agent Task
   about: Self-contained task for coding agents
   ---
   
   ## Task Description
   [Clear, specific description]
   
   ## Context
   - **Module**: [Which module/file]
   - **Related**: [Related code/docs]
   - **Why**: [Why is this needed]
   
   ## Acceptance Criteria
   - [ ] Criterion 1
   - [ ] Criterion 2
   - [ ] Tests pass
   - [ ] Documentation updated
   
   ## Test Cases
   ```python
   # Example test cases
   ```
   
   ## Hints
   - [Helpful hints for implementation]
   ```

**Priority**: Medium  
**Effort**: 2-3 days

---

## 8. Implementation Priority

### Immediate (Week 1)
1. ✅ Complete docstrings for public APIs
2. ✅ Add development scripts (Makefile)
3. ✅ Create quick start guide
4. ✅ Add health check function
5. ✅ Improve error messages

### Short Term (Weeks 2-4)
6. Add validation hooks
7. Create example patterns
8. Reorganize test structure
9. Add test fixtures
10. Create task templates

### Medium Term (Months 2-3)
11. Centralized configuration
12. Builder patterns
13. Self-diagnostic tools
14. ADR documentation
15. Complete type hints

## Conclusion

These adjustments make the codebase significantly more accessible to coding agents by:
- Providing clear structure and documentation
- Offering automated validation tools
- Establishing consistent patterns
- Creating self-contained, well-documented tasks
- Including comprehensive examples

Focus on the immediate priorities first to create a solid foundation for agent-assisted development.
