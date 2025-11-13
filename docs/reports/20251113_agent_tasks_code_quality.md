# Coding Agent Tasks - Code Quality Improvements

**Date:** November 13, 2025  
**Category:** Code Quality  
**Difficulty Range:** Easy to Medium

## Overview

This document contains self-contained code quality tasks suitable for coding agents. Each task improves code maintainability, readability, and consistency.

---

## Task 1: Add Type Hints to All Functions

**Difficulty:** Easy  
**Estimated Time:** 4-6 hours  
**Module:** All modules

### Description
Add complete type hints to all function signatures and class attributes.

### Context
Type hints improve code clarity and enable static type checking with mypy.

### Requirements
Add type hints for:
1. All function parameters
2. All function return types
3. Class attributes
4. Use typing module for complex types

### Acceptance Criteria
- [ ] All functions have parameter type hints
- [ ] All functions have return type hints
- [ ] Class attributes are typed
- [ ] Import typing constructs as needed
- [ ] mypy runs without errors (when added)
- [ ] No use of `Any` unless necessary

### Example
```python
# Before
def generate_moves(state):
    moves = []
    # ... implementation ...
    return moves

# After
from typing import List
from solitaire_analytics.models import GameState, Move

def generate_moves(state: GameState) -> List[Move]:
    """Generate all valid moves from a game state."""
    moves: List[Move] = []
    # ... implementation ...
    return moves
```

### Files to Update
- `solitaire_analytics/models/*.py`
- `solitaire_analytics/engine/*.py`
- `solitaire_analytics/solvers/*.py`
- `solitaire_analytics/analysis/*.py`

### Verification
```bash
pip install mypy
mypy solitaire_analytics --strict
```

### Related
- Task 2 (Add mypy configuration)

---

## Task 2: Add Linting Configuration

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** Project root

### Description
Add and configure linting tools (flake8, black, isort) for the project.

### Context
Consistent code style improves readability and reduces review friction.

### Requirements
1. Add flake8 configuration
2. Add black configuration
3. Add isort configuration
4. Add pre-commit hooks
5. Fix existing lint issues

### Acceptance Criteria
- [ ] `.flake8` file created with configuration
- [ ] `pyproject.toml` created with black/isort config
- [ ] `.pre-commit-config.yaml` created
- [ ] All tools added to requirements-dev.txt
- [ ] Existing code passes all checks
- [ ] Documentation updated with lint commands

### Example Configuration

`.flake8`:
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, E266, E501, W503
exclude =
    .git,
    __pycache__,
    build,
    dist,
    .venv,
    venv
per-file-ignores =
    __init__.py:F401
max-complexity = 10
```

`pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py312']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

### Installation
```bash
pip install black isort flake8 pre-commit
pre-commit install
```

### Usage
```bash
# Check formatting
black --check solitaire_analytics
isort --check solitaire_analytics
flake8 solitaire_analytics

# Fix formatting
black solitaire_analytics
isort solitaire_analytics
```

### Related
- Task 1 (Type hints)
- Task 3 (Add docstrings)

---

## Task 3: Complete Docstrings for All Public APIs

**Difficulty:** Medium  
**Estimated Time:** 6-8 hours  
**Module:** All modules

### Description
Add comprehensive Google-style docstrings to all public classes and functions.

### Context
Good documentation is essential for understanding and using the codebase.

### Requirements
Docstrings should include:
1. One-line summary
2. Detailed description
3. Args section with types and descriptions
4. Returns section with type and description
5. Raises section for exceptions
6. Example usage

### Acceptance Criteria
- [ ] All public classes have docstrings
- [ ] All public functions have docstrings
- [ ] Docstrings follow Google style
- [ ] Examples are included where helpful
- [ ] Docstrings are grammatically correct
- [ ] No missing sections

### Example
```python
def validate_move(move: Move, state: GameState) -> bool:
    """
    Validate whether a move is legal in the given game state.
    
    This function checks all Klondike Solitaire rules to determine if
    the specified move can be legally performed. It verifies:
    - Source pile has the card
    - Destination pile can accept the card
    - Move follows rank/suit rules
    
    Args:
        move: The move to validate. Must specify source and destination
            pile types and indices.
        state: The current game state. Must be a valid, non-corrupted
            game state.
    
    Returns:
        True if the move is legal and can be performed, False otherwise.
    
    Raises:
        ValueError: If move or state is None or malformed.
        InvalidStateError: If game state is corrupted.
    
    Example:
        >>> state = GameState()
        >>> move = Move(source_pile_type='stock', ...)
        >>> if validate_move(move, state):
        ...     new_state = apply_move(state, move)
    
    See Also:
        generate_moves: Get all valid moves for a state
        apply_move: Apply a validated move to a state
    """
    if move is None or state is None:
        raise ValueError("Move and state cannot be None")
    
    # ... implementation ...
```

### Modules to Document
Priority order:
1. `solitaire_analytics/__init__.py`
2. `solitaire_analytics/models/*.py`
3. `solitaire_analytics/engine/*.py`
4. `solitaire_analytics/solvers/*.py`
5. `solitaire_analytics/analysis/*.py`

### Verification
```bash
# Check for missing docstrings
pydocstyle solitaire_analytics

# Generate documentation (future)
sphinx-apidoc -o docs/api solitaire_analytics
```

### Related
- Task 10 (Generate API documentation)

---

## Task 4: Refactor Long Functions

**Difficulty:** Medium  
**Estimated Time:** 4-6 hours  
**Module:** Various

### Description
Identify and refactor functions with high complexity or length.

### Context
Long, complex functions are hard to understand and maintain. Breaking them into smaller functions improves clarity.

### Requirements
1. Find functions > 50 lines or complexity > 10
2. Extract logical sections into helper functions
3. Maintain original behavior
4. Add tests for new functions

### Acceptance Criteria
- [ ] All functions < 50 lines
- [ ] Cyclomatic complexity < 10
- [ ] Helper functions have clear names
- [ ] All tests still pass
- [ ] Code coverage maintained or improved

### Process
```bash
# Find complex functions
pip install radon
radon cc solitaire_analytics -a -nc
radon cc solitaire_analytics -s

# Find long functions
find solitaire_analytics -name "*.py" -exec wc -l {} \; | sort -rn
```

### Example Refactoring

Before:
```python
def solve_game(state: GameState) -> SolverResult:
    """Solve the game."""
    # 100+ lines of complex logic...
    
    # Initialize
    visited = set()
    queue = [state]
    best_score = 0
    
    # Main loop
    while queue:
        current = queue.pop(0)
        
        # Check if winning
        if is_winning(current):
            return SolverResult(success=True, ...)
        
        # Generate moves
        moves = generate_moves(current)
        
        # Filter and evaluate moves
        for move in moves:
            # Complex evaluation logic...
            # ... 50 more lines ...
    
    return SolverResult(success=False)
```

After:
```python
def solve_game(state: GameState) -> SolverResult:
    """Solve the game using beam search."""
    solver_state = _initialize_solver_state(state)
    
    while not solver_state.is_complete():
        _process_next_batch(solver_state)
        
        if solver_state.solution_found():
            return _create_success_result(solver_state)
    
    return _create_failure_result(solver_state)

def _initialize_solver_state(state: GameState) -> _SolverState:
    """Initialize internal solver state."""
    return _SolverState(
        initial_state=state,
        visited=set(),
        queue=[state]
    )

def _process_next_batch(solver_state: _SolverState) -> None:
    """Process next batch of states."""
    # ... extracted logic ...

def _create_success_result(solver_state: _SolverState) -> SolverResult:
    """Create successful solver result."""
    # ... extracted logic ...
```

### Candidates for Refactoring
1. `parallel_solver.py::solve()` method
2. `move_generator.py::generate_moves()`
3. Any function with complexity > 10

### Verification
```bash
pytest tests/ -v  # All tests pass
radon cc solitaire_analytics -nc  # Check improvement
```

---

## Task 5: Add Input Validation

**Difficulty:** Easy-Medium  
**Estimated Time:** 3-5 hours  
**Module:** All public APIs

### Description
Add comprehensive input validation to all public functions.

### Context
Robust input validation prevents errors and provides clear error messages.

### Requirements
Add validation for:
1. None/null checks
2. Type validation
3. Range validation
4. State invariant checks

### Acceptance Criteria
- [ ] All public functions validate inputs
- [ ] Appropriate exceptions raised
- [ ] Clear error messages
- [ ] Tests added for invalid inputs
- [ ] Documentation updated with exceptions

### Example
```python
def generate_moves(state: GameState) -> List[Move]:
    """
    Generate all valid moves.
    
    Args:
        state: Current game state.
    
    Returns:
        List of valid moves.
    
    Raises:
        ValueError: If state is None or invalid.
        InvalidStateError: If game state is corrupted.
    """
    # Validate inputs
    if state is None:
        raise ValueError("Game state cannot be None")
    
    if not isinstance(state, GameState):
        raise TypeError(
            f"Expected GameState, got {type(state).__name__}"
        )
    
    # Validate state integrity
    if not state.is_valid():
        raise InvalidStateError(
            f"Game state is invalid: {state.get_validation_errors()}"
        )
    
    # ... actual implementation ...
```

### Functions to Update
Priority:
1. `generate_moves()`
2. `validate_move()`
3. `ParallelSolver.__init__()`
4. `ParallelSolver.solve()`
5. `MoveTreeBuilder.build_tree()`
6. `DeadEndDetector.analyze_dead_end_risk()`

### Tests
```python
def test_generate_moves_validates_none():
    """Test that None state is rejected."""
    with pytest.raises(ValueError, match="cannot be None"):
        generate_moves(None)

def test_generate_moves_validates_type():
    """Test that wrong type is rejected."""
    with pytest.raises(TypeError, match="Expected GameState"):
        generate_moves("not a state")
```

---

## Task 6: Add Comprehensive Error Messages

**Difficulty:** Easy  
**Estimated Time:** 3-4 hours  
**Module:** All modules

### Description
Improve error messages to be more descriptive and actionable.

### Context
Good error messages help users understand and fix problems quickly.

### Requirements
Error messages should:
1. Explain what went wrong
2. Explain why it's wrong
3. Suggest how to fix it (if applicable)
4. Include relevant context

### Acceptance Criteria
- [ ] All error messages are descriptive
- [ ] Messages include context (values, types)
- [ ] Messages suggest solutions where applicable
- [ ] Consistent message formatting
- [ ] No bare error raises

### Examples

Bad:
```python
raise ValueError("Invalid move")
```

Good:
```python
raise ValueError(
    f"Invalid move: Cannot place {card} on {destination}. "
    f"Expected {expected_rank} of {expected_color}, "
    f"but got {card.rank} of {card.color}. "
    f"Hint: Check Solitaire stacking rules."
)
```

Bad:
```python
raise RuntimeError("Solver failed")
```

Good:
```python
raise RuntimeError(
    f"Solver failed after {states_explored} states explored in {elapsed:.2f}s. "
    f"Reason: {failure_reason}. "
    f"Try: Increase max_depth (currently {max_depth}) or beam_width ({beam_width})."
)
```

### Pattern to Follow
```python
raise ExceptionType(
    f"<What went wrong>: <Why>. "
    f"Got: <actual_value>. "
    f"Expected: <expected_value>. "
    f"Hint: <how to fix>."
)
```

### Files to Update
All files that raise exceptions.

---

## Task 7: Extract Magic Numbers to Constants

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** All modules

### Description
Replace magic numbers and strings with named constants.

### Context
Named constants make code more readable and maintainable.

### Requirements
1. Identify all magic numbers/strings
2. Extract to module-level constants
3. Use descriptive names
4. Group related constants

### Acceptance Criteria
- [ ] No unexplained magic numbers
- [ ] Constants use UPPER_CASE names
- [ ] Constants have docstrings or comments
- [ ] Related constants grouped together
- [ ] All tests pass

### Example

Before:
```python
def validate_card_rank(rank):
    if rank < 1 or rank > 13:
        raise ValueError(f"Invalid rank: {rank}")
    return True

def is_king(card):
    return card.rank == 13
```

After:
```python
# Card constants
MIN_CARD_RANK = 1  # Ace
MAX_CARD_RANK = 13  # King
KING_RANK = 13
ACE_RANK = 1

# Pile constants
NUM_TABLEAU_PILES = 7
NUM_FOUNDATION_PILES = 4

def validate_card_rank(rank: int) -> bool:
    """Validate card rank is within valid range."""
    if rank < MIN_CARD_RANK or rank > MAX_CARD_RANK:
        raise ValueError(
            f"Invalid rank: {rank}. "
            f"Must be between {MIN_CARD_RANK} and {MAX_CARD_RANK}"
        )
    return True

def is_king(card: Card) -> bool:
    """Check if card is a King."""
    return card.rank == KING_RANK
```

### Common Magic Numbers to Replace
- Card ranks (1-13)
- Number of piles (7 tableau, 4 foundations)
- Default timeouts
- Default beam widths
- Default max depths

### Files to Check
All Python files, especially:
- `models/card.py`
- `models/game_state.py`
- `solvers/parallel_solver.py`

---

## Task 8: Improve Code Comments

**Difficulty:** Easy  
**Estimated Time:** 3-4 hours  
**Module:** All modules

### Description
Add helpful comments to explain complex logic and design decisions.

### Context
Comments help future developers (and AI agents) understand code intent.

### Requirements
Add comments for:
1. Complex algorithms
2. Non-obvious design decisions
3. Performance optimizations
4. Workarounds or edge cases
5. TODO/FIXME items

### Acceptance Criteria
- [ ] Complex sections have explanatory comments
- [ ] Design decisions are documented
- [ ] No obvious code commented out
- [ ] Comments are accurate and up-to-date
- [ ] Consistent comment style

### Comment Guidelines

**DO:**
```python
# We use beam search instead of breadth-first search because:
# 1. Memory usage is bounded by beam_width
# 2. Still finds good solutions (not always optimal)
# 3. Much faster for deep searches
beam = sorted(candidates, key=score_function)[:beam_width]
```

**DON'T:**
```python
# Sort candidates
beam = sorted(candidates, key=score_function)[:beam_width]
```

**DO:**
```python
# Check for King because only Kings can be placed on empty tableau piles
# (Klondike Solitaire rule)
if card.rank == KING_RANK and destination_pile.is_empty():
    return True
```

**DON'T:**
```python
# Check for King
if card.rank == 13 and len(destination_pile) == 0:
    return True
```

### Areas Needing Comments
1. `parallel_solver.py` - Beam search algorithm
2. `move_generator.py` - Move generation logic
3. `dead_end_detector.py` - Risk scoring
4. Any performance optimizations

### Anti-patterns to Remove
```python
# def old_implementation():
#     ...  # Remove commented-out code

# x = 5  # Remove obvious comments
```

---

## Task 9: Create Configuration Dataclasses

**Difficulty:** Medium  
**Estimated Time:** 3-4 hours  
**Module:** `solitaire_analytics/config.py`

### Description
Create centralized configuration using dataclasses for better type safety.

### Context
Scattered configuration parameters are hard to manage. Centralized config improves maintainability.

### Requirements
1. Create config module with dataclasses
2. Define configuration for each component
3. Add validation
4. Update components to use config

### Acceptance Criteria
- [ ] `solitaire_analytics/config.py` created
- [ ] Dataclasses for each component
- [ ] Default values specified
- [ ] Validation in `__post_init__`
- [ ] Components updated to use config
- [ ] Tests updated

### Example Implementation

```python
# solitaire_analytics/config.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SolverConfig:
    """Configuration for parallel solver.
    
    Attributes:
        max_depth: Maximum search depth (1-100).
        n_jobs: Number of parallel jobs (-1 for all cores).
        beam_width: Beam search width (1-10000).
        timeout: Optional timeout in seconds.
        use_gpu: Whether to use GPU acceleration.
    """
    max_depth: int = 10
    n_jobs: int = -1
    beam_width: int = 100
    timeout: Optional[float] = None
    use_gpu: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_depth < 1 or self.max_depth > 100:
            raise ValueError(
                f"max_depth must be 1-100, got {self.max_depth}"
            )
        
        if self.beam_width < 1:
            raise ValueError(
                f"beam_width must be positive, got {self.beam_width}"
            )
        
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(
                f"timeout must be positive, got {self.timeout}"
            )

@dataclass
class AnalysisConfig:
    """Configuration for analysis tools."""
    max_tree_nodes: int = 1000
    max_tree_depth: int = 5
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.max_tree_nodes < 1:
            raise ValueError("max_tree_nodes must be positive")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")

@dataclass
class ProjectConfig:
    """Global project configuration."""
    solver: SolverConfig = field(default_factory=SolverConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    @classmethod
    def from_file(cls, path: str) -> 'ProjectConfig':
        """Load configuration from YAML file."""
        # Implementation
        pass
```

### Usage
```python
from solitaire_analytics.config import SolverConfig

# With defaults
solver = ParallelSolver(config=SolverConfig())

# Custom configuration
config = SolverConfig(max_depth=20, beam_width=200)
solver = ParallelSolver(config=config)
```

### Files to Update
- Create `solitaire_analytics/config.py`
- Update `solvers/parallel_solver.py`
- Update `analysis/move_tree_builder.py`
- Update `analysis/dead_end_detector.py`

---

## Task 10: Add Module-Level Docstrings

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** All modules

### Description
Add comprehensive docstrings to all Python modules.

### Context
Module docstrings help developers understand the purpose and contents of each module.

### Requirements
Each module should have:
1. Purpose description
2. Key classes/functions listed
3. Dependencies noted
4. Usage example (if applicable)

### Acceptance Criteria
- [ ] All modules have docstrings
- [ ] Docstrings describe module purpose
- [ ] Key components listed
- [ ] Dependencies documented
- [ ] Examples where helpful

### Example

```python
"""
Core game engine for Solitaire Analytics.

This module provides functions for generating and validating moves in
Klondike Solitaire. It implements the complete rules of the game and
ensures all operations maintain game state consistency.

Key Functions:
    generate_moves: Generate all valid moves from a state
    validate_move: Check if a move is legal
    apply_move: Apply a move to create new state

Dependencies:
    models: Uses Card, GameState, Move classes

Example:
    >>> from solitaire_analytics import GameState, generate_moves
    >>> state = GameState()
    >>> moves = generate_moves(state)
    >>> print(f"Found {len(moves)} possible moves")

See Also:
    models: Core data structures
    solvers: Game solving algorithms
"""

from typing import List
from solitaire_analytics.models import GameState, Move

# ... module code ...
```

### Modules to Document
- `solitaire_analytics/__init__.py`
- `solitaire_analytics/models/__init__.py`
- `solitaire_analytics/models/card.py`
- `solitaire_analytics/models/game_state.py`
- `solitaire_analytics/models/move.py`
- `solitaire_analytics/engine/__init__.py`
- `solitaire_analytics/engine/move_generator.py`
- `solitaire_analytics/engine/move_validator.py`
- `solitaire_analytics/solvers/__init__.py`
- `solitaire_analytics/solvers/parallel_solver.py`
- `solitaire_analytics/analysis/__init__.py`
- `solitaire_analytics/analysis/move_tree_builder.py`
- `solitaire_analytics/analysis/dead_end_detector.py`
- `solitaire_analytics/analysis/move_analyzer.py`

---

## Summary

These code quality tasks improve maintainability and developer experience. Recommended order:

1. **Start with:** Task 2 (Linting) - Establish standards
2. **Then:** Task 1 (Type hints) - Enable static analysis
3. **Then:** Task 3 (Docstrings) - Document APIs
4. **Then:** Task 7 (Constants) - Quick wins
5. **Then:** Tasks 5, 6 (Validation & errors) - Robustness
6. **Then:** Tasks 4, 8, 9, 10 - Polish

Each task independently improves code quality.
