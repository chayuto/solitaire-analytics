# Coding Agent Tasks - Error Handling Improvements

**Date:** November 13, 2025  
**Category:** Error Handling  
**Difficulty Range:** Easy to Medium

## Overview

This document contains self-contained error handling tasks suitable for coding agents. Each task improves robustness and user experience when errors occur.

---

## Task 1: Create Custom Exception Hierarchy

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `solitaire_analytics/exceptions.py`

### Description
Create a comprehensive hierarchy of custom exceptions for better error handling.

### Context
Generic exceptions like ValueError don't convey enough context. Custom exceptions enable better error handling and clearer error messages.

### Requirements
1. Create base exception class
2. Add specific exception types
3. Include context in exceptions
4. Document when each is raised
5. Update code to use new exceptions

### Acceptance Criteria
- [ ] Exception module created
- [ ] At least 8 exception types defined
- [ ] All exceptions have docstrings
- [ ] Code updated to use new exceptions
- [ ] Tests verify exception raising

### Implementation

```python
# solitaire_analytics/exceptions.py

"""Custom exceptions for Solitaire Analytics."""

class SolitaireError(Exception):
    """Base exception for all Solitaire Analytics errors."""
    pass

class GameStateError(SolitaireError):
    """Base exception for game state errors."""
    pass

class InvalidStateError(GameStateError):
    """
    Raised when game state is invalid or corrupted.
    
    Attributes:
        state: The invalid game state (if available).
        validation_errors: List of specific validation errors.
    """
    
    def __init__(self, message: str, state=None, validation_errors=None):
        """
        Initialize exception.
        
        Args:
            message: Error message.
            state: The invalid state (optional).
            validation_errors: List of validation errors (optional).
        """
        super().__init__(message)
        self.state = state
        self.validation_errors = validation_errors or []
    
    def __str__(self):
        """Format error message with details."""
        msg = super().__str__()
        if self.validation_errors:
            errors = "\n  - ".join(self.validation_errors)
            msg += f"\n\nValidation errors:\n  - {errors}"
        return msg

class InvalidMoveError(GameStateError):
    """
    Raised when a move is invalid according to game rules.
    
    Attributes:
        move: The invalid move.
        state: The game state where move was attempted.
        reason: Explanation of why move is invalid.
    """
    
    def __init__(self, message: str, move=None, state=None, reason=None):
        """
        Initialize exception.
        
        Args:
            message: Error message.
            move: The invalid move (optional).
            state: Game state (optional).
            reason: Detailed reason (optional).
        """
        super().__init__(message)
        self.move = move
        self.state = state
        self.reason = reason
    
    def __str__(self):
        """Format error message with details."""
        msg = super().__str__()
        if self.reason:
            msg += f"\nReason: {self.reason}"
        if self.move:
            msg += f"\nMove: {self.move}"
        return msg

class CardError(SolitaireError):
    """Base exception for card-related errors."""
    pass

class InvalidCardError(CardError):
    """
    Raised when card has invalid attributes.
    
    Attributes:
        rank: The invalid rank.
        suit: The invalid suit.
    """
    
    def __init__(self, message: str, rank=None, suit=None):
        """Initialize exception."""
        super().__init__(message)
        self.rank = rank
        self.suit = suit

class SolverError(SolitaireError):
    """Base exception for solver errors."""
    pass

class SolverTimeoutError(SolverError):
    """
    Raised when solver exceeds timeout.
    
    Attributes:
        timeout: The timeout value that was exceeded.
        states_explored: Number of states explored before timeout.
    """
    
    def __init__(self, message: str, timeout=None, states_explored=0):
        """Initialize exception."""
        super().__init__(message)
        self.timeout = timeout
        self.states_explored = states_explored

class SolverConfigError(SolverError):
    """
    Raised when solver configuration is invalid.
    
    Attributes:
        config_name: Name of invalid configuration parameter.
        config_value: The invalid value.
    """
    
    def __init__(self, message: str, config_name=None, config_value=None):
        """Initialize exception."""
        super().__init__(message)
        self.config_name = config_name
        self.config_value = config_value

class AnalysisError(SolitaireError):
    """Base exception for analysis errors."""
    pass

class TreeBuildError(AnalysisError):
    """
    Raised when tree building fails.
    
    Attributes:
        nodes_created: Number of nodes created before failure.
    """
    
    def __init__(self, message: str, nodes_created=0):
        """Initialize exception."""
        super().__init__(message)
        self.nodes_created = nodes_created

class SerializationError(SolitaireError):
    """
    Raised when serialization/deserialization fails.
    
    Attributes:
        data: The data that failed to serialize/deserialize.
        format: The format (e.g., 'json', 'binary').
    """
    
    def __init__(self, message: str, data=None, format=None):
        """Initialize exception."""
        super().__init__(message)
        self.data = data
        self.format = format

class ResourceError(SolitaireError):
    """
    Raised when system resources are insufficient.
    
    Attributes:
        resource: Name of the resource (e.g., 'memory', 'gpu').
    """
    
    def __init__(self, message: str, resource=None):
        """Initialize exception."""
        super().__init__(message)
        self.resource = resource
```

### Usage Examples

```python
# In move_validator.py
from solitaire_analytics.exceptions import InvalidMoveError

def validate_move(move: Move, state: GameState) -> bool:
    """Validate move."""
    if not can_place_on_foundation(move, state):
        raise InvalidMoveError(
            f"Cannot place card on foundation",
            move=move,
            state=state,
            reason="Card rank must be one higher than foundation top"
        )
    return True

# In game_state.py
from solitaire_analytics.exceptions import InvalidStateError

def validate(self) -> None:
    """Validate game state."""
    errors = []
    
    if self.card_count() != 52:
        errors.append(f"Expected 52 cards, found {self.card_count()}")
    
    if self._has_duplicates():
        errors.append("Duplicate cards found")
    
    if errors:
        raise InvalidStateError(
            "Game state is invalid",
            state=self,
            validation_errors=errors
        )
```

### Testing

```python
# tests/test_exceptions.py

def test_invalid_move_error():
    """Test InvalidMoveError includes context."""
    move = create_test_move()
    state = create_test_state()
    
    try:
        raise InvalidMoveError(
            "Test error",
            move=move,
            state=state,
            reason="Test reason"
        )
    except InvalidMoveError as e:
        assert e.move == move
        assert e.state == state
        assert e.reason == "Test reason"
        assert "Test reason" in str(e)

def test_invalid_state_error():
    """Test InvalidStateError formats errors."""
    errors = ["Error 1", "Error 2"]
    
    try:
        raise InvalidStateError(
            "State invalid",
            validation_errors=errors
        )
    except InvalidStateError as e:
        assert len(e.validation_errors) == 2
        err_str = str(e)
        assert "Error 1" in err_str
        assert "Error 2" in err_str
```

---

## Task 2: Add Input Validation Functions

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `solitaire_analytics/validation.py`

### Description
Create reusable validation functions for common input checks.

### Context
Validation logic is scattered across the codebase. Centralized validation improves consistency.

### Requirements
1. Create validation module
2. Add validation functions for common types
3. Include clear error messages
4. Add type hints
5. Use in existing code

### Acceptance Criteria
- [ ] Validation module created
- [ ] At least 10 validation functions
- [ ] All functions have docstrings
- [ ] Tests for each function
- [ ] Code updated to use validators

### Implementation

```python
# solitaire_analytics/validation.py

"""Input validation utilities."""

from typing import Any, List, Optional, Type
from solitaire_analytics.exceptions import InvalidCardError, SolverConfigError

def validate_card_rank(rank: int) -> None:
    """
    Validate card rank is in valid range (1-13).
    
    Args:
        rank: Card rank to validate.
    
    Raises:
        InvalidCardError: If rank is invalid.
    """
    if not isinstance(rank, int):
        raise InvalidCardError(
            f"Card rank must be an integer, got {type(rank).__name__}",
            rank=rank
        )
    
    if rank < 1 or rank > 13:
        raise InvalidCardError(
            f"Card rank must be 1-13, got {rank}",
            rank=rank
        )

def validate_positive_int(value: Any, name: str) -> None:
    """
    Validate value is a positive integer.
    
    Args:
        value: Value to validate.
        name: Parameter name (for error message).
    
    Raises:
        ValueError: If value is invalid.
    """
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    
    if value <= 0:
        raise ValueError(
            f"{name} must be positive, got {value}"
        )

def validate_range(
    value: int,
    min_val: int,
    max_val: int,
    name: str
) -> None:
    """
    Validate value is in specified range.
    
    Args:
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Parameter name (for error message).
    
    Raises:
        ValueError: If value is out of range.
    """
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    
    if value < min_val or value > max_val:
        raise ValueError(
            f"{name} must be {min_val}-{max_val}, got {value}"
        )

def validate_not_none(value: Any, name: str) -> None:
    """
    Validate value is not None.
    
    Args:
        value: Value to validate.
        name: Parameter name (for error message).
    
    Raises:
        ValueError: If value is None.
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")

def validate_type(
    value: Any,
    expected_type: Type,
    name: str
) -> None:
    """
    Validate value is of expected type.
    
    Args:
        value: Value to validate.
        expected_type: Expected type.
        name: Parameter name (for error message).
    
    Raises:
        TypeError: If value has wrong type.
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

def validate_timeout(timeout: Optional[float]) -> None:
    """
    Validate timeout value.
    
    Args:
        timeout: Timeout in seconds (None = no timeout).
    
    Raises:
        ValueError: If timeout is invalid.
    """
    if timeout is None:
        return
    
    if not isinstance(timeout, (int, float)):
        raise ValueError(
            f"Timeout must be a number, got {type(timeout).__name__}"
        )
    
    if timeout <= 0:
        raise ValueError(f"Timeout must be positive, got {timeout}")

def validate_beam_width(beam_width: int) -> None:
    """
    Validate beam width parameter.
    
    Args:
        beam_width: Beam width value.
    
    Raises:
        SolverConfigError: If beam width is invalid.
    """
    if not isinstance(beam_width, int):
        raise SolverConfigError(
            f"beam_width must be an integer",
            config_name="beam_width",
            config_value=beam_width
        )
    
    if beam_width < 1:
        raise SolverConfigError(
            f"beam_width must be at least 1, got {beam_width}",
            config_name="beam_width",
            config_value=beam_width
        )
    
    if beam_width > 10000:
        raise SolverConfigError(
            f"beam_width > 10000 may cause memory issues. Got {beam_width}",
            config_name="beam_width",
            config_value=beam_width
        )

def validate_max_depth(max_depth: int) -> None:
    """
    Validate max depth parameter.
    
    Args:
        max_depth: Maximum search depth.
    
    Raises:
        SolverConfigError: If max depth is invalid.
    """
    validate_range(max_depth, 1, 100, "max_depth")

def validate_list_not_empty(lst: List, name: str) -> None:
    """
    Validate list is not empty.
    
    Args:
        lst: List to validate.
        name: Parameter name (for error message).
    
    Raises:
        ValueError: If list is empty.
    """
    if not isinstance(lst, list):
        raise ValueError(
            f"{name} must be a list, got {type(lst).__name__}"
        )
    
    if len(lst) == 0:
        raise ValueError(f"{name} cannot be empty")

def validate_pile_index(index: int, max_index: int, pile_type: str) -> None:
    """
    Validate pile index is in valid range.
    
    Args:
        index: Pile index.
        max_index: Maximum valid index.
        pile_type: Type of pile (for error message).
    
    Raises:
        ValueError: If index is invalid.
    """
    if not isinstance(index, int):
        raise ValueError(
            f"{pile_type} index must be an integer, got {type(index).__name__}"
        )
    
    if index < 0 or index >= max_index:
        raise ValueError(
            f"{pile_type} index must be 0-{max_index-1}, got {index}"
        )
```

### Usage

```python
# In card.py
from solitaire_analytics.validation import validate_card_rank

class Card:
    def __init__(self, rank: int, suit: Suit):
        validate_card_rank(rank)
        self.rank = rank
        self.suit = suit

# In parallel_solver.py
from solitaire_analytics.validation import (
    validate_max_depth,
    validate_beam_width,
    validate_timeout
)

class ParallelSolver:
    def __init__(self, max_depth=10, beam_width=100, timeout=None):
        validate_max_depth(max_depth)
        validate_beam_width(beam_width)
        validate_timeout(timeout)
        
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.timeout = timeout
```

---

## Task 3: Add Graceful Error Recovery

**Difficulty:** Medium  
**Estimated Time:** 3-4 hours  
**Module:** Various

### Description
Add try-except blocks with graceful fallbacks for common error scenarios.

### Context
Some operations can fail in predictable ways. Graceful recovery improves user experience.

### Requirements
1. Identify operations that can fail
2. Add try-except blocks
3. Implement fallback strategies
4. Log errors appropriately
5. Test error recovery

### Acceptance Criteria
- [ ] GPU unavailable falls back to CPU
- [ ] File I/O errors handled gracefully
- [ ] Network errors (if any) handled
- [ ] Errors logged appropriately
- [ ] Tests verify recovery

### Implementation

```python
# solitaire_analytics/solvers/parallel_solver.py

import logging

logger = logging.getLogger(__name__)

class ParallelSolver:
    def __init__(self, use_gpu: bool = False, **kwargs):
        """Initialize solver with GPU fallback."""
        self.use_gpu = use_gpu
        self._gpu_available = False
        
        if use_gpu:
            try:
                import torch
                self._gpu_available = torch.cuda.is_available()
                if self._gpu_available:
                    logger.info("GPU acceleration enabled")
                else:
                    logger.warning(
                        "GPU requested but not available. "
                        "Falling back to CPU."
                    )
            except ImportError:
                logger.warning(
                    "PyTorch not available for GPU acceleration. "
                    "Falling back to CPU."
                )
                self._gpu_available = False
    
    def solve(self, state: GameState) -> SolverResult:
        """Solve with automatic fallback."""
        try:
            if self._gpu_available:
                return self._solve_gpu(state)
            else:
                return self._solve_cpu(state)
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                logger.error(
                    f"GPU solver failed: {e}. "
                    "Falling back to CPU solver."
                )
                self._gpu_available = False
                return self._solve_cpu(state)
            else:
                raise
```

```python
# solitaire_analytics/models/game_state.py

class GameState:
    @classmethod
    def from_file(cls, filepath: str) -> 'GameState':
        """Load game state with error handling."""
        from pathlib import Path
        import json
        
        path = Path(filepath)
        
        # Check file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Game state file not found: {filepath}\n"
                f"Current directory: {Path.cwd()}\n"
                f"Please verify the file path is correct."
            )
        
        # Check file is readable
        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {filepath}\n"
                f"Please provide a path to a JSON file."
            )
        
        # Load and parse JSON
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SerializationError(
                f"Invalid JSON in file: {filepath}\n"
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}\n"
                f"Please verify the file contains valid JSON.",
                format="json"
            )
        except Exception as e:
            raise SerializationError(
                f"Error reading file: {filepath}\n"
                f"Error: {e}\n"
                f"Please verify file permissions and format.",
                format="json"
            )
        
        # Parse game state
        try:
            return cls.from_json(data)
        except (KeyError, ValueError) as e:
            raise SerializationError(
                f"Invalid game state format in file: {filepath}\n"
                f"Error: {e}\n"
                f"Please verify the file contains a valid game state.",
                format="json",
                data=data
            )
```

---

## Task 4: Add Detailed Error Context

**Difficulty:** Easy-Medium  
**Estimated Time:** 3-4 hours  
**Module:** Various

### Description
Enhance error messages with detailed context to aid debugging.

### Context
Generic error messages make debugging difficult. Detailed context helps users fix issues quickly.

### Requirements
1. Include relevant values in errors
2. Suggest solutions where possible
3. Add links to documentation
4. Format errors for readability
5. Update all error sites

### Acceptance Criteria
- [ ] All errors include context
- [ ] Solutions suggested where applicable
- [ ] Error messages are clear and actionable
- [ ] Formatted for easy reading
- [ ] Examples in documentation

### Implementation

```python
# Example: Enhanced error messages

# Before
raise ValueError("Invalid beam width")

# After
raise SolverConfigError(
    f"Invalid beam_width: {beam_width}\n"
    f"\n"
    f"beam_width must be between 1 and 10000.\n"
    f"Higher values explore more states but use more memory.\n"
    f"\n"
    f"Suggestions:\n"
    f"  - For quick analysis: beam_width=50\n"
    f"  - For thorough analysis: beam_width=200\n"
    f"  - For deep search: beam_width=500\n"
    f"\n"
    f"See documentation: https://docs.example.com/solver-config",
    config_name="beam_width",
    config_value=beam_width
)

# Before
raise ValueError("Move invalid")

# After
raise InvalidMoveError(
    f"Cannot move {card} from {source} to {destination}\n"
    f"\n"
    f"Current state:\n"
    f"  Source pile: {source} has {len(source_pile)} cards\n"
    f"  Destination pile: {destination} has {len(dest_pile)} cards\n"
    f"  Top card at destination: {dest_top_card if dest_pile else 'empty'}\n"
    f"\n"
    f"Rule violation:\n"
    f"  {violation_reason}\n"
    f"\n"
    f"Valid moves from this position: {len(valid_moves)} available\n"
    f"Hint: Use generate_moves() to see all legal moves.",
    move=move,
    state=state,
    reason=violation_reason
)
```

---

## Summary

These error handling tasks improve reliability and user experience. Recommended order:

1. **Start with:** Task 1 (Custom exceptions) - Foundation
2. **Then:** Task 2 (Validation functions) - Build on exceptions
3. **Then:** Task 4 (Error context) - Enhance messages
4. **Then:** Task 3 (Recovery) - Advanced robustness

Each task makes the codebase more robust and user-friendly.
