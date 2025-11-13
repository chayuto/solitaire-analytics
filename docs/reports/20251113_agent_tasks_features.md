# Coding Agent Tasks - Feature Enhancements

**Date:** November 13, 2025  
**Category:** Features  
**Difficulty Range:** Medium to Hard

## Overview

This document contains self-contained feature enhancement tasks suitable for coding agents. Each task adds new functionality or improves existing features.

---

## Task 1: Add State Hashing for Duplicate Detection

**Difficulty:** Medium  
**Estimated Time:** 4-6 hours  
**Module:** `solitaire_analytics/models/game_state.py`

### Description
Implement state hashing to detect duplicate game positions efficiently.

### Context
The solver can reach the same game state through different move sequences. Detecting duplicates improves performance.

### Requirements
1. Add `__hash__()` method to GameState
2. Add `__eq__()` method for state comparison
3. Create canonical state representation
4. Handle face-down cards appropriately
5. Add tests for hash collisions

### Acceptance Criteria
- [ ] `__hash__()` and `__eq__()` implemented
- [ ] Hash is consistent with equality
- [ ] Hash is fast (< 1ms for typical state)
- [ ] Different move sequences to same state have same hash
- [ ] Tests verify collision resistance
- [ ] Solver updated to use state hashing

### Implementation Guide

```python
# solitaire_analytics/models/game_state.py

from typing import Tuple
import hashlib

class GameState:
    # ... existing code ...
    
    def __hash__(self) -> int:
        """
        Compute hash of game state for duplicate detection.
        
        Returns:
            Integer hash value.
        
        Note:
            Face-down card positions are included in hash,
            but their values are not (since they're unknown).
        """
        return hash(self._canonical_representation())
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two game states are equal.
        
        Args:
            other: Another game state or object.
        
        Returns:
            True if states are equivalent.
        """
        if not isinstance(other, GameState):
            return False
        
        return self._canonical_representation() == other._canonical_representation()
    
    def _canonical_representation(self) -> Tuple:
        """
        Create canonical representation of state for hashing.
        
        Returns:
            Immutable tuple representing the state.
        """
        # Sort foundations by suit for consistent ordering
        foundations = tuple(
            tuple((card.rank, card.suit.value) for card in pile)
            for pile in sorted(self.foundations, key=lambda p: min((c.suit.value for c in p), default=0))
        )
        
        # Tableau piles maintain order
        tableau = tuple(
            tuple(
                (card.rank, card.suit.value, card.face_down)
                for card in pile
            )
            for pile in self.tableau
        )
        
        # Stock and waste
        stock = tuple((card.rank, card.suit.value) for card in self.stock)
        waste = tuple((card.rank, card.suit.value) for card in self.waste)
        
        return (foundations, tableau, stock, waste)
```

### Testing

```python
# tests/test_models.py

def test_state_hash_consistency():
    """Test that same state always produces same hash."""
    state1 = create_test_state()
    state2 = create_test_state()
    
    assert hash(state1) == hash(state2)

def test_state_equality():
    """Test that equivalent states are equal."""
    state1 = create_test_state()
    state2 = create_test_state()
    
    assert state1 == state2

def test_different_states_different_hash():
    """Test that different states produce different hashes."""
    state1 = GameState()
    state2 = GameState()
    state2.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
    
    assert hash(state1) != hash(state2)
    assert state1 != state2

def test_hash_performance():
    """Test that hashing is fast."""
    import time
    state = create_complex_state()
    
    start = time.perf_counter()
    for _ in range(1000):
        _ = hash(state)
    elapsed = time.perf_counter() - start
    
    assert elapsed < 1.0, f"Hashing too slow: {elapsed}s for 1000 hashes"
```

### Integration with Solver

```python
# solitaire_analytics/solvers/parallel_solver.py

def solve(self, initial_state: GameState) -> SolverResult:
    """Solve game with duplicate detection."""
    visited = set()  # Set of state hashes
    queue = [initial_state]
    
    while queue:
        state = queue.pop(0)
        
        # Skip if we've seen this state before
        state_hash = hash(state)
        if state_hash in visited:
            continue
        visited.add(state_hash)
        
        # ... rest of solver logic ...
```

### Related
- Task 2 (Add state serialization)

---

## Task 2: Add Undo/Redo Functionality

**Difficulty:** Medium  
**Estimated Time:** 3-5 hours  
**Module:** `solitaire_analytics/models/game_history.py`

### Description
Implement move history tracking with undo/redo capabilities.

### Context
Users need to explore different move sequences and backtrack. History tracking enables this.

### Requirements
1. Create GameHistory class
2. Track move sequence
3. Implement undo() method
4. Implement redo() method
5. Support branching history

### Acceptance Criteria
- [ ] GameHistory class created
- [ ] Moves tracked automatically
- [ ] Undo restores previous state
- [ ] Redo replays undone moves
- [ ] Tests for all scenarios
- [ ] Memory efficient

### Implementation

```python
# solitaire_analytics/models/game_history.py

from typing import List, Optional
from dataclasses import dataclass
from solitaire_analytics.models import GameState, Move

@dataclass
class HistoryEntry:
    """Single entry in game history."""
    state: GameState
    move: Optional[Move]  # None for initial state
    timestamp: float

class GameHistory:
    """
    Track game state history with undo/redo support.
    
    Example:
        >>> history = GameHistory(initial_state)
        >>> history.apply_move(move1)
        >>> history.apply_move(move2)
        >>> history.undo()  # Back to state after move1
        >>> history.redo()  # Forward to state after move2
    """
    
    def __init__(self, initial_state: GameState):
        """
        Initialize history with initial state.
        
        Args:
            initial_state: Starting game state.
        """
        import time
        self._history: List[HistoryEntry] = [
            HistoryEntry(
                state=initial_state.copy(),
                move=None,
                timestamp=time.time()
            )
        ]
        self._current_index = 0
    
    def apply_move(self, move: Move) -> GameState:
        """
        Apply move and add to history.
        
        Args:
            move: Move to apply.
        
        Returns:
            New game state after move.
        
        Raises:
            ValueError: If move is invalid.
        """
        import time
        from solitaire_analytics.engine import apply_move, validate_move
        
        current_state = self.current_state()
        
        if not validate_move(move, current_state):
            raise ValueError(f"Invalid move: {move}")
        
        new_state = apply_move(current_state, move)
        
        # Remove any forward history (we're creating a new branch)
        self._history = self._history[:self._current_index + 1]
        
        # Add new entry
        self._history.append(
            HistoryEntry(
                state=new_state,
                move=move,
                timestamp=time.time()
            )
        )
        self._current_index += 1
        
        return new_state
    
    def undo(self) -> Optional[GameState]:
        """
        Undo last move.
        
        Returns:
            Previous game state, or None if at start.
        """
        if not self.can_undo():
            return None
        
        self._current_index -= 1
        return self.current_state()
    
    def redo(self) -> Optional[GameState]:
        """
        Redo previously undone move.
        
        Returns:
            Next game state, or None if at end.
        """
        if not self.can_redo():
            return None
        
        self._current_index += 1
        return self.current_state()
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self._current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self._current_index < len(self._history) - 1
    
    def current_state(self) -> GameState:
        """Get current game state."""
        return self._history[self._current_index].state
    
    def move_sequence(self) -> List[Move]:
        """
        Get sequence of moves from start to current position.
        
        Returns:
            List of moves.
        """
        return [
            entry.move
            for entry in self._history[1:self._current_index + 1]
            if entry.move is not None
        ]
    
    def get_statistics(self) -> dict:
        """
        Get history statistics.
        
        Returns:
            Dict with stats (total_states, current_position, etc.)
        """
        return {
            'total_states': len(self._history),
            'current_position': self._current_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'move_count': self._current_index
        }
```

### Testing

```python
# tests/test_game_history.py

import pytest
from solitaire_analytics.models import GameState, Move
from solitaire_analytics.models.game_history import GameHistory

def test_history_initialization():
    """Test history initializes with state."""
    state = GameState()
    history = GameHistory(state)
    
    assert history.current_state() == state
    assert not history.can_undo()
    assert not history.can_redo()

def test_apply_move():
    """Test applying moves updates history."""
    state = GameState()
    history = GameHistory(state)
    move = create_valid_move(state)
    
    new_state = history.apply_move(move)
    
    assert new_state != state
    assert history.can_undo()
    assert not history.can_redo()

def test_undo():
    """Test undo restores previous state."""
    state = GameState()
    history = GameHistory(state)
    move = create_valid_move(state)
    
    new_state = history.apply_move(move)
    previous_state = history.undo()
    
    assert previous_state == state
    assert history.current_state() == state

def test_redo():
    """Test redo replays move."""
    state = GameState()
    history = GameHistory(state)
    move = create_valid_move(state)
    
    new_state = history.apply_move(move)
    history.undo()
    redone_state = history.redo()
    
    assert redone_state == new_state

def test_move_sequence():
    """Test move sequence tracking."""
    state = GameState()
    history = GameHistory(state)
    
    moves = [create_valid_move(state) for _ in range(3)]
    for move in moves:
        history.apply_move(move)
    
    sequence = history.move_sequence()
    assert len(sequence) == 3
    assert sequence == moves
```

---

## Task 3: Add Progress Callbacks for Long Operations

**Difficulty:** Medium  
**Estimated Time:** 3-4 hours  
**Module:** `solitaire_analytics/solvers/parallel_solver.py`

### Description
Add callback system for reporting progress during long-running operations.

### Context
Users need feedback during long solves. Callbacks enable progress bars and status updates.

### Requirements
1. Define callback protocol
2. Add callbacks to solver
3. Report progress at key points
4. Support cancellation via callback
5. Add example with progress bar

### Acceptance Criteria
- [ ] Callback protocol defined
- [ ] Solver calls callback during operation
- [ ] Progress includes meaningful metrics
- [ ] Cancellation works
- [ ] Example script with tqdm
- [ ] Tests verify callback invocation

### Implementation

```python
# solitaire_analytics/types.py

from typing import Protocol, Optional

class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""
    
    def __call__(
        self,
        current: int,
        total: Optional[int],
        message: str,
        **kwargs
    ) -> bool:
        """
        Report progress.
        
        Args:
            current: Current progress value.
            total: Total expected value (None if unknown).
            message: Status message.
            **kwargs: Additional context.
        
        Returns:
            True to continue, False to cancel.
        """
        ...
```

```python
# solitaire_analytics/solvers/parallel_solver.py

from typing import Optional, Callable
from solitaire_analytics.types import ProgressCallback

class ParallelSolver:
    def __init__(
        self,
        max_depth: int = 10,
        n_jobs: int = -1,
        beam_width: int = 100,
        timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """
        Initialize solver.
        
        Args:
            progress_callback: Optional callback for progress updates.
        """
        self.progress_callback = progress_callback
        # ... other init ...
    
    def solve(self, state: GameState) -> SolverResult:
        """Solve with progress reporting."""
        states_explored = 0
        depth = 0
        cancelled = False
        
        while depth < self.max_depth and not cancelled:
            # ... solving logic ...
            
            states_explored += len(current_batch)
            depth += 1
            
            # Report progress
            if self.progress_callback:
                should_continue = self.progress_callback(
                    current=depth,
                    total=self.max_depth,
                    message=f"Depth {depth}/{self.max_depth}",
                    states_explored=states_explored,
                    depth=depth,
                    best_score=best_score
                )
                
                if not should_continue:
                    cancelled = True
                    break
        
        # ... return result ...
```

### Example Usage

```python
# examples/solver_with_progress.py

from tqdm import tqdm
from solitaire_analytics import ParallelSolver, GameState

def create_progress_callback():
    """Create progress callback with tqdm."""
    pbar = tqdm(total=100, desc="Solving")
    
    def callback(current, total, message, **kwargs):
        # Update progress bar
        if total:
            pbar.n = int(100 * current / total)
            pbar.refresh()
        
        # Update description with stats
        if 'states_explored' in kwargs:
            pbar.set_postfix(
                states=kwargs['states_explored'],
                depth=kwargs.get('depth', 0)
            )
        
        # Return True to continue
        return True
    
    return callback, pbar

# Use it
state = GameState()
callback, pbar = create_progress_callback()

solver = ParallelSolver(
    max_depth=20,
    progress_callback=callback
)

result = solver.solve(state)
pbar.close()

print(f"Success: {result.success}")
```

### Testing

```python
def test_progress_callback_called():
    """Test that progress callback is invoked."""
    calls = []
    
    def callback(current, total, message, **kwargs):
        calls.append({
            'current': current,
            'total': total,
            'message': message
        })
        return True  # Continue
    
    solver = ParallelSolver(max_depth=5, progress_callback=callback)
    solver.solve(simple_state)
    
    assert len(calls) > 0
    assert all(c['current'] <= c['total'] for c in calls)

def test_progress_callback_cancellation():
    """Test that returning False cancels operation."""
    def callback(current, total, message, **kwargs):
        return current < 3  # Cancel after 3 iterations
    
    solver = ParallelSolver(max_depth=10, progress_callback=callback)
    result = solver.solve(state)
    
    # Should have stopped early
    assert result.depth_reached < 10
```

---

## Task 4: Add JSON Import/Export for Game States

**Difficulty:** Easy-Medium  
**Estimated Time:** 3-4 hours  
**Module:** `solitaire_analytics/models/game_state.py`

### Description
Add methods to serialize/deserialize game states to/from JSON.

### Context
Users need to save and load game states for testing and analysis.

### Requirements
1. Add `to_json()` method
2. Add `from_json()` classmethod
3. Add `to_file()` method
4. Add `from_file()` classmethod
5. Handle all game state components

### Acceptance Criteria
- [ ] JSON serialization works for all states
- [ ] Deserialization reconstructs exact state
- [ ] File I/O methods work
- [ ] Error handling for invalid JSON
- [ ] Tests for round-trip conversion

### Implementation

```python
# solitaire_analytics/models/game_state.py

import json
from typing import Dict, Any, List
from pathlib import Path

class GameState:
    # ... existing code ...
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert game state to JSON-serializable dict.
        
        Returns:
            Dict representation of state.
        
        Example:
            >>> state = GameState()
            >>> json_data = state.to_json()
            >>> json.dump(json_data, open('state.json', 'w'))
        """
        return {
            'tableau': [
                [card.to_dict() for card in pile]
                for pile in self.tableau
            ],
            'foundations': [
                [card.to_dict() for card in pile]
                for pile in self.foundations
            ],
            'stock': [card.to_dict() for card in self.stock],
            'waste': [card.to_dict() for card in self.waste],
            'metadata': {
                'version': '1.0',
                'card_count': self.card_count()
            }
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'GameState':
        """
        Create game state from JSON data.
        
        Args:
            data: JSON data dict.
        
        Returns:
            GameState instance.
        
        Raises:
            ValueError: If data is invalid.
        
        Example:
            >>> with open('state.json') as f:
            ...     data = json.load(f)
            >>> state = GameState.from_json(data)
        """
        state = cls()
        
        # Validate version
        if 'metadata' in data:
            version = data['metadata'].get('version', '1.0')
            if version != '1.0':
                raise ValueError(f"Unsupported format version: {version}")
        
        # Load tableau
        state.tableau = [
            [Card.from_dict(card_data) for card_data in pile]
            for pile in data.get('tableau', [])
        ]
        
        # Load foundations
        state.foundations = [
            [Card.from_dict(card_data) for card_data in pile]
            for pile in data.get('foundations', [])
        ]
        
        # Load stock and waste
        state.stock = [
            Card.from_dict(card_data) 
            for card_data in data.get('stock', [])
        ]
        state.waste = [
            Card.from_dict(card_data)
            for card_data in data.get('waste', [])
        ]
        
        # Validate
        if not state.is_valid():
            raise ValueError("Loaded state is invalid")
        
        return state
    
    def to_file(self, filepath: str) -> None:
        """
        Save game state to JSON file.
        
        Args:
            filepath: Path to output file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f, indent=2)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GameState':
        """
        Load game state from JSON file.
        
        Args:
            filepath: Path to input file.
        
        Returns:
            GameState instance.
        
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file contains invalid data.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath) as f:
            data = json.load(f)
        
        return cls.from_json(data)
```

```python
# solitaire_analytics/models/card.py

class Card:
    # ... existing code ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert card to dict."""
        return {
            'rank': self.rank,
            'suit': self.suit.value,
            'face_down': self.face_down
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Card':
        """Create card from dict."""
        from solitaire_analytics.models.card import Suit
        return cls(
            rank=data['rank'],
            suit=Suit(data['suit']),
            face_down=data.get('face_down', False)
        )
```

### Testing

```python
def test_state_to_json():
    """Test state serialization."""
    state = create_test_state()
    json_data = state.to_json()
    
    assert 'tableau' in json_data
    assert 'foundations' in json_data
    assert 'stock' in json_data
    assert 'waste' in json_data

def test_state_from_json():
    """Test state deserialization."""
    original = create_test_state()
    json_data = original.to_json()
    restored = GameState.from_json(json_data)
    
    assert restored == original

def test_state_round_trip():
    """Test serialization round trip."""
    original = create_complex_state()
    
    # To JSON and back
    json_data = original.to_json()
    restored = GameState.from_json(json_data)
    
    # Should be identical
    assert restored.tableau == original.tableau
    assert restored.foundations == original.foundations
    assert restored.stock == original.stock
    assert restored.waste == original.waste

def test_state_file_io():
    """Test file save/load."""
    import tempfile
    
    original = create_test_state()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    try:
        # Save and load
        original.to_file(filepath)
        restored = GameState.from_file(filepath)
        
        assert restored == original
    finally:
        Path(filepath).unlink()
```

---

## Task 5: Add Move Sequence Validation

**Difficulty:** Easy  
**Estimated Time:** 2-3 hours  
**Module:** `solitaire_analytics/engine/move_validator.py`

### Description
Add function to validate an entire sequence of moves.

### Context
Users need to verify that a sequence of moves is legal before applying them.

### Requirements
1. Create `validate_move_sequence()` function
2. Check each move in sequence
3. Track intermediate states
4. Return detailed error on failure
5. Optimize for performance

### Acceptance Criteria
- [ ] Function validates sequences
- [ ] Returns True for valid sequences
- [ ] Returns False with details for invalid
- [ ] Efficient (doesn't clone unnecessarily)
- [ ] Tests for valid and invalid sequences

### Implementation

```python
# solitaire_analytics/engine/move_validator.py

from typing import List, Tuple, Optional
from solitaire_analytics.models import GameState, Move

def validate_move_sequence(
    state: GameState,
    moves: List[Move],
    strict: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate a sequence of moves.
    
    Args:
        state: Initial game state.
        moves: Sequence of moves to validate.
        strict: If True, stop at first error. If False, check all.
    
    Returns:
        Tuple of (is_valid, error_message).
        error_message is None if valid, otherwise describes the error.
    
    Example:
        >>> state = GameState()
        >>> moves = [move1, move2, move3]
        >>> valid, error = validate_move_sequence(state, moves)
        >>> if not valid:
        ...     print(f"Invalid sequence: {error}")
    """
    if not moves:
        return True, None
    
    current_state = state
    
    for i, move in enumerate(moves):
        if not validate_move(move, current_state):
            error = (
                f"Invalid move at position {i}: {move}. "
                f"Move cannot be performed in current state."
            )
            return False, error
        
        # Apply move for next iteration
        from solitaire_analytics.engine import apply_move
        current_state = apply_move(current_state, move)
    
    return True, None

def replay_move_sequence(
    state: GameState,
    moves: List[Move]
) -> List[GameState]:
    """
    Replay a move sequence and return all intermediate states.
    
    Args:
        state: Initial game state.
        moves: Sequence of moves.
    
    Returns:
        List of states (including initial state).
    
    Raises:
        ValueError: If any move is invalid.
    """
    states = [state]
    current_state = state
    
    for i, move in enumerate(moves):
        if not validate_move(move, current_state):
            raise ValueError(
                f"Invalid move at position {i}: {move}"
            )
        
        from solitaire_analytics.engine import apply_move
        current_state = apply_move(current_state, move)
        states.append(current_state)
    
    return states
```

### Testing

```python
def test_validate_empty_sequence():
    """Test that empty sequence is valid."""
    state = GameState()
    valid, error = validate_move_sequence(state, [])
    
    assert valid
    assert error is None

def test_validate_valid_sequence():
    """Test valid move sequence."""
    state = create_test_state()
    moves = [
        create_valid_move_1(state),
        create_valid_move_2(state),
    ]
    
    valid, error = validate_move_sequence(state, moves)
    
    assert valid
    assert error is None

def test_validate_invalid_sequence():
    """Test invalid move sequence."""
    state = create_test_state()
    moves = [
        create_valid_move(state),
        create_invalid_move(state),  # This one will fail
    ]
    
    valid, error = validate_move_sequence(state, moves)
    
    assert not valid
    assert error is not None
    assert "position 1" in error

def test_replay_move_sequence():
    """Test replaying move sequence."""
    state = create_test_state()
    moves = create_valid_move_sequence(state)
    
    states = replay_move_sequence(state, moves)
    
    assert len(states) == len(moves) + 1  # Initial + after each move
    assert states[0] == state
```

---

## Summary

These feature tasks add significant new capabilities. Recommended order:

1. **Start with:** Task 4 (JSON I/O) - Useful immediately
2. **Then:** Task 5 (Sequence validation) - Build on basics
3. **Then:** Task 3 (Progress callbacks) - User experience
4. **Then:** Task 2 (Undo/redo) - More complex
5. **Then:** Task 1 (State hashing) - Performance optimization

Each task is independent and adds real value to the project.
