# PlayLogger Documentation

## Overview

The `PlayLogger` class provides a configurable option to record game sessions for replay or visualization. It captures the initial game state, all moves with timestamps, and optional metadata about the game session.

## Quick Start

```python
from solitaire_analytics import PlayLogger, GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move

# Create logger (disabled by default)
logger = PlayLogger(enabled=True, metadata={"player": "Alice"})

# Start recording
state = GameState()
logger.start(state)

# Log moves
moves = generate_moves(state)
if moves:
    move = moves[0]
    new_state = apply_move(state, move)
    if new_state:
        logger.log_move(move, resulting_state=new_state)

# Save to file
logger.save('game_log.json')
```

## Configuration

### Enabling the Logger

By default, the logger is **disabled** for efficiency. Enable it explicitly:

```python
# Disabled (default) - no overhead
logger = PlayLogger()

# Enabled - records all moves
logger = PlayLogger(enabled=True)
```

### Adding Metadata

Include custom metadata about the game session:

```python
logger = PlayLogger(
    enabled=True,
    metadata={
        "player": "Alice",
        "session_id": "abc123",
        "game_type": "klondike",
        "difficulty": "hard",
        "timestamp": "2024-11-13T12:00:00Z"
    }
)
```

## Usage

### Starting a Game Session

Call `start()` with the initial game state before logging any moves:

```python
initial_state = GameState()
# ... set up game state ...
logger.start(initial_state)
```

### Logging Moves

Log moves as they occur during gameplay:

```python
# Basic logging (just the move)
logger.log_move(move)

# Include resulting state for full replay capability
logger.log_move(move, resulting_state=new_state)
```

Each logged move includes:
- **timestamp**: Relative time in seconds since game start
- **move**: Complete move details (type, source, destination, etc.)
- **resulting_state** (optional): The game state after applying the move

### Exporting Logs

Export to dictionary or JSON:

```python
# Get as dictionary
log_dict = logger.to_dict()

# Get as JSON string
json_str = logger.to_json()

# Save to file
logger.save('path/to/log.json')
```

### Loading Logs

Load a previously saved log:

```python
logger = PlayLogger.load('path/to/log.json')
print(f"Game had {len(logger.moves)} moves")
```

### Clearing Data

Clear all logged data:

```python
logger.clear()
```

## Log Format

The exported log follows this JSON structure:

```json
{
  "enabled": true,
  "metadata": {
    "player": "Alice",
    "session_id": "game_001"
  },
  "initial_state": {
    "tableau": [...],
    "foundations": [...],
    "stock": [...],
    "waste": [...],
    "move_count": 0,
    "score": 0
  },
  "moves": [
    {
      "timestamp": 0.123,
      "move": {
        "move_type": "tableau_to_foundation",
        "source_pile": 0,
        "dest_pile": 0,
        "num_cards": 1,
        "score_delta": 10,
        "description": "Move card from tableau 0 to foundation 0"
      },
      "resulting_state": {
        // Optional: full state after move
      }
    }
  ],
  "move_count": 1,
  "start_time": "2024-11-13T18:30:00.000000"
}
```

### Fields

- **enabled**: Whether logging was active
- **metadata**: Custom metadata dictionary
- **initial_state**: Complete game state at start
- **moves**: Array of move records with timestamps
- **move_count**: Total number of moves
- **start_time**: ISO format timestamp of game start

### Move Types

The logger records all Solitaire move types:
- `stock_to_waste`: Drawing from stock pile
- `tableau_to_foundation`: Moving to foundation
- `tableau_to_tableau`: Moving between tableau piles
- `waste_to_tableau`: Moving from waste to tableau
- `waste_to_foundation`: Moving waste to foundation
- `flip_tableau_card`: Flipping a tableau card face-up
- `foundation_to_tableau`: Moving from foundation (for undo)

## Use Cases

### Game Replay

Record complete game sessions for replay:

```python
logger = PlayLogger(enabled=True)
logger.start(initial_state)

# Play game...
for move in game_moves:
    new_state = apply_move(state, move)
    logger.log_move(move, resulting_state=new_state)
    state = new_state

# Save for replay
logger.save('replay.json')
```

### Visualization

Export logs for external visualization tools:

```python
logger = PlayLogger(
    enabled=True,
    metadata={"visualizer": "solitaire-viz-v1"}
)
logger.start(state)
# ... play game ...
logger.save('visualization.json')
```

### Analysis

Analyze gameplay patterns:

```python
logger = PlayLogger.load('game.json')

# Analyze timing
move_times = [m['timestamp'] for m in logger.moves]
avg_move_time = sum(move_times) / len(move_times)

# Analyze move types
move_types = [m['move']['move_type'] for m in logger.moves]
most_common = max(set(move_types), key=move_types.count)
```

### Debugging

Debug game logic by recording problematic sessions:

```python
logger = PlayLogger(
    enabled=True,
    metadata={"debug": True, "issue": "stuck_state"}
)
logger.start(problematic_state)
# ... reproduce issue ...
logger.save('debug_log.json')
```

## Performance Considerations

### When Disabled (Default)

- **Zero overhead**: All logging operations are no-ops
- No memory allocation for logs
- No timestamp calculations
- Recommended for production/analysis runs

### When Enabled

- Minimal overhead per move (~microseconds)
- Memory usage scales with game length
- Optional state recording increases memory usage
- Recommended for:
  - Recording games for replay
  - Debugging specific sessions
  - Analysis of individual games

### Best Practices

1. **Keep logging disabled by default** - Only enable when needed
2. **Use metadata wisely** - Add context without excessive data
3. **Selective state recording** - Only include resulting states when needed for replay
4. **Clean up logs** - Remove old logs to save disk space
5. **Batch operations** - Save logs after game completion, not after each move

## Integration Examples

### With Strategies

```python
from solitaire_analytics.strategies import get_strategy

logger = PlayLogger(enabled=True)
strategy = get_strategy("lookahead")

state = GameState()
logger.start(state)

while not state.is_won():
    move = strategy.select_best_move(state)
    if not move:
        break
    new_state = apply_move(state, move)
    if new_state:
        logger.log_move(move, resulting_state=new_state)
        state = new_state

logger.save('strategy_game.json')
```

### With ParallelSolver

```python
from solitaire_analytics import ParallelSolver

logger = PlayLogger(
    enabled=True,
    metadata={"solver": "parallel", "max_depth": 50}
)

state = GameState()
logger.start(state)

solver = ParallelSolver(max_depth=50)
result = solver.solve(state)

if result.success:
    for move in result.moves:
        state = apply_move(state, move)
        logger.log_move(move, resulting_state=state)

logger.save('solver_solution.json')
```

## API Reference

### Class: PlayLogger

#### Constructor

```python
PlayLogger(enabled: bool = False, metadata: Optional[Dict] = None)
```

- **enabled**: Enable logging (default: False)
- **metadata**: Optional metadata dictionary

#### Methods

##### start(initial_state: GameState) -> None

Begin logging with initial game state.

##### log_move(move: Move, resulting_state: Optional[GameState] = None) -> None

Log a move with timestamp and optional resulting state.

##### to_dict() -> Dict

Export log as dictionary.

##### to_json(indent: int = 2) -> str

Export log as JSON string.

##### save(filepath: str) -> None

Save log to JSON file.

##### clear() -> None

Clear all logged data.

##### load(filepath: str) -> PlayLogger (classmethod)

Load log from JSON file.

#### Attributes

- **enabled**: Whether logging is active
- **initial_state**: Initial game state
- **moves**: List of move records
- **metadata**: Metadata dictionary

## See Also

- Example script: `scripts/example_play_logger.py`
- Tests: `tests/test_play_logger.py`
- GameState documentation: `docs/game_state.md` (if exists)
- Move documentation: `docs/moves.md` (if exists)
