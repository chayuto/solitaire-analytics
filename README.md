# Solitaire Analytics Engine

A Python 3.12 analytics engine for Solitaire games with parallel processing support and comprehensive game analysis tools.

## Features

- **Core Models**: Card, GameState, and Move representations
- **Game Engine**: Move generation and validation logic
- **Move Selection Strategies**: Extensible framework for intelligent move selection
  - Simple greedy strategy
  - Weighted priority-based strategy
  - Lookahead sequence evaluation strategy
  - LLM-based strategy with OpenAI integration
  - Easy to create custom strategies
- **Parallel Solver**: CPU and GPU-accelerated game solving with beam search
- **Analysis Tools**:
  - Move tree builder for exploring game state space
  - Dead end detector for identifying unwinnable positions
  - Move sequence analyzer for finding optimal play paths
- **JSON Reports**: Comprehensive game state and analysis reports
- **Testing**: Comprehensive test suite with pytest markers

## Installation

```bash
# Clone the repository
git clone https://github.com/chayuto/solitaire-analytics.git
cd solitaire-analytics

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from solitaire_analytics import Card, GameState, generate_moves
from solitaire_analytics.models.card import Suit
from solitaire_analytics import ParallelSolver

# Create a game state
state = GameState()
state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))

# Generate possible moves
moves = generate_moves(state)
print(f"Found {len(moves)} possible moves")

# Solve the game
solver = ParallelSolver(max_depth=10, n_jobs=-1)
result = solver.solve(state)
print(f"Solution found: {result.success}")
```

## Project Structure

```
solitaire-analytics/
├── solitaire_analytics/
│   ├── models/          # Core data models (Card, GameState, Move)
│   ├── engine/          # Game engine (generate_moves, validate_move)
│   ├── strategies/      # Move selection strategies (NEW!)
│   ├── solvers/         # ParallelSolver with CPU+GPU support
│   └── analysis/        # MoveTreeBuilder, DeadEndDetector, analyzers
├── notebooks/           # Jupyter notebooks with examples
├── tests/               # Comprehensive test suite
├── scripts/             # Example scripts
└── requirements.txt     # Project dependencies
```

## Dependencies

- **torch** (CPU version): Neural network framework for GPU acceleration
- **joblib**: Parallel processing
- **networkx**: Graph-based move tree representation
- **numpy**: Numerical operations
- **pandas**: Data analysis
- **openai**: OpenAI API client for LLM-based strategy
- **jupyter**: Interactive notebooks
- **pytest**: Testing framework

## Usage Examples

### Using Move Selection Strategies

```python
from solitaire_analytics.strategies import get_strategy, StrategyConfig

# Simple greedy strategy
strategy = get_strategy("simple")
best_move = strategy.select_best_move(state)
print(f"Best move: {best_move}")

# Weighted strategy with custom priorities
config = StrategyConfig(
    priorities={
        "tableau_to_foundation": 200.0,
        "reveals_card": 50.0,
    }
)
strategy = get_strategy("weighted", config)
best_move = strategy.select_best_move(state)

# Lookahead strategy
config = StrategyConfig(max_depth=5)
strategy = get_strategy("lookahead", config)
sequence = strategy.select_move_sequence(state, length=5)
print(f"Best sequence: {[str(m) for m in sequence]}")

# LLM-based strategy (requires OpenAI API key)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # Set your API key

config = StrategyConfig(
    custom_params={
        "model": "gpt-4o",  # or "o1-mini", "gpt-4-turbo", etc.
        "temperature": 0.7,
        "max_tokens": 500,
    }
)
strategy = get_strategy("llm", config)
best_move = strategy.select_best_move(state)
print(f"LLM suggested move: {best_move}")
```

For more strategy examples, see `scripts/example_strategies.py`, `scripts/example_llm_strategy.py`, and `solitaire_analytics/strategies/README.md`.

### Analyzing Moves

```python
from solitaire_analytics.analysis import compute_all_possible_moves

moves = compute_all_possible_moves(state)
for move_info in moves:
    print(f"Move: {move_info['move']['description']}")
    print(f"Score delta: {move_info['score_delta']}")
```

### Building Move Trees

```python
from solitaire_analytics import MoveTreeBuilder

builder = MoveTreeBuilder(max_depth=5, max_nodes=1000)
root = builder.build_tree(state)
stats = builder.get_statistics()
print(f"Tree has {stats['total_nodes']} nodes")
```

### Detecting Dead Ends

```python
from solitaire_analytics import DeadEndDetector

detector = DeadEndDetector()
analysis = detector.analyze_dead_end_risk(state)
print(f"Dead end risk: {analysis['risk_score']:.2f}")
```

### Finding Best Move Sequences

```python
from solitaire_analytics.analysis import find_best_move_sequences

sequences = find_best_move_sequences(state, depth=3, max_sequences=5)
best_sequence = sequences[0]
print(f"Best sequence score: {best_sequence['score']}")
```

### Calculating Progression Score

```python
from solitaire_analytics.analysis import calculate_progression_score

# Calculate how close the game is to winning (0.0 to 1.0)
score = calculate_progression_score(state)
print(f"Game progression: {score:.1%}")
```

### Import and Export Game State

```python
from solitaire_analytics import GameState

# Export game state to JSON
json_str = state.to_json()

# Save to file
with open('game_state.json', 'w') as f:
    f.write(json_str)

# Load from file
with open('game_state.json', 'r') as f:
    loaded_json = f.read()

# Import game state from JSON
restored_state = GameState.from_json(loaded_json)

# Or work with dictionaries directly
state_dict = state.to_dict()
restored_state = GameState.from_dict(state_dict)
```

### Recording Play Logs for Replay/Visualization

The `PlayLogger` allows you to record game sessions with timestamps for replay or visualization elsewhere:

```python
from solitaire_analytics import PlayLogger, GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move

# Create logger (disabled by default for efficiency)
logger = PlayLogger(
    enabled=True,
    metadata={"player": "Alice", "session_id": "game_001"}
)

# Start recording with initial state
state = GameState()
logger.start(state)

# Log moves as the game progresses
moves = generate_moves(state)
if moves:
    move = moves[0]
    new_state = apply_move(state, move)
    if new_state:
        # Log with or without resulting state
        logger.log_move(move, resulting_state=new_state)
        state = new_state

# Export to JSON for use with visualizers
logger.save('/tmp/game_log.json')

# Load log later
loaded_logger = PlayLogger.load('/tmp/game_log.json')
print(f"Moves played: {len(loaded_logger.moves)}")
```

**Key Features:**
- **Disabled by default** for performance (no overhead when not needed)
- Records **initial game state/setup** for complete replay capability
- Logs **all moves with timestamps** for temporal analysis
- Supports **custom metadata** (player name, session ID, etc.)
- **JSON export** compatible with visualization tools
- Can optionally include **resulting states** for full game reconstruction

See `scripts/example_play_logger.py` for a complete demonstration.

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run specific test categories
pytest -m models
pytest -m engine
pytest -m solver
pytest -m analysis

# Run with coverage
pytest --cov=solitaire_analytics
```

### Test Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests
- `slow`: Long-running tests
- `gpu`: GPU-dependent tests
- `solver`: Solver tests
- `analysis`: Analysis tests
- `models`: Model tests
- `engine`: Engine tests

## Example Scripts

Run the example analysis script:

```bash
python scripts/example_analysis.py
```

## Jupyter Notebooks

Explore the interactive examples:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

## CI/CD

The project includes GitHub Actions CI configuration for Ubuntu that:
- Runs on Python 3.12
- Executes unit and integration tests
- Generates coverage reports
- Runs on push and pull requests

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`pytest`)
2. Code follows project style
3. New features include tests
4. Documentation is updated

## License

MIT License

## Authors

Solitaire Analytics Team
