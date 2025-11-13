# Solitaire Strategy System

The strategy module provides an extensible framework for implementing and comparing different move selection strategies in Solitaire games.

## Overview

The strategy system follows the **Strategy Pattern**, allowing you to:
- Choose from multiple built-in strategies
- Configure strategy behavior without modifying code
- Create custom strategies by subclassing the base `Strategy` class
- Compare strategy performance
- Serialize/deserialize strategy configurations
- Integrate with external systems (e.g., LLMs)

## Architecture

### Core Components

1. **Strategy (base class)**: Abstract base defining the strategy interface
2. **StrategyConfig**: Configuration dataclass for customizing strategy behavior
3. **StrategyRegistry**: Factory/registry for discovering and instantiating strategies
4. **Built-in Strategies**: Pre-implemented strategies (Simple, Weighted, Lookahead, LLM)

### Class Hierarchy

```
Strategy (ABC)
├── SimpleStrategy
├── WeightedStrategy
├── LookaheadStrategy
└── LLMStrategy
```

## Built-in Strategies

### 1. SimpleStrategy

A greedy strategy that prioritizes moves by type.

**Priority Order:**
1. Foundation moves (highest)
2. Flipping face-down cards
3. Waste to tableau moves
4. Tableau to tableau moves
5. Stock to waste (lowest)

**Usage:**
```python
from solitaire_analytics.strategies import get_strategy

strategy = get_strategy("simple")
best_move = strategy.select_best_move(game_state)
```

**Best For:** Quick decisions, baseline comparisons

### 2. WeightedStrategy

Scores moves based on configurable weighted factors.

**Scoring Factors:**
- Move type priority
- Score delta
- Cards revealed
- Foundation progress
- Move flexibility

**Default Priorities:**
```python
{
    "tableau_to_foundation": 100.0,
    "waste_to_foundation": 100.0,
    "flip_tableau_card": 50.0,
    "waste_to_tableau": 30.0,
    "tableau_to_tableau": 20.0,
    "stock_to_waste": 5.0,
    "score_delta": 1.0,
    "reveals_card": 25.0,
    "foundation_progress": 10.0,
    "move_flexibility": 2.0,
}
```

**Usage:**
```python
from solitaire_analytics.strategies import get_strategy, StrategyConfig

# Use defaults
strategy = get_strategy("weighted")

# Custom priorities
config = StrategyConfig(
    priorities={
        "tableau_to_foundation": 200.0,  # Boost foundation moves
        "reveals_card": 50.0,  # Prioritize revealing cards
    }
)
strategy = get_strategy("weighted", config)
best_move = strategy.select_best_move(game_state)
```

**Best For:** Fine-tuned control, experimenting with different priorities

### 3. LookaheadStrategy

Evaluates move sequences to find the best path forward.

**Features:**
- Depth-limited search
- Evaluates complete sequences, not just individual moves
- Configurable lookahead depth

**Usage:**
```python
from solitaire_analytics.strategies import get_strategy, StrategyConfig

config = StrategyConfig(max_depth=5)
strategy = get_strategy("lookahead", config)

# Get best first move
best_move = strategy.select_best_move(game_state)

# Get best sequence
sequence = strategy.select_move_sequence(game_state, length=5)
```

**Best For:** Finding optimal sequences, deeper analysis

**Note:** Higher depths are more computationally expensive.

### 4. LLMStrategy (Placeholder)

Interface for LLM-based move selection (future enhancement).

**Potential Features:**
- Natural language move explanations
- Context-aware suggestions
- Learning from human players
- Strategic advice

**Usage:**
```python
from solitaire_analytics.strategies import get_strategy, StrategyConfig

config = StrategyConfig(
    custom_params={
        "llm_provider": "openai",
        "model_name": "gpt-4",
        "api_key": "sk-...",
    }
)
strategy = get_strategy("llm", config)
# Currently raises NotImplementedError - integration needed
```

**Status:** Placeholder interface provided, implementation pending

## Configuration System

### StrategyConfig

The `StrategyConfig` class provides a flexible configuration system:

```python
from solitaire_analytics.strategies import StrategyConfig

config = StrategyConfig(
    know_face_down_cards=False,  # Whether strategy has perfect information
    max_depth=3,  # Lookahead depth for search-based strategies
    priorities={  # Custom priority weights
        "tableau_to_foundation": 150.0,
        "move_flexibility": 10.0,
    },
    custom_params={  # Strategy-specific parameters
        "use_heuristic": True,
        "timeout": 30,
    }
)
```

### Serialization

Configurations can be serialized for storage or transmission:

```python
# To dictionary
config_dict = config.to_dict()

# From dictionary
restored_config = StrategyConfig.from_dict(config_dict)

# JSON serialization
import json
json_str = json.dumps(config.to_dict())
config = StrategyConfig.from_dict(json.loads(json_str))
```

## Creating Custom Strategies

To create a custom strategy:

1. Subclass `Strategy`
2. Implement required methods
3. Register with `StrategyRegistry`

### Example: Random Strategy

```python
from solitaire_analytics.strategies.base import Strategy
from solitaire_analytics.engine import generate_moves
from solitaire_analytics.strategies import StrategyRegistry
import random

class RandomStrategy(Strategy):
    """Randomly selects from available moves."""
    
    def select_best_move(self, state):
        moves = generate_moves(state)
        return random.choice(moves) if moves else None
    
    def get_name(self):
        return "Random"
    
    def get_description(self):
        return "Randomly selects from available moves"

# Register the strategy
StrategyRegistry.register("random", RandomStrategy)

# Use it
from solitaire_analytics.strategies import get_strategy
strategy = get_strategy("random")
```

### Example: Monte Carlo Strategy

```python
from solitaire_analytics.strategies.base import Strategy
from solitaire_analytics.engine import generate_moves, apply_move
import random

class MonteCarloStrategy(Strategy):
    """Uses Monte Carlo simulation to evaluate moves."""
    
    def select_best_move(self, state):
        moves = generate_moves(state)
        if not moves:
            return None
        
        # Run simulations for each move
        num_simulations = self.config.custom_params.get("simulations", 100)
        best_move = None
        best_score = -float('inf')
        
        for move in moves:
            total_score = 0
            for _ in range(num_simulations):
                score = self._simulate(state, move)
                total_score += score
            
            avg_score = total_score / num_simulations
            if avg_score > best_score:
                best_score = avg_score
                best_move = move
        
        return best_move
    
    def _simulate(self, state, first_move):
        """Run a random simulation starting with first_move."""
        current_state = apply_move(state, first_move)
        if current_state is None:
            return 0
        
        # Random playthrough
        depth = 0
        max_depth = self.config.max_depth
        while depth < max_depth and not current_state.is_won():
            moves = generate_moves(current_state)
            if not moves:
                break
            
            move = random.choice(moves)
            current_state = apply_move(current_state, move)
            if current_state is None:
                break
            depth += 1
        
        # Return final score
        return current_state.score if current_state else 0
    
    def get_name(self):
        return "MonteCarlo"
    
    def get_description(self):
        return f"Monte Carlo strategy with {self.config.custom_params.get('simulations', 100)} simulations per move"
```

## Strategy Registry

The `StrategyRegistry` provides centralized strategy management:

```python
from solitaire_analytics.strategies import StrategyRegistry

# List all strategies
strategies = StrategyRegistry.get_strategy_names()
print(strategies)  # ['simple', 'weighted', 'lookahead', 'llm']

# Get strategy class
strategy_class = StrategyRegistry.get_strategy_class("simple")

# Create instance
strategy = StrategyRegistry.create_strategy("weighted", config)

# Register custom strategy
StrategyRegistry.register("custom", CustomStrategy)

# Unregister
StrategyRegistry.unregister("custom")
```

## Comparing Strategies

### Performance Comparison

```python
from solitaire_analytics.strategies import StrategyRegistry, StrategyConfig
from solitaire_analytics.models import GameState

def compare_strategies(game_state):
    """Compare all strategies on the same game state."""
    results = {}
    
    for name in StrategyRegistry.get_strategy_names():
        strategy = StrategyRegistry.create_strategy(name)
        move = strategy.select_best_move(game_state)
        results[name] = move
    
    return results

# Usage
state = GameState()
# ... initialize state ...
results = compare_strategies(state)
for name, move in results.items():
    print(f"{name}: {move}")
```

### Win Rate Testing

```python
from solitaire_analytics.engine import apply_move

def test_strategy_win_rate(strategy, num_games=100):
    """Test strategy win rate over multiple games."""
    wins = 0
    
    for _ in range(num_games):
        state = create_random_game()
        
        while not state.is_won():
            move = strategy.select_best_move(state)
            if move is None:
                break
            state = apply_move(state, move)
            if state is None:
                break
        
        if state and state.is_won():
            wins += 1
    
    return wins / num_games
```

## Best Practices

### 1. Configuration Management

Store configurations in JSON files for reproducibility:

```python
import json

# Save configuration
config = StrategyConfig(max_depth=5, priorities={"...": 1.0})
with open("config.json", "w") as f:
    json.dump(config.to_dict(), f)

# Load configuration
with open("config.json") as f:
    config = StrategyConfig.from_dict(json.load(f))
```

### 2. Strategy Selection

Choose strategies based on your needs:

- **Quick decisions**: SimpleStrategy
- **Customization**: WeightedStrategy
- **Optimal play**: LookaheadStrategy (with appropriate depth)
- **Experimentation**: Create custom strategies

### 3. Performance Optimization

- Limit lookahead depth for real-time play
- Cache strategy instances when using the same configuration
- Profile custom strategies to identify bottlenecks

### 4. Testing

Always test custom strategies:

```python
def test_custom_strategy():
    strategy = CustomStrategy()
    
    # Test on empty state
    empty_state = GameState()
    assert strategy.select_best_move(empty_state) is None
    
    # Test on state with moves
    state = create_game_with_moves()
    move = strategy.select_best_move(state)
    assert move is not None
    
    # Test sequence generation
    sequence = strategy.select_move_sequence(state, length=3)
    assert len(sequence) <= 3
```

## Future Enhancements

### Planned Features

1. **LLM Integration**: Complete implementation of LLM-based strategy
2. **Reinforcement Learning**: Strategy that learns from play
3. **Hybrid Strategies**: Combine multiple strategies
4. **Performance Profiling**: Built-in benchmarking tools
5. **Strategy Visualization**: Visualize decision-making process

### Contributing

To contribute a new strategy:

1. Create strategy class in `solitaire_analytics/strategies/`
2. Implement required methods
3. Add comprehensive tests
4. Update this documentation
5. Add example usage in `scripts/example_strategies.py`

## API Reference

### Strategy

```python
class Strategy(ABC):
    def __init__(self, config: Optional[StrategyConfig] = None)
    
    @abstractmethod
    def select_best_move(self, state: GameState) -> Optional[Move]
    
    def select_move_sequence(self, state: GameState, length: int = 3) -> List[Move]
    
    @abstractmethod
    def get_name(self) -> str
    
    def get_description(self) -> str
```

### StrategyConfig

```python
@dataclass
class StrategyConfig:
    know_face_down_cards: bool = False
    max_depth: int = 3
    priorities: Dict[str, float] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig"
```

### StrategyRegistry

```python
class StrategyRegistry:
    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy])
    
    @classmethod
    def unregister(cls, name: str)
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[Strategy]]
    
    @classmethod
    def create_strategy(cls, name: str, config: Optional[StrategyConfig] = None) -> Optional[Strategy]
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Type[Strategy]]
    
    @classmethod
    def get_strategy_names(cls) -> list
```

### Convenience Function

```python
def get_strategy(name: str, config: Optional[StrategyConfig] = None) -> Optional[Strategy]
```

## Examples

See `scripts/example_strategies.py` for comprehensive examples of all features.

## Support

For questions or issues:
- Check the docstrings in each strategy module
- Review test files in `tests/test_strategies.py`
- See example script `scripts/example_strategies.py`
- Refer to the main project documentation
