# Strategy System Architecture

## Overview

The Solitaire Analytics Strategy System provides a flexible, extensible framework for implementing and comparing different move selection strategies. This document describes the architecture, design decisions, and usage patterns.

## Design Goals

The strategy system was designed to meet the following requirements:

1. **Extensibility**: Easy to add new strategies without modifying existing code
2. **Configurability**: Customize strategy behavior without code changes
3. **Comparability**: Test and compare different strategies objectively
4. **Modularity**: Clear separation of concerns
5. **Future-proofing**: Support for LLM and other advanced techniques

## Architecture

### Strategy Pattern

The system implements the classic **Strategy Pattern** from software design patterns:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ uses
       ▼
┌─────────────────┐         ┌──────────────┐
│    Strategy     │◄────────│ StrategyImpl │
│   (abstract)    │         │   (concrete) │
└─────────────────┘         └──────────────┘
       ▲
       │
       ├─── SimpleStrategy
       ├─── WeightedStrategy
       ├─── LookaheadStrategy
       └─── LLMStrategy
```

### Core Components

#### 1. Strategy (Base Class)

Abstract base class defining the contract for all strategies:

```python
class Strategy(ABC):
    def __init__(self, config: Optional[StrategyConfig] = None)
    def select_best_move(self, state: GameState) -> Optional[Move]
    def select_move_sequence(self, state: GameState, length: int) -> List[Move]
    def get_name(self) -> str
    def get_description(self) -> str
```

**Design Decisions:**
- Abstract methods force concrete implementations to provide core functionality
- Default `select_move_sequence` implementation allows simple strategies to work without sequence planning
- Configuration passed at construction allows immutable strategies

#### 2. StrategyConfig

Dataclass for strategy configuration:

```python
@dataclass
class StrategyConfig:
    know_face_down_cards: bool = False  # Perfect information vs. realistic play
    max_depth: int = 3                   # Lookahead depth
    priorities: Dict[str, float]         # Custom priority weights
    custom_params: Dict[str, Any]        # Strategy-specific parameters
```

**Design Decisions:**
- Dataclass for clean, typed configuration
- Serializable to/from dict for storage and transmission
- Extensible via `custom_params` for strategy-specific options
- Immutable after creation (dataclass can be frozen if needed)

#### 3. StrategyRegistry

Factory and registry for strategy management:

```python
class StrategyRegistry:
    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy])
    
    @classmethod
    def create_strategy(cls, name: str, config: Optional[StrategyConfig]) -> Optional[Strategy]
    
    @classmethod
    def get_strategy_names(cls) -> List[str]
```

**Design Decisions:**
- Class methods for singleton-like behavior
- Auto-registration of built-in strategies on module import
- Support for runtime registration of custom strategies
- Name-based lookup for simple discovery

## Strategy Implementations

### 1. SimpleStrategy

**Type:** Greedy  
**Complexity:** O(n) where n = number of available moves  
**Use Case:** Quick decisions, baseline comparisons

**Algorithm:**
1. Generate all legal moves
2. Sort by fixed priority (foundation > reveal > tableau > stock)
3. Return highest priority move

**Pros:**
- Very fast
- Predictable behavior
- Good baseline

**Cons:**
- No lookahead
- May miss better sequences
- Can't be customized

### 2. WeightedStrategy

**Type:** Heuristic  
**Complexity:** O(n × m) where n = moves, m = evaluation cost  
**Use Case:** Customizable, tunable behavior

**Algorithm:**
1. Generate all legal moves
2. For each move:
   - Calculate base priority from move type
   - Apply move to get resulting state
   - Score factors: reveals, foundation progress, flexibility
   - Combine with weighted sum
3. Return highest scored move

**Configurable Weights:**
- Move type priorities (6 types)
- Score delta weight
- Card reveal bonus
- Foundation progress weight
- Move flexibility weight

**Pros:**
- Highly configurable
- Balances multiple factors
- Can be tuned for different play styles

**Cons:**
- Requires tuning
- Still no lookahead
- Evaluation cost per move

### 3. LookaheadStrategy

**Type:** Search-based  
**Complexity:** O(b^d) where b = branching factor, d = depth  
**Use Case:** Finding optimal sequences

**Algorithm:**
1. Perform depth-limited search
2. Evaluate each complete sequence:
   - Win = infinite score
   - Otherwise: weighted sum of:
     - Foundation cards × 100
     - Revealed cards × 20
     - Available moves × 5
     - Game score × 1
     - Sequence length penalty × -0.1
3. Return first move of best sequence

**Depth Control:**
- Configured via `max_depth` parameter
- Typical values: 2-5
- Higher = better but slower

**Pros:**
- Finds multi-move combinations
- Optimal within depth limit
- Good for critical decisions

**Cons:**
- Exponential complexity
- Slower than greedy
- May timeout on complex states

### 4. LLMStrategy

**Type:** AI-assisted (placeholder)  
**Status:** Interface defined, implementation pending  
**Use Case:** Natural language advice, learning from experts

**Planned Features:**
1. Game state formatting for LLM
2. Prompt engineering for move suggestions
3. Response parsing and validation
4. Fallback to simple heuristic
5. Move explanation generation

**Configuration:**
```python
StrategyConfig(
    custom_params={
        "llm_provider": "openai",
        "model_name": "gpt-4",
        "api_key": "...",
        "temperature": 0.7,
    }
)
```

## Configuration System

### Serialization

Configurations can be saved and restored:

```python
# Save to JSON
import json
config = StrategyConfig(max_depth=5)
with open("config.json", "w") as f:
    json.dump(config.to_dict(), f)

# Load from JSON
with open("config.json") as f:
    config = StrategyConfig.from_dict(json.load(f))
```

### Common Configurations

**Fast Play:**
```python
StrategyConfig()  # Simple strategy, defaults
```

**Balanced Play:**
```python
StrategyConfig(
    priorities={
        "tableau_to_foundation": 120.0,
        "reveals_card": 40.0,
        "move_flexibility": 3.0,
    }
)
```

**Optimal Play:**
```python
StrategyConfig(max_depth=5)  # Lookahead strategy
```

**Perfect Information:**
```python
StrategyConfig(
    know_face_down_cards=True,
    max_depth=4,
)
```

## Extensibility

### Creating Custom Strategies

1. **Subclass Strategy:**

```python
from solitaire_analytics.strategies.base import Strategy

class MyStrategy(Strategy):
    def select_best_move(self, state):
        # Implementation here
        pass
    
    def get_name(self):
        return "MyStrategy"
```

2. **Register Strategy:**

```python
from solitaire_analytics.strategies import StrategyRegistry

StrategyRegistry.register("my_strategy", MyStrategy)
```

3. **Use Strategy:**

```python
from solitaire_analytics.strategies import get_strategy

strategy = get_strategy("my_strategy")
```

### Example Custom Strategies

See `solitaire_analytics/strategies/README.md` for:
- RandomStrategy example
- MonteCarloStrategy example
- Guidelines for advanced strategies

## Testing

### Unit Testing

Each strategy should have comprehensive tests:

```python
def test_strategy_initialization()
def test_select_best_move_empty_state()
def test_select_best_move_with_options()
def test_select_move_sequence()
def test_configuration_handling()
```

### Integration Testing

Compare strategies on same game states:

```python
def test_all_strategies_on_same_state()
def test_strategy_win_rates()
def test_strategy_performance()
```

### Performance Testing

Benchmark strategy speed:

```python
import time

def benchmark_strategy(strategy, states, iterations):
    start = time.time()
    for _ in range(iterations):
        for state in states:
            strategy.select_best_move(state)
    return time.time() - start
```

## Performance Characteristics

| Strategy   | Time/Move | Memory | Quality | Tunable |
|-----------|-----------|--------|---------|---------|
| Simple    | ~1ms      | Low    | Basic   | No      |
| Weighted  | ~2-5ms    | Low    | Good    | Yes     |
| Lookahead | ~10-100ms | Medium | Best    | Depth   |
| LLM       | ~1-5s     | Low    | TBD     | Yes     |

*Times are approximate on typical game states*

## Best Practices

### 1. Strategy Selection

- **Interactive play**: SimpleStrategy or WeightedStrategy
- **Analysis**: LookaheadStrategy with appropriate depth
- **Research**: Compare multiple strategies
- **Production**: WeightedStrategy with tuned priorities

### 2. Configuration

- Store configs in version control
- Use meaningful config names
- Document custom priorities
- Test config changes

### 3. Custom Strategies

- Inherit from Strategy
- Implement required methods
- Add comprehensive tests
- Document algorithm
- Provide usage examples

### 4. Performance

- Profile before optimizing
- Cache expensive computations
- Limit lookahead depth appropriately
- Consider timeout mechanisms

## Future Enhancements

### Planned Features

1. **LLM Integration**
   - Complete implementation
   - Multiple provider support
   - Caching and rate limiting

2. **Reinforcement Learning**
   - Self-play training
   - Policy gradient methods
   - Model persistence

3. **Hybrid Strategies**
   - Combine multiple strategies
   - Voting/ensemble methods
   - Context-dependent selection

4. **Performance Tools**
   - Built-in benchmarking
   - Strategy profiler
   - Visualization tools

5. **Advanced Search**
   - Monte Carlo Tree Search
   - Alpha-beta pruning
   - Beam search variants

### Contributing

To contribute a new strategy:

1. Implement Strategy interface
2. Add comprehensive tests
3. Document algorithm and use cases
4. Provide configuration examples
5. Update this documentation
6. Submit PR with examples

## References

- Design Patterns: Strategy Pattern
- Game AI: Search algorithms
- Solitaire Strategy: Domain knowledge
- Testing: Unit and integration testing

## Appendix

### File Structure

```
solitaire_analytics/strategies/
├── __init__.py          # Public API
├── base.py              # Strategy and StrategyConfig
├── simple.py            # SimpleStrategy
├── weighted.py          # WeightedStrategy
├── lookahead.py         # LookaheadStrategy
├── llm.py               # LLMStrategy (placeholder)
├── registry.py          # StrategyRegistry
└── README.md            # User documentation
```

### API Summary

```python
# Getting a strategy
from solitaire_analytics.strategies import get_strategy

strategy = get_strategy("weighted", config)

# Using a strategy
best_move = strategy.select_best_move(game_state)
sequence = strategy.select_move_sequence(game_state, length=5)

# Configuration
from solitaire_analytics.strategies import StrategyConfig

config = StrategyConfig(
    know_face_down_cards=False,
    max_depth=3,
    priorities={"key": value},
)

# Registry
from solitaire_analytics.strategies import StrategyRegistry

StrategyRegistry.register("name", StrategyClass)
names = StrategyRegistry.get_strategy_names()
```

### Version History

- **v1.0** (2025-11): Initial implementation
  - 4 strategies (Simple, Weighted, Lookahead, LLM placeholder)
  - Configuration system
  - Registry pattern
  - Comprehensive tests
  - Documentation
