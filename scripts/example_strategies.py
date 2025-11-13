"""Example script demonstrating different move selection strategies."""

from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.strategies import (
    get_strategy,
    StrategyConfig,
    StrategyRegistry,
)


def create_sample_game_state():
    """Create a sample game state for demonstration."""
    state = GameState()
    
    # Add some cards to tableau
    state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))
    state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS, face_up=True))
    state.tableau[2].append(Card(rank=3, suit=Suit.DIAMONDS, face_up=True))
    
    # Add a card to waste
    state.waste.append(Card(rank=4, suit=Suit.CLUBS, face_up=True))
    
    # Add some cards to stock
    for i in range(5, 10):
        state.stock.append(Card(rank=i, suit=Suit.SPADES, face_up=False))
    
    return state


def demonstrate_simple_strategy():
    """Demonstrate the Simple greedy strategy."""
    print("=" * 60)
    print("Simple Strategy Demo")
    print("=" * 60)
    
    state = create_sample_game_state()
    strategy = get_strategy("simple")
    
    print(f"\nStrategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}\n")
    
    # Select best move
    best_move = strategy.select_best_move(state)
    if best_move:
        print(f"Best move: {best_move}")
    else:
        print("No moves available")
    
    # Select a sequence
    sequence = strategy.select_move_sequence(state, length=3)
    print(f"\nMove sequence (length {len(sequence)}):")
    for i, move in enumerate(sequence, 1):
        print(f"  {i}. {move}")


def demonstrate_weighted_strategy():
    """Demonstrate the Weighted strategy with custom priorities."""
    print("\n" + "=" * 60)
    print("Weighted Strategy Demo")
    print("=" * 60)
    
    state = create_sample_game_state()
    
    # Create custom configuration
    config = StrategyConfig(
        priorities={
            "tableau_to_foundation": 150.0,  # Boost foundation moves
            "reveals_card": 40.0,  # Prioritize revealing cards
            "move_flexibility": 5.0,  # Value flexibility more
        }
    )
    
    strategy = get_strategy("weighted", config)
    
    print(f"\nStrategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    print(f"\nCustom priorities:")
    for key, value in config.priorities.items():
        print(f"  {key}: {value}")
    
    # Select best move
    best_move = strategy.select_best_move(state)
    if best_move:
        print(f"\nBest move: {best_move}")
    else:
        print("\nNo moves available")
    
    # Select a sequence
    sequence = strategy.select_move_sequence(state, length=3)
    print(f"\nMove sequence (length {len(sequence)}):")
    for i, move in enumerate(sequence, 1):
        print(f"  {i}. {move}")


def demonstrate_lookahead_strategy():
    """Demonstrate the Lookahead strategy."""
    print("\n" + "=" * 60)
    print("Lookahead Strategy Demo")
    print("=" * 60)
    
    state = create_sample_game_state()
    
    # Create config with custom depth
    config = StrategyConfig(max_depth=4)
    strategy = get_strategy("lookahead", config)
    
    print(f"\nStrategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    print(f"\nLookahead depth: {config.max_depth}")
    
    # Select best move
    best_move = strategy.select_best_move(state)
    if best_move:
        print(f"\nBest move (first of best sequence): {best_move}")
    else:
        print("\nNo moves available")
    
    # Select a full sequence
    sequence = strategy.select_move_sequence(state, length=4)
    print(f"\nOptimized move sequence (length {len(sequence)}):")
    for i, move in enumerate(sequence, 1):
        print(f"  {i}. {move}")


def compare_strategies():
    """Compare all strategies on the same game state."""
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)
    
    state = create_sample_game_state()
    
    # Get all available strategies
    strategy_names = StrategyRegistry.get_strategy_names()
    
    print(f"\nComparing {len(strategy_names)} strategies on the same state:")
    print(f"State: {sum(len(p) for p in state.tableau)} cards in tableau, "
          f"{len(state.waste)} in waste, {len(state.stock)} in stock\n")
    
    for name in strategy_names:
        strategy = get_strategy(name)
        best_move = strategy.select_best_move(state)
        
        print(f"{strategy.get_name():12s} strategy chose: ", end="")
        if best_move:
            print(f"{best_move}")
        else:
            print("No move")


def demonstrate_custom_strategy():
    """Demonstrate creating and registering a custom strategy."""
    print("\n" + "=" * 60)
    print("Custom Strategy Demo")
    print("=" * 60)
    
    from solitaire_analytics.strategies.base import Strategy
    from solitaire_analytics.engine import generate_moves
    import random
    
    class RandomStrategy(Strategy):
        """A simple strategy that picks moves randomly."""
        
        def select_best_move(self, state):
            moves = generate_moves(state)
            return random.choice(moves) if moves else None
        
        def get_name(self):
            return "Random"
        
        def get_description(self):
            return "Randomly selects from available moves"
    
    # Register the custom strategy
    StrategyRegistry.register("random", RandomStrategy)
    
    print("\nCustom 'Random' strategy registered!")
    print(f"Available strategies: {', '.join(StrategyRegistry.get_strategy_names())}")
    
    # Use the custom strategy
    state = create_sample_game_state()
    strategy = get_strategy("random")
    
    print(f"\nStrategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    
    best_move = strategy.select_best_move(state)
    if best_move:
        print(f"\nRandomly selected move: {best_move}")
    else:
        print("\nNo moves available")
    
    # Clean up
    StrategyRegistry.unregister("random")


def demonstrate_config_serialization():
    """Demonstrate strategy configuration serialization."""
    print("\n" + "=" * 60)
    print("Configuration Serialization Demo")
    print("=" * 60)
    
    # Create a config
    original_config = StrategyConfig(
        know_face_down_cards=True,
        max_depth=5,
        priorities={
            "tableau_to_foundation": 200.0,
            "reveals_card": 50.0,
        },
        custom_params={
            "use_heuristic": True,
            "timeout": 30,
        }
    )
    
    print("\nOriginal configuration:")
    print(f"  know_face_down_cards: {original_config.know_face_down_cards}")
    print(f"  max_depth: {original_config.max_depth}")
    print(f"  priorities: {original_config.priorities}")
    print(f"  custom_params: {original_config.custom_params}")
    
    # Convert to dict
    config_dict = original_config.to_dict()
    print("\nSerialized to dict:")
    print(f"  {config_dict}")
    
    # Recreate from dict
    restored_config = StrategyConfig.from_dict(config_dict)
    print("\nRestored configuration:")
    print(f"  know_face_down_cards: {restored_config.know_face_down_cards}")
    print(f"  max_depth: {restored_config.max_depth}")
    print(f"  priorities: {restored_config.priorities}")
    print(f"  custom_params: {restored_config.custom_params}")
    
    # Use in strategy
    strategy = get_strategy("weighted", restored_config)
    print(f"\nUsing restored config in {strategy.get_name()} strategy")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Solitaire Strategy System Demonstration")
    print("=" * 60)
    print("\nThis script demonstrates the extensible strategy pattern")
    print("for move selection in Solitaire games.")
    
    # Run demonstrations
    demonstrate_simple_strategy()
    demonstrate_weighted_strategy()
    demonstrate_lookahead_strategy()
    compare_strategies()
    demonstrate_custom_strategy()
    demonstrate_config_serialization()
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Multiple strategies available (Simple, Weighted, Lookahead)")
    print("2. Strategies are configurable via StrategyConfig")
    print("3. Easy to register custom strategies")
    print("4. Strategy registry provides discovery and instantiation")
    print("5. Configurations can be serialized for storage/transmission")
    print("\nFor more information, see the documentation in:")
    print("  - solitaire_analytics/strategies/base.py")
    print("  - solitaire_analytics/strategies/README.md")
    print()


if __name__ == "__main__":
    main()
