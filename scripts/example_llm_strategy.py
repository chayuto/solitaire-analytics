#!/usr/bin/env python3
"""Example demonstrating the LLM strategy with OpenAI API.

This script shows how to use the LLM-based strategy for Solitaire move selection.
The LLM strategy uses OpenAI's language models to intelligently analyze game states
and suggest optimal moves.

Requirements:
1. OpenAI API key set in OPENAI_API_KEY environment variable
2. openai package installed (pip install openai>=1.0.0)

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."
    
    # Run the example
    python scripts/example_llm_strategy.py
"""

import os
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.strategies import get_strategy, StrategyConfig


def create_sample_game_state():
    """Create a sample game state for demonstration."""
    state = GameState()
    
    # Add some cards to tableau
    state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))  # Ace of hearts
    state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS, face_up=True))  # 2 of hearts
    state.tableau[2].append(Card(rank=13, suit=Suit.SPADES, face_up=True))  # King of spades
    state.tableau[3].append(Card(rank=3, suit=Suit.DIAMONDS, face_up=False))  # Face down
    state.tableau[3].append(Card(rank=12, suit=Suit.CLUBS, face_up=True))  # Queen of clubs
    
    # Add some cards to waste
    state.waste.append(Card(rank=11, suit=Suit.DIAMONDS, face_up=True))  # Jack of diamonds
    
    # Add some cards to stock
    for i in range(5):
        state.stock.append(Card(rank=i+4, suit=Suit.SPADES, face_up=False))
    
    return state


def example_with_default_model():
    """Example using default model (gpt-4o)."""
    print("=" * 70)
    print("Example 1: Using LLM Strategy with Default Model (gpt-4o)")
    print("=" * 70)
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Strategy will use fallback heuristic.")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
    else:
        print("✓ OPENAI_API_KEY found")
    
    print()
    
    # Create strategy with default configuration
    strategy = get_strategy("llm")
    
    print(f"Strategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    print(f"Model: {strategy.model}")
    print()
    
    # Create a game state
    state = create_sample_game_state()
    print("Game state created with:")
    print(f"  - Tableau piles with cards")
    print(f"  - Waste pile with Jack of Diamonds")
    print(f"  - Stock with 5 cards")
    print()
    
    # Select best move
    print("Selecting best move using LLM...")
    move = strategy.select_best_move(state)
    
    if move:
        print(f"✓ Selected move: {move}")
        print(f"  Move type: {move.move_type.value}")
    else:
        print("✗ No valid moves available")
    
    print()


def example_with_reasoning_model():
    """Example using reasoning model (o1-mini)."""
    print("=" * 70)
    print("Example 2: Using LLM Strategy with Reasoning Model (o1-mini)")
    print("=" * 70)
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        print()
        return
    
    print("✓ OPENAI_API_KEY found")
    print()
    
    # Create strategy with reasoning model
    config = StrategyConfig(
        custom_params={
            "model": "o1-mini",
            # Note: temperature and max_tokens not used for reasoning models
        }
    )
    strategy = get_strategy("llm", config)
    
    print(f"Strategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    print(f"Model: {strategy.model}")
    print(f"Is reasoning model: {strategy.is_reasoning_model}")
    print()
    
    # Create a game state
    state = create_sample_game_state()
    
    # Select best move
    print("Selecting best move using reasoning model...")
    move = strategy.select_best_move(state)
    
    if move:
        print(f"✓ Selected move: {move}")
        print(f"  Move type: {move.move_type.value}")
    else:
        print("✗ No valid moves available")
    
    print()


def example_with_custom_parameters():
    """Example with custom model parameters."""
    print("=" * 70)
    print("Example 3: Using LLM Strategy with Custom Parameters")
    print("=" * 70)
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        print()
        return
    
    print("✓ OPENAI_API_KEY found")
    print()
    
    # Create strategy with custom parameters
    config = StrategyConfig(
        custom_params={
            "model": "gpt-4-turbo",
            "temperature": 0.5,  # Lower temperature for more deterministic responses
            "max_tokens": 300,
            "timeout": 15,
        }
    )
    strategy = get_strategy("llm", config)
    
    print(f"Strategy: {strategy.get_name()}")
    print(f"Description: {strategy.get_description()}")
    print(f"Model: {strategy.model}")
    print(f"Temperature: {strategy.temperature}")
    print(f"Max tokens: {strategy.max_tokens}")
    print(f"Timeout: {strategy.timeout}s")
    print()
    
    # Create a game state
    state = create_sample_game_state()
    
    # Select best move
    print("Selecting best move with custom parameters...")
    move = strategy.select_best_move(state)
    
    if move:
        print(f"✓ Selected move: {move}")
        print(f"  Move type: {move.move_type.value}")
    else:
        print("✗ No valid moves available")
    
    print()


def example_comparing_strategies():
    """Example comparing LLM strategy with other strategies."""
    print("=" * 70)
    print("Example 4: Comparing LLM Strategy with Other Strategies")
    print("=" * 70)
    print()
    
    # Create a game state
    state = create_sample_game_state()
    
    # Get different strategies
    strategies = {
        "Simple": get_strategy("simple"),
        "Weighted": get_strategy("weighted"),
        "Lookahead": get_strategy("lookahead", StrategyConfig(max_depth=2)),
        "LLM": get_strategy("llm"),
    }
    
    print("Comparing move selection across different strategies:")
    print()
    
    for name, strategy in strategies.items():
        move = strategy.select_best_move(state)
        if move:
            print(f"  {name:12} → {move}")
        else:
            print(f"  {name:12} → No valid moves")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "LLM Strategy Examples" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run examples
    example_with_default_model()
    example_with_reasoning_model()
    example_with_custom_parameters()
    example_comparing_strategies()
    
    print("=" * 70)
    print("Examples completed!")
    print()
    print("Tips:")
    print("  - Set OPENAI_API_KEY environment variable to use LLM features")
    print("  - Use reasoning models (o1-mini) for complex decision-making")
    print("  - Use chat models (gpt-4o) for faster, parameter-tunable responses")
    print("  - The strategy automatically falls back to heuristics if API fails")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
