#!/usr/bin/env python3
"""Example script demonstrating PlayLogger usage for recording game sessions."""

import json
from solitaire_analytics import Card, GameState, Move, PlayLogger
from solitaire_analytics.models.card import Suit
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves, apply_move


def create_sample_game():
    """Create a sample game state for demonstration."""
    state = GameState()
    
    # Add some cards to tableau
    state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))
    state.tableau[1].append(Card(rank=2, suit=Suit.SPADES, face_up=True))
    state.tableau[2].append(Card(rank=3, suit=Suit.DIAMONDS, face_up=True))
    
    # Add some cards to stock
    state.stock.extend([
        Card(rank=4, suit=Suit.CLUBS, face_up=False),
        Card(rank=5, suit=Suit.HEARTS, face_up=False),
        Card(rank=6, suit=Suit.SPADES, face_up=False),
    ])
    
    return state


def main():
    """Main function demonstrating PlayLogger usage."""
    print("=" * 70)
    print("Solitaire Analytics - Play Logger Example")
    print("=" * 70)
    
    # Example 1: Basic usage with logging disabled (default)
    print("\n1. Logger disabled by default (for efficiency):")
    print("-" * 70)
    logger_disabled = PlayLogger()  # enabled=False by default
    print(f"   Logger enabled: {logger_disabled.enabled}")
    
    state = create_sample_game()
    logger_disabled.start(state)
    logger_disabled.log_move(Move(move_type=MoveType.STOCK_TO_WASTE))
    
    print(f"   Moves logged: {len(logger_disabled.moves)}")
    print("   (No moves recorded when disabled)")
    
    # Example 2: Enable logging to record game session
    print("\n2. Logger enabled with metadata:")
    print("-" * 70)
    logger = PlayLogger(
        enabled=True,
        metadata={
            "player": "example_player",
            "session_id": "demo_session_001",
            "game_type": "klondike",
            "difficulty": "easy"
        }
    )
    print(f"   Logger enabled: {logger.enabled}")
    print(f"   Metadata: {logger.metadata}")
    
    # Start logging with initial state
    print("\n3. Starting game with initial state:")
    print("-" * 70)
    state = create_sample_game()
    logger.start(state)
    
    print(f"   Initial tableau cards: {sum(len(pile) for pile in state.tableau)}")
    print(f"   Initial stock cards: {len(state.stock)}")
    
    # Simulate some moves
    print("\n4. Making moves and logging them:")
    print("-" * 70)
    
    current_state = state
    
    # Move 1: Draw from stock
    possible_moves = generate_moves(current_state)
    draw_move = next((m for m in possible_moves if m.move_type == MoveType.STOCK_TO_WASTE), None)
    
    if draw_move:
        print(f"   Move 1: {draw_move}")
        new_state = apply_move(current_state, draw_move)
        if new_state:
            logger.log_move(draw_move, resulting_state=new_state)
            current_state = new_state
    
    # Move 2: Move Ace to foundation
    possible_moves = generate_moves(current_state)
    foundation_move = next(
        (m for m in possible_moves if m.move_type == MoveType.TABLEAU_TO_FOUNDATION),
        None
    )
    
    if foundation_move:
        print(f"   Move 2: {foundation_move}")
        new_state = apply_move(current_state, foundation_move)
        if new_state:
            logger.log_move(foundation_move, resulting_state=new_state)
            current_state = new_state
    
    # Move 3: Another draw
    if current_state.stock:
        draw_move2 = Move(move_type=MoveType.STOCK_TO_WASTE)
        print(f"   Move 3: {draw_move2}")
        new_state = apply_move(current_state, draw_move2)
        if new_state:
            logger.log_move(draw_move2)
            current_state = new_state
    
    print(f"\n   Total moves logged: {len(logger.moves)}")
    
    # Export to dictionary
    print("\n5. Exporting log to dictionary:")
    print("-" * 70)
    log_data = logger.to_dict()
    
    print(f"   Enabled: {log_data['enabled']}")
    print(f"   Move count: {log_data['move_count']}")
    print(f"   Initial state tableau piles: {len(log_data['initial_state']['tableau'])}")
    
    # Show move details
    print("\n6. Move details with timestamps:")
    print("-" * 70)
    for i, move_record in enumerate(log_data['moves'], 1):
        print(f"   Move {i}:")
        print(f"     Timestamp: {move_record['timestamp']:.3f}s")
        print(f"     Type: {move_record['move']['move_type']}")
        print(f"     Description: {move_record['move']['description']}")
    
    # Export to JSON
    print("\n7. Exporting log to JSON:")
    print("-" * 70)
    json_output = logger.to_json()
    print(f"   JSON length: {len(json_output)} characters")
    print("\n   Sample JSON output (first 500 chars):")
    print(f"   {json_output[:500]}...")
    
    # Save to file
    print("\n8. Saving log to file:")
    print("-" * 70)
    output_file = "/tmp/solitaire_play_log.json"
    logger.save(output_file)
    print(f"   Log saved to: {output_file}")
    
    # Load from file
    print("\n9. Loading log from file:")
    print("-" * 70)
    loaded_logger = PlayLogger.load(output_file)
    print(f"   Loaded logger enabled: {loaded_logger.enabled}")
    print(f"   Loaded move count: {len(loaded_logger.moves)}")
    print(f"   Metadata matches: {loaded_logger.metadata == logger.metadata}")
    
    # Show how the log can be used for replay
    print("\n10. Log contains replay information:")
    print("-" * 70)
    print(f"   Initial state preserved: {loaded_logger.initial_state is not None}")
    print(f"   All moves recorded: {len(loaded_logger.moves)}")
    
    if loaded_logger.moves and "resulting_state" in loaded_logger.moves[0]:
        print("   ✓ Moves include resulting states for full replay")
    else:
        print("   ℹ Moves don't include resulting states (lighter logs)")
    
    # Pretty print the full log structure
    print("\n11. Complete log structure:")
    print("-" * 70)
    log_dict = loaded_logger.to_dict()
    
    # Remove large nested data for cleaner display
    display_dict = {
        "enabled": log_dict["enabled"],
        "metadata": log_dict["metadata"],
        "move_count": log_dict["move_count"],
        "start_time": log_dict["start_time"],
        "initial_state": {
            "tableau_piles": len(log_dict["initial_state"]["tableau"]),
            "foundation_piles": len(log_dict["initial_state"]["foundations"]),
            "stock_cards": len(log_dict["initial_state"]["stock"]),
            "waste_cards": len(log_dict["initial_state"]["waste"]),
        },
        "moves": [
            {
                "timestamp": m["timestamp"],
                "move_type": m["move"]["move_type"],
                "description": m["move"]["description"],
            }
            for m in log_dict["moves"]
        ]
    }
    
    print(json.dumps(display_dict, indent=2))
    
    print("\n" + "=" * 70)
    print("Play logger demonstration complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("  • Disabled by default for efficiency")
    print("  • Records initial game state/setup")
    print("  • Logs all moves with timestamps")
    print("  • Supports custom metadata")
    print("  • Can include resulting states for full replay")
    print("  • JSON export for use with visualizers")
    print("  • Save/load functionality for persistence")
    print()


if __name__ == "__main__":
    main()
