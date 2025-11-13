#!/usr/bin/env python3
"""Example script demonstrating solitaire analytics capabilities."""

import json
from solitaire_analytics import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics import ParallelSolver, MoveTreeBuilder, DeadEndDetector
from solitaire_analytics.analysis import compute_all_possible_moves, find_best_move_sequences


def create_sample_game():
    """Create a sample game state for demonstration."""
    state = GameState()
    
    # Add some cards to tableau
    state.tableau[0].append(Card(rank=13, suit=Suit.HEARTS))
    state.tableau[1].append(Card(rank=12, suit=Suit.SPADES))
    state.tableau[2].append(Card(rank=1, suit=Suit.DIAMONDS))
    state.tableau[3].append(Card(rank=2, suit=Suit.DIAMONDS))
    state.tableau[4].append(Card(rank=11, suit=Suit.HEARTS))
    
    # Add some cards to stock
    state.stock.append(Card(rank=3, suit=Suit.CLUBS))
    state.stock.append(Card(rank=4, suit=Suit.HEARTS))
    
    return state


def main():
    """Main function demonstrating analytics capabilities."""
    print("=" * 60)
    print("Solitaire Analytics Engine - Example Analysis")
    print("=" * 60)
    
    # Create sample game
    print("\n1. Creating sample game state...")
    state = create_sample_game()
    print(f"   Game state created with:")
    print(f"   - {sum(len(pile) for pile in state.tableau)} cards in tableau")
    print(f"   - {len(state.stock)} cards in stock")
    
    # Analyze possible moves
    print("\n2. Computing all possible moves...")
    moves = compute_all_possible_moves(state)
    print(f"   Found {len(moves)} possible moves")
    
    if moves:
        print("\n   Top 3 moves:")
        for i, move_info in enumerate(moves[:3]):
            print(f"   {i+1}. {move_info['move']['description']}")
            print(f"      Score delta: {move_info['score_delta']}")
    
    # Find best sequences
    print("\n3. Finding best move sequences...")
    sequences = find_best_move_sequences(state, depth=3, max_sequences=5)
    print(f"   Found {len(sequences)} sequences")
    
    if sequences:
        best = sequences[0]
        print(f"\n   Best sequence (score: {best['score']:.1f}):")
        for i, move in enumerate(best['moves']):
            print(f"   {i+1}. {move['description']}")
    
    # Build move tree
    print("\n4. Building move tree...")
    builder = MoveTreeBuilder(max_depth=5, max_nodes=100)
    root = builder.build_tree(state)
    stats = builder.get_statistics()
    
    print(f"   Tree statistics:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # Check for dead ends
    print("\n5. Analyzing dead end risk...")
    detector = DeadEndDetector()
    analysis = detector.analyze_dead_end_risk(state)
    
    print(f"   Dead end analysis:")
    print(f"   - Is dead end: {analysis['is_dead_end']}")
    print(f"   - Risk score: {analysis['risk_score']:.2f}")
    print(f"   - Available moves: {analysis['available_moves']}")
    
    # Try to solve
    print("\n6. Attempting to solve with ParallelSolver...")
    solver = ParallelSolver(max_depth=10, n_jobs=2, beam_width=50, timeout=5.0)
    result = solver.solve(state)
    
    print(f"   Solver result:")
    print(f"   - Success: {result.success}")
    print(f"   - States explored: {result.states_explored}")
    print(f"   - Time elapsed: {result.time_elapsed:.2f} seconds")
    
    if result.success:
        print(f"   - Solution found with {len(result.moves)} moves!")
    
    # Generate JSON report
    print("\n7. Generating JSON report...")
    report = {
        "game_state": {
            "tableau_cards": sum(len(pile) for pile in state.tableau),
            "foundation_cards": sum(len(pile) for pile in state.foundations),
            "stock_cards": len(state.stock),
            "waste_cards": len(state.waste),
        },
        "analysis": {
            "possible_moves": len(moves),
            "best_sequences": len(sequences),
            "dead_end_risk": analysis['risk_score'],
        },
        "solver_result": {
            "success": result.success,
            "states_explored": result.states_explored,
            "time_elapsed": result.time_elapsed,
        }
    }
    
    report_json = json.dumps(report, indent=2)
    print("\n   Report generated:")
    print(report_json)
    
    # Save report to file
    output_file = "/tmp/solitaire_analysis_report.json"
    with open(output_file, 'w') as f:
        f.write(report_json)
    print(f"\n   Report saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
