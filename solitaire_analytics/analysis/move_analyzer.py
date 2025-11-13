"""Move analysis functions for computing and ranking moves."""

from typing import List, Dict, Tuple
import json

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move


def compute_all_possible_moves(state: GameState) -> List[Dict]:
    """Compute all possible moves from a game state with analysis.
    
    Args:
        state: Current game state
        
    Returns:
        List of dictionaries containing move information and analysis
    """
    moves = generate_moves(state)
    analyzed_moves = []
    
    for move in moves:
        new_state = apply_move(state, move)
        if new_state is None:
            continue
        
        move_info = {
            "move": move.to_dict(),
            "score_delta": move.score_delta,
            "new_score": new_state.score,
            "new_foundation_cards": sum(len(f) for f in new_state.foundations),
            "new_face_down_cards": new_state.count_face_down_cards(),
            "available_moves_after": len(generate_moves(new_state)),
            "wins_game": new_state.is_won()
        }
        
        analyzed_moves.append(move_info)
    
    return analyzed_moves


def find_best_move_sequences(
    state: GameState,
    depth: int = 3,
    max_sequences: int = 10
) -> List[Dict]:
    """Find the best sequences of moves from a game state.
    
    Uses a depth-limited search to find promising move sequences.
    
    Args:
        state: Current game state
        depth: How many moves to look ahead
        max_sequences: Maximum number of sequences to return
        
    Returns:
        List of dictionaries containing move sequences and their evaluations
    """
    sequences = []
    
    def search(current_state: GameState, moves: List[Move], current_depth: int):
        """Recursive search for move sequences."""
        if current_depth >= depth:
            # Evaluate this sequence
            score = _evaluate_sequence(current_state, moves)
            sequences.append({
                "moves": [m.to_dict() for m in moves],
                "score": score,
                "final_state_analysis": {
                    "game_won": current_state.is_won(),
                    "total_score": current_state.score,
                    "foundation_cards": sum(len(f) for f in current_state.foundations),
                    "face_down_cards": current_state.count_face_down_cards(),
                    "available_moves": len(generate_moves(current_state))
                }
            })
            return
        
        # If game is won, record and stop
        if current_state.is_won():
            score = _evaluate_sequence(current_state, moves)
            sequences.append({
                "moves": [m.to_dict() for m in moves],
                "score": score,
                "final_state_analysis": {
                    "game_won": True,
                    "total_score": current_state.score,
                    "foundation_cards": 52,
                    "face_down_cards": 0,
                    "available_moves": 0
                }
            })
            return
        
        # Generate and explore moves
        available_moves = generate_moves(current_state)
        
        # Limit branching factor
        if len(available_moves) > 5:
            # Prioritize foundation moves
            available_moves = _prioritize_moves(available_moves, current_state)[:5]
        
        for move in available_moves:
            new_state = apply_move(current_state, move)
            if new_state is not None:
                search(new_state, moves + [move], current_depth + 1)
    
    # Start search
    search(state, [], 0)
    
    # Sort by score and return top sequences
    sequences.sort(key=lambda x: x["score"], reverse=True)
    return sequences[:max_sequences]


def _evaluate_sequence(state: GameState, moves: List[Move]) -> float:
    """Evaluate the quality of a move sequence.
    
    Args:
        state: Final state after sequence
        moves: Sequence of moves
        
    Returns:
        Evaluation score (higher is better)
    """
    score = 0.0
    
    # Winning is best
    if state.is_won():
        return 10000.0
    
    # Cards in foundations (most important)
    foundation_cards = sum(len(f) for f in state.foundations)
    score += foundation_cards * 100
    
    # Fewer face-down cards is better
    face_down = state.count_face_down_cards()
    score += (28 - face_down) * 20  # 28 is max face-down at start
    
    # More available moves is better (flexibility)
    available_moves = len(generate_moves(state))
    score += available_moves * 5
    
    # Shorter sequences are better (tie-breaker)
    score -= len(moves) * 0.1
    
    # Game score
    score += state.score
    
    return score


def _prioritize_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Prioritize moves by likely value.
    
    Args:
        moves: List of moves to prioritize
        state: Current game state
        
    Returns:
        Prioritized list of moves
    """
    from solitaire_analytics.models.move import MoveType
    
    # Priority order
    priority = {
        MoveType.TABLEAU_TO_FOUNDATION: 5,
        MoveType.WASTE_TO_FOUNDATION: 5,
        MoveType.WASTE_TO_TABLEAU: 3,
        MoveType.TABLEAU_TO_TABLEAU: 2,
        MoveType.FLIP_TABLEAU_CARD: 4,
        MoveType.STOCK_TO_WASTE: 1,
    }
    
    # Sort by priority
    return sorted(moves, key=lambda m: priority.get(m.move_type, 0), reverse=True)


def generate_move_report(state: GameState, output_file: str = None) -> str:
    """Generate a JSON report of all moves and analysis.
    
    Args:
        state: Game state to analyze
        output_file: Optional file path to save report
        
    Returns:
        JSON string of the report
    """
    report = {
        "current_state": {
            "move_count": state.move_count,
            "score": state.score,
            "foundation_cards": sum(len(f) for f in state.foundations),
            "face_down_cards": state.count_face_down_cards(),
            "game_won": state.is_won()
        },
        "all_moves": compute_all_possible_moves(state),
        "best_sequences": find_best_move_sequences(state, depth=3, max_sequences=5)
    }
    
    json_str = json.dumps(report, indent=2)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_str)
    
    return json_str
