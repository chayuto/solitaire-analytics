"""Lookahead strategy that evaluates move sequences."""

from typing import Optional, List, Tuple

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move
from solitaire_analytics.strategies.base import Strategy


class LookaheadStrategy(Strategy):
    """Strategy that looks ahead multiple moves to find best sequences.
    
    This strategy performs a limited depth search to evaluate the quality
    of move sequences, not just individual moves. It uses the existing
    move sequence evaluation logic but wraps it in the strategy pattern.
    
    The depth of lookahead is configurable via max_depth in the config.
    """
    
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select best move by evaluating future sequences.
        
        Args:
            state: Current game state
            
        Returns:
            First move of the best sequence, or None if no moves available
        """
        sequences = self._find_best_sequences(
            state,
            depth=self.config.max_depth,
            max_sequences=5
        )
        
        if not sequences:
            return None
        
        # Return first move of best sequence
        best_sequence = sequences[0]
        if best_sequence["moves"]:
            return best_sequence["moves"][0]
        
        return None
    
    def select_move_sequence(
        self,
        state: GameState,
        length: int = 3
    ) -> List[Move]:
        """Select a sequence of best moves using lookahead.
        
        Args:
            state: Current game state
            length: Desired sequence length
            
        Returns:
            List of moves forming the best sequence
        """
        sequences = self._find_best_sequences(
            state,
            depth=length,
            max_sequences=1
        )
        
        if not sequences:
            return []
        
        return sequences[0]["moves"]
    
    def _find_best_sequences(
        self,
        state: GameState,
        depth: int,
        max_sequences: int
    ) -> List[dict]:
        """Find the best move sequences using depth-limited search.
        
        Args:
            state: Current game state
            depth: How many moves to look ahead
            max_sequences: Maximum number of sequences to return
            
        Returns:
            List of dictionaries containing move sequences and scores
        """
        sequences = []
        
        def search(current_state: GameState, moves: List[Move], current_depth: int):
            """Recursive search for move sequences."""
            # If game is won, record and stop
            if current_state.is_won():
                score = self._evaluate_sequence(current_state, moves)
                sequences.append({
                    "moves": moves[:],
                    "score": score,
                    "final_state": {
                        "game_won": True,
                        "foundation_cards": 52,
                        "face_down_cards": 0,
                    }
                })
                return
            
            # Generate and explore moves
            available_moves = generate_moves(current_state)
            
            # If no moves available or reached depth, evaluate current sequence
            if not available_moves or current_depth >= depth:
                # Only record if we have at least one move
                if moves:
                    score = self._evaluate_sequence(current_state, moves)
                    sequences.append({
                        "moves": moves[:],  # Copy the list
                        "score": score,
                        "final_state": {
                            "game_won": current_state.is_won(),
                            "foundation_cards": sum(len(f) for f in current_state.foundations),
                            "face_down_cards": current_state.count_face_down_cards(),
                        }
                    })
                return
            
            # Limit branching factor
            if len(available_moves) > 5:
                # Prioritize foundation moves
                available_moves = self._prioritize_moves(available_moves)[:5]
            
            for move in available_moves:
                new_state = apply_move(current_state, move)
                if new_state is not None:
                    search(new_state, moves + [move], current_depth + 1)
        
        # Start search
        search(state, [], 0)
        
        # Sort by score and return top sequences
        sequences.sort(key=lambda x: x["score"], reverse=True)
        return sequences[:max_sequences]
    
    def _evaluate_sequence(self, state: GameState, moves: List[Move]) -> float:
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
    
    def _prioritize_moves(self, moves: List[Move]) -> List[Move]:
        """Prioritize moves by likely value.
        
        Args:
            moves: List of moves to prioritize
            
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
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "Lookahead"
    
    def get_description(self) -> str:
        """Get strategy description."""
        return (
            f"Lookahead strategy that evaluates move sequences up to "
            f"{self.config.max_depth} moves ahead to find the best path forward."
        )
