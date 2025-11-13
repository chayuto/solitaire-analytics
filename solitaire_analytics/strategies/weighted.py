"""Weighted strategy with configurable priorities."""

from typing import Optional, Dict

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves, apply_move
from solitaire_analytics.strategies.base import Strategy, StrategyConfig


class WeightedStrategy(Strategy):
    """Strategy that scores moves based on configurable weighted priorities.
    
    This strategy allows fine-tuned control over move selection by assigning
    weights to different factors:
    - Move type priorities (foundation moves, revealing cards, etc.)
    - Score delta from the move
    - Cards revealed after the move
    - Available moves after the move (flexibility)
    - Foundation progress
    
    The weights can be customized via the strategy configuration.
    """
    
    DEFAULT_PRIORITIES = {
        # Move type weights
        "tableau_to_foundation": 100.0,
        "waste_to_foundation": 100.0,
        "flip_tableau_card": 50.0,
        "waste_to_tableau": 30.0,
        "tableau_to_tableau": 20.0,
        "stock_to_waste": 5.0,
        # Factor weights
        "score_delta": 1.0,
        "reveals_card": 25.0,
        "foundation_progress": 10.0,
        "move_flexibility": 2.0,
    }
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize weighted strategy.
        
        Args:
            config: Strategy configuration with priorities
        """
        super().__init__(config)
        
        # Merge default priorities with custom ones
        self.priorities = self.DEFAULT_PRIORITIES.copy()
        if self.config.priorities:
            self.priorities.update(self.config.priorities)
    
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select the best move using weighted scoring.
        
        Args:
            state: Current game state
            
        Returns:
            The highest scored move, or None if no moves available
        """
        moves = generate_moves(state)
        
        if not moves:
            return None
        
        # Score each move
        scored_moves = []
        for move in moves:
            score = self._score_move(state, move)
            scored_moves.append((score, move))
        
        # Sort by score descending
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Return best move
        return scored_moves[0][1]
    
    def _score_move(self, state: GameState, move: Move) -> float:
        """Score a move based on weighted priorities.
        
        Args:
            state: Current game state
            move: Move to score
            
        Returns:
            Weighted score for the move
        """
        score = 0.0
        
        # Move type priority
        move_type_key = self._get_move_type_key(move.move_type)
        score += self.priorities.get(move_type_key, 0.0)
        
        # Score delta
        score += move.score_delta * self.priorities.get("score_delta", 1.0)
        
        # Simulate the move to get more information
        new_state = apply_move(state, move)
        if new_state is None:
            return score  # Invalid move, return base score
        
        # Check if move reveals a card
        old_face_down = state.count_face_down_cards()
        new_face_down = new_state.count_face_down_cards()
        if new_face_down < old_face_down:
            score += self.priorities.get("reveals_card", 0.0)
        
        # Foundation progress
        old_foundation = sum(len(f) for f in state.foundations)
        new_foundation = sum(len(f) for f in new_state.foundations)
        foundation_delta = new_foundation - old_foundation
        score += foundation_delta * self.priorities.get("foundation_progress", 0.0)
        
        # Move flexibility (more available moves after = better)
        if not self.config.know_face_down_cards:
            # Only consider flexibility if we don't know face-down cards
            available_after = len(generate_moves(new_state))
            score += available_after * self.priorities.get("move_flexibility", 0.0)
        
        return score
    
    def _get_move_type_key(self, move_type: MoveType) -> str:
        """Convert MoveType to priority key.
        
        Args:
            move_type: Type of move
            
        Returns:
            String key for priority lookup
        """
        type_map = {
            MoveType.TABLEAU_TO_FOUNDATION: "tableau_to_foundation",
            MoveType.WASTE_TO_FOUNDATION: "waste_to_foundation",
            MoveType.FLIP_TABLEAU_CARD: "flip_tableau_card",
            MoveType.WASTE_TO_TABLEAU: "waste_to_tableau",
            MoveType.TABLEAU_TO_TABLEAU: "tableau_to_tableau",
            MoveType.STOCK_TO_WASTE: "stock_to_waste",
        }
        return type_map.get(move_type, "unknown")
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "Weighted"
    
    def get_description(self) -> str:
        """Get strategy description."""
        return (
            "Weighted strategy that scores moves based on configurable priorities "
            "including move type, score changes, card reveals, and move flexibility."
        )
