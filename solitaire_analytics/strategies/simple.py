"""Simple greedy strategy for move selection."""

from typing import Optional

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves
from solitaire_analytics.strategies.base import Strategy


class SimpleStrategy(Strategy):
    """Simple greedy strategy that prioritizes moves by type.
    
    This strategy uses a fixed priority order:
    1. Moves to foundation (highest priority)
    2. Flipping face-down cards
    3. Moves from waste to tableau
    4. Moves between tableau piles
    5. Drawing from stock (lowest priority)
    
    Within each priority level, the first available move is selected.
    """
    
    # Fixed priority order for move types
    PRIORITY_ORDER = {
        MoveType.TABLEAU_TO_FOUNDATION: 5,
        MoveType.WASTE_TO_FOUNDATION: 5,
        MoveType.FLIP_TABLEAU_CARD: 4,
        MoveType.WASTE_TO_TABLEAU: 3,
        MoveType.TABLEAU_TO_TABLEAU: 2,
        MoveType.STOCK_TO_WASTE: 1,
    }
    
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select the best move using simple greedy approach.
        
        Args:
            state: Current game state
            
        Returns:
            The highest priority move, or None if no moves available
        """
        moves = generate_moves(state)
        
        if not moves:
            return None
        
        # Sort moves by priority
        moves.sort(key=lambda m: self.PRIORITY_ORDER.get(m.move_type, 0), reverse=True)
        
        # Return highest priority move
        return moves[0]
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "Simple"
    
    def get_description(self) -> str:
        """Get strategy description."""
        return (
            "Simple greedy strategy that prioritizes moves by type: "
            "foundation moves first, then revealing cards, then tableau moves, "
            "and stock drawing as last resort."
        )
