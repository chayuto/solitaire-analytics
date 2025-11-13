"""GameState model representing the complete state of a Solitaire game."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json

from solitaire_analytics.models.card import Card


@dataclass
class GameState:
    """Represents the complete state of a Solitaire game.
    
    Attributes:
        tableau: Seven piles of cards in the main playing area
        foundations: Four foundation piles (one per suit)
        stock: Draw pile
        waste: Discard pile
        move_count: Number of moves made
        score: Current game score
    """
    tableau: List[List[Card]] = field(default_factory=lambda: [[] for _ in range(7)])
    foundations: List[List[Card]] = field(default_factory=lambda: [[] for _ in range(4)])
    stock: List[Card] = field(default_factory=list)
    waste: List[Card] = field(default_factory=list)
    move_count: int = 0
    score: int = 0

    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        return GameState(
            tableau=[pile[:] for pile in self.tableau],
            foundations=[pile[:] for pile in self.foundations],
            stock=self.stock[:],
            waste=self.waste[:],
            move_count=self.move_count,
            score=self.score
        )

    def is_won(self) -> bool:
        """Check if the game is won (all cards in foundations)."""
        return all(len(foundation) == 13 for foundation in self.foundations)

    def get_top_card(self, pile_type: str, pile_index: int) -> Optional[Card]:
        """Get the top card from a specified pile.
        
        Args:
            pile_type: Type of pile ('tableau', 'foundation', 'waste')
            pile_index: Index of the pile
            
        Returns:
            Top card or None if pile is empty
        """
        if pile_type == "tableau":
            pile = self.tableau[pile_index] if 0 <= pile_index < 7 else []
        elif pile_type == "foundation":
            pile = self.foundations[pile_index] if 0 <= pile_index < 4 else []
        elif pile_type == "waste":
            pile = self.waste
        else:
            return None
        
        return pile[-1] if pile else None

    def count_face_down_cards(self) -> int:
        """Count the number of face-down cards in tableau."""
        count = 0
        for pile in self.tableau:
            count += sum(1 for card in pile if not card.face_up)
        return count

    def get_available_moves_count(self) -> int:
        """Get a quick estimate of available moves (for heuristics)."""
        from solitaire_analytics.engine import generate_moves
        return len(generate_moves(self))

    def to_dict(self) -> Dict:
        """Convert game state to dictionary for serialization."""
        return {
            "tableau": [
                [{"rank": card.rank, "suit": card.suit.value, "face_up": card.face_up} 
                 for card in pile]
                for pile in self.tableau
            ],
            "foundations": [
                [{"rank": card.rank, "suit": card.suit.value} 
                 for card in pile]
                for pile in self.foundations
            ],
            "stock": [
                {"rank": card.rank, "suit": card.suit.value, "face_up": card.face_up}
                for card in self.stock
            ],
            "waste": [
                {"rank": card.rank, "suit": card.suit.value}
                for card in self.waste
            ],
            "move_count": self.move_count,
            "score": self.score
        }

    def to_json(self) -> str:
        """Convert game state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __hash__(self) -> int:
        """Generate hash for state comparison and memoization."""
        # Create a hashable representation of the state
        tableau_hash = tuple(
            tuple((c.rank, c.suit.value, c.face_up) for c in pile)
            for pile in self.tableau
        )
        foundations_hash = tuple(
            tuple((c.rank, c.suit.value) for c in pile)
            for pile in self.foundations
        )
        stock_hash = tuple((c.rank, c.suit.value, c.face_up) for c in self.stock)
        waste_hash = tuple((c.rank, c.suit.value) for c in self.waste)
        
        return hash((tableau_hash, foundations_hash, stock_hash, waste_hash))

    def __eq__(self, other) -> bool:
        """Check equality of game states."""
        if not isinstance(other, GameState):
            return False
        return hash(self) == hash(other)
