"""Move model representing a game move in Solitaire."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MoveType(Enum):
    """Types of moves in Solitaire."""
    TABLEAU_TO_TABLEAU = "tableau_to_tableau"
    TABLEAU_TO_FOUNDATION = "tableau_to_foundation"
    WASTE_TO_TABLEAU = "waste_to_tableau"
    WASTE_TO_FOUNDATION = "waste_to_foundation"
    STOCK_TO_WASTE = "stock_to_waste"
    FLIP_TABLEAU_CARD = "flip_tableau_card"
    FOUNDATION_TO_TABLEAU = "foundation_to_tableau"  # Undo support


@dataclass(frozen=True)
class Move:
    """Represents a move in Solitaire.
    
    Attributes:
        move_type: Type of move
        source_pile: Index of source pile (for tableau/foundation)
        dest_pile: Index of destination pile
        num_cards: Number of cards to move (for multi-card tableau moves)
        score_delta: Change in score from this move
    """
    move_type: MoveType
    source_pile: Optional[int] = None
    dest_pile: Optional[int] = None
    num_cards: int = 1
    score_delta: int = 0

    def __str__(self) -> str:
        """String representation of the move."""
        if self.move_type == MoveType.STOCK_TO_WASTE:
            return "Draw from stock"
        elif self.move_type == MoveType.FLIP_TABLEAU_CARD:
            return f"Flip card on tableau pile {self.source_pile}"
        elif self.move_type == MoveType.TABLEAU_TO_TABLEAU:
            return f"Move {self.num_cards} card(s) from tableau {self.source_pile} to tableau {self.dest_pile}"
        elif self.move_type == MoveType.TABLEAU_TO_FOUNDATION:
            return f"Move card from tableau {self.source_pile} to foundation {self.dest_pile}"
        elif self.move_type == MoveType.WASTE_TO_TABLEAU:
            return f"Move card from waste to tableau {self.dest_pile}"
        elif self.move_type == MoveType.WASTE_TO_FOUNDATION:
            return f"Move card from waste to foundation {self.dest_pile}"
        elif self.move_type == MoveType.FOUNDATION_TO_TABLEAU:
            return f"Move card from foundation {self.source_pile} to tableau {self.dest_pile}"
        else:
            return f"Unknown move type: {self.move_type}"

    def __repr__(self) -> str:
        """Detailed representation of the move."""
        return (
            f"Move(type={self.move_type.value}, "
            f"source={self.source_pile}, dest={self.dest_pile}, "
            f"num_cards={self.num_cards}, score_delta={self.score_delta})"
        )

    def to_dict(self) -> dict:
        """Convert move to dictionary for serialization."""
        return {
            "move_type": self.move_type.value,
            "source_pile": self.source_pile,
            "dest_pile": self.dest_pile,
            "num_cards": self.num_cards,
            "score_delta": self.score_delta,
            "description": str(self)
        }
