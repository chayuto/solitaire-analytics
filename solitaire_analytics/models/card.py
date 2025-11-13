"""Card model representing a playing card in Solitaire."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Suit(Enum):
    """Card suits."""
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class Color(Enum):
    """Card colors."""
    RED = "red"
    BLACK = "black"


@dataclass(frozen=True)
class Card:
    """Represents a playing card in Solitaire.
    
    Attributes:
        rank: Card rank (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King)
        suit: Card suit
        face_up: Whether the card is face up
    """
    rank: int
    suit: Suit
    face_up: bool = True

    def __post_init__(self):
        """Validate card attributes."""
        if not 1 <= self.rank <= 13:
            raise ValueError(f"Invalid rank: {self.rank}. Must be between 1 and 13.")

    @property
    def color(self) -> Color:
        """Get the color of the card."""
        if self.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return Color.RED
        return Color.BLACK

    @property
    def rank_name(self) -> str:
        """Get the name of the rank."""
        names = {
            1: "Ace",
            11: "Jack",
            12: "Queen",
            13: "King"
        }
        return names.get(self.rank, str(self.rank))

    def can_stack_on(self, other: Optional["Card"]) -> bool:
        """Check if this card can be stacked on another card in tableau.
        
        Args:
            other: The card to stack on, or None for an empty pile
            
        Returns:
            True if this card can be stacked on the other card
        """
        if other is None:
            # Only Kings can be placed on empty tableau piles
            return self.rank == 13
        
        # Must be opposite color and one rank lower
        return (
            self.color != other.color
            and self.rank == other.rank - 1
        )

    def can_place_on_foundation(self, other: Optional["Card"]) -> bool:
        """Check if this card can be placed on a foundation pile.
        
        Args:
            other: The top card of the foundation, or None for empty foundation
            
        Returns:
            True if this card can be placed on the foundation
        """
        if other is None:
            # Only Aces can start a foundation
            return self.rank == 1
        
        # Must be same suit and one rank higher
        return (
            self.suit == other.suit
            and self.rank == other.rank + 1
        )

    def __str__(self) -> str:
        """String representation of the card."""
        return f"{self.rank_name} of {self.suit.value}"

    def __repr__(self) -> str:
        """Detailed representation of the card."""
        face = "up" if self.face_up else "down"
        return f"Card(rank={self.rank}, suit={self.suit.value}, face_{face})"
