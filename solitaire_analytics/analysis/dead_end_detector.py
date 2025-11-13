"""Dead end detection for Solitaire games."""

from typing import Set, List, Dict
from solitaire_analytics.models import GameState, Card
from solitaire_analytics.models.card import Suit
from solitaire_analytics.engine import generate_moves


class DeadEndDetector:
    """Detector for identifying dead-end game states.
    
    A dead-end state is one where no winning sequence of moves exists.
    This class implements various heuristics to detect such states.
    """
    
    def __init__(self, strict: bool = False):
        """Initialize the dead end detector.
        
        Args:
            strict: If True, use stricter dead-end detection criteria
        """
        self.strict = strict
    
    def is_dead_end(self, state: GameState) -> bool:
        """Check if a game state is a dead end.
        
        Args:
            state: Game state to check
            
        Returns:
            True if the state is likely a dead end
        """
        # If game is won, it's not a dead end
        if state.is_won():
            return False
        
        # Check various dead-end conditions
        if self._has_no_moves(state):
            return True
        
        if self._has_blocking_cards(state):
            return True
        
        if self._has_unreachable_cards(state):
            return True
        
        if self.strict:
            if self._has_poor_configuration(state):
                return True
        
        return False
    
    def _has_no_moves(self, state: GameState) -> bool:
        """Check if there are no legal moves available.
        
        Args:
            state: Game state
            
        Returns:
            True if no moves are available
        """
        moves = generate_moves(state)
        
        # If stock is empty and no other moves, it's a dead end
        if not moves:
            return True
        
        # If only move is to draw from empty stock, it's a dead end
        if len(moves) == 1 and not state.stock:
            return True
        
        return False
    
    def _has_blocking_cards(self, state: GameState) -> bool:
        """Check if there are cards that permanently block others.
        
        A card is blocking if it must be moved before cards below it,
        but cannot be moved due to game rules.
        
        Args:
            state: Game state
            
        Returns:
            True if blocking cards exist
        """
        for pile in state.tableau:
            if not pile:
                continue
            
            # Check each face-down card
            for i, card in enumerate(pile):
                if not card.face_up:
                    # Check if cards above this one can ever be moved
                    cards_above = pile[i+1:]
                    if not self._can_cards_be_cleared(cards_above, state):
                        return True
        
        return False
    
    def _can_cards_be_cleared(self, cards: List[Card], state: GameState) -> bool:
        """Check if a sequence of cards can potentially be cleared.
        
        Args:
            cards: Cards to check
            state: Current game state
            
        Returns:
            True if cards can potentially be cleared
        """
        # Empty sequence can always be cleared
        if not cards:
            return True
        
        # If top card is face-down, we can't determine
        if not cards[0].face_up:
            return True  # Optimistically assume it can be cleared
        
        # Check if sequence is valid (alternating colors, descending ranks)
        for i in range(len(cards) - 1):
            if not cards[i+1].can_stack_on(cards[i]):
                return False
        
        return True
    
    def _has_unreachable_cards(self, state: GameState) -> bool:
        """Check if there are cards that can never be accessed.
        
        Args:
            state: Game state
            
        Returns:
            True if unreachable cards exist
        """
        # Cards in stock can always be reached
        # Cards in waste can be reached
        # Check tableau for cards that are permanently buried
        
        for pile in state.tableau:
            if len(pile) < 2:
                continue
            
            # Find first face-up card
            first_face_up = None
            for i, card in enumerate(pile):
                if card.face_up:
                    first_face_up = i
                    break
            
            if first_face_up is None:
                continue
            
            # Check if any face-down cards are below a King
            for i in range(first_face_up):
                if not pile[i].face_up:
                    # If there's a King above this, the card might be unreachable
                    if any(c.rank == 13 and c.face_up for c in pile[i+1:]):
                        # This is too conservative, skip for now
                        pass
        
        return False
    
    def _has_poor_configuration(self, state: GameState) -> bool:
        """Check if the game has a poor configuration (strict mode).
        
        This uses more aggressive heuristics that might have false positives.
        
        Args:
            state: Game state
            
        Returns:
            True if configuration is poor
        """
        # Check if foundations are far behind
        foundation_cards = sum(len(f) for f in state.foundations)
        total_face_up = sum(
            sum(1 for c in pile if c.face_up)
            for pile in state.tableau
        )
        
        # If we have many face-up cards but few in foundations, might be stuck
        if total_face_up > 20 and foundation_cards < 4:
            # Check if we have many moves available
            moves = generate_moves(state)
            if len(moves) < 3:
                return True
        
        return False
    
    def analyze_dead_end_risk(self, state: GameState) -> Dict:
        """Analyze the risk of a state being a dead end.
        
        Args:
            state: Game state to analyze
            
        Returns:
            Dictionary with risk analysis
        """
        analysis = {
            "is_dead_end": self.is_dead_end(state),
            "no_moves": self._has_no_moves(state),
            "has_blocking_cards": self._has_blocking_cards(state),
            "has_unreachable_cards": self._has_unreachable_cards(state),
            "available_moves": len(generate_moves(state)),
            "face_down_cards": state.count_face_down_cards(),
            "foundation_progress": sum(len(f) for f in state.foundations) / 52.0,
        }
        
        # Calculate risk score (0-1, higher is worse)
        risk_score = 0.0
        
        if analysis["no_moves"]:
            risk_score += 0.5
        if analysis["has_blocking_cards"]:
            risk_score += 0.3
        if analysis["has_unreachable_cards"]:
            risk_score += 0.2
        
        if analysis["available_moves"] == 0:
            risk_score = 1.0
        elif analysis["available_moves"] < 3:
            risk_score += 0.2
        
        analysis["risk_score"] = min(risk_score, 1.0)
        
        return analysis
