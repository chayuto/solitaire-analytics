"""Base strategy class for move selection in Solitaire games."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from solitaire_analytics.models import GameState, Move


@dataclass
class StrategyConfig:
    """Configuration for move selection strategies.
    
    Attributes:
        know_face_down_cards: Whether strategy has knowledge of face-down cards
        max_depth: Maximum depth for lookahead strategies
        priorities: Dictionary mapping move types to priority weights
        custom_params: Additional strategy-specific parameters
    """
    know_face_down_cards: bool = False
    max_depth: int = 3
    priorities: Dict[str, float] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "know_face_down_cards": self.know_face_down_cards,
            "max_depth": self.max_depth,
            "priorities": self.priorities.copy(),
            "custom_params": self.custom_params.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Create config from dictionary."""
        return cls(
            know_face_down_cards=data.get("know_face_down_cards", False),
            max_depth=data.get("max_depth", 3),
            priorities=data.get("priorities", {}),
            custom_params=data.get("custom_params", {}),
        )


class Strategy(ABC):
    """Abstract base class for move selection strategies.
    
    All strategies must implement the select_best_move method which
    analyzes a game state and returns the best move according to the
    strategy's algorithm.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration. If None, uses default config.
        """
        self.config = config or StrategyConfig()
    
    @abstractmethod
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select the best move for the given game state.
        
        Args:
            state: Current game state
            
        Returns:
            The best move according to this strategy, or None if no valid moves exist
        """
        pass
    
    def select_move_sequence(
        self,
        state: GameState,
        length: int = 3
    ) -> List[Move]:
        """Select a sequence of best moves.
        
        This default implementation repeatedly calls select_best_move and applies
        moves to get a sequence. Strategies can override for more sophisticated
        sequence planning.
        
        Args:
            state: Current game state
            length: Desired sequence length
            
        Returns:
            List of moves forming the best sequence
        """
        from solitaire_analytics.engine import apply_move
        
        sequence = []
        current_state = state
        
        for _ in range(length):
            if current_state.is_won():
                break
                
            best_move = self.select_best_move(current_state)
            if best_move is None:
                break
                
            sequence.append(best_move)
            current_state = apply_move(current_state, best_move)
            
            if current_state is None:
                # Invalid move was selected, truncate sequence
                sequence.pop()
                break
        
        return sequence
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the strategy name.
        
        Returns:
            Name of the strategy
        """
        pass
    
    def get_description(self) -> str:
        """Get a description of the strategy.
        
        Returns:
            Description of how the strategy works
        """
        return f"{self.get_name()} strategy"
