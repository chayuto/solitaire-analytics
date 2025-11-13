"""Play logger for recording game moves and state for replay/visualization."""

import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from solitaire_analytics.models import GameState, Move


class PlayLogger:
    """Records game moves and state for replay or visualization.
    
    This logger captures the initial game state and all subsequent moves
    with timestamps, allowing games to be replayed or analyzed elsewhere.
    
    Attributes:
        enabled: Whether logging is active (default: False for efficiency)
        initial_state: The starting game state
        moves: List of move records with timestamps
        metadata: Optional metadata about the game session
    """
    
    def __init__(self, enabled: bool = False, metadata: Optional[Dict] = None):
        """Initialize the play logger.
        
        Args:
            enabled: Whether to enable logging (default: False)
            metadata: Optional metadata to include in the log (e.g., player name, session ID)
        """
        self.enabled = enabled
        self.initial_state: Optional[GameState] = None
        self.moves: List[Dict] = []
        self.metadata = metadata or {}
        self._start_time: Optional[datetime] = None
    
    def start(self, initial_state: GameState) -> None:
        """Begin logging with the initial game state.
        
        Args:
            initial_state: The starting game state to record
        """
        if not self.enabled:
            return
        
        self.initial_state = initial_state.copy()
        self._start_time = datetime.now()
        self.moves = []
    
    def log_move(self, move: Move, resulting_state: Optional[GameState] = None) -> None:
        """Log a move with timestamp.
        
        Args:
            move: The move that was made
            resulting_state: Optional resulting state after the move
        """
        if not self.enabled:
            return
        
        if self._start_time is None:
            raise ValueError("Logger not started. Call start() with initial state first.")
        
        timestamp = (datetime.now() - self._start_time).total_seconds()
        
        move_record = {
            "timestamp": timestamp,
            "move": move.to_dict(),
        }
        
        # Optionally include the resulting state for full replay capability
        if resulting_state is not None:
            move_record["resulting_state"] = resulting_state.to_dict()
        
        self.moves.append(move_record)
    
    def to_dict(self) -> Dict:
        """Export the play log as a dictionary.
        
        Returns:
            Dictionary containing initial state, moves, and metadata
        """
        if not self.enabled or self.initial_state is None:
            return {
                "enabled": False,
                "message": "Logging was not enabled or no game was recorded"
            }
        
        return {
            "enabled": True,
            "metadata": self.metadata,
            "initial_state": self.initial_state.to_dict(),
            "moves": self.moves,
            "move_count": len(self.moves),
            "start_time": self._start_time.isoformat() if self._start_time else None,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export the play log as a JSON string.
        
        Args:
            indent: Number of spaces for indentation (default: 2)
            
        Returns:
            JSON string representation of the play log
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save the play log to a JSON file.
        
        Args:
            filepath: Path to save the log file
        """
        if not self.enabled:
            raise ValueError("Cannot save log when logging is not enabled")
        
        if self.initial_state is None:
            raise ValueError("Cannot save log before starting (no initial state recorded)")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    def clear(self) -> None:
        """Clear all logged data."""
        self.initial_state = None
        self.moves = []
        self._start_time = None
    
    @classmethod
    def load(cls, filepath: str) -> "PlayLogger":
        """Load a play log from a JSON file.
        
        Args:
            filepath: Path to the log file
            
        Returns:
            PlayLogger instance with loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data.get("enabled", False):
            logger = cls(enabled=False)
            return logger
        
        logger = cls(enabled=True, metadata=data.get("metadata", {}))
        
        # Reconstruct initial state
        if "initial_state" in data:
            logger.initial_state = GameState.from_dict(data["initial_state"])
        
        # Load moves
        logger.moves = data.get("moves", [])
        
        # Reconstruct start time
        if data.get("start_time"):
            logger._start_time = datetime.fromisoformat(data["start_time"])
        
        return logger
