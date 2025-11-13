"""Tests for PlayLogger functionality."""

import json
import pytest
import tempfile
import os
from datetime import datetime

from solitaire_analytics import Card, GameState, Move, PlayLogger
from solitaire_analytics.models.card import Suit
from solitaire_analytics.models.move import MoveType


class TestPlayLogger:
    """Test suite for PlayLogger class."""
    
    def test_logger_initialization_disabled(self):
        """Test that logger can be created in disabled state (default)."""
        logger = PlayLogger()
        assert logger.enabled is False
        assert logger.initial_state is None
        assert logger.moves == []
        assert logger.metadata == {}
    
    def test_logger_initialization_enabled(self):
        """Test that logger can be created in enabled state."""
        metadata = {"player": "test_player", "session": "test_session"}
        logger = PlayLogger(enabled=True, metadata=metadata)
        assert logger.enabled is True
        assert logger.metadata == metadata
    
    def test_start_disabled_logger(self):
        """Test that starting disabled logger does nothing."""
        logger = PlayLogger(enabled=False)
        state = GameState()
        logger.start(state)
        assert logger.initial_state is None
    
    def test_start_enabled_logger(self):
        """Test that starting enabled logger records initial state."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        logger.start(state)
        
        assert logger.initial_state is not None
        assert len(logger.initial_state.tableau[0]) == 1
        assert logger.moves == []
    
    def test_log_move_disabled(self):
        """Test that logging moves when disabled does nothing."""
        logger = PlayLogger(enabled=False)
        state = GameState()
        logger.start(state)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        logger.log_move(move)
        
        assert logger.moves == []
    
    def test_log_move_without_start_raises_error(self):
        """Test that logging a move before start raises error."""
        logger = PlayLogger(enabled=True)
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        
        with pytest.raises(ValueError, match="Logger not started"):
            logger.log_move(move)
    
    def test_log_move_records_move(self):
        """Test that logging a move records it correctly."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        logger.start(state)
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_FOUNDATION,
            source_pile=0,
            dest_pile=0,
            score_delta=10
        )
        logger.log_move(move)
        
        assert len(logger.moves) == 1
        assert logger.moves[0]["move"]["move_type"] == "tableau_to_foundation"
        assert logger.moves[0]["move"]["source_pile"] == 0
        assert logger.moves[0]["move"]["dest_pile"] == 0
        assert "timestamp" in logger.moves[0]
    
    def test_log_move_with_resulting_state(self):
        """Test that logging a move can include resulting state."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        logger.start(state)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        resulting_state = state.copy()
        resulting_state.move_count += 1
        
        logger.log_move(move, resulting_state=resulting_state)
        
        assert len(logger.moves) == 1
        assert "resulting_state" in logger.moves[0]
        assert logger.moves[0]["resulting_state"]["move_count"] == 1
    
    def test_log_multiple_moves(self):
        """Test that multiple moves can be logged."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        logger.start(state)
        
        for i in range(3):
            move = Move(move_type=MoveType.STOCK_TO_WASTE)
            logger.log_move(move)
        
        assert len(logger.moves) == 3
        # Timestamps should be in order
        assert logger.moves[0]["timestamp"] <= logger.moves[1]["timestamp"]
        assert logger.moves[1]["timestamp"] <= logger.moves[2]["timestamp"]
    
    def test_to_dict_disabled(self):
        """Test to_dict when logger is disabled."""
        logger = PlayLogger(enabled=False)
        data = logger.to_dict()
        
        assert data["enabled"] is False
        assert "message" in data
    
    def test_to_dict_enabled_no_start(self):
        """Test to_dict when logger is enabled but not started."""
        logger = PlayLogger(enabled=True)
        data = logger.to_dict()
        
        assert data["enabled"] is False
        assert "message" in data
    
    def test_to_dict_with_data(self):
        """Test to_dict with recorded data."""
        metadata = {"player": "test"}
        logger = PlayLogger(enabled=True, metadata=metadata)
        
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        logger.start(state)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        logger.log_move(move)
        
        data = logger.to_dict()
        
        assert data["enabled"] is True
        assert data["metadata"] == metadata
        assert "initial_state" in data
        assert data["move_count"] == 1
        assert len(data["moves"]) == 1
        assert "start_time" in data
    
    def test_to_json(self):
        """Test JSON export."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        logger.start(state)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        logger.log_move(move)
        
        json_str = logger.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["enabled"] is True
        assert data["move_count"] == 1
    
    def test_save_and_load(self):
        """Test saving and loading play log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_log.json")
            
            # Create and save log
            metadata = {"player": "test", "difficulty": "hard"}
            logger = PlayLogger(enabled=True, metadata=metadata)
            
            state = GameState()
            state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
            state.tableau[1].append(Card(rank=2, suit=Suit.SPADES))
            logger.start(state)
            
            move1 = Move(move_type=MoveType.STOCK_TO_WASTE)
            move2 = Move(
                move_type=MoveType.TABLEAU_TO_FOUNDATION,
                source_pile=0,
                dest_pile=0
            )
            logger.log_move(move1)
            logger.log_move(move2)
            
            logger.save(filepath)
            
            # Load log
            loaded_logger = PlayLogger.load(filepath)
            
            assert loaded_logger.enabled is True
            assert loaded_logger.metadata == metadata
            assert loaded_logger.initial_state is not None
            assert len(loaded_logger.initial_state.tableau[0]) == 1
            assert len(loaded_logger.initial_state.tableau[1]) == 1
            assert len(loaded_logger.moves) == 2
            assert loaded_logger.moves[0]["move"]["move_type"] == "stock_to_waste"
            assert loaded_logger.moves[1]["move"]["move_type"] == "tableau_to_foundation"
    
    def test_save_without_enabling_raises_error(self):
        """Test that saving disabled logger raises error."""
        logger = PlayLogger(enabled=False)
        
        with pytest.raises(ValueError, match="not enabled"):
            logger.save("/tmp/test.json")
    
    def test_save_without_start_raises_error(self):
        """Test that saving before start raises error."""
        logger = PlayLogger(enabled=True)
        
        with pytest.raises(ValueError, match="no initial state"):
            logger.save("/tmp/test.json")
    
    def test_clear(self):
        """Test clearing logged data."""
        logger = PlayLogger(enabled=True)
        state = GameState()
        logger.start(state)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        logger.log_move(move)
        
        assert logger.initial_state is not None
        assert len(logger.moves) > 0
        
        logger.clear()
        
        assert logger.initial_state is None
        assert logger.moves == []
    
    def test_load_disabled_log(self):
        """Test loading a disabled log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "disabled_log.json")
            
            # Create disabled log data
            with open(filepath, 'w') as f:
                json.dump({"enabled": False, "message": "Not enabled"}, f)
            
            loaded_logger = PlayLogger.load(filepath)
            assert loaded_logger.enabled is False


@pytest.mark.unit
class TestPlayLoggerIntegration:
    """Integration tests for PlayLogger with game play."""
    
    def test_full_game_logging(self):
        """Test logging a full game session."""
        logger = PlayLogger(
            enabled=True,
            metadata={"game_type": "klondike", "player": "test_user"}
        )
        
        # Set up initial state
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))
        state.tableau[1].append(Card(rank=2, suit=Suit.SPADES, face_up=True))
        state.stock.extend([
            Card(rank=3, suit=Suit.CLUBS, face_up=False),
            Card(rank=4, suit=Suit.DIAMONDS, face_up=False),
        ])
        
        logger.start(state)
        
        # Simulate some moves
        moves = [
            Move(move_type=MoveType.STOCK_TO_WASTE),
            Move(
                move_type=MoveType.TABLEAU_TO_FOUNDATION,
                source_pile=0,
                dest_pile=0,
                score_delta=10
            ),
            Move(
                move_type=MoveType.TABLEAU_TO_TABLEAU,
                source_pile=1,
                dest_pile=0,
                num_cards=1
            ),
        ]
        
        for move in moves:
            logger.log_move(move)
        
        # Verify log
        data = logger.to_dict()
        assert data["move_count"] == 3
        assert len(data["moves"]) == 3
        
        # Verify initial state preserved
        assert len(data["initial_state"]["tableau"][0]) == 1
        assert len(data["initial_state"]["stock"]) == 2
        
        # Verify timestamps are reasonable
        for i in range(len(moves) - 1):
            assert data["moves"][i]["timestamp"] <= data["moves"][i + 1]["timestamp"]
    
    def test_replay_from_log(self):
        """Test that a log contains enough information for replay."""
        logger = PlayLogger(enabled=True)
        
        # Create a game state
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))
        state.tableau[1].append(Card(rank=13, suit=Suit.SPADES, face_up=True))
        
        logger.start(state)
        
        # Log a move with resulting state
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=0,
            dest_pile=1,
            num_cards=1
        )
        
        new_state = state.copy()
        card = new_state.tableau[0].pop()
        new_state.tableau[1].append(card)
        new_state.move_count += 1
        
        logger.log_move(move, resulting_state=new_state)
        
        # Export and verify
        data = logger.to_dict()
        
        # Should have initial state
        assert len(data["initial_state"]["tableau"][0]) == 1
        assert len(data["initial_state"]["tableau"][1]) == 1
        
        # Should have move with resulting state
        assert data["moves"][0]["move"]["move_type"] == "tableau_to_tableau"
        assert "resulting_state" in data["moves"][0]
        assert len(data["moves"][0]["resulting_state"]["tableau"][0]) == 0
        assert len(data["moves"][0]["resulting_state"]["tableau"][1]) == 2
