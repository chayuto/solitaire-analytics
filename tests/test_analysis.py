"""Tests for analysis components."""

import pytest
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.analysis import (
    MoveTreeBuilder,
    DeadEndDetector,
    compute_all_possible_moves,
    find_best_move_sequences
)


@pytest.mark.unit
@pytest.mark.analysis
class TestMoveTreeBuilder:
    """Test MoveTreeBuilder."""
    
    def test_build_tree_empty_state(self):
        """Test building tree from empty state."""
        state = GameState()
        builder = MoveTreeBuilder(max_depth=2, max_nodes=10)
        
        root = builder.build_tree(state)
        
        assert root is not None
        assert root.state == state
        assert root.depth == 0
    
    def test_build_tree_with_moves(self):
        """Test building tree with available moves."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        builder = MoveTreeBuilder(max_depth=2, max_nodes=100)
        root = builder.build_tree(state)
        
        # Should have children since moves are available
        assert len(root.children) > 0
    
    def test_get_statistics(self):
        """Test getting tree statistics."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        builder = MoveTreeBuilder(max_depth=2, max_nodes=50)
        builder.build_tree(state)
        
        stats = builder.get_statistics()
        
        assert "total_nodes" in stats
        assert "max_depth" in stats
        assert "leaf_nodes" in stats
        assert stats["total_nodes"] > 0
    
    def test_to_networkx_graph(self):
        """Test converting tree to NetworkX graph."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        builder = MoveTreeBuilder(max_depth=2, max_nodes=20)
        builder.build_tree(state)
        
        graph = builder.to_networkx_graph()
        
        assert graph is not None
        assert graph.number_of_nodes() > 0


@pytest.mark.unit
@pytest.mark.analysis
class TestDeadEndDetector:
    """Test DeadEndDetector."""
    
    def test_empty_state_is_dead_end(self):
        """Test that empty state is a dead end."""
        state = GameState()
        detector = DeadEndDetector()
        
        # Empty state with no moves is a dead end
        assert detector.is_dead_end(state)
    
    def test_won_game_not_dead_end(self):
        """Test that won game is not a dead end."""
        state = GameState()
        
        # Fill foundations
        for i in range(4):
            for rank in range(1, 14):
                state.foundations[i].append(Card(rank=rank, suit=list(Suit)[i]))
        
        detector = DeadEndDetector()
        assert not detector.is_dead_end(state)
    
    def test_state_with_moves_not_dead_end(self):
        """Test that state with available moves is not a dead end."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        detector = DeadEndDetector()
        assert not detector.is_dead_end(state)
    
    def test_analyze_dead_end_risk(self):
        """Test analyzing dead end risk."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        detector = DeadEndDetector()
        analysis = detector.analyze_dead_end_risk(state)
        
        assert "is_dead_end" in analysis
        assert "risk_score" in analysis
        assert "available_moves" in analysis
        assert 0 <= analysis["risk_score"] <= 1


@pytest.mark.unit
@pytest.mark.analysis
class TestMoveAnalyzer:
    """Test move analyzer functions."""
    
    def test_compute_all_possible_moves_empty_state(self):
        """Test computing moves on empty state."""
        state = GameState()
        moves = compute_all_possible_moves(state)
        
        assert isinstance(moves, list)
        assert len(moves) == 0
    
    def test_compute_all_possible_moves_with_cards(self):
        """Test computing moves with cards."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        moves = compute_all_possible_moves(state)
        
        assert len(moves) > 0
        
        # Check move info structure
        move_info = moves[0]
        assert "move" in move_info
        assert "score_delta" in move_info
        assert "wins_game" in move_info
    
    def test_find_best_move_sequences(self):
        """Test finding best move sequences."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        
        sequences = find_best_move_sequences(state, depth=2, max_sequences=5)
        
        assert isinstance(sequences, list)
        
        if len(sequences) > 0:
            seq = sequences[0]
            assert "moves" in seq
            assert "score" in seq
            assert "final_state_analysis" in seq
