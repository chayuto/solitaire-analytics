"""Tests for analysis components."""

import pytest
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.analysis import (
    MoveTreeBuilder,
    DeadEndDetector,
    compute_all_possible_moves,
    find_best_move_sequences,
    calculate_progression_score
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


@pytest.mark.unit
@pytest.mark.analysis
class TestProgressionScore:
    """Test progression score calculation."""
    
    def test_empty_state_progression_score(self):
        """Test progression score for empty state."""
        state = GameState()
        score = calculate_progression_score(state)
        
        # Empty state has moderate progression due to no face-down cards and no stock/waste
        # (reveal_progress=1.0 * 0.2 + stock_progress=1.0 * 0.1 = 0.3)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert 0.3 <= score <= 0.5  # Empty state has some baseline progress
    
    def test_won_game_progression_score(self):
        """Test progression score for won game."""
        state = GameState()
        
        # Fill all foundations with complete suits
        for i in range(4):
            for rank in range(1, 14):
                state.foundations[i].append(Card(rank=rank, suit=list(Suit)[i]))
        
        score = calculate_progression_score(state)
        
        # Won game should have score of 1.0
        assert score == 1.0
    
    def test_progression_score_with_some_foundation_cards(self):
        """Test progression score with cards in foundations."""
        state = GameState()
        
        # Add some cards to foundations (e.g., 13 cards = 1 complete suit)
        for rank in range(1, 14):
            state.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        
        score = calculate_progression_score(state)
        
        # Should have meaningful progress (13/52 ≈ 0.25 of foundation progress)
        # Plus baseline from empty state
        assert 0.4 <= score <= 0.7
    
    def test_progression_score_increases_with_foundation_cards(self):
        """Test that progression score increases as more cards go to foundations."""
        state1 = GameState()
        state2 = GameState()
        
        # State 1: Few foundation cards
        for rank in range(1, 4):
            state1.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        
        # State 2: More foundation cards
        for rank in range(1, 10):
            state2.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        
        score1 = calculate_progression_score(state1)
        score2 = calculate_progression_score(state2)
        
        # Score should increase with more foundation cards
        assert score2 > score1
    
    def test_progression_score_with_face_down_cards(self):
        """Test progression score considering face-down cards."""
        state1 = GameState()
        state2 = GameState()
        
        # State 1: Many face-down cards
        for i in range(7):
            for j in range(i + 1):
                card = Card(rank=j+1, suit=Suit.HEARTS, face_up=(j == i))
                state1.tableau[i].append(card)
        
        # State 2: Same but with more cards face-up
        for i in range(7):
            for j in range(i + 1):
                card = Card(rank=j+1, suit=Suit.HEARTS, face_up=True)
                state2.tableau[i].append(card)
        
        score1 = calculate_progression_score(state1)
        score2 = calculate_progression_score(state2)
        
        # More face-up cards should give higher score
        assert score2 > score1
    
    def test_progression_score_with_empty_piles(self):
        """Test progression score considering empty tableau piles."""
        state1 = GameState()
        state2 = GameState()
        
        # State 1: Some cards in tableau
        state1.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state1.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        
        # State 2: Same foundation progress but more empty piles
        state2.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        # Leave other piles empty
        
        score1 = calculate_progression_score(state1)
        score2 = calculate_progression_score(state2)
        
        # More empty piles should give slightly higher score
        assert score2 >= score1
    
    def test_progression_score_with_stock_and_waste(self):
        """Test progression score considering stock and waste."""
        state1 = GameState()
        state2 = GameState()
        
        # State 1: Many cards in stock
        for i in range(24):
            state1.stock.append(Card(rank=(i % 13) + 1, suit=Suit.HEARTS))
        
        # State 2: Fewer cards in stock
        for i in range(10):
            state2.stock.append(Card(rank=(i % 13) + 1, suit=Suit.HEARTS))
        
        score1 = calculate_progression_score(state1)
        score2 = calculate_progression_score(state2)
        
        # Fewer stock cards should give higher score
        assert score2 >= score1
    
    def test_progression_score_range(self):
        """Test that progression score is always in valid range."""
        # Test various game states
        states = []
        
        # Empty state
        states.append(GameState())
        
        # State with some cards
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        states.append(state)
        
        # State with foundation cards
        state = GameState()
        for rank in range(1, 6):
            state.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        states.append(state)
        
        # Test all states
        for state in states:
            score = calculate_progression_score(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"
    
    def test_progression_score_monotonicity(self):
        """Test that progression score generally increases toward winning."""
        # Create a sequence of states moving toward completion
        states = []
        
        # State 0: Empty
        states.append(GameState())
        
        # State 1: Some foundation cards
        state = GameState()
        for rank in range(1, 7):
            state.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        states.append(state)
        
        # State 2: More foundation cards
        state = GameState()
        for rank in range(1, 14):
            state.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
        states.append(state)
        
        # State 3: Even more foundation cards (2 suits)
        state = GameState()
        for rank in range(1, 14):
            state.foundations[0].append(Card(rank=rank, suit=Suit.HEARTS))
            state.foundations[1].append(Card(rank=rank, suit=Suit.DIAMONDS))
        states.append(state)
        
        # Calculate scores
        scores = [calculate_progression_score(s) for s in states]
        
        # Scores should generally increase
        for i in range(len(scores) - 1):
            assert scores[i+1] > scores[i], \
                f"Score should increase: {scores[i]} -> {scores[i+1]}"
