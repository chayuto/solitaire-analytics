"""Tests for game engine."""

import pytest
from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.models.card import Suit
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves, validate_move, apply_move


@pytest.mark.unit
@pytest.mark.engine
class TestMoveGenerator:
    """Test move generation."""
    
    def test_generate_moves_empty_state(self):
        """Test move generation on empty state."""
        state = GameState()
        moves = generate_moves(state)
        
        # Empty state should have no moves
        assert len(moves) == 0
    
    def test_generate_tableau_to_foundation_moves(self):
        """Test generating tableau to foundation moves."""
        state = GameState()
        
        # Add an Ace to tableau
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        moves = generate_moves(state)
        
        # Should have moves to place Ace on foundations
        foundation_moves = [m for m in moves if m.move_type == MoveType.TABLEAU_TO_FOUNDATION]
        assert len(foundation_moves) > 0
    
    def test_generate_tableau_to_tableau_moves(self):
        """Test generating tableau to tableau moves."""
        state = GameState()
        
        # Setup: Red King on pile 0, Black Queen on pile 1
        state.tableau[0].append(Card(rank=13, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=12, suit=Suit.SPADES))
        
        moves = generate_moves(state)
        
        # Should be able to move Black Queen onto Red King
        tableau_moves = [m for m in moves if m.move_type == MoveType.TABLEAU_TO_TABLEAU]
        assert len(tableau_moves) > 0
    
    def test_generate_stock_to_waste_move(self):
        """Test generating stock to waste move."""
        state = GameState()
        
        # Add cards to stock
        state.stock.append(Card(rank=1, suit=Suit.HEARTS))
        
        moves = generate_moves(state)
        
        # Should have a draw move
        stock_moves = [m for m in moves if m.move_type == MoveType.STOCK_TO_WASTE]
        assert len(stock_moves) == 1


@pytest.mark.unit
@pytest.mark.engine
class TestMoveValidator:
    """Test move validation."""
    
    def test_validate_stock_to_waste(self):
        """Test validating stock to waste move."""
        state = GameState()
        state.stock.append(Card(rank=1, suit=Suit.HEARTS))
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        assert validate_move(state, move)
        
        # Should be invalid with empty stock
        empty_state = GameState()
        assert not validate_move(empty_state, move)
    
    def test_validate_tableau_to_foundation(self):
        """Test validating tableau to foundation move."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_FOUNDATION,
            source_pile=0,
            dest_pile=0
        )
        
        assert validate_move(state, move)
    
    def test_validate_tableau_to_tableau(self):
        """Test validating tableau to tableau move."""
        state = GameState()
        state.tableau[0].append(Card(rank=13, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=12, suit=Suit.SPADES))
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=1,
            dest_pile=0,
            num_cards=1
        )
        
        assert validate_move(state, move)
    
    def test_validate_invalid_move(self):
        """Test validating invalid moves."""
        state = GameState()
        
        # Invalid source pile
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=10,
            dest_pile=0
        )
        assert not validate_move(state, move)


@pytest.mark.unit
@pytest.mark.engine
class TestMoveApplication:
    """Test move application."""
    
    def test_apply_stock_to_waste(self):
        """Test applying stock to waste move."""
        state = GameState()
        card = Card(rank=5, suit=Suit.HEARTS)
        state.stock.append(card)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        new_state = apply_move(state, move)
        
        assert new_state is not None
        assert len(new_state.stock) == 0
        assert len(new_state.waste) == 1
        assert new_state.waste[0].rank == 5
    
    def test_apply_tableau_to_foundation(self):
        """Test applying tableau to foundation move."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_FOUNDATION,
            source_pile=0,
            dest_pile=0,
            score_delta=10
        )
        
        new_state = apply_move(state, move)
        
        assert new_state is not None
        assert len(new_state.tableau[0]) == 0
        assert len(new_state.foundations[0]) == 1
        assert new_state.score == 10
        assert new_state.move_count == 1
    
    def test_apply_tableau_to_tableau(self):
        """Test applying tableau to tableau move."""
        state = GameState()
        state.tableau[0].append(Card(rank=13, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=12, suit=Suit.SPADES))
        state.tableau[1].append(Card(rank=11, suit=Suit.HEARTS))
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=1,
            dest_pile=0,
            num_cards=2
        )
        
        new_state = apply_move(state, move)
        
        assert new_state is not None
        assert len(new_state.tableau[1]) == 0
        assert len(new_state.tableau[0]) == 3
        assert new_state.tableau[0][-1].rank == 11
    
    def test_apply_invalid_move_returns_none(self):
        """Test that applying invalid move returns None."""
        state = GameState()
        
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=0,
            dest_pile=1
        )
        
        new_state = apply_move(state, move)
        assert new_state is None
    
    def test_state_immutability(self):
        """Test that applying move doesn't modify original state."""
        state = GameState()
        card = Card(rank=5, suit=Suit.HEARTS)
        state.stock.append(card)
        
        move = Move(move_type=MoveType.STOCK_TO_WASTE)
        new_state = apply_move(state, move)
        
        # Original state should be unchanged
        assert len(state.stock) == 1
        assert len(state.waste) == 0
        
        # New state should have changes
        assert len(new_state.stock) == 0
        assert len(new_state.waste) == 1
