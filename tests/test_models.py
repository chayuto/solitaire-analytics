"""Tests for data models."""

import pytest
from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.models.card import Suit, Color
from solitaire_analytics.models.move import MoveType


@pytest.mark.unit
@pytest.mark.models
class TestCard:
    """Test Card model."""
    
    def test_card_creation(self):
        """Test creating a card."""
        card = Card(rank=1, suit=Suit.HEARTS)
        assert card.rank == 1
        assert card.suit == Suit.HEARTS
        assert card.face_up is True
    
    def test_card_color(self):
        """Test card color property."""
        red_card = Card(rank=1, suit=Suit.HEARTS)
        assert red_card.color == Color.RED
        
        black_card = Card(rank=1, suit=Suit.SPADES)
        assert black_card.color == Color.BLACK
    
    def test_card_rank_name(self):
        """Test card rank name property."""
        ace = Card(rank=1, suit=Suit.HEARTS)
        assert ace.rank_name == "Ace"
        
        jack = Card(rank=11, suit=Suit.HEARTS)
        assert jack.rank_name == "Jack"
        
        five = Card(rank=5, suit=Suit.HEARTS)
        assert five.rank_name == "5"
    
    def test_invalid_rank(self):
        """Test that invalid ranks raise ValueError."""
        with pytest.raises(ValueError):
            Card(rank=0, suit=Suit.HEARTS)
        
        with pytest.raises(ValueError):
            Card(rank=14, suit=Suit.HEARTS)
    
    def test_can_stack_on_tableau(self):
        """Test tableau stacking rules."""
        red_king = Card(rank=13, suit=Suit.HEARTS)
        black_queen = Card(rank=12, suit=Suit.SPADES)
        red_jack = Card(rank=11, suit=Suit.DIAMONDS)
        
        # King can go on empty pile
        assert red_king.can_stack_on(None)
        
        # Queen cannot go on empty pile
        assert not black_queen.can_stack_on(None)
        
        # Black Queen can go on red King
        assert black_queen.can_stack_on(red_king)
        
        # Red Jack can go on black Queen
        assert red_jack.can_stack_on(black_queen)
        
        # Same color cannot stack
        assert not red_jack.can_stack_on(red_king)
    
    def test_can_place_on_foundation(self):
        """Test foundation placement rules."""
        ace_hearts = Card(rank=1, suit=Suit.HEARTS)
        two_hearts = Card(rank=2, suit=Suit.HEARTS)
        two_spades = Card(rank=2, suit=Suit.SPADES)
        
        # Ace can start foundation
        assert ace_hearts.can_place_on_foundation(None)
        
        # Two cannot start foundation
        assert not two_hearts.can_place_on_foundation(None)
        
        # Two of hearts can go on Ace of hearts
        assert two_hearts.can_place_on_foundation(ace_hearts)
        
        # Different suit cannot go on foundation
        assert not two_spades.can_place_on_foundation(ace_hearts)


@pytest.mark.unit
@pytest.mark.models
class TestGameState:
    """Test GameState model."""
    
    def test_game_state_creation(self):
        """Test creating a game state."""
        state = GameState()
        assert len(state.tableau) == 7
        assert len(state.foundations) == 4
        assert state.move_count == 0
        assert state.score == 0
    
    def test_game_state_copy(self):
        """Test copying a game state."""
        state = GameState()
        state.score = 100
        state.move_count = 5
        
        copy = state.copy()
        assert copy.score == 100
        assert copy.move_count == 5
        
        # Modify copy shouldn't affect original
        copy.score = 200
        assert state.score == 100
    
    def test_is_won(self):
        """Test win condition."""
        state = GameState()
        assert not state.is_won()
        
        # Fill foundations with 13 cards each
        for i in range(4):
            for rank in range(1, 14):
                state.foundations[i].append(Card(rank=rank, suit=list(Suit)[i]))
        
        assert state.is_won()
    
    def test_get_top_card(self):
        """Test getting top card from piles."""
        state = GameState()
        
        # Empty pile returns None
        assert state.get_top_card("tableau", 0) is None
        
        # Add a card and test
        card = Card(rank=5, suit=Suit.HEARTS)
        state.tableau[0].append(card)
        
        top_card = state.get_top_card("tableau", 0)
        assert top_card.rank == 5
        assert top_card.suit == Suit.HEARTS
    
    def test_count_face_down_cards(self):
        """Test counting face-down cards."""
        state = GameState()
        assert state.count_face_down_cards() == 0
        
        # Add some face-down cards
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=False))
        state.tableau[0].append(Card(rank=2, suit=Suit.HEARTS, face_up=False))
        state.tableau[1].append(Card(rank=3, suit=Suit.SPADES, face_up=False))
        
        assert state.count_face_down_cards() == 3
    
    def test_game_state_hash(self):
        """Test game state hashing."""
        state1 = GameState()
        state2 = GameState()
        
        # Same states should have same hash
        assert hash(state1) == hash(state2)
        
        # Different states should have different hash
        state1.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        assert hash(state1) != hash(state2)


@pytest.mark.unit
@pytest.mark.models
class TestMove:
    """Test Move model."""
    
    def test_move_creation(self):
        """Test creating a move."""
        move = Move(
            move_type=MoveType.TABLEAU_TO_TABLEAU,
            source_pile=0,
            dest_pile=1,
            num_cards=3
        )
        
        assert move.move_type == MoveType.TABLEAU_TO_TABLEAU
        assert move.source_pile == 0
        assert move.dest_pile == 1
        assert move.num_cards == 3
    
    def test_move_string(self):
        """Test move string representation."""
        move = Move(
            move_type=MoveType.STOCK_TO_WASTE
        )
        
        assert "Draw from stock" in str(move)
    
    def test_move_to_dict(self):
        """Test move serialization."""
        move = Move(
            move_type=MoveType.TABLEAU_TO_FOUNDATION,
            source_pile=0,
            dest_pile=1,
            score_delta=10
        )
        
        move_dict = move.to_dict()
        assert move_dict["move_type"] == "tableau_to_foundation"
        assert move_dict["source_pile"] == 0
        assert move_dict["dest_pile"] == 1
        assert move_dict["score_delta"] == 10
