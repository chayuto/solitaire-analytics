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
    
    def test_to_dict(self):
        """Test converting game state to dictionary."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS, face_up=True))
        state.tableau[0].append(Card(rank=2, suit=Suit.SPADES, face_up=False))
        state.foundations[0].append(Card(rank=1, suit=Suit.DIAMONDS))
        state.stock.append(Card(rank=3, suit=Suit.CLUBS, face_up=False))
        state.waste.append(Card(rank=4, suit=Suit.HEARTS))
        state.move_count = 5
        state.score = 100
        
        data = state.to_dict()
        
        assert len(data["tableau"]) == 7
        assert len(data["tableau"][0]) == 2
        assert data["tableau"][0][0]["rank"] == 1
        assert data["tableau"][0][0]["suit"] == "hearts"
        assert data["tableau"][0][0]["face_up"] is True
        assert data["tableau"][0][1]["face_up"] is False
        
        assert len(data["foundations"]) == 4
        assert len(data["foundations"][0]) == 1
        assert data["foundations"][0][0]["rank"] == 1
        assert data["foundations"][0][0]["suit"] == "diamonds"
        
        assert len(data["stock"]) == 1
        assert data["stock"][0]["rank"] == 3
        assert data["stock"][0]["suit"] == "clubs"
        assert data["stock"][0]["face_up"] is False
        
        assert len(data["waste"]) == 1
        assert data["waste"][0]["rank"] == 4
        assert data["waste"][0]["suit"] == "hearts"
        
        assert data["move_count"] == 5
        assert data["score"] == 100
    
    def test_to_json(self):
        """Test converting game state to JSON string."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.score = 100
        
        json_str = state.to_json()
        
        assert isinstance(json_str, str)
        assert "hearts" in json_str
        assert "100" in json_str
    
    def test_from_dict(self):
        """Test creating game state from dictionary."""
        data = {
            "tableau": [
                [
                    {"rank": 1, "suit": "hearts", "face_up": True},
                    {"rank": 2, "suit": "spades", "face_up": False}
                ],
                [], [], [], [], [], []
            ],
            "foundations": [
                [{"rank": 1, "suit": "diamonds"}],
                [], [], []
            ],
            "stock": [
                {"rank": 3, "suit": "clubs", "face_up": False}
            ],
            "waste": [
                {"rank": 4, "suit": "hearts"}
            ],
            "move_count": 5,
            "score": 100
        }
        
        state = GameState.from_dict(data)
        
        assert len(state.tableau) == 7
        assert len(state.tableau[0]) == 2
        assert state.tableau[0][0].rank == 1
        assert state.tableau[0][0].suit == Suit.HEARTS
        assert state.tableau[0][0].face_up is True
        assert state.tableau[0][1].rank == 2
        assert state.tableau[0][1].suit == Suit.SPADES
        assert state.tableau[0][1].face_up is False
        
        assert len(state.foundations) == 4
        assert len(state.foundations[0]) == 1
        assert state.foundations[0][0].rank == 1
        assert state.foundations[0][0].suit == Suit.DIAMONDS
        
        assert len(state.stock) == 1
        assert state.stock[0].rank == 3
        assert state.stock[0].suit == Suit.CLUBS
        assert state.stock[0].face_up is False
        
        assert len(state.waste) == 1
        assert state.waste[0].rank == 4
        assert state.waste[0].suit == Suit.HEARTS
        
        assert state.move_count == 5
        assert state.score == 100
    
    def test_from_json(self):
        """Test creating game state from JSON string."""
        json_str = """
        {
            "tableau": [
                [{"rank": 1, "suit": "hearts", "face_up": true}],
                [], [], [], [], [], []
            ],
            "foundations": [[], [], [], []],
            "stock": [],
            "waste": [],
            "move_count": 0,
            "score": 0
        }
        """
        
        state = GameState.from_json(json_str)
        
        assert len(state.tableau[0]) == 1
        assert state.tableau[0][0].rank == 1
        assert state.tableau[0][0].suit == Suit.HEARTS
        assert state.move_count == 0
        assert state.score == 0
    
    def test_import_export_roundtrip(self):
        """Test that exporting and importing preserves state."""
        # Create a complex game state
        original_state = GameState()
        original_state.tableau[0].append(Card(rank=13, suit=Suit.HEARTS, face_up=True))
        original_state.tableau[0].append(Card(rank=12, suit=Suit.SPADES, face_up=False))
        original_state.tableau[1].append(Card(rank=5, suit=Suit.DIAMONDS, face_up=True))
        original_state.foundations[0].append(Card(rank=1, suit=Suit.CLUBS))
        original_state.foundations[0].append(Card(rank=2, suit=Suit.CLUBS))
        original_state.stock.append(Card(rank=7, suit=Suit.HEARTS, face_up=False))
        original_state.stock.append(Card(rank=8, suit=Suit.SPADES, face_up=False))
        original_state.waste.append(Card(rank=9, suit=Suit.DIAMONDS))
        original_state.move_count = 10
        original_state.score = 250
        
        # Export to JSON and import back
        json_str = original_state.to_json()
        restored_state = GameState.from_json(json_str)
        
        # Verify all fields match
        assert len(restored_state.tableau) == len(original_state.tableau)
        for i in range(7):
            assert len(restored_state.tableau[i]) == len(original_state.tableau[i])
            for j in range(len(original_state.tableau[i])):
                assert restored_state.tableau[i][j].rank == original_state.tableau[i][j].rank
                assert restored_state.tableau[i][j].suit == original_state.tableau[i][j].suit
                assert restored_state.tableau[i][j].face_up == original_state.tableau[i][j].face_up
        
        assert len(restored_state.foundations) == len(original_state.foundations)
        for i in range(4):
            assert len(restored_state.foundations[i]) == len(original_state.foundations[i])
            for j in range(len(original_state.foundations[i])):
                assert restored_state.foundations[i][j].rank == original_state.foundations[i][j].rank
                assert restored_state.foundations[i][j].suit == original_state.foundations[i][j].suit
        
        assert len(restored_state.stock) == len(original_state.stock)
        for i in range(len(original_state.stock)):
            assert restored_state.stock[i].rank == original_state.stock[i].rank
            assert restored_state.stock[i].suit == original_state.stock[i].suit
            assert restored_state.stock[i].face_up == original_state.stock[i].face_up
        
        assert len(restored_state.waste) == len(original_state.waste)
        for i in range(len(original_state.waste)):
            assert restored_state.waste[i].rank == original_state.waste[i].rank
            assert restored_state.waste[i].suit == original_state.waste[i].suit
        
        assert restored_state.move_count == original_state.move_count
        assert restored_state.score == original_state.score
    
    def test_from_dict_empty_state(self):
        """Test creating empty game state from minimal dictionary."""
        data = {
            "tableau": [[], [], [], [], [], [], []],
            "foundations": [[], [], [], []],
            "stock": [],
            "waste": []
        }
        
        state = GameState.from_dict(data)
        
        assert all(len(pile) == 0 for pile in state.tableau)
        assert all(len(pile) == 0 for pile in state.foundations)
        assert len(state.stock) == 0
        assert len(state.waste) == 0
        assert state.move_count == 0
        assert state.score == 0


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
