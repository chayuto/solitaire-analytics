"""Tests for strategy module."""

import pytest
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.strategies import (
    Strategy,
    StrategyConfig,
    SimpleStrategy,
    WeightedStrategy,
    LookaheadStrategy,
    StrategyRegistry,
    get_strategy,
)


@pytest.mark.unit
class TestStrategyConfig:
    """Test StrategyConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StrategyConfig()
        
        assert config.know_face_down_cards is False
        assert config.max_depth == 3
        assert isinstance(config.priorities, dict)
        assert isinstance(config.custom_params, dict)
    
    def test_custom_config(self):
        """Test custom configuration."""
        priorities = {"tableau_to_foundation": 100.0}
        custom = {"param1": "value1"}
        
        config = StrategyConfig(
            know_face_down_cards=True,
            max_depth=5,
            priorities=priorities,
            custom_params=custom,
        )
        
        assert config.know_face_down_cards is True
        assert config.max_depth == 5
        assert config.priorities == priorities
        assert config.custom_params == custom
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = StrategyConfig(max_depth=4)
        data = config.to_dict()
        
        assert isinstance(data, dict)
        assert data["max_depth"] == 4
        assert "priorities" in data
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "know_face_down_cards": True,
            "max_depth": 7,
            "priorities": {"test": 1.0},
        }
        
        config = StrategyConfig.from_dict(data)
        
        assert config.know_face_down_cards is True
        assert config.max_depth == 7
        assert config.priorities == {"test": 1.0}


@pytest.mark.unit
class TestSimpleStrategy:
    """Test SimpleStrategy."""
    
    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        strategy = SimpleStrategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.get_name() == "Simple"
    
    def test_select_best_move_empty_state(self):
        """Test selecting move from empty state."""
        strategy = SimpleStrategy()
        state = GameState()
        
        move = strategy.select_best_move(state)
        
        assert move is None
    
    def test_select_best_move_with_options(self):
        """Test selecting best move when options available."""
        strategy = SimpleStrategy()
        state = GameState()
        
        # Add an Ace to tableau (can go to foundation)
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        move = strategy.select_best_move(state)
        
        assert move is not None
        # Should prioritize foundation move
        assert move.move_type in [
            MoveType.TABLEAU_TO_FOUNDATION,
            MoveType.WASTE_TO_FOUNDATION,
        ]
    
    def test_select_move_sequence(self):
        """Test selecting a sequence of moves."""
        strategy = SimpleStrategy()
        state = GameState()
        
        # Add cards
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        
        sequence = strategy.select_move_sequence(state, length=2)
        
        assert isinstance(sequence, list)
        # Might be empty if moves aren't valid


@pytest.mark.unit
class TestWeightedStrategy:
    """Test WeightedStrategy."""
    
    def test_strategy_initialization(self):
        """Test strategy with default config."""
        strategy = WeightedStrategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.get_name() == "Weighted"
        assert len(strategy.priorities) > 0
    
    def test_strategy_with_custom_priorities(self):
        """Test strategy with custom priorities."""
        config = StrategyConfig(
            priorities={"tableau_to_foundation": 200.0}
        )
        strategy = WeightedStrategy(config)
        
        # Custom priority should override default
        assert strategy.priorities["tableau_to_foundation"] == 200.0
    
    def test_select_best_move_empty_state(self):
        """Test selecting move from empty state."""
        strategy = WeightedStrategy()
        state = GameState()
        
        move = strategy.select_best_move(state)
        
        assert move is None
    
    def test_select_best_move_with_options(self):
        """Test selecting best move with weighted scoring."""
        strategy = WeightedStrategy()
        state = GameState()
        
        # Add cards
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        move = strategy.select_best_move(state)
        
        assert move is not None


@pytest.mark.unit
class TestLookaheadStrategy:
    """Test LookaheadStrategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = LookaheadStrategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.get_name() == "Lookahead"
        assert strategy.config.max_depth == 3
    
    def test_strategy_with_custom_depth(self):
        """Test strategy with custom lookahead depth."""
        config = StrategyConfig(max_depth=5)
        strategy = LookaheadStrategy(config)
        
        assert strategy.config.max_depth == 5
    
    def test_select_best_move_empty_state(self):
        """Test selecting move from empty state."""
        strategy = LookaheadStrategy()
        state = GameState()
        
        move = strategy.select_best_move(state)
        
        assert move is None
    
    def test_select_best_move_with_options(self):
        """Test selecting best move with lookahead."""
        config = StrategyConfig(max_depth=2)
        strategy = LookaheadStrategy(config)
        state = GameState()
        
        # Add cards
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        
        move = strategy.select_best_move(state)
        
        # Should find a move
        assert move is not None
    
    def test_select_move_sequence(self):
        """Test selecting sequence with lookahead."""
        config = StrategyConfig(max_depth=3)
        strategy = LookaheadStrategy(config)
        state = GameState()
        
        # Add cards
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        
        sequence = strategy.select_move_sequence(state, length=2)
        
        assert isinstance(sequence, list)


@pytest.mark.unit
class TestStrategyRegistry:
    """Test StrategyRegistry."""
    
    def test_builtin_strategies_registered(self):
        """Test that built-in strategies are auto-registered."""
        names = StrategyRegistry.get_strategy_names()
        
        assert "simple" in names
        assert "weighted" in names
        assert "lookahead" in names
    
    def test_get_strategy_class(self):
        """Test getting strategy class."""
        strategy_class = StrategyRegistry.get_strategy_class("simple")
        
        assert strategy_class is not None
        assert strategy_class == SimpleStrategy
    
    def test_create_strategy(self):
        """Test creating strategy instance."""
        strategy = StrategyRegistry.create_strategy("simple")
        
        assert strategy is not None
        assert isinstance(strategy, SimpleStrategy)
    
    def test_create_strategy_with_config(self):
        """Test creating strategy with config."""
        config = StrategyConfig(max_depth=5)
        strategy = StrategyRegistry.create_strategy("lookahead", config)
        
        assert strategy is not None
        assert isinstance(strategy, LookaheadStrategy)
        assert strategy.config.max_depth == 5
    
    def test_create_nonexistent_strategy(self):
        """Test creating strategy that doesn't exist."""
        strategy = StrategyRegistry.create_strategy("nonexistent")
        
        assert strategy is None
    
    def test_list_strategies(self):
        """Test listing all strategies."""
        strategies = StrategyRegistry.list_strategies()
        
        assert isinstance(strategies, dict)
        assert len(strategies) >= 3  # At least the built-in ones
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        
        class CustomStrategy(Strategy):
            def select_best_move(self, state):
                return None
            
            def get_name(self):
                return "Custom"
        
        # Register
        StrategyRegistry.register("custom_test", CustomStrategy)
        
        # Verify it's registered
        assert "custom_test" in StrategyRegistry.get_strategy_names()
        
        # Create instance
        strategy = StrategyRegistry.create_strategy("custom_test")
        assert isinstance(strategy, CustomStrategy)
        
        # Clean up
        StrategyRegistry.unregister("custom_test")
        assert "custom_test" not in StrategyRegistry.get_strategy_names()
    
    def test_get_strategy_convenience_function(self):
        """Test convenience function for getting strategy."""
        strategy = get_strategy("simple")
        
        assert strategy is not None
        assert isinstance(strategy, SimpleStrategy)
    
    def test_get_strategy_with_config(self):
        """Test convenience function with config."""
        config = StrategyConfig(max_depth=4)
        strategy = get_strategy("lookahead", config)
        
        assert strategy is not None
        assert strategy.config.max_depth == 4


@pytest.mark.integration
class TestStrategyComparison:
    """Integration tests comparing different strategies."""
    
    def test_all_strategies_can_select_moves(self):
        """Test that all strategies can handle same state."""
        state = GameState()
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.DIAMONDS))
        
        strategies = [
            SimpleStrategy(),
            WeightedStrategy(),
            LookaheadStrategy(),
        ]
        
        for strategy in strategies:
            move = strategy.select_best_move(state)
            assert move is not None, f"{strategy.get_name()} failed to select move"
    
    def test_strategies_may_differ(self):
        """Test that different strategies can make different choices."""
        state = GameState()
        
        # Create a complex state with multiple valid moves
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.HEARTS))
        state.tableau[2].append(Card(rank=3, suit=Suit.DIAMONDS))
        state.waste.append(Card(rank=4, suit=Suit.CLUBS))
        
        simple = SimpleStrategy()
        weighted = WeightedStrategy()
        lookahead = LookaheadStrategy(StrategyConfig(max_depth=2))
        
        move1 = simple.select_best_move(state)
        move2 = weighted.select_best_move(state)
        move3 = lookahead.select_best_move(state)
        
        # All should find a move
        assert move1 is not None
        assert move2 is not None
        assert move3 is not None
        
        # Note: They might choose the same move, that's okay
        # The test just verifies they all work
