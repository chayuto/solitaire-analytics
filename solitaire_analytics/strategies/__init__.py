"""Strategy module for selecting best moves in Solitaire games."""

from solitaire_analytics.strategies.base import Strategy, StrategyConfig
from solitaire_analytics.strategies.simple import SimpleStrategy
from solitaire_analytics.strategies.weighted import WeightedStrategy
from solitaire_analytics.strategies.lookahead import LookaheadStrategy
from solitaire_analytics.strategies.registry import StrategyRegistry, get_strategy

__all__ = [
    "Strategy",
    "StrategyConfig",
    "SimpleStrategy",
    "WeightedStrategy",
    "LookaheadStrategy",
    "StrategyRegistry",
    "get_strategy",
]
