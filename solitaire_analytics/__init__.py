"""Solitaire Analytics Engine - A Python 3.12 analytics engine for Solitaire games."""

__version__ = "0.1.0"

from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.engine import generate_moves, validate_move
from solitaire_analytics.solvers import ParallelSolver
from solitaire_analytics.analysis import MoveTreeBuilder, DeadEndDetector
from solitaire_analytics.strategies import (
    Strategy,
    StrategyConfig,
    get_strategy,
    StrategyRegistry,
)
from solitaire_analytics.play_logger import PlayLogger
from solitaire_analytics.game import (
    GameSession,
    ObservationConfig,
    deal_klondike,
)
from solitaire_analytics.harvest import (
    DecisionHarvest,
    DecisionRecord,
    build_decision_record,
)
from solitaire_analytics.session_registry import (
    SessionEntry,
    SessionRegistry,
    SessionRegistryError,
)

__all__ = [
    "Card",
    "GameState",
    "Move",
    "generate_moves",
    "validate_move",
    "ParallelSolver",
    "MoveTreeBuilder",
    "DeadEndDetector",
    "Strategy",
    "StrategyConfig",
    "get_strategy",
    "StrategyRegistry",
    "PlayLogger",
    "GameSession",
    "ObservationConfig",
    "deal_klondike",
    "DecisionHarvest",
    "DecisionRecord",
    "build_decision_record",
    "SessionEntry",
    "SessionRegistry",
    "SessionRegistryError",
]
