"""Solitaire Analytics Engine - A Python 3.12 analytics engine for Solitaire games."""

__version__ = "0.1.0"

from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.engine import generate_moves, validate_move
from solitaire_analytics.solvers import ParallelSolver
from solitaire_analytics.analysis import MoveTreeBuilder, DeadEndDetector

__all__ = [
    "Card",
    "GameState",
    "Move",
    "generate_moves",
    "validate_move",
    "ParallelSolver",
    "MoveTreeBuilder",
    "DeadEndDetector",
]
