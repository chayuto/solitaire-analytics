"""Tests for solver components."""

import pytest
from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit
from solitaire_analytics.solvers import ParallelSolver


@pytest.mark.unit
@pytest.mark.solver
class TestParallelSolver:
    """Test ParallelSolver."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = ParallelSolver(max_depth=10, n_jobs=1)
        
        assert solver.max_depth == 10
        assert solver.n_jobs == 1
    
    def test_solve_empty_state(self):
        """Test solving empty state."""
        state = GameState()
        solver = ParallelSolver(max_depth=2, n_jobs=1, beam_width=10)
        
        result = solver.solve(state)
        
        assert result is not None
        assert result.success is False
        assert result.states_explored >= 0
        assert result.time_elapsed >= 0
    
    def test_solve_simple_winning_state(self):
        """Test solving a simple near-winning state."""
        state = GameState()
        
        # Setup: Almost won, just need to place a few cards
        for i in range(4):
            # Place 12 cards in each foundation
            for rank in range(1, 13):
                state.foundations[i].append(Card(rank=rank, suit=list(Suit)[i]))
            
            # Place the King in tableau
            state.tableau[i].append(Card(rank=13, suit=list(Suit)[i]))
        
        solver = ParallelSolver(max_depth=10, n_jobs=1, beam_width=100)
        result = solver.solve(state)
        
        assert result is not None
        assert result.success is True
        assert len(result.moves) > 0
    
    def test_solver_timeout(self):
        """Test solver timeout functionality."""
        state = GameState()
        # Add some cards to make it non-trivial
        state.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
        state.tableau[1].append(Card(rank=2, suit=Suit.SPADES))
        
        solver = ParallelSolver(max_depth=50, n_jobs=1, timeout=0.1)
        result = solver.solve(state)
        
        # Should complete within timeout
        assert result.time_elapsed <= 1.0  # Allow some buffer
    
    def test_heuristic_score(self):
        """Test heuristic scoring function."""
        solver = ParallelSolver()
        
        state1 = GameState()
        state2 = GameState()
        
        # State with more foundation cards should score higher
        state2.foundations[0].append(Card(rank=1, suit=Suit.HEARTS))
        
        score1 = solver._heuristic_score(state1)
        score2 = solver._heuristic_score(state2)
        
        assert score2 > score1


@pytest.mark.slow
@pytest.mark.solver
class TestParallelSolverIntegration:
    """Integration tests for ParallelSolver."""
    
    def test_solve_with_stock(self):
        """Test solving with stock cards."""
        state = GameState()
        
        # Add some cards to stock and tableau
        state.stock.append(Card(rank=1, suit=Suit.HEARTS))
        state.stock.append(Card(rank=2, suit=Suit.HEARTS))
        state.tableau[0].append(Card(rank=3, suit=Suit.SPADES))
        
        solver = ParallelSolver(max_depth=5, n_jobs=1, beam_width=50)
        result = solver.solve(state)
        
        assert result is not None
        assert result.states_explored > 0
