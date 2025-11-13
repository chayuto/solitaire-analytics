"""Parallel solver for Solitaire games with CPU and GPU support."""

import torch
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
from joblib import Parallel, delayed
import time

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move


@dataclass
class SolverResult:
    """Result from solving a Solitaire game.
    
    Attributes:
        success: Whether a solution was found
        moves: Sequence of moves to win (if success=True)
        states_explored: Number of game states explored
        time_elapsed: Time taken to solve (seconds)
        final_state: Final game state reached
    """
    success: bool
    moves: List[Move]
    states_explored: int
    time_elapsed: float
    final_state: GameState


class ParallelSolver:
    """Parallel solver for Solitaire games with CPU and GPU support.
    
    This solver uses breadth-first search with parallel processing to find
    winning sequences of moves. It can utilize both CPU cores via joblib
    and GPU acceleration via PyTorch for state evaluation.
    """
    
    def __init__(
        self,
        max_depth: int = 100,
        n_jobs: int = -1,
        use_gpu: bool = False,
        beam_width: int = 1000,
        timeout: Optional[float] = None
    ):
        """Initialize the parallel solver.
        
        Args:
            max_depth: Maximum search depth
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            use_gpu: Whether to use GPU acceleration for heuristics
            beam_width: Maximum number of states to keep at each depth
            timeout: Maximum time to run solver (seconds), None for no limit
        """
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.beam_width = beam_width
        self.timeout = timeout
        
        # Setup device for PyTorch
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
    
    def solve(self, initial_state: GameState) -> SolverResult:
        """Solve a Solitaire game starting from the given state.
        
        Args:
            initial_state: Starting game state
            
        Returns:
            SolverResult with solution details
        """
        start_time = time.time()
        states_explored = 0
        visited: Set[int] = set()
        
        # Queue: (state, move_sequence)
        queue: List[Tuple[GameState, List[Move]]] = [(initial_state, [])]
        visited.add(hash(initial_state))
        
        for depth in range(self.max_depth):
            if not queue:
                break
            
            # Check timeout
            if self.timeout and (time.time() - start_time) > self.timeout:
                break
            
            # Process current level in parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._explore_state)(state, moves)
                for state, moves in queue
            )
            
            # Collect new states
            new_queue = []
            for state_moves_list in results:
                for new_state, move_sequence in state_moves_list:
                    states_explored += 1
                    
                    # Check if we won
                    if new_state.is_won():
                        time_elapsed = time.time() - start_time
                        return SolverResult(
                            success=True,
                            moves=move_sequence,
                            states_explored=states_explored,
                            time_elapsed=time_elapsed,
                            final_state=new_state
                        )
                    
                    # Add to queue if not visited
                    state_hash = hash(new_state)
                    if state_hash not in visited:
                        visited.add(state_hash)
                        new_queue.append((new_state, move_sequence))
            
            # Prune queue using beam search if needed
            if len(new_queue) > self.beam_width:
                new_queue = self._prune_states(new_queue)
            
            queue = new_queue
        
        # No solution found
        time_elapsed = time.time() - start_time
        final_state = queue[0][0] if queue else initial_state
        
        return SolverResult(
            success=False,
            moves=[],
            states_explored=states_explored,
            time_elapsed=time_elapsed,
            final_state=final_state
        )
    
    def _explore_state(
        self,
        state: GameState,
        move_sequence: List[Move]
    ) -> List[Tuple[GameState, List[Move]]]:
        """Explore all possible moves from a given state.
        
        Args:
            state: Current game state
            move_sequence: Sequence of moves that led to this state
            
        Returns:
            List of (new_state, new_move_sequence) tuples
        """
        results = []
        moves = generate_moves(state)
        
        for move in moves:
            new_state = apply_move(state, move)
            if new_state is not None:
                new_sequence = move_sequence + [move]
                results.append((new_state, new_sequence))
        
        return results
    
    def _prune_states(
        self,
        states: List[Tuple[GameState, List[Move]]]
    ) -> List[Tuple[GameState, List[Move]]]:
        """Prune states using heuristic evaluation.
        
        Args:
            states: List of (state, move_sequence) tuples
            
        Returns:
            Pruned list of best states
        """
        # Calculate heuristic scores
        if self.use_gpu and len(states) > 100:
            scores = self._gpu_evaluate_states([s[0] for s in states])
        else:
            scores = [self._heuristic_score(s[0]) for s in states]
        
        # Sort by score (higher is better)
        scored_states = list(zip(scores, states))
        scored_states.sort(key=lambda x: x[0], reverse=True)
        
        # Return top beam_width states
        return [s[1] for s in scored_states[:self.beam_width]]
    
    def _heuristic_score(self, state: GameState) -> float:
        """Calculate heuristic score for a game state.
        
        Higher scores indicate states closer to winning.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Heuristic score
        """
        score = 0.0
        
        # Cards in foundations (highest priority)
        foundation_cards = sum(len(f) for f in state.foundations)
        score += foundation_cards * 100
        
        # Face-up cards in tableau
        face_up_cards = sum(
            sum(1 for card in pile if card.face_up)
            for pile in state.tableau
        )
        score += face_up_cards * 10
        
        # Empty tableau piles (good for Kings)
        empty_piles = sum(1 for pile in state.tableau if not pile)
        score += empty_piles * 20
        
        # Cards in waste (slightly negative)
        score -= len(state.waste) * 2
        
        # Cards in stock (negative)
        score -= len(state.stock) * 5
        
        return score
    
    def _gpu_evaluate_states(self, states: List[GameState]) -> List[float]:
        """Evaluate multiple states using GPU acceleration.
        
        Args:
            states: List of game states to evaluate
            
        Returns:
            List of heuristic scores
        """
        # Convert states to tensor representation
        features = []
        for state in states:
            feature = self._state_to_tensor(state)
            features.append(feature)
        
        # Batch process on GPU
        features_tensor = torch.stack(features).to(self.device)
        
        # Simple linear combination of features (can be replaced with neural network)
        weights = torch.tensor([100.0, 10.0, 20.0, -2.0, -5.0], device=self.device)
        scores = torch.matmul(features_tensor, weights)
        
        return scores.cpu().tolist()
    
    def _state_to_tensor(self, state: GameState) -> torch.Tensor:
        """Convert game state to tensor representation.
        
        Args:
            state: Game state
            
        Returns:
            Tensor representation
        """
        features = [
            float(sum(len(f) for f in state.foundations)),  # Foundation cards
            float(sum(sum(1 for c in p if c.face_up) for p in state.tableau)),  # Face-up cards
            float(sum(1 for p in state.tableau if not p)),  # Empty piles
            float(len(state.waste)),  # Waste cards
            float(len(state.stock)),  # Stock cards
        ]
        return torch.tensor(features, dtype=torch.float32)
