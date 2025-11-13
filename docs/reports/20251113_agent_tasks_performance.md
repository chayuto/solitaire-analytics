# Coding Agent Tasks - Performance Optimizations

**Date:** November 13, 2025  
**Category:** Performance  
**Difficulty Range:** Medium to Hard

## Overview

This document contains self-contained performance optimization tasks suitable for coding agents. Each task improves execution speed, memory usage, or scalability.

---

## Task 1: Add Caching for Move Generation

**Difficulty:** Medium  
**Estimated Time:** 3-4 hours  
**Module:** `solitaire_analytics/engine/move_generator.py`

### Description
Implement LRU caching for expensive move generation operations.

### Context
Move generation is called repeatedly for the same states. Caching results improves performance significantly.

### Requirements
1. Add functools.lru_cache decorator
2. Make GameState hashable (if not done)
3. Configure cache size appropriately
4. Add cache statistics method
5. Add tests verifying cache hits

### Acceptance Criteria
- [ ] Caching implemented
- [ ] Cache hit rate > 50% in solver
- [ ] Performance improves by > 20%
- [ ] Memory usage acceptable
- [ ] Tests verify caching behavior

### Implementation

```python
# solitaire_analytics/engine/move_generator.py

from functools import lru_cache
from typing import List
from solitaire_analytics.models import GameState, Move

# Global cache size (adjustable based on memory constraints)
MOVE_CACHE_SIZE = 10000

@lru_cache(maxsize=MOVE_CACHE_SIZE)
def generate_moves_cached(state_hash: int, state: GameState) -> tuple:
    """
    Generate moves with caching.
    
    Args:
        state_hash: Hash of state for cache key.
        state: Game state.
    
    Returns:
        Tuple of moves (immutable for caching).
    
    Note:
        Returns tuple instead of list for hashability.
    """
    moves = _generate_moves_impl(state)
    return tuple(moves)

def generate_moves(state: GameState) -> List[Move]:
    """
    Generate all valid moves from state (cached).
    
    Args:
        state: Current game state.
    
    Returns:
        List of valid moves.
    """
    # Use hash for cache key
    state_hash = hash(state)
    moves_tuple = generate_moves_cached(state_hash, state)
    return list(moves_tuple)

def _generate_moves_impl(state: GameState) -> List[Move]:
    """Actual move generation implementation (uncached)."""
    moves = []
    
    # ... existing move generation logic ...
    
    return moves

def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dict with hits, misses, size, etc.
    """
    info = generate_moves_cached.cache_info()
    return {
        'hits': info.hits,
        'misses': info.misses,
        'size': info.currsize,
        'maxsize': info.maxsize,
        'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
    }

def clear_cache() -> None:
    """Clear the move generation cache."""
    generate_moves_cached.cache_clear()
```

### Testing

```python
# tests/test_engine.py

def test_move_generation_caching():
    """Test that move generation uses cache."""
    from solitaire_analytics.engine.move_generator import clear_cache, get_cache_stats
    
    clear_cache()
    state = create_test_state()
    
    # First call - cache miss
    moves1 = generate_moves(state)
    stats1 = get_cache_stats()
    assert stats1['misses'] == 1
    assert stats1['hits'] == 0
    
    # Second call with same state - cache hit
    moves2 = generate_moves(state)
    stats2 = get_cache_stats()
    assert stats2['hits'] == 1
    assert stats2['misses'] == 1
    
    # Results should be identical
    assert moves1 == moves2

def test_cache_performance_improvement():
    """Test that caching improves performance."""
    import time
    from solitaire_analytics.engine.move_generator import clear_cache
    
    state = create_complex_state()
    clear_cache()
    
    # Measure uncached performance
    start = time.perf_counter()
    for _ in range(100):
        clear_cache()
        generate_moves(state)
    uncached_time = time.perf_counter() - start
    
    # Measure cached performance
    clear_cache()
    generate_moves(state)  # Prime cache
    
    start = time.perf_counter()
    for _ in range(100):
        generate_moves(state)
    cached_time = time.perf_counter() - start
    
    # Cached should be much faster
    speedup = uncached_time / cached_time
    assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"

def test_cache_memory_bounded():
    """Test that cache size is bounded."""
    from solitaire_analytics.engine.move_generator import clear_cache, get_cache_stats
    
    clear_cache()
    
    # Generate moves for many different states
    for i in range(15000):  # More than cache size
        state = create_unique_state(i)
        generate_moves(state)
    
    stats = get_cache_stats()
    assert stats['size'] <= stats['maxsize']
```

---

## Task 2: Optimize State Cloning

**Difficulty:** Medium  
**Estimated Time:** 3-4 hours  
**Module:** `solitaire_analytics/models/game_state.py`

### Description
Optimize the state copy/clone operation which is called frequently.

### Context
State cloning happens thousands of times during solving. Optimizing it significantly improves overall performance.

### Requirements
1. Profile current copy() method
2. Implement shallow copy where possible
3. Use copy-on-write for large structures
4. Add fast path for common cases
5. Benchmark improvements

### Acceptance Criteria
- [ ] Copy is 2-3x faster
- [ ] Correctness maintained
- [ ] Memory usage not increased significantly
- [ ] Benchmark shows improvement
- [ ] Tests verify correctness

### Implementation

```python
# solitaire_analytics/models/game_state.py

from typing import List
import copy

class GameState:
    """Game state with optimized cloning."""
    
    __slots__ = ['tableau', 'foundations', 'stock', 'waste', '_hash_cache']
    
    def __init__(self):
        """Initialize with empty state."""
        self.tableau: List[List[Card]] = [[] for _ in range(7)]
        self.foundations: List[List[Card]] = [[] for _ in range(4)]
        self.stock: List[Card] = []
        self.waste: List[Card] = []
        self._hash_cache: Optional[int] = None
    
    def copy(self) -> 'GameState':
        """
        Create a deep copy of the game state (optimized).
        
        Returns:
            New GameState instance.
        
        Note:
            This is optimized for performance. Uses shallow copies
            where safe and fast paths for common cases.
        """
        new_state = GameState.__new__(GameState)
        
        # Copy tableau (deep copy needed as lists are mutable)
        new_state.tableau = [
            pile.copy() for pile in self.tableau
        ]
        
        # Copy foundations (deep copy)
        new_state.foundations = [
            pile.copy() for pile in self.foundations
        ]
        
        # Copy stock and waste
        new_state.stock = self.stock.copy()
        new_state.waste = self.waste.copy()
        
        # Clear hash cache (state changed)
        new_state._hash_cache = None
        
        return new_state
    
    def copy_with_move(self, move: Move) -> 'GameState':
        """
        Create copy with move applied (optimized).
        
        This is faster than copy() + apply_move() because it only
        copies the parts that change.
        
        Args:
            move: Move to apply.
        
        Returns:
            New state with move applied.
        """
        new_state = GameState.__new__(GameState)
        
        # Identify which piles are affected
        src_type = move.source_pile_type
        src_idx = move.source_pile_index
        dst_type = move.destination_pile_type
        dst_idx = move.destination_pile_index
        
        # Copy only changed piles, share unchanged ones
        if src_type == 'tableau':
            new_state.tableau = self.tableau.copy()
            new_state.tableau[src_idx] = self.tableau[src_idx].copy()
        else:
            new_state.tableau = self.tableau  # Share unchanged
        
        # ... similar for other piles ...
        
        # Apply the move
        # ... move logic ...
        
        new_state._hash_cache = None
        return new_state
    
    def __hash__(self) -> int:
        """Cached hash computation."""
        if self._hash_cache is None:
            self._hash_cache = hash(self._canonical_representation())
        return self._hash_cache
```

### Benchmarking

```python
# tests/test_performance.py

import pytest
import time

@pytest.mark.performance
def test_copy_performance():
    """Benchmark state copying."""
    state = create_complex_state()
    
    # Warm up
    for _ in range(100):
        state.copy()
    
    # Measure
    start = time.perf_counter()
    for _ in range(10000):
        clone = state.copy()
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / 10000
    print(f"\nAverage copy time: {avg_time*1000:.3f}ms")
    
    # Should be fast
    assert avg_time < 0.001, f"Copy too slow: {avg_time*1000:.2f}ms"

@pytest.mark.performance
def test_copy_correctness():
    """Verify optimized copy is correct."""
    original = create_complex_state()
    clone = original.copy()
    
    # Should be equal but different objects
    assert clone == original
    assert clone is not original
    assert clone.tableau is not original.tableau
    
    # Modifying clone shouldn't affect original
    clone.tableau[0].append(Card(rank=1, suit=Suit.HEARTS))
    assert clone != original
```

---

## Task 3: Add Beam Search Optimization

**Difficulty:** Hard  
**Estimated Time:** 5-6 hours  
**Module:** `solitaire_analytics/solvers/parallel_solver.py`

### Description
Optimize the beam search algorithm for better pruning and state evaluation.

### Context
Current beam search can be improved with better heuristics and pruning strategies.

### Requirements
1. Implement dominance checking
2. Add better state scoring
3. Optimize beam selection
4. Add adaptive beam width
5. Benchmark improvements

### Acceptance Criteria
- [ ] Finds solutions faster
- [ ] Explores fewer states
- [ ] Solution quality maintained or improved
- [ ] Benchmarks show 30%+ improvement
- [ ] Tests verify correctness

### Implementation

```python
# solitaire_analytics/solvers/beam_search.py

from typing import List, Tuple
import heapq
from solitaire_analytics.models import GameState

def evaluate_state(state: GameState) -> float:
    """
    Evaluate state with improved heuristic.
    
    Higher scores are better. Combines multiple factors:
    - Cards in foundations (primary)
    - Cards revealed in tableau
    - Empty tableau piles
    - Card ordering potential
    
    Args:
        state: Game state to evaluate.
    
    Returns:
        Evaluation score (higher is better).
    """
    score = 0.0
    
    # Foundation cards (most important)
    foundation_cards = sum(len(pile) for pile in state.foundations)
    score += foundation_cards * 100
    
    # Revealed cards in tableau
    revealed = sum(
        1 for pile in state.tableau 
        for card in pile 
        if not card.face_down
    )
    score += revealed * 10
    
    # Empty tableau piles (valuable for Kings)
    empty_piles = sum(1 for pile in state.tableau if len(pile) == 0)
    score += empty_piles * 50
    
    # Cards in proper sequence in tableau
    sequence_bonus = 0
    for pile in state.tableau:
        for i in range(len(pile) - 1):
            if pile[i].rank == pile[i+1].rank + 1:
                if pile[i].is_red() != pile[i+1].is_red():
                    sequence_bonus += 5
    score += sequence_bonus
    
    # Penalty for cards in stock
    score -= len(state.stock) * 2
    
    return score

def dominates(state1: GameState, state2: GameState) -> bool:
    """
    Check if state1 dominates state2.
    
    State1 dominates state2 if it's strictly better in all aspects.
    This allows pruning of dominated states.
    
    Args:
        state1: First state.
        state2: Second state.
    
    Returns:
        True if state1 dominates state2.
    """
    # Compare foundations
    for f1, f2 in zip(state1.foundations, state2.foundations):
        if len(f1) < len(f2):
            return False
    
    # If foundations are better or equal, state1 might dominate
    # Add more sophisticated checks here
    
    return True

def select_beam(
    candidates: List[Tuple[GameState, float]],
    beam_width: int,
    use_diversity: bool = True
) -> List[GameState]:
    """
    Select best states for beam search.
    
    Args:
        candidates: List of (state, score) tuples.
        beam_width: Number of states to keep.
        use_diversity: If True, promote diverse states.
    
    Returns:
        List of selected states.
    """
    if len(candidates) <= beam_width:
        return [state for state, score in candidates]
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    if not use_diversity:
        # Simple: take top N
        return [state for state, score in candidates[:beam_width]]
    
    # Diverse beam selection
    selected = []
    selected_hashes = set()
    
    # Always take the best
    best_state, best_score = candidates[0]
    selected.append(best_state)
    selected_hashes.add(hash(best_state))
    
    # Select diverse states from remaining
    for state, score in candidates[1:]:
        if len(selected) >= beam_width:
            break
        
        state_hash = hash(state)
        
        # Check if too similar to already selected
        if state_hash not in selected_hashes:
            # Check diversity with all selected states
            is_diverse = True
            for sel_state in selected:
                if not is_diverse_from(state, sel_state):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(state)
                selected_hashes.add(state_hash)
    
    # Fill remaining slots with best remaining
    while len(selected) < beam_width and len(candidates) > len(selected):
        for state, score in candidates[len(selected):]:
            if hash(state) not in selected_hashes:
                selected.append(state)
                selected_hashes.add(hash(state))
                break
    
    return selected

def is_diverse_from(state1: GameState, state2: GameState, threshold: float = 0.3) -> bool:
    """
    Check if two states are diverse (different enough).
    
    Args:
        state1: First state.
        state2: Second state.
        threshold: Diversity threshold (0-1).
    
    Returns:
        True if states are sufficiently different.
    """
    differences = 0
    total = 0
    
    # Compare tableau
    for p1, p2 in zip(state1.tableau, state2.tableau):
        if len(p1) != len(p2):
            differences += abs(len(p1) - len(p2))
        total += max(len(p1), len(p2))
    
    # Compare foundations
    for f1, f2 in zip(state1.foundations, state2.foundations):
        if len(f1) != len(f2):
            differences += abs(len(f1) - len(f2))
        total += max(len(f1), len(f2))
    
    if total == 0:
        return False
    
    diversity = differences / total
    return diversity >= threshold
```

### Testing

```python
@pytest.mark.performance
def test_beam_search_optimization():
    """Test that optimized beam search is faster."""
    state = create_solvable_state()
    
    # Old solver
    old_solver = ParallelSolver(max_depth=15, use_optimization=False)
    start = time.time()
    old_result = old_solver.solve(state)
    old_time = time.time() - start
    
    # Optimized solver
    new_solver = ParallelSolver(max_depth=15, use_optimization=True)
    start = time.time()
    new_result = new_solver.solve(state)
    new_time = time.time() - start
    
    # Should be faster
    speedup = old_time / new_time
    print(f"\nSpeedup: {speedup:.2f}x")
    assert speedup > 1.3, f"Expected >30% speedup, got {speedup:.2f}x"
    
    # Both should find solution
    assert old_result.success == new_result.success
```

---

## Task 4: Add Parallel Tree Building

**Difficulty:** Hard  
**Estimated Time:** 4-5 hours  
**Module:** `solitaire_analytics/analysis/move_tree_builder.py`

### Description
Parallelize move tree construction for better performance on multi-core systems.

### Context
Tree building is CPU-intensive. Parallelization can provide near-linear speedup.

### Requirements
1. Use joblib for parallelization
2. Implement work stealing
3. Handle shared state correctly
4. Add parallel/sequential toggle
5. Benchmark on multi-core

### Acceptance Criteria
- [ ] Parallel version implemented
- [ ] Speedup scales with cores
- [ ] Results identical to sequential
- [ ] No race conditions
- [ ] Tests verify correctness

### Implementation

```python
# solitaire_analytics/analysis/move_tree_builder.py

from typing import List, Optional
import joblib
from joblib import Parallel, delayed
import multiprocessing as mp

class MoveTreeBuilder:
    """Build move trees with optional parallelization."""
    
    def __init__(
        self,
        max_depth: int = 5,
        max_nodes: int = 1000,
        n_jobs: int = 1
    ):
        """
        Initialize tree builder.
        
        Args:
            max_depth: Maximum tree depth.
            max_nodes: Maximum number of nodes.
            n_jobs: Number of parallel jobs (1 = sequential,
                   -1 = all cores).
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.n_jobs = n_jobs if n_jobs != 1 else 1
        
        self._nodes_created = 0
        self._lock = mp.Lock() if n_jobs != 1 else None
    
    def build_tree(self, root_state: GameState) -> TreeNode:
        """
        Build move tree from root state.
        
        Args:
            root_state: Initial game state.
        
        Returns:
            Root node of tree.
        """
        if self.n_jobs == 1:
            return self._build_sequential(root_state)
        else:
            return self._build_parallel(root_state)
    
    def _build_sequential(self, root_state: GameState) -> TreeNode:
        """Build tree sequentially."""
        root = TreeNode(state=root_state, depth=0)
        queue = [root]
        
        while queue and self._nodes_created < self.max_nodes:
            node = queue.pop(0)
            
            if node.depth >= self.max_depth:
                continue
            
            # Generate child nodes
            moves = generate_moves(node.state)
            for move in moves:
                if self._nodes_created >= self.max_nodes:
                    break
                
                child_state = apply_move(node.state, move)
                child = TreeNode(
                    state=child_state,
                    depth=node.depth + 1,
                    parent=node,
                    move=move
                )
                node.children.append(child)
                queue.append(child)
                self._nodes_created += 1
        
        return root
    
    def _build_parallel(self, root_state: GameState) -> TreeNode:
        """Build tree in parallel."""
        root = TreeNode(state=root_state, depth=0)
        
        # Build first few levels sequentially
        initial_depth = min(2, self.max_depth)
        frontier = self._build_to_depth(root, initial_depth)
        
        # Process frontier in parallel
        n_jobs = self.n_jobs if self.n_jobs > 0 else mp.cpu_count()
        
        # Split frontier into batches
        batch_size = max(1, len(frontier) // n_jobs)
        batches = [
            frontier[i:i+batch_size]
            for i in range(0, len(frontier), batch_size)
        ]
        
        # Process batches in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_batch)(batch)
            for batch in batches
        )
        
        # Merge results back into tree
        for batch_results in results:
            for node, children in batch_results:
                node.children.extend(children)
        
        return root
    
    def _build_to_depth(
        self,
        root: TreeNode,
        target_depth: int
    ) -> List[TreeNode]:
        """
        Build tree to specified depth, return frontier nodes.
        
        Args:
            root: Root node.
            target_depth: Depth to build to.
        
        Returns:
            List of frontier nodes at target depth.
        """
        frontier = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            
            if node.depth >= target_depth:
                frontier.append(node)
                continue
            
            if self._nodes_created >= self.max_nodes:
                break
            
            moves = generate_moves(node.state)
            for move in moves:
                if self._nodes_created >= self.max_nodes:
                    break
                
                child_state = apply_move(node.state, move)
                child = TreeNode(
                    state=child_state,
                    depth=node.depth + 1,
                    parent=node,
                    move=move
                )
                node.children.append(child)
                queue.append(child)
                
                with self._lock:
                    self._nodes_created += 1
        
        return frontier
    
    def _process_batch(
        self,
        nodes: List[TreeNode]
    ) -> List[Tuple[TreeNode, List[TreeNode]]]:
        """
        Process a batch of nodes.
        
        Args:
            nodes: Nodes to expand.
        
        Returns:
            List of (node, children) tuples.
        """
        results = []
        
        for node in nodes:
            if self._nodes_created >= self.max_nodes:
                break
            
            children = []
            moves = generate_moves(node.state)
            
            for move in moves:
                if self._nodes_created >= self.max_nodes:
                    break
                
                child_state = apply_move(node.state, move)
                child = TreeNode(
                    state=child_state,
                    depth=node.depth + 1,
                    parent=node,
                    move=move
                )
                children.append(child)
                
                with self._lock:
                    self._nodes_created += 1
            
            results.append((node, children))
        
        return results
```

### Testing

```python
@pytest.mark.performance
def test_parallel_tree_building():
    """Test parallel tree building performance."""
    state = create_test_state()
    
    # Sequential
    builder_seq = MoveTreeBuilder(max_depth=5, max_nodes=1000, n_jobs=1)
    start = time.time()
    tree_seq = builder_seq.build_tree(state)
    time_seq = time.time() - start
    
    # Parallel
    builder_par = MoveTreeBuilder(max_depth=5, max_nodes=1000, n_jobs=-1)
    start = time.time()
    tree_par = builder_par.build_tree(state)
    time_par = time.time() - start
    
    # Should be faster
    speedup = time_seq / time_par
    print(f"\nSpeedup: {speedup:.2f}x")
    assert speedup > 1.5, f"Expected >50% speedup, got {speedup:.2f}x"
    
    # Should have same number of nodes
    assert count_nodes(tree_seq) == count_nodes(tree_par)
```

---

## Summary

These performance tasks significantly improve execution speed. Recommended order:

1. **Start with:** Task 1 (Move caching) - Easy wins, big impact
2. **Then:** Task 2 (State cloning) - Foundation for other optimizations
3. **Then:** Task 4 (Parallel tree) - Good performance gains
4. **Then:** Task 3 (Beam search) - Most complex, highest potential

Each task provides measurable performance improvements.
