"""Move tree builder for analyzing game state transitions."""

from typing import List, Dict, Optional, Set
import networkx as nx

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.engine import generate_moves, apply_move


class MoveTreeNode:
    """Node in the move tree representing a game state.
    
    Attributes:
        state: Game state at this node
        parent_move: Move that led to this state (None for root)
        depth: Depth in the tree
        children: List of child nodes
    """
    
    def __init__(
        self,
        state: GameState,
        parent_move: Optional[Move] = None,
        depth: int = 0
    ):
        """Initialize a move tree node.
        
        Args:
            state: Game state
            parent_move: Move that led to this state
            depth: Depth in the tree
        """
        self.state = state
        self.parent_move = parent_move
        self.depth = depth
        self.children: List['MoveTreeNode'] = []
        self.state_hash = hash(state)
    
    def add_child(self, child: 'MoveTreeNode') -> None:
        """Add a child node."""
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0


class MoveTreeBuilder:
    """Builder for constructing and analyzing move trees.
    
    A move tree represents all possible game states reachable from an initial
    state through a series of moves.
    """
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 10000):
        """Initialize the move tree builder.
        
        Args:
            max_depth: Maximum depth to build the tree
            max_nodes: Maximum number of nodes in the tree
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root: Optional[MoveTreeNode] = None
        self.visited: Set[int] = set()
        self.node_count = 0
    
    def build_tree(self, initial_state: GameState) -> MoveTreeNode:
        """Build a move tree from an initial game state.
        
        Args:
            initial_state: Starting game state
            
        Returns:
            Root node of the move tree
        """
        self.root = MoveTreeNode(initial_state, depth=0)
        self.visited = {hash(initial_state)}
        self.node_count = 1
        
        self._build_recursive(self.root)
        
        return self.root
    
    def _build_recursive(self, node: MoveTreeNode) -> None:
        """Recursively build the tree from a node.
        
        Args:
            node: Current node
        """
        # Stop if max depth or max nodes reached
        if node.depth >= self.max_depth or self.node_count >= self.max_nodes:
            return
        
        # Stop if game is won
        if node.state.is_won():
            return
        
        # Generate and explore all moves
        moves = generate_moves(node.state)
        
        for move in moves:
            new_state = apply_move(node.state, move)
            if new_state is None:
                continue
            
            state_hash = hash(new_state)
            
            # Skip if we've seen this state (avoid cycles)
            if state_hash in self.visited:
                continue
            
            self.visited.add(state_hash)
            self.node_count += 1
            
            # Create child node
            child_node = MoveTreeNode(
                state=new_state,
                parent_move=move,
                depth=node.depth + 1
            )
            node.add_child(child_node)
            
            # Recursively build from child
            if self.node_count < self.max_nodes:
                self._build_recursive(child_node)
    
    def get_all_leaf_states(self) -> List[GameState]:
        """Get all leaf states in the tree.
        
        Returns:
            List of game states at leaf nodes
        """
        if self.root is None:
            return []
        
        leaves = []
        self._collect_leaves(self.root, leaves)
        return [node.state for node in leaves]
    
    def _collect_leaves(self, node: MoveTreeNode, leaves: List[MoveTreeNode]) -> None:
        """Recursively collect leaf nodes.
        
        Args:
            node: Current node
            leaves: List to append leaf nodes to
        """
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)
    
    def get_winning_paths(self) -> List[List[Move]]:
        """Get all paths that lead to winning states.
        
        Returns:
            List of move sequences that lead to wins
        """
        if self.root is None:
            return []
        
        winning_paths = []
        self._find_winning_paths(self.root, [], winning_paths)
        return winning_paths
    
    def _find_winning_paths(
        self,
        node: MoveTreeNode,
        current_path: List[Move],
        winning_paths: List[List[Move]]
    ) -> None:
        """Recursively find winning paths.
        
        Args:
            node: Current node
            current_path: Current sequence of moves
            winning_paths: List to append winning paths to
        """
        # Add current move to path (if not root)
        if node.parent_move is not None:
            current_path = current_path + [node.parent_move]
        
        # Check if this is a winning state
        if node.state.is_won():
            winning_paths.append(current_path)
            return
        
        # Recurse to children
        for child in node.children:
            self._find_winning_paths(child, current_path, winning_paths)
    
    def to_networkx_graph(self) -> nx.DiGraph:
        """Convert the move tree to a NetworkX directed graph.
        
        Returns:
            NetworkX directed graph representation
        """
        G = nx.DiGraph()
        
        if self.root is None:
            return G
        
        self._add_to_graph(self.root, G, node_id=0)
        return G
    
    def _add_to_graph(
        self,
        node: MoveTreeNode,
        graph: nx.DiGraph,
        node_id: int,
        parent_id: Optional[int] = None
    ) -> int:
        """Recursively add nodes to NetworkX graph.
        
        Args:
            node: Current tree node
            graph: NetworkX graph
            node_id: ID for current node
            parent_id: ID of parent node
            
        Returns:
            Next available node ID
        """
        # Add node with attributes
        graph.add_node(
            node_id,
            depth=node.depth,
            won=node.state.is_won(),
            score=node.state.score,
            move_count=node.state.move_count
        )
        
        # Add edge from parent
        if parent_id is not None and node.parent_move is not None:
            graph.add_edge(
                parent_id,
                node_id,
                move=str(node.parent_move),
                move_type=node.parent_move.move_type.value
            )
        
        # Process children
        next_id = node_id + 1
        for child in node.children:
            next_id = self._add_to_graph(child, graph, next_id, node_id)
        
        return next_id
    
    def get_statistics(self) -> Dict:
        """Get statistics about the move tree.
        
        Returns:
            Dictionary with tree statistics
        """
        if self.root is None:
            return {}
        
        stats = {
            "total_nodes": self.node_count,
            "max_depth": self.max_depth,
            "leaf_nodes": len(self.get_all_leaf_states()),
            "winning_paths": len(self.get_winning_paths()),
            "visited_states": len(self.visited)
        }
        
        return stats
