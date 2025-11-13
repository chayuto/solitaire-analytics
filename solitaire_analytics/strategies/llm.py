"""LLM-based strategy placeholder for future enhancement.

This module provides a placeholder interface for LLM-based move selection.
In the future, this can be integrated with language models to provide
intelligent, context-aware move suggestions.
"""

from typing import Optional

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.engine import generate_moves
from solitaire_analytics.strategies.base import Strategy


class LLMStrategy(Strategy):
    """Placeholder for LLM-based move selection strategy.
    
    This strategy provides an interface for integrating Large Language Models
    (LLMs) into the move selection process. The actual LLM integration is left
    as a future enhancement.
    
    Potential approaches for LLM integration:
    1. Use LLM to evaluate game state and suggest high-level strategies
    2. Ask LLM to rank possible moves based on game context
    3. Use LLM for explaining move choices to users
    4. Combine LLM suggestions with traditional heuristics
    
    Configuration options (via config.custom_params):
    - llm_provider: Name of LLM provider (e.g., "openai", "anthropic")
    - model_name: Specific model to use
    - api_key: API key for the LLM service
    - prompt_template: Template for formatting game state for LLM
    - temperature: Sampling temperature for LLM responses
    - max_tokens: Maximum tokens in LLM response
    
    Example future usage:
        config = StrategyConfig(
            custom_params={
                "llm_provider": "openai",
                "model_name": "gpt-4",
                "api_key": "sk-...",
                "temperature": 0.7,
            }
        )
        strategy = LLMStrategy(config)
        best_move = strategy.select_best_move(game_state)
    """
    
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select best move (currently falls back to simple heuristic).
        
        This is a placeholder implementation. In the future, this would:
        1. Format the game state for the LLM
        2. Send a prompt asking for move suggestions
        3. Parse the LLM response
        4. Validate and return the suggested move
        
        Args:
            state: Current game state
            
        Returns:
            The suggested move, or None if no moves available
            
        Raises:
            NotImplementedError: LLM integration is not yet implemented
        """
        # Check if LLM is configured
        llm_provider = self.config.custom_params.get("llm_provider")
        
        if llm_provider:
            # LLM integration would go here
            raise NotImplementedError(
                "LLM integration is not yet implemented. "
                "To use LLM-based strategies, you need to implement the "
                "LLM client integration in this method."
            )
        
        # Fallback to simple heuristic
        moves = generate_moves(state)
        if not moves:
            return None
        
        # Use simple priority-based selection as fallback
        from solitaire_analytics.models.move import MoveType
        
        priority_order = {
            MoveType.TABLEAU_TO_FOUNDATION: 5,
            MoveType.WASTE_TO_FOUNDATION: 5,
            MoveType.FLIP_TABLEAU_CARD: 4,
            MoveType.WASTE_TO_TABLEAU: 3,
            MoveType.TABLEAU_TO_TABLEAU: 2,
            MoveType.STOCK_TO_WASTE: 1,
        }
        
        moves.sort(key=lambda m: priority_order.get(m.move_type, 0), reverse=True)
        return moves[0]
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "LLM"
    
    def get_description(self) -> str:
        """Get strategy description."""
        llm_provider = self.config.custom_params.get("llm_provider", "not configured")
        return (
            f"LLM-based strategy (provider: {llm_provider}). "
            "This is currently a placeholder that falls back to simple heuristics. "
            "Full LLM integration is a future enhancement."
        )


def _format_game_state_for_llm(state: GameState) -> str:
    """Format game state as text for LLM input.
    
    This helper function converts the game state into a human-readable
    format suitable for LLM prompts.
    
    Args:
        state: Game state to format
        
    Returns:
        Formatted string description of the game state
    """
    lines = ["Current Solitaire Game State:", ""]
    
    # Foundations
    lines.append("Foundations:")
    for i, foundation in enumerate(state.foundations):
        if foundation:
            top_card = foundation[-1]
            lines.append(f"  Foundation {i}: {len(foundation)} cards, top: {top_card}")
        else:
            lines.append(f"  Foundation {i}: empty")
    
    lines.append("")
    
    # Tableau
    lines.append("Tableau:")
    for i, pile in enumerate(state.tableau):
        if pile:
            face_up = [c for c in pile if c.face_up]
            face_down = len(pile) - len(face_up)
            lines.append(f"  Pile {i}: {face_down} face-down, {len(face_up)} face-up")
            if face_up:
                lines.append(f"    Top card: {face_up[-1]}")
        else:
            lines.append(f"  Pile {i}: empty")
    
    lines.append("")
    
    # Stock and waste
    lines.append(f"Stock: {len(state.stock)} cards")
    if state.waste:
        lines.append(f"Waste: {len(state.waste)} cards, top: {state.waste[-1]}")
    else:
        lines.append("Waste: empty")
    
    lines.append("")
    lines.append(f"Score: {state.score}")
    lines.append(f"Moves made: {state.move_count}")
    
    return "\n".join(lines)


def _create_llm_prompt(state: GameState) -> str:
    """Create a prompt for asking LLM to suggest a move.
    
    This helper function creates a structured prompt for the LLM.
    
    Args:
        state: Current game state
        
    Returns:
        Formatted prompt string
    """
    state_desc = _format_game_state_for_llm(state)
    
    prompt = f"""You are an expert Solitaire player. Given the following game state, 
suggest the best next move. Consider:
- Prioritizing moves to foundations when possible
- Revealing face-down cards
- Creating empty tableau piles for kings
- Maintaining move flexibility

{state_desc}

Based on this state, what is the best move to make? Please explain your reasoning 
and suggest a specific move (e.g., "Move card from tableau pile 2 to foundation 0").
"""
    
    return prompt
