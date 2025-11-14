"""LLM-based strategy using OpenAI API for intelligent move selection.

This module integrates with OpenAI's API to provide AI-powered move suggestions
for Solitaire games. It supports both reasoning models (o1, o3) and standard
chat models (gpt-4o, gpt-4-turbo).
"""

import os
import json
from typing import Optional, Dict, Any

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import generate_moves
from solitaire_analytics.strategies.base import Strategy


class LLMStrategy(Strategy):
    """LLM-based move selection strategy using OpenAI API.
    
    This strategy uses OpenAI's language models to intelligently select moves
    in Solitaire games. It supports both reasoning models (o1, o3) and standard
    chat models (gpt-4o, gpt-4-turbo).
    
    Configuration options (via config.custom_params):
    - api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    - model: Model to use (default: "gpt-4o")
    - temperature: Sampling temperature (default: 0.7, not used for reasoning models)
    - max_tokens: Maximum tokens in response (default: 500, not used for reasoning models)
    - timeout: Request timeout in seconds (default: 30)
    
    Supported models:
    - Reasoning models: o1, o3-mini (use thinking process)
    - Chat models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo (support parameters)
    
    Example usage:
        from solitaire_analytics.strategies import get_strategy, StrategyConfig
        
        config = StrategyConfig(
            custom_params={
                "model": "gpt-4o",
                "temperature": 0.7,
            }
        )
        strategy = get_strategy("llm", config)
        best_move = strategy.select_best_move(game_state)
    """
    
    # Reasoning models that don't support temperature/max_tokens
    REASONING_MODELS = {"o1", "o1-preview", "o1-mini", "o3", "o3-mini"}
    
    def __init__(self, config=None):
        """Initialize LLM strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = self.config.custom_params.get(
            "api_key", 
            os.environ.get("OPENAI_API_KEY")
        )
        
        # Get model configuration
        self.model = self.config.custom_params.get("model", "gpt-4o")
        self.temperature = self.config.custom_params.get("temperature", 0.7)
        self.max_tokens = self.config.custom_params.get("max_tokens", 500)
        self.timeout = self.config.custom_params.get("timeout", 30)
        
        # Check if this is a reasoning model
        self.is_reasoning_model = any(
            self.model.startswith(rm) for rm in self.REASONING_MODELS
        )
        
        # Initialize OpenAI client if API key is available
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            except ImportError:
                # OpenAI package not installed
                pass
    
    def select_best_move(self, state: GameState) -> Optional[Move]:
        """Select best move using OpenAI LLM.
        
        This method:
        1. Formats the game state for the LLM
        2. Sends a prompt asking for move suggestions
        3. Parses the LLM response (JSON mode)
        4. Validates and returns the suggested move
        
        Args:
            state: Current game state
            
        Returns:
            The suggested move, or None if no moves available or LLM fails
        """
        # Get all available moves
        moves = generate_moves(state)
        if not moves:
            return None
        
        # Check if OpenAI client is available
        if not self._client:
            # Fallback to simple heuristic
            return self._fallback_move(moves)
        
        try:
            # Format game state and available moves for LLM
            game_state_text = _format_game_state_for_llm(state)
            moves_text = _format_moves_for_llm(moves)
            
            # Create the prompt
            user_prompt = f"""{game_state_text}

Available moves:
{moves_text}

Analyze the current game state and select the best move from the available options. Consider:
1. Prioritizing moves to foundations
2. Revealing face-down cards
3. Creating empty tableau piles for Kings
4. Maintaining flexibility for future moves

Respond with JSON containing the move index (0-based) and reasoning."""
            
            # Call OpenAI API with appropriate parameters
            if self.is_reasoning_model:
                # Reasoning models don't support temperature/max_tokens
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
            else:
                # Standard chat models support all parameters
                system_prompt = _create_system_prompt()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
            
            # Parse the response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract move index
            move_index = result.get("move_index", 0)
            
            # Validate and return the move
            if 0 <= move_index < len(moves):
                return moves[move_index]
            else:
                # Invalid index, return first move
                return moves[0]
                
        except Exception as e:
            # On any error, fall back to simple heuristic
            print(f"LLM strategy error: {e}")
            return self._fallback_move(moves)
    
    def _fallback_move(self, moves: list) -> Optional[Move]:
        """Select move using simple heuristic fallback.
        
        Args:
            moves: List of available moves
            
        Returns:
            Best move according to simple priority
        """
        if not moves:
            return None
        
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
        if self._client:
            return f"LLM-based strategy using OpenAI {self.model}"
        else:
            return "LLM-based strategy (OpenAI not configured, using fallback)"


def _create_system_prompt() -> str:
    """Create system prompt for non-reasoning models.
    
    Returns:
        System prompt string
    """
    return """You are an expert Solitaire player and advisor. Your goal is to help players 
make the best possible moves in Klondike Solitaire. You have deep understanding of:

1. Solitaire rules and objectives
2. Strategic principles (building foundations, revealing cards, maintaining flexibility)
3. Tactical decision-making (evaluating trade-offs between moves)

When analyzing game states, consider:
- Immediate opportunities to build foundations
- Revealing face-down cards to create more options
- Creating empty tableau piles for Kings
- Maintaining move flexibility for future turns
- Avoiding moves that block important cards

Always respond with valid JSON containing your selected move and reasoning."""


def _format_game_state_for_llm(state: GameState) -> str:
    """Format game state as text for LLM input from user perspective.
    
    This helper function converts the game state into a human-readable
    format suitable for LLM prompts, showing the board from the player's perspective.
    
    Args:
        state: Game state to format
        
    Returns:
        Formatted string description of the game state
    """
    lines = ["=== Current Solitaire Game State ===", ""]
    
    # Foundations (goal piles)
    lines.append("FOUNDATIONS (Goal):")
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    for i, foundation in enumerate(state.foundations):
        suit_name = suits[i] if i < len(suits) else f"Suit {i}"
        if foundation:
            top_card = foundation[-1]
            lines.append(f"  {suit_name}: {_format_card(top_card)} ({len(foundation)} cards)")
        else:
            lines.append(f"  {suit_name}: Empty")
    
    lines.append("")
    
    # Tableau (main playing area)
    lines.append("TABLEAU (Main Playing Area):")
    for i, pile in enumerate(state.tableau):
        if pile:
            face_down_count = sum(1 for c in pile if not c.face_up)
            face_up_cards = [c for c in pile if c.face_up]
            
            pile_desc = f"  Pile {i+1}: "
            if face_down_count > 0:
                pile_desc += f"[{face_down_count} face-down] "
            if face_up_cards:
                pile_desc += " -> ".join(_format_card(c) for c in face_up_cards)
            else:
                pile_desc += "[no face-up cards]"
            lines.append(pile_desc)
        else:
            lines.append(f"  Pile {i+1}: Empty (can place King here)")
    
    lines.append("")
    
    # Stock and waste
    lines.append("DRAW PILE:")
    lines.append(f"  Stock: {len(state.stock)} cards remaining")
    if state.waste:
        top_waste = state.waste[-1]
        lines.append(f"  Waste (top card): {_format_card(top_waste)}")
    else:
        lines.append("  Waste: Empty")
    
    lines.append("")
    lines.append(f"Current Score: {state.score}")
    lines.append(f"Moves Made: {state.move_count}")
    
    return "\n".join(lines)


def _format_card(card) -> str:
    """Format a card for human-readable display.
    
    Args:
        card: Card to format
        
    Returns:
        String representation of the card
    """
    rank_names = {
        1: "A",
        11: "J",
        12: "Q",
        13: "K"
    }
    rank = rank_names.get(card.rank, str(card.rank))
    
    suit_symbols = {
        "hearts": "♥",
        "diamonds": "♦",
        "clubs": "♣",
        "spades": "♠"
    }
    suit = suit_symbols.get(card.suit.value, card.suit.value[0].upper())
    
    return f"{rank}{suit}"


def _format_moves_for_llm(moves: list) -> str:
    """Format available moves for LLM display.
    
    Args:
        moves: List of available Move objects
        
    Returns:
        Formatted string listing all moves with indices
    """
    lines = []
    for i, move in enumerate(moves):
        lines.append(f"{i}. {move}")
    return "\n".join(lines)
