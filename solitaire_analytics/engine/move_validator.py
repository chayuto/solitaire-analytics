"""Move validation and application logic for Solitaire game."""

from typing import Optional

from solitaire_analytics.models import GameState, Move
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.models.card import Card, Suit


def validate_move(game_state: GameState, move: Move) -> bool:
    """Validate if a move is legal in the given game state.
    
    Args:
        game_state: Current game state
        move: Move to validate
        
    Returns:
        True if the move is legal, False otherwise
    """
    try:
        if move.move_type == MoveType.STOCK_TO_WASTE:
            return len(game_state.stock) > 0
        
        elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
            return _validate_tableau_to_foundation(game_state, move)
        
        elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
            return _validate_waste_to_foundation(game_state, move)
        
        elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
            return _validate_tableau_to_tableau(game_state, move)
        
        elif move.move_type == MoveType.WASTE_TO_TABLEAU:
            return _validate_waste_to_tableau(game_state, move)
        
        elif move.move_type == MoveType.FLIP_TABLEAU_CARD:
            return _validate_flip_tableau_card(game_state, move)
        
        return False
    except (IndexError, ValueError):
        return False


def _validate_tableau_to_foundation(game_state: GameState, move: Move) -> bool:
    """Validate tableau to foundation move."""
    if move.source_pile is None or move.dest_pile is None:
        return False
    
    if not (0 <= move.source_pile < 7 and 0 <= move.dest_pile < 4):
        return False
    
    source_pile = game_state.tableau[move.source_pile]
    if not source_pile or not source_pile[-1].face_up:
        return False
    
    moving_card = source_pile[-1]
    foundation = game_state.foundations[move.dest_pile]
    foundation_top = foundation[-1] if foundation else None
    
    return moving_card.can_place_on_foundation(foundation_top)


def _validate_waste_to_foundation(game_state: GameState, move: Move) -> bool:
    """Validate waste to foundation move."""
    if move.dest_pile is None or not game_state.waste:
        return False
    
    if not (0 <= move.dest_pile < 4):
        return False
    
    moving_card = game_state.waste[-1]
    foundation = game_state.foundations[move.dest_pile]
    foundation_top = foundation[-1] if foundation else None
    
    return moving_card.can_place_on_foundation(foundation_top)


def _validate_tableau_to_tableau(game_state: GameState, move: Move) -> bool:
    """Validate tableau to tableau move."""
    if move.source_pile is None or move.dest_pile is None:
        return False
    
    if not (0 <= move.source_pile < 7 and 0 <= move.dest_pile < 7):
        return False
    
    if move.source_pile == move.dest_pile:
        return False
    
    source_pile = game_state.tableau[move.source_pile]
    if len(source_pile) < move.num_cards:
        return False
    
    # Check that all cards being moved are face up
    moving_cards = source_pile[-move.num_cards:]
    if not all(card.face_up for card in moving_cards):
        return False
    
    # Check that the first card can be placed on the destination
    moving_card = moving_cards[0]
    dest_pile = game_state.tableau[move.dest_pile]
    dest_top = dest_pile[-1] if dest_pile else None
    
    return moving_card.can_stack_on(dest_top)


def _validate_waste_to_tableau(game_state: GameState, move: Move) -> bool:
    """Validate waste to tableau move."""
    if move.dest_pile is None or not game_state.waste:
        return False
    
    if not (0 <= move.dest_pile < 7):
        return False
    
    moving_card = game_state.waste[-1]
    dest_pile = game_state.tableau[move.dest_pile]
    dest_top = dest_pile[-1] if dest_pile else None
    
    return moving_card.can_stack_on(dest_top)


def _validate_flip_tableau_card(game_state: GameState, move: Move) -> bool:
    """Validate flip tableau card move."""
    if move.source_pile is None:
        return False
    
    if not (0 <= move.source_pile < 7):
        return False
    
    pile = game_state.tableau[move.source_pile]
    return pile and not pile[-1].face_up


def apply_move(game_state: GameState, move: Move) -> Optional[GameState]:
    """Apply a move to a game state and return the new state.
    
    Args:
        game_state: Current game state
        move: Move to apply
        
    Returns:
        New game state after applying the move, or None if move is invalid
    """
    if not validate_move(game_state, move):
        return None
    
    # Create a copy of the game state
    new_state = game_state.copy()
    new_state.move_count += 1
    new_state.score += move.score_delta
    
    if move.move_type == MoveType.STOCK_TO_WASTE:
        card = new_state.stock.pop()
        new_state.waste.append(card)
    
    elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
        card = new_state.tableau[move.source_pile].pop()
        new_state.foundations[move.dest_pile].append(card)
        _flip_top_card_if_needed(new_state.tableau[move.source_pile])
    
    elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
        card = new_state.waste.pop()
        new_state.foundations[move.dest_pile].append(card)
    
    elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
        cards = new_state.tableau[move.source_pile][-move.num_cards:]
        new_state.tableau[move.source_pile] = new_state.tableau[move.source_pile][:-move.num_cards]
        new_state.tableau[move.dest_pile].extend(cards)
        _flip_top_card_if_needed(new_state.tableau[move.source_pile])
    
    elif move.move_type == MoveType.WASTE_TO_TABLEAU:
        card = new_state.waste.pop()
        new_state.tableau[move.dest_pile].append(card)
    
    elif move.move_type == MoveType.FLIP_TABLEAU_CARD:
        pile = new_state.tableau[move.source_pile]
        if pile:
            # Create new card with face_up=True
            old_card = pile[-1]
            new_card = Card(rank=old_card.rank, suit=old_card.suit, face_up=True)
            pile[-1] = new_card
            new_state.score += 5  # Bonus for flipping a card
    
    return new_state


def _flip_top_card_if_needed(pile: list) -> None:
    """Flip the top card of a pile if it's face down.
    
    Args:
        pile: Pile to check and modify
    """
    if pile and not pile[-1].face_up:
        old_card = pile[-1]
        pile[-1] = Card(rank=old_card.rank, suit=old_card.suit, face_up=True)
