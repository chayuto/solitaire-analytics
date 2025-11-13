"""Move generation logic for Solitaire game."""

from typing import List

from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.models.move import MoveType


def generate_moves(game_state: GameState) -> List[Move]:
    """Generate all possible legal moves for a given game state.
    
    Args:
        game_state: Current game state
        
    Returns:
        List of all legal moves
    """
    moves = []
    
    # Generate tableau to foundation moves
    moves.extend(_generate_tableau_to_foundation_moves(game_state))
    
    # Generate waste to foundation moves
    moves.extend(_generate_waste_to_foundation_moves(game_state))
    
    # Generate tableau to tableau moves
    moves.extend(_generate_tableau_to_tableau_moves(game_state))
    
    # Generate waste to tableau moves
    moves.extend(_generate_waste_to_tableau_moves(game_state))
    
    # Generate stock to waste move
    if game_state.stock:
        moves.append(Move(
            move_type=MoveType.STOCK_TO_WASTE,
            score_delta=0
        ))
    
    return moves


def _generate_tableau_to_foundation_moves(game_state: GameState) -> List[Move]:
    """Generate moves from tableau to foundation."""
    moves = []
    
    for tableau_idx, pile in enumerate(game_state.tableau):
        if not pile:
            continue
        
        top_card = pile[-1]
        if not top_card.face_up:
            continue
        
        # Try to place on each foundation
        for foundation_idx, foundation in enumerate(game_state.foundations):
            foundation_top = foundation[-1] if foundation else None
            
            if top_card.can_place_on_foundation(foundation_top):
                moves.append(Move(
                    move_type=MoveType.TABLEAU_TO_FOUNDATION,
                    source_pile=tableau_idx,
                    dest_pile=foundation_idx,
                    score_delta=10
                ))
    
    return moves


def _generate_waste_to_foundation_moves(game_state: GameState) -> List[Move]:
    """Generate moves from waste to foundation."""
    moves = []
    
    if not game_state.waste:
        return moves
    
    waste_top = game_state.waste[-1]
    
    # Try to place on each foundation
    for foundation_idx, foundation in enumerate(game_state.foundations):
        foundation_top = foundation[-1] if foundation else None
        
        if waste_top.can_place_on_foundation(foundation_top):
            moves.append(Move(
                move_type=MoveType.WASTE_TO_FOUNDATION,
                dest_pile=foundation_idx,
                score_delta=10
            ))
    
    return moves


def _generate_tableau_to_tableau_moves(game_state: GameState) -> List[Move]:
    """Generate moves from tableau to tableau."""
    moves = []
    
    for source_idx, source_pile in enumerate(game_state.tableau):
        if not source_pile:
            continue
        
        # Find the first face-up card
        first_face_up = None
        for i, card in enumerate(source_pile):
            if card.face_up:
                first_face_up = i
                break
        
        if first_face_up is None:
            continue
        
        # Try moving each sequence starting from face-up cards
        for card_idx in range(first_face_up, len(source_pile)):
            moving_card = source_pile[card_idx]
            num_cards = len(source_pile) - card_idx
            
            # Try placing on each tableau pile
            for dest_idx, dest_pile in enumerate(game_state.tableau):
                if dest_idx == source_idx:
                    continue
                
                dest_top = dest_pile[-1] if dest_pile else None
                
                if moving_card.can_stack_on(dest_top):
                    # Don't move Kings to empty piles if source would be empty
                    # (no point in just moving cards around)
                    if not dest_pile and not source_pile[:card_idx]:
                        continue
                    
                    moves.append(Move(
                        move_type=MoveType.TABLEAU_TO_TABLEAU,
                        source_pile=source_idx,
                        dest_pile=dest_idx,
                        num_cards=num_cards,
                        score_delta=0
                    ))
    
    return moves


def _generate_waste_to_tableau_moves(game_state: GameState) -> List[Move]:
    """Generate moves from waste to tableau."""
    moves = []
    
    if not game_state.waste:
        return moves
    
    waste_top = game_state.waste[-1]
    
    # Try placing on each tableau pile
    for dest_idx, dest_pile in enumerate(game_state.tableau):
        dest_top = dest_pile[-1] if dest_pile else None
        
        if waste_top.can_stack_on(dest_top):
            moves.append(Move(
                move_type=MoveType.WASTE_TO_TABLEAU,
                dest_pile=dest_idx,
                score_delta=5
            ))
    
    return moves
