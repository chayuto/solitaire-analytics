"""Dealing logic for setting up a fresh Klondike Solitaire game."""

import random
from typing import List, Optional

from solitaire_analytics.models import Card, GameState
from solitaire_analytics.models.card import Suit


def make_deck() -> List[Card]:
    """Build a standard, ordered 52-card deck (all face down).

    Returns:
        List of 52 unique Card objects.
    """
    return [
        Card(rank=rank, suit=suit, face_up=False)
        for suit in Suit
        for rank in range(1, 14)
    ]


def deal_klondike(seed: Optional[int] = None) -> GameState:
    """Deal a fresh, standard Klondike Solitaire game.

    The seven tableau piles receive 1..7 cards. Only the last card of each
    pile is dealt face up; the remaining 24 cards form the face-down stock.

    Args:
        seed: Optional RNG seed. Passing the same seed always produces the
            same deal, which makes games reproducible for agents and tests.

    Returns:
        A freshly dealt GameState.
    """
    deck = make_deck()
    random.Random(seed).shuffle(deck)

    state = GameState()
    idx = 0
    for pile_index in range(7):
        for position in range(pile_index + 1):
            card = deck[idx]
            idx += 1
            face_up = position == pile_index
            state.tableau[pile_index].append(
                Card(rank=card.rank, suit=card.suit, face_up=face_up)
            )

    for card in deck[idx:]:
        state.stock.append(Card(rank=card.rank, suit=card.suit, face_up=False))

    return state
