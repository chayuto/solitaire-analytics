"""Interactive game layer for agentic Solitaire players.

This package adds the pieces the analytics engine intentionally leaves out so
an AI agent can actually *play* a game end to end:

* :func:`deal_klondike` -- deal a fresh, reproducible Klondike game
* :class:`GameSession` -- a stateful, turn-based playable game
* :class:`ObservationConfig` -- control the information level revealed to the
  agent (face-down cards, stock, waste history)
"""

from solitaire_analytics.game.dealer import deal_klondike, make_deck
from solitaire_analytics.game.observation import (
    ObservationConfig,
    card_code,
    card_to_dict,
    observe,
    render_text,
)
from solitaire_analytics.game.session import GameAction, GameSession

__all__ = [
    "deal_klondike",
    "make_deck",
    "ObservationConfig",
    "card_code",
    "card_to_dict",
    "observe",
    "render_text",
    "GameAction",
    "GameSession",
]
