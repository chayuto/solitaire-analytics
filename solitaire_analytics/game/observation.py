"""Configurable observations of a Solitaire game for agentic players.

The :class:`ObservationConfig` controls the *information level* an agent
player is allowed to see. This lets you study how agents behave under
realistic (imperfect) information versus perfect information:

* ``face_down`` -- whether the identity of face-down tableau cards is revealed
* ``stock`` -- whether the stock is hidden, shown as a count, or fully revealed
* ``waste`` -- whether only the top waste card or the whole drawn pile is shown
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from solitaire_analytics.models import Card, GameState


FACE_DOWN_LEVELS = ("count", "revealed")
STOCK_LEVELS = ("hidden", "count", "revealed")
WASTE_LEVELS = ("top", "full")

_SUIT_CODE = {"hearts": "H", "diamonds": "D", "clubs": "C", "spades": "S"}
_RANK_CODE = {1: "A", 10: "T", 11: "J", 12: "Q", 13: "K"}


@dataclass
class ObservationConfig:
    """Controls how much game information is revealed to an agent player.

    Attributes:
        face_down: Visibility of face-down tableau cards.
            ``"count"`` reveals only how many are hidden (what a human sees);
            ``"revealed"`` exposes their identities (perfect information).
        stock: Visibility of the stock (draw) pile.
            ``"hidden"`` reveals nothing, ``"count"`` reveals only the size,
            ``"revealed"`` exposes the full ordered contents.
        waste: Visibility of the waste pile of previously drawn cards.
            ``"top"`` shows only the playable top card; ``"full"`` shows the
            entire stack of previously drawn cards.
        include_legal_moves: Whether observations embed the list of legal moves.
    """

    face_down: str = "count"
    stock: str = "count"
    waste: str = "full"
    include_legal_moves: bool = True

    def __post_init__(self) -> None:
        if self.face_down not in FACE_DOWN_LEVELS:
            raise ValueError(
                f"face_down must be one of {FACE_DOWN_LEVELS}, got {self.face_down!r}"
            )
        if self.stock not in STOCK_LEVELS:
            raise ValueError(
                f"stock must be one of {STOCK_LEVELS}, got {self.stock!r}"
            )
        if self.waste not in WASTE_LEVELS:
            raise ValueError(
                f"waste must be one of {WASTE_LEVELS}, got {self.waste!r}"
            )

    @classmethod
    def human(cls) -> "ObservationConfig":
        """Information a human player sees: hidden cards stay hidden."""
        return cls(face_down="count", stock="count", waste="full")

    @classmethod
    def perfect_information(cls) -> "ObservationConfig":
        """Everything revealed -- useful for solver-style or 'cheating' agents."""
        return cls(face_down="revealed", stock="revealed", waste="full")

    @classmethod
    def minimal(cls) -> "ObservationConfig":
        """The most restrictive level: no stock info, only the top waste card."""
        return cls(face_down="count", stock="hidden", waste="top")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a plain dictionary."""
        return {
            "face_down": self.face_down,
            "stock": self.stock,
            "waste": self.waste,
            "include_legal_moves": self.include_legal_moves,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationConfig":
        """Build a config from a dictionary, falling back to defaults."""
        return cls(
            face_down=data.get("face_down", "count"),
            stock=data.get("stock", "count"),
            waste=data.get("waste", "full"),
            include_legal_moves=data.get("include_legal_moves", True),
        )


def card_code(card: Card) -> str:
    """Return a compact card code such as ``AH`` (Ace of Hearts) or ``TS``."""
    rank = _RANK_CODE.get(card.rank, str(card.rank))
    return f"{rank}{_SUIT_CODE[card.suit.value]}"


def card_to_dict(card: Card) -> Dict[str, Any]:
    """Serialize a card to a dictionary for an agent observation."""
    return {
        "rank": card.rank,
        "suit": card.suit.value,
        "color": card.color.value,
        "code": card_code(card),
    }


def _foundation_view(foundation: List[Card], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "count": len(foundation),
        "top": card_to_dict(foundation[-1]) if foundation else None,
    }


def _tableau_view(pile: List[Card], index: int, config: ObservationConfig) -> Dict[str, Any]:
    face_down = [c for c in pile if not c.face_up]
    face_up = [c for c in pile if c.face_up]
    view: Dict[str, Any] = {
        "index": index,
        "size": len(pile),
        "face_down_count": len(face_down),
        "face_up": [card_to_dict(c) for c in face_up],
    }
    if config.face_down == "revealed" and face_down:
        view["face_down_cards"] = [card_to_dict(c) for c in face_down]
    return view


def _stock_view(stock: List[Card], config: ObservationConfig) -> Dict[str, Any]:
    if config.stock == "hidden":
        return {"hidden": True}
    view: Dict[str, Any] = {"count": len(stock)}
    if config.stock == "revealed":
        view["cards"] = [card_to_dict(c) for c in stock]
    return view


def _waste_view(waste: List[Card], config: ObservationConfig) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        "count": len(waste),
        "top": card_to_dict(waste[-1]) if waste else None,
    }
    if config.waste == "full":
        view["cards"] = [card_to_dict(c) for c in waste]
    return view


def observe(
    state: GameState,
    config: ObservationConfig,
    actions: Optional[List[Any]] = None,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Produce an agent-facing observation of a game state.

    Args:
        state: The game state to observe.
        config: The information level controlling what is revealed.
        actions: Optional list of legal :class:`GameAction` objects.
        session: Optional :class:`GameSession`, used to surface session-level
            facts (draw count, redeal count, stuck status).

    Returns:
        A JSON-serializable dictionary describing the game.
    """
    observation: Dict[str, Any] = {
        "foundations": [
            _foundation_view(f, i) for i, f in enumerate(state.foundations)
        ],
        "tableau": [
            _tableau_view(p, i, config) for i, p in enumerate(state.tableau)
        ],
        "stock": _stock_view(state.stock, config),
        "waste": _waste_view(state.waste, config),
        "move_count": state.move_count,
        "score": state.score,
        "won": state.is_won(),
        "info_level": config.to_dict(),
    }

    if session is not None:
        observation["draw_count"] = session.draw_count
        observation["redeal_count"] = session.redeal_count

    if actions is not None:
        observation["stuck"] = len(actions) == 0 and not state.is_won()
        if config.include_legal_moves:
            observation["legal_moves"] = [a.to_dict() for a in actions]

    return observation


def render_text(state: GameState, config: ObservationConfig) -> str:
    """Render a compact, human-readable board respecting the information level."""
    lines = ["=== Solitaire ==="]

    foundations = " ".join(
        card_code(f[-1]) if f else "--" for f in state.foundations
    )
    lines.append(f"Foundations: {foundations}")

    lines.append("Tableau:")
    for i, pile in enumerate(state.tableau):
        face_down = [c for c in pile if not c.face_up]
        face_up = [c for c in pile if c.face_up]
        if config.face_down == "revealed":
            down = " ".join(f"({card_code(c)})" for c in face_down)
        else:
            down = "## " * len(face_down)
        up = " ".join(card_code(c) for c in face_up)
        parts = " ".join(p for p in (down.strip(), up) if p) or "(empty)"
        lines.append(f"  T{i}: {parts}")

    if config.stock == "hidden":
        lines.append("Stock: (hidden)")
    elif config.stock == "revealed":
        lines.append(
            f"Stock ({len(state.stock)}): "
            + " ".join(card_code(c) for c in state.stock)
        )
    else:
        lines.append(f"Stock: {len(state.stock)} cards")

    if not state.waste:
        lines.append("Waste: (empty)")
    elif config.waste == "full":
        lines.append(
            f"Waste ({len(state.waste)}): "
            + " ".join(card_code(c) for c in state.waste)
        )
    else:
        lines.append(f"Waste: {card_code(state.waste[-1])} (+{len(state.waste) - 1} below)")

    lines.append(f"Score: {state.score}  Moves: {state.move_count}")
    return "\n".join(lines)
