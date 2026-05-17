"""Interactive Klondike Solitaire session for agentic AI players.

A :class:`GameSession` wraps a :class:`GameState` and the move engine into a
stateful, turn-based interface suitable for an agent: deal a game, ask for the
legal moves, apply one by index, repeat. It also handles the two mechanics the
pure analytics engine leaves out -- drawing several cards at once and recycling
the waste pile back into an exhausted stock.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from solitaire_analytics.models import Card, GameState, Move
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.engine import apply_move, generate_moves
from solitaire_analytics.game.dealer import deal_klondike
from solitaire_analytics.game.observation import (
    ObservationConfig,
    observe,
    render_text,
)


@dataclass
class GameAction:
    """A single action an agent may take, addressable by stable index.

    Attributes:
        index: Position of this action in the current legal-action list.
        kind: One of ``"move"`` (a tableau/foundation/waste move),
            ``"draw"`` (deal from stock to waste), or ``"recycle"``
            (turn the waste pile back into the stock).
        description: Human-readable description of the action.
        move: The underlying engine :class:`Move`, for ``"move"`` and
            ``"draw"`` kinds. ``None`` for ``"recycle"``.
    """

    index: int
    kind: str
    description: str
    move: Optional[Move] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the action for an agent (the engine move is omitted)."""
        return {
            "index": self.index,
            "kind": self.kind,
            "description": self.description,
        }


class GameSession:
    """A stateful, playable Klondike Solitaire game."""

    def __init__(
        self,
        state: GameState,
        draw_count: int = 1,
        observation_config: Optional[ObservationConfig] = None,
        seed: Optional[int] = None,
        log: bool = True,
    ):
        """Wrap an existing game state in a session.

        Args:
            state: The game state to play.
            draw_count: Cards moved from stock to waste per draw (1 or 3).
            observation_config: Default information level for observations.
            seed: The seed used to deal this game, if known (for reference).
            log: Whether to record a session log of every action taken.
        """
        if draw_count < 1:
            raise ValueError(f"draw_count must be >= 1, got {draw_count}")
        self.state = state
        self.draw_count = int(draw_count)
        self.observation_config = observation_config or ObservationConfig()
        self.seed = seed
        self.redeal_count = 0

        # Session log. ``initial_state`` is a snapshot of the dealt game --
        # i.e. the full starting deck arrangement -- captured up front so the
        # game can always be reconstructed even when no seed was supplied.
        self.log_enabled = log
        self.initial_state = state.copy()
        self.created_at = datetime.now(timezone.utc)
        self._action_log: List[Dict[str, Any]] = []

    @classmethod
    def new_game(
        cls,
        seed: Optional[int] = None,
        draw_count: int = 1,
        observation_config: Optional[ObservationConfig] = None,
        log: bool = True,
    ) -> "GameSession":
        """Deal and return a fresh game session.

        Args:
            seed: Optional RNG seed for a reproducible deal.
            draw_count: Cards drawn from stock to waste per draw.
            observation_config: Default information level for observations.
            log: Whether to record a session log of every action taken.
        """
        return cls(
            state=deal_klondike(seed),
            draw_count=draw_count,
            observation_config=observation_config,
            seed=seed,
            log=log,
        )

    def legal_actions(self) -> List[GameAction]:
        """Return every action the agent may currently take, with stable indices.

        The list is deterministic for a given state, so an index returned here
        stays valid until the next action is applied.
        """
        actions: List[GameAction] = []
        for move in generate_moves(self.state):
            if move.move_type == MoveType.STOCK_TO_WASTE:
                drawn = min(self.draw_count, len(self.state.stock))
                actions.append(
                    GameAction(0, "draw", f"Draw {drawn} card(s) from stock", move)
                )
            else:
                actions.append(GameAction(0, "move", str(move), move))

        if not self.state.stock and self.state.waste:
            actions.append(
                GameAction(0, "recycle", "Recycle the waste pile back into the stock")
            )

        for i, action in enumerate(actions):
            action.index = i
        return actions

    def apply_action(self, index: int) -> Dict[str, Any]:
        """Apply the legal action at ``index`` and advance the game.

        Args:
            index: Index into the list returned by :meth:`legal_actions`.

        Returns:
            A dictionary describing what happened.

        Raises:
            ValueError: If the index is out of range or the move is illegal.
        """
        actions = self.legal_actions()
        if not 0 <= index < len(actions):
            raise ValueError(
                f"Invalid move index {index}; {len(actions)} legal action(s) available"
            )

        action = actions[index]
        if action.kind == "draw":
            self._draw()
        elif action.kind == "recycle":
            self._recycle()
        else:
            new_state = apply_move(self.state, action.move)
            if new_state is None:
                raise ValueError(f"Illegal move: {action.description}")
            self.state = new_state

        if self.log_enabled:
            self._action_log.append(
                {
                    "seq": len(self._action_log) + 1,
                    "kind": action.kind,
                    "description": action.description,
                    "move": action.move.to_dict() if action.move is not None else None,
                    "score": self.state.score,
                    "move_count": self.state.move_count,
                    "resulting_state": self.state.to_dict(),
                }
            )

        return {
            "applied": action.description,
            "kind": action.kind,
            "won": self.state.is_won(),
            "stuck": self.is_stuck(),
        }

    def _draw(self) -> None:
        """Move up to ``draw_count`` cards from stock to waste (face up)."""
        new_state = self.state.copy()
        count = min(self.draw_count, len(new_state.stock))
        for _ in range(count):
            card = new_state.stock.pop()
            new_state.waste.append(
                Card(rank=card.rank, suit=card.suit, face_up=True)
            )
        new_state.move_count += 1
        self.state = new_state

    def _recycle(self) -> None:
        """Turn the waste pile back into a face-down stock.

        Cards are reversed so the next pass draws them in the original order.
        """
        new_state = self.state.copy()
        new_state.stock = [
            Card(rank=c.rank, suit=c.suit, face_up=False)
            for c in reversed(new_state.waste)
        ]
        new_state.waste = []
        new_state.move_count += 1
        self.redeal_count += 1
        self.state = new_state

    def is_won(self) -> bool:
        """Return True if every foundation is complete."""
        return self.state.is_won()

    def is_stuck(self) -> bool:
        """Return True if the game is unwinnable from here (no legal actions)."""
        return not self.is_won() and len(self.legal_actions()) == 0

    def observation(self, config: Optional[ObservationConfig] = None) -> Dict[str, Any]:
        """Return an agent-facing observation at the given information level.

        Args:
            config: Information level to use. Defaults to the session config.
        """
        cfg = config or self.observation_config
        return observe(self.state, cfg, self.legal_actions(), session=self)

    def render(self, config: Optional[ObservationConfig] = None) -> str:
        """Return a human-readable text board respecting the information level."""
        return render_text(self.state, config or self.observation_config)

    def get_log(self) -> Dict[str, Any]:
        """Return the full session log as a JSON-serializable dictionary.

        The log captures everything needed to reproduce and audit the game:

        * ``seed`` -- the RNG seed (the same seed re-deals the identical game)
        * ``initial_state`` -- the dealt starting deck and full game state
        * ``actions`` -- every action taken, with the resulting state
        * ``result`` -- the final outcome and counters
        """
        return {
            "seed": self.seed,
            "draw_count": self.draw_count,
            "created_at": self.created_at.isoformat(),
            "observation_config": self.observation_config.to_dict(),
            "initial_state": self.initial_state.to_dict(),
            "actions": list(self._action_log),
            "result": {
                "won": self.is_won(),
                "stuck": self.is_stuck(),
                "score": self.state.score,
                "move_count": self.state.move_count,
                "redeal_count": self.redeal_count,
                "total_actions": len(self._action_log),
            },
        }

    def save_log(self, filepath: str, indent: int = 2) -> None:
        """Write the session log to a JSON file.

        Args:
            filepath: Destination path; parent directories are created.
            indent: JSON indentation for readability.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as handle:
            json.dump(self.get_log(), handle, indent=indent)
