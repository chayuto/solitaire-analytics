"""MCP server exposing Solitaire as a tool-driven game for agentic AI players.

This server lets an MCP-capable agent play Klondike Solitaire move by move.
The key feature is a configurable *information level*: when starting a game you
choose how much the agent is allowed to see -- the identity of face-down
tableau cards, the contents of the stock, and the history of previously drawn
(waste) cards. This makes it easy to study agent play under realistic
imperfect information versus perfect information.

Run it over stdio::

    python -m solitaire_analytics.mcp_server

or, once the package is installed::

    solitaire-mcp
"""

import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from solitaire_analytics.game import GameSession, ObservationConfig
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.server_analytics import ServerAnalyticsLog
from solitaire_analytics.strategies import get_strategy

mcp = FastMCP("solitaire-analytics")

# A single active game per server process. new_game() replaces it.
_session: Optional[GameSession] = None

# Server-side analytics: a cross-game event stream for detailed analysis.
# Set SOLITAIRE_MCP_LOG_FILE to also append events to a JSON Lines file.
_analytics = ServerAnalyticsLog.from_env()
_game_seq = 0
_current_game_id: Optional[str] = None


def _require_session() -> GameSession:
    if _session is None:
        raise ValueError("No game in progress. Call new_game first.")
    return _session


@mcp.tool()
def new_game(
    seed: Optional[int] = None,
    draw_count: int = 1,
    face_down: str = "count",
    stock: str = "count",
    waste: str = "full",
    include_legal_moves: bool = True,
) -> Dict[str, Any]:
    """Deal a fresh Klondike Solitaire game and return the first observation.

    The information-level arguments decide what the agent player may see:

    Args:
        seed: Optional RNG seed. The same seed always deals the same game.
        draw_count: Cards moved from stock to waste per draw (1 or 3).
        face_down: Visibility of face-down tableau cards -- "count" (only how
            many are hidden, as a human sees) or "revealed" (their identities,
            i.e. perfect information).
        stock: Visibility of the stock pile -- "hidden", "count", or "revealed".
        waste: Visibility of previously drawn cards -- "top" (only the playable
            top card) or "full" (the entire drawn pile).
        include_legal_moves: Whether observations embed the legal-move list.

    Returns:
        The initial observation of the dealt game.
    """
    global _session, _game_seq, _current_game_id

    # Record any unfinished prior game as abandoned before replacing it.
    if _session is not None and _current_game_id is not None:
        if not (_session.is_won() or _session.is_stuck()):
            _analytics.record(
                "game_abandoned",
                game_id=_current_game_id,
                move_count=_session.state.move_count,
                score=_session.state.score,
            )

    config = ObservationConfig(
        face_down=face_down,
        stock=stock,
        waste=waste,
        include_legal_moves=include_legal_moves,
    )
    _session = GameSession.new_game(
        seed=seed, draw_count=draw_count, observation_config=config
    )
    _game_seq += 1
    _current_game_id = f"game-{_game_seq}"
    _analytics.record(
        "game_started",
        game_id=_current_game_id,
        seed=seed,
        draw_count=draw_count,
        info_level=config.to_dict(),
    )
    return _session.observation()


@mcp.tool()
def get_observation() -> Dict[str, Any]:
    """Return the current game observation at the configured information level."""
    return _require_session().observation()


@mcp.tool()
def get_legal_moves() -> List[Dict[str, Any]]:
    """Return the legal actions, each with a stable index to pass to play_move."""
    return [action.to_dict() for action in _require_session().legal_actions()]


@mcp.tool()
def play_move(move_index: int) -> Dict[str, Any]:
    """Apply the legal action at the given index and return the new state.

    Args:
        move_index: Index from get_legal_moves (or the embedded legal_moves).

    Returns:
        A dictionary with the applied action, win/stuck flags, and the
        resulting observation.
    """
    session = _require_session()
    result = session.apply_action(move_index)

    _analytics.record(
        "move",
        game_id=_current_game_id,
        kind=result["kind"],
        description=result["applied"],
        score=session.state.score,
        move_count=session.state.move_count,
        won=result["won"],
        stuck=result["stuck"],
    )
    if result["won"] or result["stuck"]:
        _analytics.record(
            "game_ended",
            game_id=_current_game_id,
            result="won" if result["won"] else "stuck",
            score=session.state.score,
            move_count=session.state.move_count,
            redeal_count=session.redeal_count,
        )

    result["observation"] = session.observation()
    return result


@mcp.tool()
def set_information_level(
    face_down: str = "count",
    stock: str = "count",
    waste: str = "full",
    include_legal_moves: bool = True,
) -> Dict[str, Any]:
    """Change the information level revealed to the agent for the current game.

    Args:
        face_down: "count" or "revealed".
        stock: "hidden", "count", or "revealed".
        waste: "top" or "full".
        include_legal_moves: Whether observations embed the legal-move list.

    Returns:
        A fresh observation at the new information level.
    """
    session = _require_session()
    session.observation_config = ObservationConfig(
        face_down=face_down,
        stock=stock,
        waste=waste,
        include_legal_moves=include_legal_moves,
    )
    return session.observation()


@mcp.tool()
def render_board() -> str:
    """Return a compact, human-readable text rendering of the current board."""
    return _require_session().render()


@mcp.tool()
def game_status() -> Dict[str, Any]:
    """Return a short status summary: won, stuck, score, and move/redeal counts."""
    session = _require_session()
    return {
        "won": session.is_won(),
        "stuck": session.is_stuck(),
        "score": session.state.score,
        "move_count": session.state.move_count,
        "redeal_count": session.redeal_count,
        "draw_count": session.draw_count,
        "legal_move_count": len(session.legal_actions()),
    }


@mcp.tool()
def get_game_log() -> Dict[str, Any]:
    """Return the full session log of the current game.

    The log captures the seed, the dealt starting deck and game state, every
    action taken with its resulting state, and the final result -- enough to
    reproduce and audit the game.
    """
    return _require_session().get_log()


@mcp.tool()
def save_game_log(path: str) -> Dict[str, Any]:
    """Write the current game's session log to a JSON file.

    Args:
        path: Destination file path.

    Returns:
        A confirmation with the path and number of logged actions.
    """
    session = _require_session()
    session.save_log(path)
    return {"saved": True, "path": path, "actions_logged": len(session.get_log()["actions"])}


@mcp.tool()
def get_server_analytics() -> Dict[str, Any]:
    """Return aggregate analytics across every game played this server session.

    Includes games started/completed/won/stuck/abandoned, win rate, the count
    of logged actions broken down by kind, and average game length. Set the
    SOLITAIRE_MCP_LOG_FILE environment variable to also persist the raw event
    stream to a JSON Lines file for offline analysis.
    """
    return _analytics.summary()


@mcp.tool()
def suggest_move() -> Dict[str, Any]:
    """Suggest a reasonable move using the built-in greedy strategy (a hint).

    Returns:
        The suggested action's index and description, or a note if the only
        option is to recycle the waste or the game is over.
    """
    session = _require_session()
    actions = session.legal_actions()
    if not actions:
        return {"suggestion": None, "reason": "No legal moves; the game is over."}

    suggested = get_strategy("simple").select_best_move(session.state)
    if suggested is None:
        recycle = next((a for a in actions if a.kind == "recycle"), None)
        if recycle is not None:
            return {"suggestion": recycle.to_dict(), "reason": "Recycle the waste pile."}
        return {"suggestion": actions[0].to_dict(), "reason": "Fallback to first legal move."}

    for action in actions:
        move = action.move
        if move is None:
            continue
        if move.move_type == MoveType.STOCK_TO_WASTE == suggested.move_type:
            return {"suggestion": action.to_dict(), "reason": "Greedy strategy pick."}
        if (
            move.move_type == suggested.move_type
            and move.source_pile == suggested.source_pile
            and move.dest_pile == suggested.dest_pile
            and move.num_cards == suggested.num_cards
        ):
            return {"suggestion": action.to_dict(), "reason": "Greedy strategy pick."}

    return {"suggestion": actions[0].to_dict(), "reason": "Fallback to first legal move."}


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    # Logs go to stderr -- stdout is reserved for the MCP protocol.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
