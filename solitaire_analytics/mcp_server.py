"""MCP server exposing Solitaire as a tool-driven game for agentic AI players.

Each connected agent plays its own concurrent game: ``new_game`` returns a
``session_id`` and every subsequent tool call passes it back. Per-session locks
keep moves within one game serial while allowing different sessions to make
progress in parallel.

When every move is made by an AI (the Gemma 4 E2B distillation use case), the
server doubles as the training-data source: each ``play_move`` emits a
:class:`~solitaire_analytics.harvest.DecisionRecord` to the JSONL file given by
``SOLITAIRE_MCP_HARVEST_FILE`` -- carrying the seed, observation, full legal
move list, chosen index, agent identity, and any agent-supplied reasoning.

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
from solitaire_analytics.harvest import DecisionHarvest, build_decision_record
from solitaire_analytics.models.move import MoveType
from solitaire_analytics.server_analytics import ServerAnalyticsLog
from solitaire_analytics.session_registry import SessionRegistry
from solitaire_analytics.strategies import get_strategy

mcp = FastMCP("solitaire-analytics")

#: Global multi-session state. The MCP server process owns one of each.
_registry = SessionRegistry()
_analytics = ServerAnalyticsLog.from_env()
_harvest = DecisionHarvest.from_env()


def _get_entry(session_id: str):
    """Look up the registry entry or raise a tool-friendly error."""
    return _registry.get(session_id)


@mcp.tool()
def new_game(
    seed: Optional[int] = None,
    draw_count: int = 1,
    face_down: str = "count",
    stock: str = "count",
    waste: str = "full",
    include_legal_moves: bool = True,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    app_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Deal a fresh Klondike game for this caller and return the first observation.

    Multiple agents may keep concurrent games open; each new_game call mints a
    new ``session_id`` you must pass to every other tool. Agent-identity fields
    (``agent_id``, ``model``, ``provider``, ``app_commit``) are stamped onto
    every harvested decision record for downstream training analysis.

    Args:
        seed: Optional RNG seed. The same seed always deals the same game.
        draw_count: Cards moved from stock to waste per draw (1 or 3).
        face_down: Visibility of face-down tableau cards -- "count" (only how
            many are hidden) or "revealed" (their identities; perfect info).
        stock: Visibility of the stock pile -- "hidden", "count", or "revealed".
        waste: Visibility of previously drawn cards -- "top" or "full".
        include_legal_moves: Whether observations embed the legal-move list.
        agent_id: Optional caller identifier (e.g. an LLM agent name).
        model: Model name powering the agent (e.g. ``"gemma-4-31b-it"``).
        provider: Model provider (e.g. ``"gemini"``).
        app_commit: Caller commit hash for reproducibility.

    Returns:
        ``{"session_id", "observation", "agent": {...}}``.
    """
    config = ObservationConfig(
        face_down=face_down,
        stock=stock,
        waste=waste,
        include_legal_moves=include_legal_moves,
    )
    session = GameSession.new_game(
        seed=seed, draw_count=draw_count, observation_config=config
    )
    entry = _registry.create(
        session,
        agent_id=agent_id,
        model=model,
        provider=provider,
        app_commit=app_commit,
    )
    _analytics.record(
        "game_started",
        session_id=entry.session_id,
        agent_id=agent_id,
        model=model,
        provider=provider,
        seed=seed,
        draw_count=draw_count,
        info_level=config.to_dict(),
    )
    return {
        "session_id": entry.session_id,
        "agent": {
            "agent_id": entry.agent_id,
            "model": entry.model,
            "provider": entry.provider,
            "app_commit": entry.app_commit,
        },
        "observation": session.observation(),
    }


@mcp.tool()
def list_sessions() -> List[Dict[str, Any]]:
    """Return summaries of every active session on this server."""
    return _registry.list()


@mcp.tool()
def end_session(session_id: str) -> Dict[str, Any]:
    """Close a session and free its registry slot.

    Records a ``game_abandoned`` analytics event if the game was unfinished.
    """
    entry = _registry.get(session_id)
    with entry.lock:
        session = entry.session
        if not (session.is_won() or session.is_stuck()):
            _analytics.record(
                "game_abandoned",
                session_id=entry.session_id,
                move_count=session.state.move_count,
                score=session.state.score,
            )
        _registry.end(session_id)
        return {
            "ended": True,
            "session_id": entry.session_id,
            "move_count": session.state.move_count,
            "score": session.state.score,
            "won": session.is_won(),
            "stuck": session.is_stuck(),
        }


@mcp.tool()
def get_observation(session_id: str) -> Dict[str, Any]:
    """Return the current observation for ``session_id``."""
    entry = _get_entry(session_id)
    with entry.lock:
        return entry.session.observation()


@mcp.tool()
def get_legal_moves(session_id: str) -> List[Dict[str, Any]]:
    """Return ``session_id``'s legal actions, each with a stable index."""
    entry = _get_entry(session_id)
    with entry.lock:
        return [action.to_dict() for action in entry.session.legal_actions()]


@mcp.tool()
def play_move(
    session_id: str,
    move_index: int,
    decision_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply the legal action at ``move_index`` and harvest a training record.

    Args:
        session_id: The session returned by ``new_game``.
        move_index: Index from ``get_legal_moves`` (or the embedded
            ``legal_moves`` block).
        decision_meta: Optional agent-supplied fields recorded with the
            decision (e.g. ``{"confidence": 0.92, "reasoning": "...",
            "boardAnalysis": "...", "alternativeMoveIndex": 2,
            "thinkingText": "..."}``). Used for distillation training data.

    Returns:
        ``{"applied", "kind", "won", "stuck", "observation", "decision_id"}``.
    """
    entry = _get_entry(session_id)
    with entry.lock:
        session = entry.session

        # Snapshot the pre-move state for the harvest record. The
        # legal_actions list and observation here are exactly what the agent
        # chose from -- capturing them after apply_action would be the *next*
        # turn's state and useless for training.
        pre_actions = session.legal_actions()
        if not 0 <= move_index < len(pre_actions):
            raise ValueError(
                f"Invalid move_index {move_index}; "
                f"{len(pre_actions)} legal action(s) available"
            )
        legal_snapshot = [a.to_dict() for a in pre_actions]
        chosen = pre_actions[move_index]
        pre_observation = session.observation()
        turn_index = session.state.move_count

        result = session.apply_action(move_index)

        record = build_decision_record(
            session_id=entry.session_id,
            turn_index=turn_index,
            legal_actions_snapshot=legal_snapshot,
            chosen_index=move_index,
            chosen_kind=chosen.kind,
            chosen_description=chosen.description,
            observation=pre_observation,
            draw_count=session.draw_count,
            seed=session.seed,
            info_level=session.observation_config.to_dict(),
            agent_id=entry.agent_id,
            model=entry.model,
            provider=entry.provider,
            app_commit=entry.app_commit,
            decision_meta=decision_meta,
        )
        _harvest.record(record)

        _analytics.record(
            "move",
            session_id=entry.session_id,
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
                session_id=entry.session_id,
                result="won" if result["won"] else "stuck",
                score=session.state.score,
                move_count=session.state.move_count,
                redeal_count=session.redeal_count,
            )

        result["observation"] = session.observation()
        result["decision_id"] = record.id
        return result


@mcp.tool()
def set_information_level(
    session_id: str,
    face_down: str = "count",
    stock: str = "count",
    waste: str = "full",
    include_legal_moves: bool = True,
) -> Dict[str, Any]:
    """Change ``session_id``'s information level and return a fresh observation."""
    entry = _get_entry(session_id)
    with entry.lock:
        entry.session.observation_config = ObservationConfig(
            face_down=face_down,
            stock=stock,
            waste=waste,
            include_legal_moves=include_legal_moves,
        )
        return entry.session.observation()


@mcp.tool()
def render_board(session_id: str) -> str:
    """Return a compact, human-readable text rendering of the board."""
    entry = _get_entry(session_id)
    with entry.lock:
        return entry.session.render()


@mcp.tool()
def game_status(session_id: str) -> Dict[str, Any]:
    """Return a short status: won, stuck, score, move and redeal counts."""
    entry = _get_entry(session_id)
    with entry.lock:
        session = entry.session
        return {
            "session_id": entry.session_id,
            "won": session.is_won(),
            "stuck": session.is_stuck(),
            "score": session.state.score,
            "move_count": session.state.move_count,
            "redeal_count": session.redeal_count,
            "draw_count": session.draw_count,
            "legal_move_count": len(session.legal_actions()),
        }


@mcp.tool()
def get_game_log(session_id: str) -> Dict[str, Any]:
    """Return the full per-game session log (seed, initial state, every action)."""
    entry = _get_entry(session_id)
    with entry.lock:
        return entry.session.get_log()


@mcp.tool()
def save_game_log(session_id: str, path: str) -> Dict[str, Any]:
    """Write ``session_id``'s log to a JSON file."""
    entry = _get_entry(session_id)
    with entry.lock:
        entry.session.save_log(path)
        return {
            "saved": True,
            "session_id": entry.session_id,
            "path": path,
            "actions_logged": len(entry.session.get_log()["actions"]),
        }


@mcp.tool()
def get_server_analytics() -> Dict[str, Any]:
    """Return aggregate analytics across every game played this server session.

    Set ``SOLITAIRE_MCP_LOG_FILE`` to also persist the raw event stream as a
    JSON Lines file, and ``SOLITAIRE_MCP_HARVEST_FILE`` to capture one decision
    record per move for distillation training.
    """
    summary = _analytics.summary()
    summary["harvest"] = _harvest.summary()
    summary["active_sessions"] = len(_registry)
    return summary


@mcp.tool()
def suggest_move(session_id: str) -> Dict[str, Any]:
    """Suggest a reasonable move using the built-in greedy strategy (a hint)."""
    entry = _get_entry(session_id)
    with entry.lock:
        session = entry.session
        actions = session.legal_actions()
        if not actions:
            return {"suggestion": None, "reason": "No legal moves; the game is over."}

        suggested = get_strategy("simple").select_best_move(session.state)
        if suggested is None:
            recycle = next((a for a in actions if a.kind == "recycle"), None)
            if recycle is not None:
                return {
                    "suggestion": recycle.to_dict(),
                    "reason": "Recycle the waste pile.",
                }
            return {
                "suggestion": actions[0].to_dict(),
                "reason": "Fallback to first legal move.",
            }

        for action in actions:
            move = action.move
            if move is None:
                continue
            if move.move_type == MoveType.STOCK_TO_WASTE == suggested.move_type:
                return {
                    "suggestion": action.to_dict(),
                    "reason": "Greedy strategy pick.",
                }
            if (
                move.move_type == suggested.move_type
                and move.source_pile == suggested.source_pile
                and move.dest_pile == suggested.dest_pile
                and move.num_cards == suggested.num_cards
            ):
                return {
                    "suggestion": action.to_dict(),
                    "reason": "Greedy strategy pick.",
                }

        return {
            "suggestion": actions[0].to_dict(),
            "reason": "Fallback to first legal move.",
        }


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
