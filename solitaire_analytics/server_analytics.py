"""Server-side analytics logging for the Solitaire MCP server.

While each :class:`~solitaire_analytics.game.GameSession` keeps its own
per-game log, this module records a *cross-game event stream* for the server
as a whole -- one structured event per game lifecycle step (game started,
move played, game ended/abandoned). Events are kept in memory for live
summaries and, optionally, appended to a JSON Lines file for offline analysis.

Important: a stdio MCP server must never write logs to stdout (it is reserved
for the protocol). This module writes only to a file and/or the standard
``logging`` framework, which is configured to use stderr.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Environment variable naming the JSON Lines file to append events to.
ENV_LOG_FILE = "SOLITAIRE_MCP_LOG_FILE"


class ServerAnalyticsLog:
    """Records and summarizes a cross-game event stream for the MCP server."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Create an analytics log.

        Args:
            log_file: Optional path to a JSON Lines file events are appended
                to. Parent directories are created. If ``None``, events are
                kept in memory and emitted via ``logging`` only.
            logger: Logger used to emit events (defaults to ``solitaire.mcp``).
        """
        self.log_file = log_file
        self.logger = logger or logging.getLogger("solitaire.mcp")
        self.events: List[Dict[str, Any]] = []
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ServerAnalyticsLog":
        """Create a log, taking the JSON Lines file path from the environment."""
        return cls(log_file=os.environ.get(ENV_LOG_FILE) or None)

    def record(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Record one analytics event.

        Args:
            event_type: The kind of event, e.g. ``"game_started"``, ``"move"``,
                ``"game_ended"``, ``"game_abandoned"``.
            **fields: Arbitrary JSON-serializable event details.

        Returns:
            The recorded event, including its UTC timestamp.
        """
        event: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
        }
        event.update(fields)
        self.events.append(event)

        self.logger.info("%s %s", event_type, json.dumps(fields, default=str))
        if self.log_file:
            with open(self.log_file, "a") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        return event

    def summary(self) -> Dict[str, Any]:
        """Return aggregate analytics over every event recorded so far."""
        started = [e for e in self.events if e["event"] == "game_started"]
        ended = [e for e in self.events if e["event"] == "game_ended"]
        abandoned = [e for e in self.events if e["event"] == "game_abandoned"]
        moves = [e for e in self.events if e["event"] == "move"]
        won = [e for e in ended if e.get("result") == "won"]
        stuck = [e for e in ended if e.get("result") == "stuck"]

        actions_by_kind: Dict[str, int] = {}
        for move in moves:
            kind = move.get("kind", "unknown")
            actions_by_kind[kind] = actions_by_kind.get(kind, 0) + 1

        move_counts = [
            e["move_count"] for e in ended if e.get("move_count") is not None
        ]
        completed = len(ended)

        return {
            "total_events": len(self.events),
            "games_started": len(started),
            "games_completed": completed,
            "games_won": len(won),
            "games_stuck": len(stuck),
            "games_abandoned": len(abandoned),
            "win_rate": len(won) / completed if completed else 0.0,
            "actions_logged": len(moves),
            "actions_by_kind": actions_by_kind,
            "avg_moves_per_completed_game": (
                sum(move_counts) / len(move_counts) if move_counts else 0.0
            ),
            "log_file": self.log_file,
        }
