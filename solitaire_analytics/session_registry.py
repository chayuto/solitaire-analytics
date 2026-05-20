"""In-process registry of concurrent Solitaire game sessions.

The MCP server used to hold a single global :class:`GameSession`; this registry
lets many agents each play their own game over the same server process. Every
session is identified by a server-issued ``session_id`` (UUID4) and carries
optional agent-supplied metadata (``agent_id``, ``model``, ``provider``,
``app_commit``) that the harvest emitter stamps onto each decision record.

Each session is guarded by its own :class:`threading.RLock` so two MCP tool
calls for different sessions can run concurrently without serialising on a
single global lock; calls for the *same* session still serialise so a moves
sequence within a game stays consistent. Use :meth:`SessionRegistry.lock` as a
context manager whenever you read or mutate session state from a request
handler.
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from solitaire_analytics.game import GameSession


class SessionRegistryError(RuntimeError):
    """Raised for session-registry problems (full, missing, duplicate)."""


@dataclass
class SessionEntry:
    """A single registered session and its agent-declared metadata.

    Attributes:
        session_id: Server-issued opaque identifier the agent passes back.
        session: The wrapped :class:`GameSession`.
        agent_id: Optional caller-supplied agent identifier.
        model: Optional model name used by the agent (e.g. ``"gemma-4-31b-it"``).
        provider: Optional model provider (e.g. ``"gemini"``, ``"openai"``).
        app_commit: Optional commit hash of the caller, for reproducibility.
        created_at: UTC creation timestamp.
        lock: Per-session reentrant lock; held during any state-mutating call.
    """

    session_id: str
    session: GameSession
    agent_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    app_commit: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def info(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary suitable for ``list_sessions``."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "provider": self.provider,
            "app_commit": self.app_commit,
            "created_at": self.created_at.isoformat(),
            "move_count": self.session.state.move_count,
            "score": self.session.state.score,
            "won": self.session.is_won(),
            "stuck": self.session.is_stuck(),
            "draw_count": self.session.draw_count,
            "seed": self.session.seed,
        }


class SessionRegistry:
    """Thread-safe registry of concurrent :class:`GameSession` instances."""

    def __init__(self, max_sessions: int = 100):
        """Create a registry.

        Args:
            max_sessions: Hard cap on simultaneously open sessions. Reached
                when ``create()`` would push the count over the cap; callers
                must explicitly ``end()`` finished games to free a slot.
        """
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}")
        self.max_sessions = int(max_sessions)
        self._entries: Dict[str, SessionEntry] = {}
        self._registry_lock = threading.Lock()

    def create(
        self,
        session: GameSession,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        app_commit: Optional[str] = None,
    ) -> SessionEntry:
        """Register a new session and return its entry (with the ``session_id``).

        Raises:
            SessionRegistryError: If the registry is at ``max_sessions``.
        """
        with self._registry_lock:
            if len(self._entries) >= self.max_sessions:
                raise SessionRegistryError(
                    f"Session registry full ({self.max_sessions} active); "
                    "end an existing session before starting a new one."
                )
            session_id = uuid.uuid4().hex
            entry = SessionEntry(
                session_id=session_id,
                session=session,
                agent_id=agent_id,
                model=model,
                provider=provider,
                app_commit=app_commit,
            )
            self._entries[session_id] = entry
            return entry

    def get(self, session_id: str) -> SessionEntry:
        """Return the entry for ``session_id``.

        Raises:
            SessionRegistryError: If no such session exists.
        """
        with self._registry_lock:
            entry = self._entries.get(session_id)
        if entry is None:
            raise SessionRegistryError(
                f"Unknown session_id {session_id!r}; "
                "call new_game to start a session."
            )
        return entry

    def end(self, session_id: str) -> SessionEntry:
        """Remove and return the session entry. Raises if missing."""
        with self._registry_lock:
            entry = self._entries.pop(session_id, None)
        if entry is None:
            raise SessionRegistryError(
                f"Unknown session_id {session_id!r}; cannot end."
            )
        return entry

    def list(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all active sessions as summary dicts."""
        with self._registry_lock:
            entries = list(self._entries.values())
        return [entry.info() for entry in entries]

    def __len__(self) -> int:
        with self._registry_lock:
            return len(self._entries)

    def __contains__(self, session_id: str) -> bool:
        with self._registry_lock:
            return session_id in self._entries
