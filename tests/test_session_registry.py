"""Tests for the multi-session registry."""

import threading

import pytest

from solitaire_analytics.game import GameSession
from solitaire_analytics.session_registry import (
    SessionEntry,
    SessionRegistry,
    SessionRegistryError,
)


def _make_session(seed: int = 1) -> GameSession:
    return GameSession.new_game(seed=seed)


@pytest.mark.unit
class TestSessionRegistry:
    """Unit tests for SessionRegistry create/get/list/end semantics."""

    def test_create_returns_entry_with_unique_id(self):
        registry = SessionRegistry()
        a = registry.create(_make_session(1))
        b = registry.create(_make_session(2))
        assert isinstance(a, SessionEntry)
        assert a.session_id != b.session_id
        assert len(registry) == 2

    def test_get_returns_registered_entry(self):
        registry = SessionRegistry()
        entry = registry.create(_make_session(1), agent_id="alice", model="m1")
        fetched = registry.get(entry.session_id)
        assert fetched is entry
        assert fetched.agent_id == "alice"
        assert fetched.model == "m1"

    def test_get_unknown_raises(self):
        registry = SessionRegistry()
        with pytest.raises(SessionRegistryError):
            registry.get("not-a-session")

    def test_end_removes_entry(self):
        registry = SessionRegistry()
        entry = registry.create(_make_session(1))
        registry.end(entry.session_id)
        assert entry.session_id not in registry
        assert len(registry) == 0
        with pytest.raises(SessionRegistryError):
            registry.get(entry.session_id)

    def test_end_unknown_raises(self):
        registry = SessionRegistry()
        with pytest.raises(SessionRegistryError):
            registry.end("not-a-session")

    def test_max_sessions_enforced(self):
        registry = SessionRegistry(max_sessions=2)
        registry.create(_make_session(1))
        registry.create(_make_session(2))
        with pytest.raises(SessionRegistryError):
            registry.create(_make_session(3))

    def test_max_sessions_after_end_frees_slot(self):
        registry = SessionRegistry(max_sessions=2)
        a = registry.create(_make_session(1))
        registry.create(_make_session(2))
        registry.end(a.session_id)
        # Should succeed now that a slot is free.
        registry.create(_make_session(3))
        assert len(registry) == 2

    def test_list_returns_summaries_with_metadata(self):
        registry = SessionRegistry()
        registry.create(
            _make_session(1),
            agent_id="alice",
            model="m1",
            provider="gemini",
            app_commit="abc123",
        )
        registry.create(_make_session(2), agent_id="bob")

        rows = registry.list()
        assert len(rows) == 2
        agents = {row["agent_id"] for row in rows}
        assert agents == {"alice", "bob"}
        alice_row = next(row for row in rows if row["agent_id"] == "alice")
        assert alice_row["model"] == "m1"
        assert alice_row["provider"] == "gemini"
        assert alice_row["app_commit"] == "abc123"
        assert alice_row["seed"] == 1
        assert "session_id" in alice_row

    def test_per_session_lock_is_reentrant(self):
        registry = SessionRegistry()
        entry = registry.create(_make_session(1))
        # An RLock can be acquired twice by the same thread; the second acquire
        # must not deadlock. This guards against accidentally using a plain
        # Lock, which would hang here.
        with entry.lock:
            with entry.lock:
                assert True

    def test_concurrent_access_to_different_sessions_does_not_serialise(self):
        # Two threads each hold their own session lock at the same time;
        # neither blocks the other. This proves the registry uses per-session
        # locks rather than a single global one.
        registry = SessionRegistry()
        a = registry.create(_make_session(1))
        b = registry.create(_make_session(2))

        both_in_critical = threading.Event()
        started = threading.Barrier(2)

        def hold(entry, peer_in_critical: threading.Event):
            with entry.lock:
                started.wait(timeout=1.0)
                # Signal we hold our lock and wait briefly for the peer to do
                # the same. If the registry serialised, this barrier would
                # never balance.
                peer_in_critical.wait(timeout=1.0)
                both_in_critical.set()

        peer_a = threading.Event()
        peer_b = threading.Event()
        ta = threading.Thread(target=hold, args=(a, peer_b))
        tb = threading.Thread(target=hold, args=(b, peer_a))
        ta.start()
        tb.start()
        peer_a.set()
        peer_b.set()
        ta.join(timeout=2.0)
        tb.join(timeout=2.0)
        assert both_in_critical.is_set()
