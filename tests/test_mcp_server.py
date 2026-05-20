"""Tests for the multi-session MCP server module.

These exercise the underlying tool functions directly -- we do not spin up
FastMCP's stdio transport. The MCP server holds module-level singletons
(registry, analytics, harvest), so each test resets them in a fixture.
"""

import json

import pytest

from solitaire_analytics import mcp_server as server
from solitaire_analytics.harvest import DecisionHarvest
from solitaire_analytics.server_analytics import ServerAnalyticsLog
from solitaire_analytics.session_registry import (
    SessionRegistry,
    SessionRegistryError,
)


@pytest.fixture
def fresh_server(monkeypatch, tmp_path):
    """Reset the MCP server's module-level singletons for each test.

    Returns the (registry, analytics, harvest) triple so tests can introspect
    them. A harvest file is wired up so we can assert on disk output without
    side-effecting the real environment.
    """
    harvest_path = tmp_path / "harvest.jsonl"
    registry = SessionRegistry(max_sessions=10)
    analytics = ServerAnalyticsLog()
    harvest = DecisionHarvest(harvest_file=str(harvest_path))
    monkeypatch.setattr(server, "_registry", registry)
    monkeypatch.setattr(server, "_analytics", analytics)
    monkeypatch.setattr(server, "_harvest", harvest)
    return registry, analytics, harvest, harvest_path


@pytest.mark.unit
class TestNewGame:
    """new_game registers a session and returns its identity."""

    def test_returns_session_id_and_observation(self, fresh_server):
        registry, *_ = fresh_server
        result = server.new_game(seed=1)
        assert "session_id" in result
        assert "observation" in result
        assert result["session_id"] in registry

    def test_records_agent_metadata(self, fresh_server):
        registry, *_ = fresh_server
        result = server.new_game(
            seed=1,
            agent_id="alice",
            model="gemma-4-31b-it",
            provider="gemini",
            app_commit="abc",
        )
        assert result["agent"]["agent_id"] == "alice"
        assert result["agent"]["model"] == "gemma-4-31b-it"
        entry = registry.get(result["session_id"])
        assert entry.model == "gemma-4-31b-it"

    def test_emits_game_started_event(self, fresh_server):
        _, analytics, *_ = fresh_server
        server.new_game(seed=1, model="m")
        events = [e for e in analytics.events if e["event"] == "game_started"]
        assert len(events) == 1
        assert events[0]["model"] == "m"
        assert "session_id" in events[0]


@pytest.mark.unit
class TestMultiSessionIsolation:
    """Two sessions on one server stay independent."""

    def test_two_sessions_have_distinct_state(self, fresh_server):
        a = server.new_game(seed=1)["session_id"]
        b = server.new_game(seed=2)["session_id"]
        assert a != b

        # Apply a move in session A only; session B's move_count stays at 0.
        server.play_move(a, 0)
        status_a = server.game_status(a)
        status_b = server.game_status(b)
        assert status_a["move_count"] >= 1
        assert status_b["move_count"] == 0

    def test_list_sessions_reflects_live_state(self, fresh_server):
        a = server.new_game(seed=1)["session_id"]
        b = server.new_game(seed=2)["session_id"]
        rows = server.list_sessions()
        ids = {row["session_id"] for row in rows}
        assert ids == {a, b}

    def test_unknown_session_raises(self, fresh_server):
        with pytest.raises(SessionRegistryError):
            server.play_move("not-a-session", 0)


@pytest.mark.unit
class TestPlayMoveHarvest:
    """play_move records one DecisionRecord per move."""

    def test_decision_record_emitted(self, fresh_server):
        _, _, harvest, harvest_path = fresh_server
        sid = server.new_game(
            seed=1, agent_id="alice", model="gemma-4-31b-it"
        )["session_id"]
        result = server.play_move(
            sid,
            0,
            decision_meta={"confidence": 0.95, "reasoning": "test"},
        )
        assert result["decision_id"]
        assert len(harvest.records) == 1
        record = harvest.records[0]
        assert record["sessionId"] == sid
        assert record["model"] == "gemma-4-31b-it"
        assert record["decision"]["moveIndex"] == 0
        assert record["decision"]["confidence"] == 0.95
        assert record["turnIndex"] == 0  # state.move_count before the move

    def test_decision_record_persisted_to_jsonl(self, fresh_server):
        _, _, _, harvest_path = fresh_server
        sid = server.new_game(seed=1)["session_id"]
        server.play_move(sid, 0)
        server.play_move(sid, 0)
        lines = harvest_path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["sessionId"] == sid

    def test_invalid_move_index_does_not_emit_record(self, fresh_server):
        _, _, harvest, _ = fresh_server
        sid = server.new_game(seed=1)["session_id"]
        with pytest.raises(ValueError):
            server.play_move(sid, 999)
        assert harvest.records == []

    def test_legal_moves_snapshot_matches_choice(self, fresh_server):
        # The harvest record must capture the legal-action list the agent
        # *saw* at decision time, not the post-move list. Re-checking the
        # chosen index against the snapshot should yield the same description.
        _, _, harvest, _ = fresh_server
        sid = server.new_game(seed=1)["session_id"]
        legal_before = server.get_legal_moves(sid)
        chosen_idx = next(i for i, a in enumerate(legal_before) if a["kind"] == "draw")
        result = server.play_move(sid, chosen_idx)

        record = harvest.records[0]
        snapshot = record["prompt"]["legalMoves"]
        assert snapshot == legal_before
        assert snapshot[chosen_idx]["description"] == result["applied"]


@pytest.mark.unit
class TestEndSession:
    """end_session frees the slot and tags abandonments."""

    def test_end_removes_from_registry(self, fresh_server):
        registry, *_ = fresh_server
        sid = server.new_game(seed=1)["session_id"]
        server.end_session(sid)
        assert sid not in registry

    def test_end_unfinished_records_abandoned_event(self, fresh_server):
        _, analytics, *_ = fresh_server
        sid = server.new_game(seed=1)["session_id"]
        server.end_session(sid)
        abandoned = [e for e in analytics.events if e["event"] == "game_abandoned"]
        assert len(abandoned) == 1
        assert abandoned[0]["session_id"] == sid

    def test_end_unknown_raises(self, fresh_server):
        with pytest.raises(SessionRegistryError):
            server.end_session("not-a-session")


@pytest.mark.unit
class TestGetServerAnalytics:
    """get_server_analytics aggregates events + harvest + session count."""

    def test_summary_includes_harvest_and_active_sessions(self, fresh_server):
        server.new_game(seed=1)
        server.new_game(seed=2)
        summary = server.get_server_analytics()
        assert summary["active_sessions"] == 2
        assert "harvest" in summary
        assert summary["games_started"] == 2
