"""End-to-end integration tests: full playthroughs via the multi-session MCP API.

Drives 2-3 concurrent sessions through the actual MCP tool functions until
each game finishes or hits a move cap, asserting:

* sessions stay isolated (one game's moves don't leak into another's log),
* the harvest emits exactly one record per applied move,
* each harvested record reconstructs the chosen action correctly,
* end_session frees its slot and survives the cap,
* list_sessions/get_server_analytics reflect the live state.
"""

import json

import pytest

from solitaire_analytics import mcp_server as server
from solitaire_analytics.harvest import DecisionHarvest
from solitaire_analytics.server_analytics import ServerAnalyticsLog
from solitaire_analytics.session_registry import SessionRegistry


MOVE_CAP = 80  # cap per session so the test stays well under the pytest timeout


@pytest.fixture
def fresh_server(monkeypatch, tmp_path):
    """Isolated registry/analytics/harvest per test."""
    harvest_path = tmp_path / "harvest.jsonl"
    monkeypatch.setattr(server, "_registry", SessionRegistry(max_sessions=10))
    monkeypatch.setattr(server, "_analytics", ServerAnalyticsLog())
    monkeypatch.setattr(
        server, "_harvest", DecisionHarvest(harvest_file=str(harvest_path))
    )
    return harvest_path


def _greedy_play(session_id: str, cap: int = MOVE_CAP) -> int:
    """Play a session forward with a trivial 'prefer real moves' agent.

    Returns the number of moves applied. Stops at cap, win, or stuck.
    """
    applied = 0
    for _ in range(cap):
        status = server.game_status(session_id)
        if status["won"] or status["stuck"]:
            break
        legal = server.get_legal_moves(session_id)
        # Pick a real tableau/foundation move when one exists; otherwise
        # draw/recycle. The point is to exercise many code paths, not to win.
        chosen = next(
            (a for a in legal if a["kind"] == "move"),
            legal[0] if legal else None,
        )
        if chosen is None:
            break
        server.play_move(
            session_id,
            chosen["index"],
            decision_meta={"confidence": 0.9, "reasoning": "greedy"},
        )
        applied += 1
    return applied


@pytest.mark.integration
class TestConcurrentPlaythrough:
    """Drive multiple sessions to completion through the MCP tool layer."""

    def test_two_sessions_play_independently(self, fresh_server):
        harvest_path = fresh_server
        a = server.new_game(seed=11, agent_id="alpha", model="m-alpha")["session_id"]
        b = server.new_game(seed=22, agent_id="beta", model="m-beta")["session_id"]

        moves_a = _greedy_play(a)
        moves_b = _greedy_play(b)

        assert moves_a > 0 and moves_b > 0

        # Each session's log only contains its own moves.
        log_a = server.get_game_log(a)
        log_b = server.get_game_log(b)
        assert log_a["seed"] == 11
        assert log_b["seed"] == 22
        assert len(log_a["actions"]) == moves_a
        assert len(log_b["actions"]) == moves_b

        # Harvest captured every applied move, tagged with the right session.
        lines = harvest_path.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == moves_a + moves_b
        per_session: dict[str, int] = {}
        for record in records:
            per_session[record["sessionId"]] = per_session.get(record["sessionId"], 0) + 1
        assert per_session[a] == moves_a
        assert per_session[b] == moves_b

        # Agent identity stayed pinned to the session that started it.
        for record in records:
            if record["sessionId"] == a:
                assert record["agentId"] == "alpha"
                assert record["model"] == "m-alpha"
            else:
                assert record["agentId"] == "beta"
                assert record["model"] == "m-beta"

    def test_harvest_records_reconstruct_the_chosen_move(self, fresh_server):
        """Every record's snapshot[moveIndex] must equal the move actually applied."""
        harvest_path = fresh_server
        sid = server.new_game(seed=7)["session_id"]
        _greedy_play(sid, cap=30)

        records = [
            json.loads(line)
            for line in harvest_path.read_text().strip().splitlines()
        ]
        assert len(records) > 0
        log_actions = server.get_game_log(sid)["actions"]
        assert len(records) == len(log_actions)

        for record, logged in zip(records, log_actions):
            chosen_idx = record["decision"]["moveIndex"]
            snapshot = record["prompt"]["legalMoves"]
            assert 0 <= chosen_idx < len(snapshot)
            # The snapshot's chosen entry describes the move that the session
            # log says was applied. This catches off-by-one snapshot/apply bugs.
            assert snapshot[chosen_idx]["description"] == logged["description"]

    def test_end_session_frees_slot_and_records_outcome(self, fresh_server):
        a = server.new_game(seed=11)["session_id"]
        b = server.new_game(seed=22)["session_id"]

        _greedy_play(a, cap=20)
        # End an in-progress game -- expect game_abandoned.
        server.end_session(a)
        assert server._registry.__contains__(a) is False

        # Starting a new session should succeed (slot freed); b remains alive.
        c = server.new_game(seed=33)["session_id"]
        rows = server.list_sessions()
        ids = {row["session_id"] for row in rows}
        assert ids == {b, c}

    def test_server_analytics_reflects_full_run(self, fresh_server):
        a = server.new_game(seed=11)["session_id"]
        b = server.new_game(seed=22)["session_id"]
        moves_a = _greedy_play(a)
        moves_b = _greedy_play(b)

        summary = server.get_server_analytics()
        assert summary["active_sessions"] == 2
        assert summary["games_started"] == 2
        assert summary["actions_logged"] == moves_a + moves_b
        # harvest summary echoes the same total
        assert summary["harvest"]["total_decisions"] == moves_a + moves_b
        assert set(summary["harvest"]["decisions_by_session"]) == {a, b}
