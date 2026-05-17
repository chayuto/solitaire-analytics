"""Tests for the interactive game layer (dealer, observation, session)."""

import pytest

from solitaire_analytics.game import (
    GameSession,
    ObservationConfig,
    deal_klondike,
    make_deck,
    observe,
    render_text,
)


@pytest.mark.unit
class TestDealer:
    """Tests for dealing a fresh Klondike game."""

    def test_deck_has_52_unique_cards(self):
        deck = make_deck()
        assert len(deck) == 52
        assert len({(c.rank, c.suit) for c in deck}) == 52

    def test_deal_pile_sizes(self):
        state = deal_klondike(seed=1)
        assert [len(p) for p in state.tableau] == [1, 2, 3, 4, 5, 6, 7]
        assert len(state.stock) == 24
        assert len(state.waste) == 0

    def test_deal_uses_all_52_cards(self):
        state = deal_klondike(seed=7)
        cards = list(state.stock)
        for pile in state.tableau:
            cards.extend(pile)
        assert len({(c.rank, c.suit) for c in cards}) == 52

    def test_only_top_tableau_card_face_up(self):
        state = deal_klondike(seed=3)
        for pile in state.tableau:
            assert pile[-1].face_up
            assert all(not c.face_up for c in pile[:-1])
        assert all(not c.face_up for c in state.stock)

    def test_seed_is_reproducible(self):
        a = deal_klondike(seed=42)
        b = deal_klondike(seed=42)
        c = deal_klondike(seed=43)
        assert a == b
        assert a != c


@pytest.mark.unit
class TestObservationConfig:
    """Tests for the information-level configuration."""

    def test_rejects_invalid_levels(self):
        with pytest.raises(ValueError):
            ObservationConfig(face_down="bogus")
        with pytest.raises(ValueError):
            ObservationConfig(stock="bogus")
        with pytest.raises(ValueError):
            ObservationConfig(waste="bogus")

    def test_presets(self):
        assert ObservationConfig.perfect_information().face_down == "revealed"
        assert ObservationConfig.perfect_information().stock == "revealed"
        assert ObservationConfig.minimal().stock == "hidden"
        assert ObservationConfig.minimal().waste == "top"
        assert ObservationConfig.human().face_down == "count"

    def test_roundtrip_dict(self):
        config = ObservationConfig(face_down="revealed", stock="hidden", waste="top")
        assert ObservationConfig.from_dict(config.to_dict()) == config


@pytest.mark.unit
class TestObservationInformationLevel:
    """Tests that observations respect the configured information level."""

    def test_face_down_count_hides_card_identities(self):
        state = deal_klondike(seed=5)
        obs = observe(state, ObservationConfig(face_down="count"))
        last_pile = obs["tableau"][6]
        assert last_pile["face_down_count"] == 6
        assert "face_down_cards" not in last_pile

    def test_face_down_revealed_exposes_card_identities(self):
        state = deal_klondike(seed=5)
        obs = observe(state, ObservationConfig(face_down="revealed"))
        last_pile = obs["tableau"][6]
        assert len(last_pile["face_down_cards"]) == 6

    def test_stock_hidden_count_and_revealed(self):
        state = deal_klondike(seed=5)
        assert observe(state, ObservationConfig(stock="hidden"))["stock"] == {
            "hidden": True
        }
        count_view = observe(state, ObservationConfig(stock="count"))["stock"]
        assert count_view == {"count": 24}
        revealed_view = observe(state, ObservationConfig(stock="revealed"))["stock"]
        assert revealed_view["count"] == 24
        assert len(revealed_view["cards"]) == 24

    def test_waste_top_versus_full(self):
        session = GameSession.new_game(seed=5, draw_count=1)
        for _ in range(3):
            draw = next(a for a in session.legal_actions() if a.kind == "draw")
            session.apply_action(draw.index)

        top_only = session.observation(ObservationConfig(waste="top"))["waste"]
        assert top_only["top"] is not None
        assert "cards" not in top_only

        full = session.observation(ObservationConfig(waste="full"))["waste"]
        assert len(full["cards"]) == 3


@pytest.mark.unit
class TestGameSession:
    """Tests for the playable session interface."""

    def test_new_game_has_legal_moves(self):
        session = GameSession.new_game(seed=1)
        actions = session.legal_actions()
        assert len(actions) > 0
        assert all(a.index == i for i, a in enumerate(actions))

    def test_draw_moves_cards_to_waste(self):
        session = GameSession.new_game(seed=1, draw_count=3)
        draw = next(a for a in session.legal_actions() if a.kind == "draw")
        session.apply_action(draw.index)
        assert len(session.state.waste) == 3
        assert len(session.state.stock) == 21
        assert all(c.face_up for c in session.state.waste)

    def test_invalid_index_raises(self):
        session = GameSession.new_game(seed=1)
        with pytest.raises(ValueError):
            session.apply_action(999)

    def test_recycle_offered_when_stock_empty(self):
        session = GameSession.new_game(seed=1, draw_count=3)
        # Exhaust the stock by drawing repeatedly.
        for _ in range(20):
            draw = next(
                (a for a in session.legal_actions() if a.kind == "draw"), None
            )
            if draw is None:
                break
            session.apply_action(draw.index)

        assert len(session.state.stock) == 0
        recycle = next(
            (a for a in session.legal_actions() if a.kind == "recycle"), None
        )
        assert recycle is not None

        waste_before = len(session.state.waste)
        session.apply_action(recycle.index)
        assert len(session.state.stock) == waste_before
        assert len(session.state.waste) == 0
        assert session.redeal_count == 1

    def test_observation_reports_session_facts(self):
        session = GameSession.new_game(seed=1, draw_count=3)
        obs = session.observation()
        assert obs["draw_count"] == 3
        assert obs["redeal_count"] == 0
        assert obs["won"] is False
        assert "legal_moves" in obs

    def test_render_text_runs(self):
        state = deal_klondike(seed=2)
        rendered = render_text(state, ObservationConfig.perfect_information())
        assert "Foundations:" in rendered
        assert "Tableau:" in rendered


@pytest.mark.unit
class TestSessionLog:
    """Tests for the session log (starting deck, game state, seed, actions)."""

    def test_log_records_seed_and_starting_deck(self):
        session = GameSession.new_game(seed=99, draw_count=3)
        log = session.get_log()
        assert log["seed"] == 99
        assert log["draw_count"] == 3
        # The initial_state captures the full dealt deck.
        initial = log["initial_state"]
        assert [len(p) for p in initial["tableau"]] == [1, 2, 3, 4, 5, 6, 7]
        assert len(initial["stock"]) == 24

    def test_log_records_actions_with_resulting_state(self):
        session = GameSession.new_game(seed=11, draw_count=1)
        for _ in range(3):
            actions = session.legal_actions()
            chosen = next((a for a in actions if a.kind == "move"), actions[0])
            session.apply_action(chosen.index)

        log = session.get_log()
        assert len(log["actions"]) == 3
        first = log["actions"][0]
        assert first["seq"] == 1
        assert "description" in first
        assert "resulting_state" in first
        assert log["result"]["total_actions"] == 3

    def test_log_initial_state_reproduces_with_seed(self):
        from solitaire_analytics import GameState
        from solitaire_analytics.game import deal_klondike

        session = GameSession.new_game(seed=123)
        logged_initial = GameState.from_dict(session.get_log()["initial_state"])
        assert logged_initial == deal_klondike(seed=123)

    def test_logging_can_be_disabled(self):
        session = GameSession.new_game(seed=1, log=False)
        actions = session.legal_actions()
        session.apply_action(actions[0].index)
        assert session.get_log()["actions"] == []

    def test_save_log_writes_file(self, tmp_path):
        import json

        session = GameSession.new_game(seed=5)
        session.apply_action(session.legal_actions()[0].index)
        path = tmp_path / "logs" / "game.json"
        session.save_log(str(path))

        data = json.loads(path.read_text())
        assert data["seed"] == 5
        assert len(data["actions"]) == 1


@pytest.mark.unit
class TestServerAnalyticsLog:
    """Tests for the cross-game server-side analytics log."""

    def _log(self):
        from solitaire_analytics.server_analytics import ServerAnalyticsLog

        return ServerAnalyticsLog()

    def test_record_adds_event_with_timestamp(self):
        log = self._log()
        event = log.record("game_started", game_id="game-1", seed=7)
        assert event["event"] == "game_started"
        assert event["seed"] == 7
        assert "timestamp" in event
        assert len(log.events) == 1

    def test_summary_aggregates_games_and_win_rate(self):
        log = self._log()
        log.record("game_started", game_id="game-1")
        log.record("move", game_id="game-1", kind="draw")
        log.record("move", game_id="game-1", kind="move")
        log.record("game_ended", game_id="game-1", result="won", move_count=120)
        log.record("game_started", game_id="game-2")
        log.record("game_ended", game_id="game-2", result="stuck", move_count=80)

        summary = log.summary()
        assert summary["games_started"] == 2
        assert summary["games_completed"] == 2
        assert summary["games_won"] == 1
        assert summary["games_stuck"] == 1
        assert summary["win_rate"] == 0.5
        assert summary["actions_logged"] == 2
        assert summary["actions_by_kind"] == {"draw": 1, "move": 1}
        assert summary["avg_moves_per_completed_game"] == 100.0

    def test_appends_to_jsonl_file(self, tmp_path):
        import json
        from solitaire_analytics.server_analytics import ServerAnalyticsLog

        path = tmp_path / "events" / "analytics.jsonl"
        log = ServerAnalyticsLog(log_file=str(path))
        log.record("game_started", game_id="game-1", seed=1)
        log.record("game_ended", game_id="game-1", result="won")

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "game_started"
        assert json.loads(lines[1])["result"] == "won"


@pytest.mark.integration
class TestSessionPlaythrough:
    """Integration test: drive a session forward many moves."""

    def test_session_advances_without_error(self):
        session = GameSession.new_game(seed=11, draw_count=1)
        for _ in range(200):
            if session.is_won() or session.is_stuck():
                break
            actions = session.legal_actions()
            # Prefer a real move over draw/recycle to make progress.
            chosen = next((a for a in actions if a.kind == "move"), actions[0])
            result = session.apply_action(chosen.index)
            assert "applied" in result
        assert session.state.move_count > 0
