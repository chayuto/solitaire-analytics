"""Tests for the per-decision harvest emitter."""

import json
import os
from unittest.mock import patch

import pytest

from solitaire_analytics.harvest import (
    ENV_HARVEST_FILE,
    HARVEST_SCHEMA,
    DecisionHarvest,
    DecisionRecord,
    build_decision_record,
)


def _sample_record(session_id: str = "s1", turn_index: int = 0) -> DecisionRecord:
    return build_decision_record(
        session_id=session_id,
        turn_index=turn_index,
        legal_actions_snapshot=[
            {"index": 0, "kind": "move", "description": "AS->2H"},
            {"index": 1, "kind": "draw", "description": "Draw 1"},
        ],
        chosen_index=0,
        chosen_kind="move",
        chosen_description="AS->2H",
        observation={"tableau": []},
        draw_count=1,
        seed=42,
        info_level={"face_down": "count"},
        agent_id="alice",
        model="gemma-4-31b-it",
        provider="gemini",
        app_commit="abc1234",
        decision_meta={
            "confidence": 0.9,
            "reasoning": "expose face-down",
            "thinkingText": "thinking...",
            "alternativeMoveIndex": 1,
        },
    )


@pytest.mark.unit
class TestBuildDecisionRecord:
    """build_decision_record packs ingest-shaped rows."""

    def test_record_has_required_ingest_fields(self):
        record = _sample_record().to_dict()
        for field in (
            "id",
            "sessionId",
            "turnIndex",
            "timestamp",
            "schemaTier",
            "outcome",
            "agentId",
            "model",
            "provider",
            "appCommit",
            "drawCount",
            "seed",
            "infoLevel",
            "prompt",
            "decision",
            "thinkingText",
        ):
            assert field in record, f"missing {field}"

    def test_decision_block_captures_chosen_move(self):
        record = _sample_record().to_dict()
        assert record["decision"]["moveIndex"] == 0
        assert record["decision"]["chosenKind"] == "move"
        assert record["decision"]["chosenDescribe"] == "AS->2H"
        assert record["decision"]["confidence"] == 0.9
        assert record["decision"]["reasoning"] == "expose face-down"
        assert record["decision"]["alternativeMoveIndex"] == 1

    def test_prompt_block_carries_legal_moves(self):
        record = _sample_record().to_dict()
        assert len(record["prompt"]["legalMoves"]) == 2
        assert record["prompt"]["legalMoves"][0]["description"] == "AS->2H"

    def test_thinking_text_lifted_out_of_decision_meta(self):
        record = _sample_record().to_dict()
        # thinkingText is a top-level field in the ingest schema, not under
        # "decision" -- the builder pops it from decision_meta accordingly.
        assert record["thinkingText"] == "thinking..."
        assert "thinkingText" not in record["decision"]

    def test_schema_tier_marked_for_server_harvest(self):
        record = _sample_record().to_dict()
        assert record["schemaTier"] == HARVEST_SCHEMA

    def test_agent_metadata_passed_through(self):
        record = _sample_record().to_dict()
        assert record["agentId"] == "alice"
        assert record["model"] == "gemma-4-31b-it"
        assert record["provider"] == "gemini"
        assert record["appCommit"] == "abc1234"


@pytest.mark.unit
class TestDecisionHarvest:
    """DecisionHarvest writes JSONL and aggregates summaries."""

    def test_record_kept_in_memory(self):
        harvest = DecisionHarvest()
        harvest.record(_sample_record())
        assert len(harvest.records) == 1
        assert harvest.records[0]["sessionId"] == "s1"

    def test_record_writes_jsonl_line(self, tmp_path):
        path = tmp_path / "events" / "harvest.jsonl"
        harvest = DecisionHarvest(harvest_file=str(path))
        harvest.record(_sample_record(turn_index=0))
        harvest.record(_sample_record(turn_index=1))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["sessionId"] == "s1"
        assert row["turnIndex"] == 0
        assert row["schemaTier"] == HARVEST_SCHEMA

    def test_summary_breaks_down_by_session_and_model(self):
        harvest = DecisionHarvest()
        harvest.record(_sample_record(session_id="s1"))
        harvest.record(_sample_record(session_id="s1", turn_index=1))
        harvest.record(_sample_record(session_id="s2"))

        summary = harvest.summary()
        assert summary["total_decisions"] == 3
        assert summary["decisions_by_session"] == {"s1": 2, "s2": 1}
        assert summary["decisions_by_model"] == {"gemma-4-31b-it": 3}

    def test_from_env_reads_path(self, tmp_path):
        path = tmp_path / "from_env.jsonl"
        with patch.dict(os.environ, {ENV_HARVEST_FILE: str(path)}):
            harvest = DecisionHarvest.from_env()
        assert harvest.harvest_file == str(path)

    def test_from_env_with_no_var_keeps_memory_only(self):
        with patch.dict(os.environ, {}, clear=True):
            harvest = DecisionHarvest.from_env()
        assert harvest.harvest_file is None
