"""Per-decision harvest emitter for AI-played Solitaire games.

When every move on the MCP server is made by an AI agent (as in the Gemma 4 E2B
distillation pipeline), the server itself becomes the authoritative source of
training data. This module captures one structured :class:`DecisionRecord` per
``play_move`` call -- the (board observation, legal-move list, chosen index,
agent reasoning) tuple needed to fine-tune a student model.

The on-disk format is JSON Lines (one decision per line) at the path given by
the ``SOLITAIRE_MCP_HARVEST_FILE`` environment variable. Records are shaped to
align with the fields ``scripts/ingest_exports.py`` derives from the external
collection harness (``id``, ``sessionId``, ``turnIndex``, ``model``,
``provider``, ``appCommit``, plus the parsed ``decision`` and ``prompt`` blocks)
so the ingest pipeline can pick them up with minimal adaptation.
"""

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Environment variable naming the JSON Lines harvest file.
ENV_HARVEST_FILE = "SOLITAIRE_MCP_HARVEST_FILE"

#: Schema tag for records emitted by this server. Bump when fields change.
HARVEST_SCHEMA = "server-harvest-v1"


@dataclass
class DecisionRecord:
    """One agent decision, harvest-ready.

    Mirrors the per-interaction shape consumed by ``scripts/ingest_exports.py``
    so the JSONL output drops straight into ``data/raw/`` for distillation.

    Attributes mirror the ingest schema names (camelCase) where they overlap
    so downstream tooling does not have to translate field names.
    """

    id: str
    sessionId: str
    turnIndex: int
    timestamp: int
    schemaTier: str
    outcome: str
    agentId: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    appCommit: Optional[str]
    drawCount: int
    seed: Optional[int]
    infoLevel: Dict[str, Any]
    prompt: Dict[str, Any]
    decision: Dict[str, Any]
    thinkingText: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionHarvest:
    """Records per-move decisions for offline distillation training.

    Each :meth:`record` call appends one JSON Lines row to the configured
    harvest file (and keeps an in-memory copy for tests and ``summary()``).
    Writes are serialised with a lock so concurrent sessions cannot interleave
    a single record's bytes on disk.
    """

    def __init__(
        self,
        harvest_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Create a harvest emitter.

        Args:
            harvest_file: Optional JSONL destination. Parent directories are
                created. ``None`` keeps records in memory only.
            logger: Logger for diagnostic info (defaults to ``solitaire.harvest``).
        """
        self.harvest_file = harvest_file
        self.logger = logger or logging.getLogger("solitaire.harvest")
        self.records: List[Dict[str, Any]] = []
        self._write_lock = threading.Lock()
        if self.harvest_file:
            Path(self.harvest_file).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DecisionHarvest":
        """Create a harvest emitter, reading the destination from the env."""
        return cls(harvest_file=os.environ.get(ENV_HARVEST_FILE) or None)

    def record(self, record: DecisionRecord) -> Dict[str, Any]:
        """Persist one decision and return the stored dict."""
        row = record.to_dict()
        line = json.dumps(row, default=str)
        with self._write_lock:
            self.records.append(row)
            if self.harvest_file:
                with open(self.harvest_file, "a") as handle:
                    handle.write(line + "\n")
        return row

    def summary(self) -> Dict[str, Any]:
        """Return aggregate stats over all harvested decisions this session."""
        by_session: Dict[str, int] = {}
        by_model: Dict[str, int] = {}
        for row in self.records:
            sid = row.get("sessionId") or "(none)"
            by_session[sid] = by_session.get(sid, 0) + 1
            mdl = row.get("model") or "(none)"
            by_model[mdl] = by_model.get(mdl, 0) + 1
        return {
            "total_decisions": len(self.records),
            "decisions_by_session": by_session,
            "decisions_by_model": by_model,
            "harvest_file": self.harvest_file,
        }


def build_decision_record(
    *,
    session_id: str,
    turn_index: int,
    legal_actions_snapshot: List[Dict[str, Any]],
    chosen_index: int,
    chosen_kind: str,
    chosen_description: str,
    observation: Dict[str, Any],
    draw_count: int,
    seed: Optional[int],
    info_level: Dict[str, Any],
    agent_id: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    app_commit: Optional[str],
    decision_meta: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Assemble a :class:`DecisionRecord` from the pre-move snapshot.

    ``decision_meta`` carries optional agent-supplied fields (``confidence``,
    ``reasoning``, ``boardAnalysis``, ``alternativeMoveIndex``, ``thinkingText``);
    unknown keys are passed through under ``decision`` unchanged.
    """
    meta = dict(decision_meta or {})
    thinking_text = meta.pop("thinkingText", None)
    decision_block: Dict[str, Any] = {
        "moveIndex": chosen_index,
        "chosenKind": chosen_kind,
        "chosenDescribe": chosen_description,
    }
    decision_block.update(meta)

    prompt_block: Dict[str, Any] = {
        "legalMoves": legal_actions_snapshot,
        "observation": observation,
    }

    return DecisionRecord(
        id=uuid.uuid4().hex,
        sessionId=session_id,
        turnIndex=turn_index,
        timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
        schemaTier=HARVEST_SCHEMA,
        outcome="success",
        agentId=agent_id,
        model=model,
        provider=provider,
        appCommit=app_commit,
        drawCount=draw_count,
        seed=seed,
        infoLevel=info_level,
        prompt=prompt_block,
        decision=decision_block,
        thinkingText=thinking_text,
    )
