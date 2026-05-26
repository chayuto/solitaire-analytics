#!/usr/bin/env python3
"""Ingest Solitaire AI export logs into a deduplicated, interaction-keyed store.

Exports overlap in practice: the same interaction can appear in more than one
file, a single game session can span several files, and some files are partial
or full re-exports of others. Every interaction carries a globally-unique
UUIDv7 ``id``, though, so the reliable unit of deduplication is the
interaction, not the file. This script unions all interactions by ``id`` --
overlap becomes harmless and nothing is double-counted or lost, whatever the
exporter's chunking rule turns out to be.

NOTHING IS EVER DISCARDED. Every file in data/raw/ is ingested in full; every
interaction is kept in the store. The "training dataset" is a *selection* over
that store -- decided by SELECTION CRITERIA below -- and every success decision
is tagged with whether it is eligible and, if not, why.

Two datasets are derived from the one store, each a named selection:
  * LOCAL set    -- teacher-model decisions on the current schema, for the
                    local fine-tune. Narrow and on-target.
  * PUBLISHING set -- every teacher decision, all models and schemas: a broader
                    community corpus, published to Hugging Face under CC-BY-4.0
                    with a dataset card. Rows are published as-is (no field
                    stripping).

Layout (paths resolve relative to the repo root):

    data/raw/                       raw exports, immutable -- gitignored
    data/store/interactions.jsonl   ALL interactions, deduped by id
    data/index/manifest.jsonl       provenance: one row per ingested raw file
    data/dataset/decisions.jsonl    ALL success decisions, tagged in/out + reason
    data/dataset/training.jsonl     the LOCAL set (selected training subset)
    data/publish/                   the PUBLISHING set -- Hugging Face dataset
    data/publish/README.md          HF dataset card (CC-BY-4.0)
    data/SUMMARY.md                 auto-generated stats (do not hand-edit)

The run is idempotent. Default mode is incremental (files already in the
manifest, matched by sha256, are skipped); ``--rebuild`` reprocesses every
file in data/raw/.

Usage:
    python scripts/ingest_exports.py
    python scripts/ingest_exports.py --rebuild
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
STORE = DATA / "store" / "interactions.jsonl"
MANIFEST = DATA / "index" / "manifest.jsonl"
DECISIONS = DATA / "dataset" / "decisions.jsonl"
TRAINING = DATA / "dataset" / "training.jsonl"
PUBLISH_DIR = DATA / "publish"
PUBLISH_CARD = PUBLISH_DIR / "README.md"

# Hugging Face publishing -- three configs under one dataset path. Config names
# are provenance-prefixed (``client_v1_*``) so server-collected data can slot
# in later as ``server_v1_*`` without renaming what already shipped.
PUBLISH_FULL_RAW       = PUBLISH_DIR / "client_v1_full_corpus_raw.jsonl"
PUBLISH_CLEAN_RAW      = PUBLISH_DIR / "client_v1_teacher_clean_raw.jsonl"
PUBLISH_CLEAN_LEAN     = PUBLISH_DIR / "client_v1_teacher_clean_lean.jsonl"

# Legacy single-file artefact -- kept so users who pinned the old filename keep
# working until the rename has propagated.
PUBLISH_LEGACY_ALIAS   = PUBLISH_DIR / "solitaire_advisor_decisions.jsonl"
SUMMARY = DATA / "SUMMARY.md"

# Hugging Face publishing config.
HF_LICENSE = "cc-by-4.0"
HF_PRETTY_NAME = "Klondike Solitaire LLM Advisor Decisions"

# --------------------------------------------------------- SELECTION CRITERIA
# These decide only what enters the *training* set. They never affect the
# store: legacy-schema rows and other-model rows are kept and queryable.
#
# Teacher being distilled -- only this model's decisions are training-eligible.
TEACHER_MODEL = "gemma-4-31b-it"
# Schema contract for the "current" exporter format (schema v3 onward): a row
# is current when it carries all these identity fields. Older exports (v1/v2)
# lack appCommit/sessionId/turnIndex; they stay in the store as reference but
# are not training-eligible. Designing from the latest schema onward means new
# builds keep flowing in as long as they still emit these fields.
SCHEMA_CONTRACT_FIELDS = ("id", "sessionId", "turnIndex", "appCommit")

# Stall filter (per GAME_PROGRESS_METRIC_2026-05-19.md). A session is "stalled"
# when foundationCards (sum of foundation ranks, 0..52) and faceDownTotal
# (sum of faceDownCount across the 7 columns, 21..0) have BOTH been unchanged
# for at least STALL_TURNS consecutive interactions. Decisions inside the
# stalled stretch are kept in the store and the publish set as a research
# baseline, but they are excluded from the LOCAL training set: every stalled
# decision in the harvest so far is a doom-loop draw, and training on those
# teaches the model to loop. See DEAD_DEAL_ANALYSIS_2026-05-20.md.
STALL_TURNS = 25


# --------------------------------------------------------------------------- io
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    """Row count for a JSONL file, or 0 if it doesn't exist yet."""
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _delta(new: int, old: int) -> str:
    """Format a row-count delta as `(+N)`, `(-N)`, or `` when unchanged."""
    diff = new - old
    if diff == 0:
        return ""
    return f" ({diff:+d})"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalise_schema(rows: list[dict]) -> list[dict]:
    """Return rows widened to the union of keys, type-stably.

    HuggingFace's Arrow loader infers a schema per shard. Two failure
    modes occur with mixed-schema rows:
      * Some shards have a key, others don't -> 'column names don't match'.
      * A key is None in the early rows and list/dict in later rows ->
        'Couldn't cast array of type list<item: string> to null'.

    We fix both by widening every row to the union of keys AND substituting
    a typed empty value (``[]`` for lists, ``{}`` for dicts) where the row
    lacks the field. Scalars and strings stay as None.
    """
    # Type sniff: for each key, find the first non-None value's container shape.
    list_keys: set[str] = set()
    dict_keys: set[str] = set()
    all_keys: set[str] = set()
    for r in rows:
        for k, v in r.items():
            all_keys.add(k)
            if k not in list_keys and k not in dict_keys and v is not None:
                if isinstance(v, list):
                    list_keys.add(k)
                elif isinstance(v, dict):
                    dict_keys.add(k)

    def fill(r: dict) -> dict:
        out = {}
        for k in all_keys:
            if k in r and r[k] is not None:
                out[k] = r[k]
            elif k in list_keys:
                out[k] = []
            elif k in dict_keys:
                out[k] = {}
            else:
                out[k] = None
        return out

    return [fill(r) for r in rows]


def _front_load_rich(rows: list[dict]) -> list[dict]:
    """Sort rows so the schema-richest appear first.

    Richness == count of non-None / non-empty fields. Stable within ties on
    ``timestamp`` so equally-rich rows keep their natural ordering.

    Reason: ``datasets`` infers Arrow types from the first read batch (~600
    rows). If the first batch only has empty lists/dicts for a field, the
    inferred type is ``list<null>`` or ``struct<>``, and later rows with
    populated values fail to cast. Putting rich rows first sidesteps that.
    """
    def richness(r: dict) -> int:
        return sum(1 for v in r.values() if v not in (None, [], {}))

    return sorted(rows, key=lambda r: (-richness(r), r.get("timestamp") or 0))


# -------------------------------------------------------------- schema + parse
def classify_file(doc: dict) -> str:
    if isinstance(doc, dict) and "interactions" in doc:
        return "ai_log"
    if isinstance(doc, dict) and "moveHistory" in doc:
        return "win_record"
    return "unknown"


def schema_tier(it: dict) -> str:
    """'current' if the row satisfies the latest-schema field contract."""
    return ("current"
            if all(it.get(k) not in (None, "") for k in SCHEMA_CONTRACT_FIELDS)
            else "legacy")


def parse_game(prompt: str) -> dict | None:
    """Pull the board state out of a prompt.

    Two prompt layouts are supported: the original schema embedded a
    ``CURRENT GAME (JSON):`` blob; the ``de7dc06+`` 'hybrid-v1' layout
    inlines plain-text ``FOUNDATIONS / TABLEAU / LEGAL MOVES / PROGRESS``
    blocks instead. Return shape is normalised across both -- a dict with
    ``foundations``, ``tableau``, ``legalMoves``, and ``metrics`` keys --
    so downstream callers don't need to branch on schema.
    """
    marker = "CURRENT GAME (JSON):"
    i = prompt.find(marker)
    if i >= 0:
        rest = prompt[i + len(marker):]
        j = rest.find("Now choose")
        blob = (rest[:j] if j >= 0 else rest).strip()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    return _parse_game_from_text(prompt)


# Plain-text board parsing for the hybrid-v1 prompt layout (appCommit de7dc06+).
# We anchor on the second occurrence of each section header (the first
# occurrence in each prompt is inside the rules/format preamble that lists
# possible sections).
_FND_SUIT_PAIR_RE = re.compile(r'\b([HDCS]):\s*(--|[AKQJT2-9][HDCS])')
_TABLEAU_BLOCK_RE = re.compile(
    r'TABLEAU:\s*\n(.+?)(?=\n[A-Z][A-Z ]+(?::|\b)|\Z)',
    re.DOTALL,
)
_TABLEAU_COL_RE = re.compile(r'^\s*col\d+:\s*(.*)$', re.MULTILINE)
_LEGAL_BLOCK_RE = re.compile(
    r'LEGAL MOVES[^\n]*:\s*\n(.+?)(?=\n[A-Z][A-Z ]+(?::|\b)|\Z)',
    re.DOTALL,
)
_LEGAL_LINE_RE = re.compile(r'\s*\[(\d+)\]\s+(\S+)\s+(.+)')
_PROGRESS_RE = re.compile(
    r'PROGRESS:\s*foundation=(\d+)/52,\s*face-down remaining=(\d+),\s*completion=(\d+)%'
)


def _parse_game_from_text(prompt: str) -> dict | None:
    """Fallback parser for the hybrid-v1 plain-text prompt layout."""
    # Skip the rules preamble: find FOUNDATIONS as a section header. The first
    # match in the prompt is usually the rules list ("FOUNDATIONS, STOCK, ..."),
    # the second is the actual data row ("FOUNDATIONS:   H: AH ...").
    body_start = 0
    fnd_idx = prompt.find("FOUNDATIONS:")
    if fnd_idx < 0:
        return None
    body_start = fnd_idx

    # Foundations line (single line starting at fnd_idx)
    line_end = prompt.find("\n", fnd_idx)
    if line_end < 0:
        line_end = len(prompt)
    fnd_line = prompt[fnd_idx:line_end]
    foundations: dict[str, str] = {}
    for suit, val in _FND_SUIT_PAIR_RE.findall(fnd_line):
        if not val.startswith("-"):
            foundations[suit] = val

    # Tableau: count "??" per column line
    tableau: list[dict] = []
    tab_m = _TABLEAU_BLOCK_RE.search(prompt, body_start)
    if tab_m:
        for line in _TABLEAU_COL_RE.findall(tab_m.group(1)):
            tableau.append({"faceDownCount": line.count("??")})

    # Legal moves: parse "[idx] type   describe"
    legal: list[dict] = []
    lm_m = _LEGAL_BLOCK_RE.search(prompt, body_start)
    if lm_m:
        for line in lm_m.group(1).split("\n"):
            mm = _LEGAL_LINE_RE.match(line)
            if mm:
                legal.append({"type": mm.group(2), "describe": mm.group(3).strip()})

    # Progress line gives a precomputed completion percentage
    metrics: dict = {}
    pm = _PROGRESS_RE.search(prompt, body_start)
    if pm:
        metrics["completionProgress"] = int(pm.group(3))

    if not foundations and not tableau and not legal and not metrics:
        return None
    return {
        "foundations": foundations,
        "tableau": tableau,
        "legalMoves": legal,
        "metrics": metrics,
    }


_RANK_FROM_CARD = {"A": 1, "T": 10, "J": 11, "Q": 12, "K": 13,
                   **{str(n): n for n in range(2, 10)}}


def progress_components(game: dict | None) -> tuple[int | None, int | None]:
    """Return (foundationCards, faceDownTotal) for a parsed board JSON.

    foundationCards is the sum of foundation ranks (0..52). faceDownTotal is
    the sum of ``faceDownCount`` across the 7 tableau columns (21 down to 0).
    Either component returns ``None`` when its source field is missing or
    malformed.
    """
    if not isinstance(game, dict):
        return None, None
    f = game.get("foundations") or {}
    fc: int | None
    try:
        fc = sum(_RANK_FROM_CARD[v[0]] for v in f.values() if v) if isinstance(f, dict) else None
    except (KeyError, TypeError, IndexError):
        fc = None
    tab = game.get("tableau") or []
    fd: int | None
    try:
        fd = sum(int(c.get("faceDownCount", 0)) for c in tab) if isinstance(tab, list) else None
    except (TypeError, ValueError):
        fd = None
    return fc, fd


def progress_score(fc: int | None, fd: int | None) -> float | None:
    """Blended 0..100 progress (see GAME_PROGRESS_METRIC_2026-05-19.md):
    ``100 * (0.65 * foundationCards / 52 + 0.35 * (21 - faceDown) / 21)``."""
    if fc is None or fd is None:
        return None
    return round(100 * (0.65 * fc / 52 + 0.35 * (21 - fd) / 21), 2)


def compute_stall_info(interactions: list[dict]) -> dict[str, dict]:
    """Per-interaction progress and stall annotations, keyed by interaction id.

    Within each ``sessionId``, interactions are ordered by ``turnIndex`` and
    each is annotated with ``turnsSinceProgress`` -- the number of consecutive
    prior interactions for which foundationCards AND faceDownTotal were both
    unchanged. A row is ``stalled`` once ``turnsSinceProgress >= STALL_TURNS``.

    Returns ``{id: {foundationCards, faceDownTotal, progressScore,
                    turnsSinceProgress, stalled}}``.
    """
    out: dict[str, dict] = {}
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for it in interactions:
        if it.get("id"):
            by_sess[it.get("sessionId") or ""].append(it)

    def _sort_key(r: dict) -> tuple[int, int]:
        ti = r.get("turnIndex")
        ts = r.get("timestamp") or 0
        return (ti if isinstance(ti, int) else 0, ts if isinstance(ts, int) else 0)

    for rows in by_sess.values():
        rows = sorted(rows, key=_sort_key)
        prev_fc: int | None = None
        prev_fd: int | None = None
        flat = 0
        for r in rows:
            game = parse_game(r.get("prompt", "") or "")
            fc, fd = progress_components(game)
            if fc is None or fd is None:
                out[r["id"]] = {"foundationCards": fc, "faceDownTotal": fd,
                                "progressScore": None,
                                "turnsSinceProgress": None, "stalled": False}
                continue
            if prev_fc is None or fc != prev_fc or fd != prev_fd:
                flat = 0
            else:
                flat += 1
            prev_fc, prev_fd = fc, fd
            out[r["id"]] = {
                "foundationCards": fc,
                "faceDownTotal": fd,
                "progressScore": progress_score(fc, fd),
                "turnsSinceProgress": flat,
                "stalled": flat >= STALL_TURNS,
            }
    return out


def exclude_reasons(it: dict, stall: dict | None = None) -> list[str]:
    """Why a success decision is NOT training-eligible (empty list == eligible)."""
    reasons = []
    if schema_tier(it) != "current":
        reasons.append("legacy-schema")
    if it.get("model") != TEACHER_MODEL:
        reasons.append(f"non-teacher-model:{it.get('model')}")
    if stall and stall.get("stalled"):
        reasons.append("stalled-game")
    return reasons


def derive_decision(it: dict, stall_info: dict[str, dict] | None = None) -> dict:
    """Lean, analysis-friendly row for a successful interaction, tagged in/out."""
    dec = it.get("decision") or {}
    game = parse_game(it.get("prompt", "")) or {}
    legal = game.get("legalMoves", []) if isinstance(game, dict) else []
    metrics = game.get("metrics", {}) if isinstance(game, dict) else {}
    mi = dec.get("moveIndex")
    chosen = legal[mi] if isinstance(mi, int) and 0 <= mi < len(legal) else None
    stall = (stall_info or {}).get(it.get("id")) or {}
    reasons = exclude_reasons(it, stall)
    return {
        "id": it.get("id"),
        "sessionId": it.get("sessionId"),
        "timestamp": it.get("timestamp"),
        "turnIndex": it.get("turnIndex"),
        "model": it.get("model"),
        "provider": it.get("provider"),
        "appCommit": it.get("appCommit"),
        "schemaTier": schema_tier(it),
        "trainingEligible": not reasons,
        "excludeReasons": reasons,
        "moveIndex": mi,
        "chosenMoveType": chosen.get("type") if chosen else None,
        "chosenMoveDescribe": chosen.get("describe") if chosen else None,
        "nLegalMoves": len(legal),
        "confidence": dec.get("confidence"),
        "alternativeMoveIndex": dec.get("alternativeMoveIndex"),
        "completionProgress": metrics.get("completionProgress"),
        "moveCount": metrics.get("moveCount"),
        "perceivedDifficulty": metrics.get("perceivedDifficulty"),
        "foundationCards": stall.get("foundationCards"),
        "faceDownTotal": stall.get("faceDownTotal"),
        "progressScore": stall.get("progressScore"),
        "turnsSinceProgress": stall.get("turnsSinceProgress"),
        "thinkingText": it.get("thinkingText"),
        "boardAnalysis": dec.get("boardAnalysis"),
        "reasoning": dec.get("reasoning"),
    }


# ------------------------------------------------------------------ summary md
def render_summary(store: dict[str, dict], manifest: list[dict],
                    decisions: list[dict]) -> str:
    interactions = list(store.values())
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    A = lines.append

    A("# Dataset Summary")
    A("")
    A(f"_Auto-generated by `scripts/ingest_exports.py` on {now}. Do not hand-edit._")
    A("")
    ai = [m for m in manifest if m["type"] == "ai_log"]
    win = [m for m in manifest if m["type"] == "win_record"]
    A(f"- Raw files ingested: **{len(manifest)}** ({len(ai)} ai_log, {len(win)} win_record)")
    A(f"- Unique interactions in store: **{len(interactions)}** (nothing discarded)")
    success = [it for it in interactions if it.get("outcome") == "success"]
    A(f"- Success interactions: **{len(success)}**")
    elig = [d for d in decisions if d["trainingEligible"]]
    A(f"- **Training-eligible decisions: {len(elig)}**")
    A("")

    A("## Training-set selection funnel")
    A("")
    cur = [d for d in decisions if d["schemaTier"] == "current"]
    A(f"All success decisions: {len(decisions)}")
    A(f"  -> current schema (v3+): {len(cur)}  "
      f"(legacy excluded: {len(decisions) - len(cur)})")
    teacher = [d for d in cur if d["model"] == TEACHER_MODEL]
    A(f"  -> teacher model `{TEACHER_MODEL}`: {len(teacher)}  "
      f"(other models excluded: {len(cur) - len(teacher)})")
    not_stalled = [d for d in teacher if "stalled-game" not in d["excludeReasons"]]
    stalled_dropped = len(teacher) - len(not_stalled)
    A(f"  -> not stalled (foundations and faceDown unchanged for <{STALL_TURNS} "
      f"turns): {len(not_stalled)}  "
      f"(stalled-game excluded: {stalled_dropped})")
    A(f"  => **local set (training.jsonl): {len(not_stalled)}**")
    A("")
    A(f"**Publishing set** (all success decisions, every model and schema): "
      f"**{len(decisions)}** rows -> `data/publish/` (Hugging Face, CC-BY-4.0).")
    A("")

    A("## By schema tier")
    A("")
    A("| Tier | Interactions | Success |")
    A("|---|---|---|")
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for it in interactions:
        by_tier[schema_tier(it)].append(it)
    for tier, rows in sorted(by_tier.items()):
        ok = sum(1 for r in rows if r.get("outcome") == "success")
        A(f"| {tier} | {len(rows)} | {ok} |")
    A("")

    A("## By model")
    A("")
    A("| Model | Interactions | Success | Success rate |")
    A("|---|---|---|---|")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for it in interactions:
        by_model[it.get("model") or "(unknown)"].append(it)
    for model, rows in sorted(by_model.items()):
        ok = sum(1 for r in rows if r.get("outcome") == "success")
        A(f"| `{model}` | {len(rows)} | {ok} | {100 * ok / len(rows):.0f}% |")
    A("")

    A("## By session")
    A("")
    A("`Coverage` is the fraction of turnIndex values within `[min, max]` that "
      "we have a logged interaction for. Gaps come from harvester filtering "
      "(auto-played moves, or possible silent drops -- open question with the "
      "harvest team) and from export-window boundaries; cap-related boundary "
      "loss will diminish in future exports as the harvester's log-size cap is "
      "raised, but legacy exports keep their gap profile.")
    A("")
    A("| Session | Interactions | Success | Max progress | Turn range | Coverage |")
    A("|---|---|---|---|---|---|")
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for it in interactions:
        by_sess[it.get("sessionId") or "(legacy/no-session)"].append(it)
    for sess, rows in sorted(by_sess.items(),
                             key=lambda kv: min(r.get("timestamp", 0) for r in kv[1])):
        ok = sum(1 for r in rows if r.get("outcome") == "success")
        progs = []
        for r in rows:
            game = parse_game(r.get("prompt", "")) or {}
            p = game.get("metrics", {}).get("completionProgress") if isinstance(game, dict) else None
            if isinstance(p, (int, float)):
                progs.append(p)
        turns = {r.get("turnIndex") for r in rows if isinstance(r.get("turnIndex"), int)}
        if turns:
            mn, mx = min(turns), max(turns)
            span = mx - mn + 1
            cov = f"{len(turns)}/{span} ({100 * len(turns) / span:.0f}%)"
            turn_range = f"{mn}..{mx}"
        else:
            cov = "(legacy)"
            turn_range = "—"
        label = sess if len(sess) <= 14 else "..." + sess[-12:]
        A(f"| `{label}` | {len(rows)} | {ok} | {max(progs) if progs else '?'}% | "
          f"{turn_range} | {cov} |")
    A("")

    A("## Error breakdown")
    A("")
    errs = Counter(
        (it.get("errorMessage") or it.get("errorKind") or f"httpStatus={it.get('httpStatus')}")
        for it in interactions if it.get("outcome") != "success"
    )
    if errs:
        A("| Error | Count |")
        A("|---|---|")
        for msg, c in errs.most_common():
            A(f"| {msg} | {c} |")
    else:
        A("_No error rows._")
    A("")

    A("## Chosen move types (training set)")
    A("")
    moves = Counter(d["chosenMoveType"] for d in elig if d["chosenMoveType"])
    if moves:
        total = sum(moves.values())
        A("| Move type | Count | Share |")
        A("|---|---|---|")
        for mt, c in moves.most_common():
            A(f"| `{mt}` | {c} | {100 * c / total:.0f}% |")
    else:
        A("_Training set is empty -- no rows match the selection criteria yet._")
    A("")
    confs = sorted(d["confidence"] for d in elig
                   if isinstance(d["confidence"], (int, float)))
    if confs:
        mean = sum(confs) / len(confs)
        A(f"**Confidence (training set):** n={len(confs)}, min={confs[0]:.2f}, "
          f"median={confs[len(confs) // 2]:.2f}, mean={mean:.2f}, max={confs[-1]:.2f}."
          + (" Saturated/miscalibrated -- treat as suspect."
             if confs and confs[0] >= 0.6 else ""))
    A("")
    return "\n".join(lines)


# ----------------------------------------------------------- hf dataset card
def _size_category(n: int) -> str:
    for hi, label in [(1_000, "n<1K"), (10_000, "1K<n<10K"),
                       (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")]:
        if n < hi:
            return label
    return "n>1M"


def render_dataset_card(
    full_raw: list[dict],
    clean_raw: list[dict],
    clean_lean: list[dict],
) -> str:
    """Hugging Face dataset card -- three configs under one dataset path.

    Config names are provenance-prefixed (``client_v1_*``) so server-collected
    data can slot in later as ``server_v1_*`` without renaming what already
    shipped. The default config is ``client_v1_full_corpus_raw`` -- back-compat
    with the prior single-file pub.
    """
    n_full = len(full_raw)
    n_raw = len(clean_raw)
    n_lean = len(clean_lean)
    models = Counter(r.get("model") or "(unknown)" for r in full_raw)
    tiers = Counter(schema_tier(r) for r in full_raw)
    sessions = sorted({(r.get("sessionId") or "(none)") for r in full_raw})
    moves: Counter = Counter()
    confs: list[float] = []
    ts = [r["timestamp"] for r in full_raw if isinstance(r.get("timestamp"), int)]
    for r in full_raw:
        d = derive_decision(r)
        if d["chosenMoveType"]:
            moves[d["chosenMoveType"]] += 1
        if isinstance(d["confidence"], (int, float)):
            confs.append(d["confidence"])
    span = ""
    if ts:
        lo = dt.datetime.fromtimestamp(min(ts) / 1000, dt.timezone.utc).date()
        hi = dt.datetime.fromtimestamp(max(ts) / 1000, dt.timezone.utc).date()
        span = f"{lo} to {hi}"

    fm = [
        "---",
        f"license: {HF_LICENSE}",
        f"pretty_name: {HF_PRETTY_NAME}",
        "language:",
        "- en",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- solitaire",
        "- klondike",
        "- game-playing",
        "- llm-decisions",
        "- reasoning",
        "- distillation",
        "- failure-modes",
        "size_categories:",
        f"- {_size_category(n_full)}",
        "configs:",
        "- config_name: client_v1_full_corpus_raw",
        "  data_files: client_v1_full_corpus_raw.jsonl",
        "  default: true",
        "- config_name: client_v1_teacher_clean_raw",
        "  data_files: client_v1_teacher_clean_raw.jsonl",
        "- config_name: client_v1_teacher_clean_lean",
        "  data_files: client_v1_teacher_clean_lean.jsonl",
        "---",
    ]

    body: list[str] = []
    B = body.append
    B(f"# {HF_PRETTY_NAME}")
    B("")
    B("Per-decision traces from large language models acting as advisors in "
      "Klondike Solitaire, collected to support **distillation research** and "
      "the study of **LLM failure modes in sequential decision tasks**. Every "
      "row records one advisor call against a reproducible game state.")
    B("")
    B("## Configs at a glance")
    B("")
    B("Three subsets under one dataset path. Pick the one that fits your "
      "use-case; researchers who want everything should use the default.")
    B("")
    B("| Config | Rows | Schema | Best for |")
    B("|---|---:|---|---|")
    B(f"| `client_v1_full_corpus_raw` (default) | **{n_full}** | full interaction "
      "(prompt + rawResponse + decision blob + call metadata) | failure-mode "
      "research, replay, end-to-end audit |")
    B(f"| `client_v1_teacher_clean_raw` | **{n_raw}** | full interaction | fine-tuning, "
      "honest training-quality subset (single teacher model, current schema, "
      "non-stalled) |")
    B(f"| `client_v1_teacher_clean_lean` | **{n_lean}** | derived per-decision "
      "(flat schema; see *Fields*) | quick analytics, lightweight loading, "
      "headline-statistics work |")
    B("")
    B("```python")
    B("from datasets import load_dataset")
    B("")
    B("# Default -- the full corpus, including failure modes")
    B(f'full = load_dataset("chayuto/klondike-llm-decisions")  # {n_full} rows')
    B("")
    B("# The training-friendly subset (filtered, single teacher)")
    B(f'clean_raw  = load_dataset("chayuto/klondike-llm-decisions", '
      f'"client_v1_teacher_clean_raw")   # {n_raw} rows')
    B(f'clean_lean = load_dataset("chayuto/klondike-llm-decisions", '
      f'"client_v1_teacher_clean_lean")  # {n_lean} rows, flat schema')
    B("```")
    B("")
    B("### Filtering by model")
    B("")
    B("Every row in `*_raw` configs carries a `model` field "
      "(e.g. `\"gemma-4-31b-it\"`, `\"gemini-3.1-flash-lite\"`). Use the "
      "standard HF `.filter()` to subset:")
    B("")
    B("```python")
    B('ds = load_dataset("chayuto/klondike-llm-decisions")  # full corpus')
    B('teacher_only = ds["train"].filter(lambda r: r["model"] == "gemma-4-31b-it")')
    B('other_only   = ds["train"].filter(lambda r: r["model"] != "gemma-4-31b-it")')
    B("```")
    B("")
    B("The `client_v1_teacher_clean_*` configs are already filtered to a "
      "single teacher model (currently `gemma-4-31b-it`); use them if you "
      "want a homogeneous training subset without writing a filter.")
    B("")
    B("## Collection method (`client_v1_*`)")
    B("")
    B(f"Collected via an external client-side harness (closed-source) "
      "running the Klondike app and capturing every teacher-advisor call. "
      "Each game seeds a reproducible deal. Rows are deduplicated by their "
      "UUIDv7 `id` across re-exports; nothing is discarded.")
    B("")
    if span:
        B(f"- **Collection window**: {span}")
    B(f"- **Sessions**: {len(sessions)} distinct game sessions")
    B(f"- **Models**: " + ", ".join(f"`{m}` ({c})" for m, c in models.most_common()))
    B(f"- **Schema tiers**: " + ", ".join(f"{t} ({c})" for t, c in tiers.most_common()))
    B("")
    B("### Planned: `server_v1_*` configs")
    B("")
    B("A second collection method is being prepared using the open-source MCP "
      "server in this project's parent codebase "
      "(`solitaire_analytics.mcp_server`). Server-collected rows will "
      "ship as `server_v1_*` configs under this same dataset path. They will "
      "carry agent identity (`agent_id`, `model`, `provider`, `app_commit`) "
      "stamped per decision, plus an `infoLevel` block per session so "
      "perfect-vs-imperfect-information runs are unambiguous. Not yet "
      "published.")
    B("")
    B("## Fields")
    B("")
    B("### `*_raw` configs")
    B("")
    B("Verbatim interaction records as captured by the harness.")
    B("")
    B("- `id`: globally unique UUIDv7 for the interaction")
    B("- `sessionId`, `turnIndex`: game session and move number "
      "(current-schema rows)")
    B("- `model`, `provider`: the advisor model")
    B("- `prompt`: full prompt: Klondike rules + board state JSON + "
      "legal-move list")
    B("- `rawResponse`: the model's raw text reply")
    B("- `decision`: parsed `moveIndex`, `confidence`, "
      "`alternativeMoveIndex`, `boardAnalysis`, `reasoning`")
    B("- `outcome`, token counts, timing: call metadata")
    B("")
    B("### `*_lean` config")
    B("")
    B("Derived per-decision rows, flattened. Built by joining each successful "
      "interaction against its parsed prompt + decision.")
    B("")
    B("- `id`, `sessionId`, `turnIndex`, `timestamp`, `model`, `provider`, "
      "`appCommit`")
    B("- `chosenMoveType`, `chosenMoveDescribe`, `moveIndex`, `nLegalMoves`")
    B("- `confidence`, `alternativeMoveIndex`")
    B("- `completionProgress`, `moveCount`, `perceivedDifficulty`: "
      "from the prompt metrics block")
    B("- `foundationCards`, `faceDownTotal`, `progressScore`, "
      "`turnsSinceProgress`: computed by the ingest from board state")
    B("- `boardAnalysis`, `reasoning`, `thinkingText`: agent's natural-"
      "language fields")
    B("")
    if moves:
        total = sum(moves.values())
        B("## Chosen-move distribution (full corpus)")
        B("")
        B("| Move type | Count | Share |")
        B("|---|---:|---:|")
        for mt, c in moves.most_common():
            B(f"| `{mt}` | {c} | {100 * c / total:.0f}% |")
        B("")
    B("## Failure modes are a feature of `*_full_corpus_raw`, not a bug")
    B("")
    B("The full corpus deliberately includes sessions where the teacher fails "
      "to make progress. These are research signal, not noise. The cleaned "
      "configs (`*_teacher_clean_*`) filter them out per a stall heuristic; "
      "the full corpus keeps them so you can study the failure modes directly.")
    B("")
    B("Two documented pathologies recur in the corpus:")
    B("")
    B("1. **Doom-loop / oscillation.** The teacher rationalises a two-card "
      "shuffle (e.g. moving `5C`/`4D` back and forth between two columns) as "
      "a 'setup move' even when `recentMoves` clearly shows the exact "
      "reversal was just played. Confidence stays saturated at 0.9+ "
      "throughout. Example: session with the longest plateau in the corpus "
      "carries 75 consecutive turns of foundation/face-down unchanged.")
    B("")
    B("2. **Honest hunt that degrades.** Some sessions begin with "
      "draw-dominated card hunting (correct behaviour when needed cards are "
      "still hidden) and only descend into oscillation after extended "
      "no-progress windows. A plateau-only stall detector would over-fire on "
      "these; a *shuffle-fraction* gate is needed alongside the plateau gate "
      "to discriminate honest hunt from doom-loop.")
    B("")
    B("Use the `progressScore` / `turnsSinceProgress` columns in the `*_lean` "
      "config to locate stalled stretches; use the `chosenMoveType` "
      "distribution within those stretches to classify them.")
    B("")
    B("## Known limitations")
    B("")
    if confs:
        confs.sort()
        mean = sum(confs) / len(confs)
        B(f"- **Confidence is miscalibrated.** Reported `confidence` spans "
          f"{confs[0]:.2f} to {confs[-1]:.2f} (mean {mean:.2f}); the teacher signals "
          f"near-certainty regardless of board state. Do not treat it as a "
          f"calibrated probability; in our experience using it as a "
          f"training-time signal teaches student models to be overconfident.")
    B("- **Mixed schema versions** in `client_v1_full_corpus_raw`. Older rows "
      "lack `sessionId` / `turnIndex` / `appCommit`. Filter on field presence "
      "if you need a homogeneous subset, or use the `client_v1_teacher_clean_*` "
      "configs which exclude legacy schema rows.")
    B("- **Outcome skew.** Most logged games were lost or stalled; winning "
      "play is under-represented. End-game (foundation_cards > ~10) is "
      "particularly sparse. Student models trained on this corpus will lack "
      "guidance for late-game transitions.")
    B("- **Mixed information modes.** A few early sessions had "
      "perfect-information game state exposed to the advisor; most run under "
      "imperfect information. The `client_v1_teacher_clean_*` configs select "
      "a single information mode.")
    B("- **Move-type skew toward `draw_card`.** Draws are ~50 to 66% of "
      "eligible rows in the cleaned configs, reflecting the teacher's "
      "tendency to keep drawing when no productive tableau move is "
      "obvious. Apply your own re-weighting if this matters for your task.")
    B("")
    B("## Build on top of this")
    B("")
    B("This corpus is intentionally public so others can study or build on "
      "Gemma 4 31B's Klondike behaviour without reproducing the harvest "
      "infrastructure from scratch. The license is permissive (CC-BY-4.0); "
      "attribution is the only ask. Some specific ways the data is set up "
      "to be useful:")
    B("")
    B("### Replay any seed in your browser")
    B("")
    B("Every row carries a `sessionId` (and most carry a `seed` derivable "
      "from the source repo's `data/index/manifest.jsonl`). The harvester "
      "web UI at `https://solitaire.chayuto.com/?seed=<seed>` deals the "
      "same board deterministically: you can load any seed from this "
      "corpus and play or feed it to your own model, then compare your "
      "model's decisions against the rows here turn for turn.")
    B("")
    B("### Run your own kill-or-continue analysis")
    B("")
    B("The source repo at "
      "[`chayuto/solitaire-analytics`](https://github.com/chayuto/solitaire-analytics) "
      "publishes the tooling used to produce this corpus, including:")
    B("")
    B("- `scripts/ingest_exports.py`: the dedup + stall-filter pipeline that "
      "produced these configs from raw exports.")
    B("- `.claude/skills/solitaire-analyst/`: a Claude Code skill that reads "
      "any raw export and produces a kill-or-continue verdict with failure-"
      "mode classification. Includes a Monte Carlo solvability check via "
      "`pyksolve` (DFS with dominance pruning, ~10 ms per sample) at "
      "`.claude/skills/solitaire-analyst/scripts/check_winnability.py`.")
    B("- `data/DATASET_NOTES.md`: the long-form taxonomy of every documented "
      "session in the corpus. Each entry calls out the failure class "
      "(behavioural-doom-loop, dead-deal-flailing, honest-hunt, "
      "self-rescue-fails) with the specific evidence that drove the call. "
      "Useful if you want to know which sessions are which kind of failure "
      "before pulling them.")
    B("")
    B("### Compare a model on the same boards")
    B("")
    B("A 20-state Klondike-state benchmark used by this project's "
      "distillation evaluations lives in the source repo under "
      "`experiments/a4_phase1.5_2026_05_24/prompts/C0/`. Five early-game, "
      "eight midgame, seven oscillation-prone states; each state's reference "
      "answer is the teacher model's pick scored on a six-level tier (`foundation` > "
      "`reveal` > `waste_play` > `shuffle` > `draw` > `illegal`). If you want "
      "to bench your own Klondike-playing model on the same positions and "
      "compare apples-to-apples against `gemma-4-31b-it`, this is the "
      "fastest way.")
    B("")
    B("### Cite if you publish")
    B("")
    B("If this corpus shows up in a paper, blog post, or model card, please "
      "cite the dataset URL (`https://huggingface.co/datasets/chayuto/klondike-llm-decisions`) "
      "and link to the source repo (`https://github.com/chayuto/solitaire-analytics`) "
      "so readers can find the analysis context. The corpus continues to "
      "grow; pin a specific revision (`load_dataset(..., revision=...)`) "
      "if your work depends on a fixed snapshot.")
    B("")
    B("### Talk to us")
    B("")
    B("Issues, comparisons, replay videos, alternative analyses are all "
      "welcome. Open an issue on the source repo or comment on the dataset "
      "discussion tab.")
    B("")
    B("## License")
    B("")
    B(f"Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). "
      f"Free to use, share, and adapt with attribution.")
    B("")
    B("_Card and data generated by `scripts/ingest_exports.py`._")
    return "\n".join(fm) + "\n\n" + "\n".join(body) + "\n"


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the manifest and reprocess every file in data/raw/")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        store: dict[str, dict] = {}
        manifest: list[dict] = []
        seen_sha: set[str] = set()
    else:
        store = {it["id"]: it for it in read_jsonl(STORE)}
        manifest = read_jsonl(MANIFEST)
        seen_sha = {m["sha256"] for m in manifest}

    raw_files = sorted(RAW.glob("*.json"))
    if not raw_files:
        print(f"No exports found in {RAW}. Drop raw JSON exports there and re-run.")
        return 1

    new_files = 0
    conflicts = 0
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    for path in raw_files:
        digest = sha256_file(path)
        if digest in seen_sha:
            continue
        seen_sha.add(digest)
        new_files += 1

        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  SKIP {path.name}: invalid JSON ({exc})")
            manifest.append({"file": path.name, "sha256": digest, "type": "unknown",
                             "ingestedAt": now_iso})
            continue

        kind = classify_file(doc)
        row = {
            "file": path.name,
            "sha256": digest,
            "type": kind,
            "exportedAt": doc.get("exportedAt"),
            "appCommit": doc.get("appCommit"),
            "appBuildTime": doc.get("appBuildTime"),
            "ingestedAt": now_iso,
        }

        if kind == "ai_log":
            interactions = doc.get("interactions", [])
            sessions, added, dup, usable = set(), 0, 0, 0
            tiers: Counter = Counter()
            for it in interactions:
                iid = it.get("id")
                if not iid:
                    continue
                sessions.add(it.get("sessionId"))
                tiers[schema_tier(it)] += 1
                if it.get("outcome") == "success":
                    usable += 1
                if iid in store:
                    if store[iid] != it:
                        conflicts += 1
                        print(f"  CONFLICT {path.name}: id {iid} differs from a "
                              f"prior copy -- keeping the earlier one")
                    dup += 1
                else:
                    store[iid] = it
                    added += 1
            row.update(rows=len(interactions), usable=usable, new=added, duplicate=dup,
                       schemaTiers=dict(tiers),
                       sessions=sorted(s for s in sessions if s))
            print(f"  + {path.name}: {len(interactions)} rows, {added} new, "
                  f"{dup} known, schema {dict(tiers)}")
        elif kind == "win_record":
            row.update(moves=len(doc.get("moveHistory", [])),
                       gameWon=doc.get("gameWon"),
                       sessionId=doc.get("gameSessionId"))
            print(f"  + {path.name}: win_record, {row['moves']} moves "
                  f"(won={row['gameWon']})")
        else:
            print(f"  + {path.name}: unrecognised schema, catalogued only")

        manifest.append(row)

    # Snapshot prior-run row counts BEFORE writing, so the summary can show
    # what changed. Files that don't exist yet count as 0 (first-run diffs
    # against an empty baseline are fine).
    prior = {
        "store":      count_lines(STORE),
        "decisions":  count_lines(DECISIONS),
        "training":   count_lines(TRAINING),
        "full_raw":   count_lines(PUBLISH_FULL_RAW),
        "clean_raw":  count_lines(PUBLISH_CLEAN_RAW),
        "clean_lean": count_lines(PUBLISH_CLEAN_LEAN),
    }

    # canonical store -- ALL interactions, deterministic order for stable diffs
    ordered = sorted(store.values(),
                     key=lambda it: (it.get("timestamp", 0), it.get("id", "")))
    write_jsonl(STORE, ordered)

    # per-interaction stall annotations, computed once and shared across the
    # decision rows and the local-set filter
    stall_info = compute_stall_info(ordered)

    # every success decision, tagged with training eligibility
    decisions = [derive_decision(it, stall_info) for it in ordered
                 if it.get("outcome") == "success" and it.get("decision")]
    write_jsonl(DECISIONS, decisions)

    # LOCAL set -- full interaction records for the training-eligible subset,
    # so prepare_dataset.py can consume training.jsonl directly.
    eligible_ids = {d["id"] for d in decisions if d["trainingEligible"]}
    training = [it for it in ordered if it.get("id") in eligible_ids]
    write_jsonl(TRAINING, training)

    # PUBLISHING set -- three Hugging Face configs under one dataset path.
    # See render_dataset_card() for the naming + provenance rationale.
    publish_full_raw = [it for it in ordered
                        if it.get("outcome") == "success" and it.get("decision")]
    publish_clean_raw = [it for it in publish_full_raw if it.get("id") in eligible_ids]
    publish_clean_lean = [d for d in decisions if d["trainingEligible"]]

    # Normalise raw rows so every row carries the same key set (with None for
    # absent fields). Mixed-schema rows would otherwise fail Arrow's per-shard
    # schema inference inside `datasets.load_dataset`. Schema-rich rows are
    # sorted to the front: Arrow infers types from the first batch, so rich
    # rows there guarantee struct/list types resolve correctly before the
    # poor rows (with empty containers) slot in as nulls.
    publish_full_raw   = _front_load_rich(_normalise_schema(publish_full_raw))
    publish_clean_raw  = _front_load_rich(_normalise_schema(publish_clean_raw))

    write_jsonl(PUBLISH_FULL_RAW,   publish_full_raw)
    write_jsonl(PUBLISH_CLEAN_RAW,  publish_clean_raw)
    write_jsonl(PUBLISH_CLEAN_LEAN, publish_clean_lean)
    # Legacy alias for the old filename. Same content as the full-raw config.
    write_jsonl(PUBLISH_LEGACY_ALIAS, publish_full_raw)
    PUBLISH_CARD.write_text(render_dataset_card(
        publish_full_raw, publish_clean_raw, publish_clean_lean))

    write_jsonl(MANIFEST, manifest)
    SUMMARY.write_text(render_summary(store, manifest, decisions))

    print()
    print(f"Ingested {new_files} new file(s).")
    print(f"  store      {len(store):5d} interactions{_delta(len(store), prior['store'])} "
          f"(all kept) -> {STORE.relative_to(REPO_ROOT)}")
    print(f"  decisions  {len(decisions):5d} success decisions"
          f"{_delta(len(decisions), prior['decisions'])} (tagged) "
          f"-> {DECISIONS.relative_to(REPO_ROOT)}")
    print(f"  local set  {len(training):5d} selected rows"
          f"{_delta(len(training), prior['training'])} "
          f"-> {TRAINING.relative_to(REPO_ROOT)}")
    print(f"  publish    {len(publish_full_raw):5d}{_delta(len(publish_full_raw), prior['full_raw'])} "
          f"/ {len(publish_clean_raw):5d}{_delta(len(publish_clean_raw), prior['clean_raw'])} "
          f"/ {len(publish_clean_lean):5d}{_delta(len(publish_clean_lean), prior['clean_lean'])} "
          f"rows (full / clean-raw / clean-lean) "
          f"+ HF card ({HF_LICENSE}) -> {PUBLISH_DIR.relative_to(REPO_ROOT)}/")
    if conflicts:
        print(f"  WARNING: {conflicts} id conflict(s) -- see lines above.")
    print(f"  summary    -> {SUMMARY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
