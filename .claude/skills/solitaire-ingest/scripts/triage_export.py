#!/usr/bin/env python3
"""Triage harvester export files BEFORE any deep analysis or ingest.

The harvester drops up to three kinds of file per session, all sharing the same
6-char id token in the filename:

    solitaire-ai-log-<id>-<ts>.json   per-interaction decision log (analyst input)
    solitaire-win-<id>-<ts>.json      TERMINAL full-state record (gameWon true/false)
    solitaire-game-<id>-<ts>.json     MID-GAME SNAPSHOT (NOT a terminal record)

Why this script exists: the ingest pipeline (scripts/ingest_exports.py) cannot
tell a terminal record from a snapshot. `classify_file` labels anything carrying
a `moveHistory` as a "win_record" and copies `gameWon` verbatim. So a
solitaire-game snapshot captured at 77% (gameWon=false) of a game that LATER WON
is indistinguishable, to the pipeline, from a terminal loss. Session #a11e74 is
the standing proof: its 77% gameWon=false snapshot preceded the actual win by an
hour. Trusting the pipeline's label there would record a loss for a won game.

This script makes the distinction the pipeline can't, and reads
model/build/template PER FILE so a mixed-build batch is never collapsed to one
assumed attribution. It prints, per session: which artifacts are present, the
per-file attribution, a record class, and a one-line triage verdict telling you
whether to ingest-and-catalog, hand off for a kill/continue call, or wait for the
terminal export. Read this first; do the deep dive second.

Run:
    python scripts/triage_export.py <file> [<file> ...]
    python scripts/triage_export.py ~/Downloads/solitaire-*.json

It is read-only: it never moves, writes, or ingests anything.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_RANK = {"A": 1, "T": 10, "J": 11, "Q": 12, "K": 13, **{str(n): n for n in range(2, 10)}}
_FNAME_RE = re.compile(r"solitaire-(ai-log|win|game)-([0-9a-z]{6})-(\d+)\.json$")


# ----------------------------------------------------------------------
# Loading + light parsing (handles both the state-file and ai-log shapes)
# ----------------------------------------------------------------------

def _file_kind(path: Path, doc: dict) -> str:
    """ai-log | win | game, from the filename first, content as a fallback."""
    m = _FNAME_RE.search(path.name)
    if m:
        return m.group(1)
    if "interactions" in doc and "session" in doc:
        return "ai-log"
    if "moveHistory" in doc:
        # No filename hint: a state file. We cannot be sure win vs game, so lean
        # on content -- a clearly mid-game board (stock left / progress < 100)
        # is treated as a snapshot to stay on the safe side.
        prog = _progress(doc)
        if doc.get("gameWon") is not True and (prog is None or prog < 100):
            return "game"
        return "win"
    return "ai-log"


def _short_id(path: Path, doc: dict) -> str:
    m = _FNAME_RE.search(path.name)
    if m:
        return m.group(2)
    sid = doc.get("gameSessionId") or (doc.get("session") or {}).get("sessionId") or ""
    return sid[-12:] if sid else "??????"


def _full_id(doc: dict) -> str:
    return doc.get("gameSessionId") or (doc.get("session") or {}).get("sessionId") or "(unknown)"


def _progress(doc: dict) -> int | None:
    p = doc.get("completionProgress")
    if isinstance(p, (int, float)):
        return int(p)
    p = (doc.get("session") or {}).get("finalProgress")
    return int(p) if isinstance(p, (int, float)) else None


def _facedown(doc: dict) -> int | None:
    """Terminal face-down count from a state file's tableau (None for ai-logs)."""
    tab = doc.get("tableau")
    if not isinstance(tab, list) or not tab:
        return None
    n = 0
    for col in tab:
        if isinstance(col, list):
            n += sum(1 for c in col if isinstance(c, dict) and not c.get("faceUp"))
    return n


def _template(doc: dict) -> str | None:
    """First prompt-template hash found anywhere in the doc, abbreviated."""
    stack: list[Any] = [doc]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k in ("templateHash", "promptTemplateHash"):
                if isinstance(o.get(k), str):
                    return o[k][:8] + "…"
            for k in ("promptTemplateVersion", "promptLayoutVersion"):
                if isinstance(o.get(k), str) and o[k] not in ("hybrid-v1",):
                    # version string is a useful fallback when no hash is present
                    return o[k]
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return None


def _attribution(kind: str, doc: dict) -> dict:
    if kind == "ai-log":
        sess = doc.get("session") or {}
        model = sess.get("model") or ""
        seed = sess.get("seed")
    else:
        model = (doc.get("aiConfig") or {}).get("model") or ""
        seed = doc.get("seed")
    return {
        "model": model or "(unknown)",
        "seed": seed,
        "build": doc.get("appCommit") or "(unknown)",
        "buildTime": (doc.get("appBuildTime") or "")[:10] or "?",
        "template": _template(doc) or "(none)",
    }


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

def classify(kind: str, doc: dict) -> tuple[str, str]:
    """Return (CLASS, one-line verdict) for a state file or ai-log-only session.

    CLASS is one of TERMINAL-WIN, TERMINAL-LOSS, PENDING-SNAPSHOT, AI-LOG-ONLY.
    """
    gw = doc.get("gameWon")
    prog = _progress(doc)
    fd = _facedown(doc)

    if kind == "game":
        # Game files are snapshots by construction. Even gameWon=true here would
        # be a post-hoc snapshot; treat the outcome as not-yet-final regardless.
        return ("PENDING-SNAPSHOT",
                "NOT terminal. Do NOT record an outcome (no win, no loss) from this "
                "file. If the session is still live, this is a kill/continue call: "
                "hand the ai-log to the solitaire-analyst skill. Await the terminal "
                "solitaire-win-* export before cataloging. (See #a11e74: a 77% "
                "snapshot that later WON.)")

    if kind == "win":
        if gw is True:
            ok = (prog == 100) and (fd == 0)
            note = "" if ok else (f"  [VERIFY: gameWon=true but progress={prog}, "
                                  f"faceDown={fd}; expected 100/0]")
            return ("TERMINAL-WIN",
                    "Ingest. Game is over, so there is no kill decision. Catalog "
                    "under '## Won sessions' in DATASET_NOTES.md." + note)
        if gw is False:
            return ("TERMINAL-LOSS",
                    "Terminal loss (a win-file with gameWon=false). Ingest as a "
                    "loss and catalog under '## Known doom-loop sessions'. If a twin "
                    "of this session is still running, that is a KILL.")
        return ("TERMINAL-WIN?",
                f"win-file with gameWon={gw!r} -- unexpected; inspect before cataloging.")

    # ai-log only (no state file in this drop)
    outcome = (doc.get("session") or {}).get("outcome") or ""
    if outcome == "won":
        return ("AI-LOG-ONLY",
                "ai-log reports outcome=won but no solitaire-win-* file is in this "
                "drop. Ingest the ai-log; the terminal win record may arrive "
                "separately. Catalog under '## Won sessions'.")
    return ("AI-LOG-ONLY",
            f"ai-log only (outcome={outcome or 'unset'}). This is a kill/continue "
            "judgement, not a clean ingest: run the solitaire-analyst skill on it.")


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _rows(doc: dict) -> tuple[int, int]:
    inter = doc.get("interactions") or []
    succ = sum(1 for r in inter if r.get("outcome") == "success")
    return len(inter), succ


def _ailog_stats(doc: dict) -> dict:
    """Liveness + resign stats for an ai-log: what a kill record needs to quote."""
    inter = doc.get("interactions") or []
    sess = doc.get("session") or {}
    resigns = 0
    for r in inter:
        d = r.get("decision")
        if isinstance(d, dict) and d.get("move_index") == -1:
            resigns += 1
        elif isinstance(d, str) and '"move_index": -1' in d:
            resigns += 1
    ts = [r.get("timestamp") for r in inter if isinstance(r.get("timestamp"), (int, float))]
    last = None
    if ts:
        from datetime import datetime, timezone
        last = datetime.fromtimestamp(max(ts) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "outcome": sess.get("outcome") or "unset",
        "moveCount": sess.get("moveCount"),
        "finalProgress": sess.get("finalProgress"),
        "resigns": resigns,
        "last": last or "?",
    }


def _manifest_known_ids() -> dict[str, int]:
    """Short-id -> prior file count from data/index/manifest.jsonl, if reachable.

    A hit means this drop contains a RE-EXPORT of an already-ingested session.
    Ingest anyway (UUID dedup unions rows), but check DATASET_NOTES for an
    existing entry to UPDATE rather than duplicate -- a re-export can show a
    'finished' session is still alive (#92762f: 195 -> 422 moves across two
    exports of the same proven-dead board).
    """
    counts: dict[str, int] = defaultdict(int)
    for base in (Path.cwd(), *Path.cwd().parents):
        manifest = base / "data" / "index" / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text().splitlines():
            try:
                fname = json.loads(line).get("file", "")
            except json.JSONDecodeError:
                continue
            m = _FNAME_RE.search(fname)
            if m:
                counts[m.group(2)] += 1
        break
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="export JSON files (win/game/ai-log)")
    args = ap.parse_args(argv)

    # Load everything, grouped by session short id.
    sessions: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": {}})
    for raw in args.paths:
        p = Path(raw)
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"!! could not read {p.name}: {e}")
            continue
        kind = _file_kind(p, doc)
        sid = _short_id(p, doc)
        sessions[sid]["files"][kind] = (p, doc)
        sessions[sid]["full"] = _full_id(doc)

    models: Counter[str] = Counter()
    builds: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    known = _manifest_known_ids()
    reexports: list[str] = []

    print(f"\n=== TRIAGE: {len(sessions)} session(s) across {len(args.paths)} file(s) ===\n")
    for sid, info in sessions.items():
        files = info["files"]
        # The state file that decides the verdict: prefer win > game; else ai-log.
        if "win" in files:
            decide_kind, (dp, ddoc) = "win", files["win"]
        elif "game" in files:
            decide_kind, (dp, ddoc) = "game", files["game"]
        else:
            decide_kind, (dp, ddoc) = "ai-log", files["ai-log"]

        attr = _attribution(decide_kind, ddoc)
        # The prompt-template hash reliably lives in the ai-log, not the state
        # file. Pull it from the richest artifact this session has.
        for k in ("ai-log", "win", "game"):
            if k in files:
                t = _template(files[k][1])
                if t:
                    attr["template"] = t
                    break
        cls, verdict = classify(decide_kind, ddoc)
        classes[cls] += 1
        models[attr["model"]] += 1
        builds[attr["build"]] += 1
        templates[attr["template"]] += 1

        artifacts = []
        for k in ("ai-log", "win", "game"):
            if k in files:
                _, kd = files[k]
                if k == "ai-log":
                    t, s = _rows(kd)
                    artifacts.append(f"ai-log({t} rows, {s} success)")
                elif k == "win":
                    artifacts.append(f"win-record({len(kd.get('moveHistory', []))} moves)")
                else:
                    artifacts.append(f"game-snapshot({len(kd.get('moveHistory', []))} moves)")

        print(f"#{sid}  (full {info.get('full')})")
        print(f"  artifacts : {' + '.join(artifacts)}")
        print(f"  attrib    : {attr['model']} | build {attr['build']} ({attr['buildTime']}) "
              f"| prompt {attr['template']} | seed {attr['seed']}")
        gw = ddoc.get("gameWon")
        if decide_kind in ("win", "game"):
            print(f"  state     : gameWon={gw}  progress={_progress(ddoc)}  "
                  f"faceDown={_facedown(ddoc)}  drawPile={len(ddoc.get('drawPile', []))}  "
                  f"recycle={ddoc.get('recycleCount')}  [{decide_kind}-file]")
        if "ai-log" in files:
            st = _ailog_stats(files["ai-log"][1])
            print(f"  session   : outcome={st['outcome']}  moveCount={st['moveCount']}  "
                  f"progress={st['finalProgress']}%  resigns={st['resigns']}  "
                  f"last activity {st['last']}")
        if known.get(sid):
            reexports.append(sid)
            print(f"  RE-EXPORT : {known[sid]} prior file(s) for #{sid} already in the "
                  "manifest. Ingest anyway (UUID dedup unions rows), but check "
                  "DATASET_NOTES for an existing entry to UPDATE, not duplicate; "
                  "compare moveCount/last-activity above against the entry -- the "
                  "session may still be alive (cf. #92762f, 195 -> 422 moves).")
        print(f"  >> {cls}")
        print(f"     {verdict}")
        print()

    # Batch-level heterogeneity + training-mix note.
    print("=== BATCH SUMMARY ===")
    print(f"  classes  : {dict(classes)}")
    print(f"  models   : {dict(models)}")
    print(f"  builds   : {dict(builds)}" +
          ("   <-- HETEROGENEOUS build: read attribution per file"
           if len(builds) > 1 else ""))
    known_templates = {t for t in templates if t != "(none)"}
    if len(known_templates) > 1:
        tmpl_flag = "   <-- HETEROGENEOUS prompt template"
    elif known_templates and "(none)" in templates:
        tmpl_flag = "   (template unknown for some files; pass the ai-log to resolve)"
    else:
        tmpl_flag = ""
    print(f"  templates: {dict(templates)}" + tmpl_flag)
    non_teacher = [m for m in models if m and m != "gemma-4-31b-it"]
    if non_teacher:
        print(f"  note     : {non_teacher} present. TEACHER_MODEL=gemma-4-31b-it, so "
              "these are catalogued but excluded from the default training set.")
    wins = classes.get("TERMINAL-WIN", 0)
    terminal = wins + classes.get("TERMINAL-LOSS", 0)
    if terminal:
        print(f"  win-rate : {wins}/{terminal} terminal record(s) won "
              "(count wins from terminal win-files only, not snapshots).")
    if classes.get("PENDING-SNAPSHOT"):
        print("  caution  : a PENDING-SNAPSHOT is present -- its outcome is unknown. "
              "Ingest is fine (full-stream), but do not record it as a win or loss.")
    if reexports:
        print(f"  re-export: {['#' + s for s in reexports]} already in the manifest; "
              "update their existing DATASET_NOTES entries.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
