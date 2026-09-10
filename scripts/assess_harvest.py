"""Assess a harvest: separate what the model got wrong from what the harness did to it.

`data/SUMMARY.md` reports a success rate per model. That number answers "how many
calls produced a usable decision", which is not the same question as "how good is
this model at Klondike" -- and for MiniMax-M3 the two answers are very far apart.
This script reports both, side by side, for any set of export files:

  infrastructure   timeouts, transport failures, proxy errors, empty replies
  parse            replies that arrived but could not be read
  decision         replies that were read but chose an illegal or absent move
  play             progress actually made on decks with a known outcome

Run it over the existing corpus and over a new harvest to compare like with like:

    .venv/bin/python scripts/assess_harvest.py data/raw --model MiniMax-M3
    .venv/bin/python scripts/assess_harvest.py path/to/harvest --model MiniMax-M3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Failures that say something about the pipe, not about the player.
INFRA_PATTERNS = [
    (re.compile(r"did not respond within", re.I), "client timeout"),
    (re.compile(r"could not reach|network connection", re.I), "transport unreachable"),
    (re.compile(r"rejected the API key|401|403", re.I), "auth rejected"),
    (re.compile(r"temporarily unavailable|5\d\d", re.I), "server error"),
    (re.compile(r"empty answer|no message content", re.I), "empty reply"),
    (re.compile(r"was cancelled", re.I), "cancelled"),
]
INFRA_CLASSES = {"stall", "deadline", "network", "http_5xx", "http_429", "http_4xx",
                 "empty_content", "timeout"}
PARSE_CLASSES = {"unparseable", "not JSON"}
DECISION_CLASSES = {"bad_index", "illegal_move"}


def classify(rec: dict) -> tuple[str, str]:
    """(bucket, label) for one interaction."""
    if rec.get("outcome") == "success":
        return "success", "usable decision"
    cls = rec.get("errorClass")
    msg = str(rec.get("errorMessage") or "")
    if cls:
        if cls in INFRA_CLASSES:
            return "infrastructure", cls
        if cls in PARSE_CLASSES:
            return "parse", cls
        if cls in DECISION_CLASSES:
            return "decision", cls
        return "other", cls
    for pat, label in INFRA_PATTERNS:
        if pat.search(msg):
            return "infrastructure", label
    if re.search(r"not JSON|parse", msg, re.I):
        return "parse", "unparseable"
    return "other", (msg[:48] or "unlabelled error")


def load(paths: list[str]) -> list[dict]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    out = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "interactions" in d:
            out.append(d)
    return out


def pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="export files or directories")
    ap.add_argument("--model", help="only sessions whose model contains this string")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    docs = load(args.paths)
    if args.model:
        docs = [d for d in docs if args.model.lower() in str((d.get("session") or {}).get("model", "")).lower()]
    if not docs:
        print("no matching exports", file=sys.stderr)
        return 2

    buckets: Counter = Counter()
    labels: dict[str, Counter] = defaultdict(Counter)
    lat_ok: list[float] = []
    lat_fail: list[float] = []
    thought: list[int] = []
    sessions = []
    total = 0
    retries = 0

    for d in docs:
        s = d.get("session") or {}
        sessions.append(s)
        for rec in d.get("interactions") or []:
            if rec.get("event") in ("forced_move", "stall_terminated"):
                continue                      # not a model turn
            total += 1
            bucket, label = classify(rec)
            buckets[bucket] += 1
            labels[bucket][label] += 1
            ms = rec.get("durationMs") or 0
            if ms:
                (lat_ok if bucket == "success" else lat_fail).append(ms)
            if rec.get("thoughtTokens"):
                thought.append(rec["thoughtTokens"])
            retries += max(0, (rec.get("attempt") or 1) - 1)

    if not total:
        print("no model turns found", file=sys.stderr)
        return 2

    report = {
        "model": args.model or "all",
        "sessions": len(sessions),
        "modelTurns": total,
        "buckets": {k: {"count": v, "pct": round(v / total * 100, 1), "detail": dict(labels[k])}
                    for k, v in buckets.most_common()},
        "retriesSpent": retries,
    }

    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print(f"model      {report['model']}")
    print(f"sessions   {len(sessions)}   model turns {total}   retries spent {retries}")
    print()
    ok = buckets.get("success", 0)
    infra = buckets.get("infrastructure", 0)
    print(f"{'usable decisions':<20} {ok:6d}  {ok / total * 100:5.1f} %")
    for b in ("infrastructure", "parse", "decision", "other"):
        n = buckets.get(b, 0)
        if not n:
            continue
        print(f"{b:<20} {n:6d}  {n / total * 100:5.1f} %")
        for lab, c in labels[b].most_common(6):
            print(f"    {c:6d}  {lab}")
    print()

    # The number the summary should carry: how the model did on the calls that reached it.
    reached = total - infra
    if reached:
        print(f"of the {reached} calls that actually reached the model and came back, "
              f"{ok} were usable ({ok / reached * 100:.1f} %)")
        if infra:
            print(f"  the headline {ok / total * 100:.1f} % is dragged down by {infra} "
                  f"infrastructure failures that say nothing about the model")
    print()

    if lat_ok:
        print(f"latency on success   p50 {pctile(lat_ok, .5)/1000:6.0f}s   "
              f"p90 {pctile(lat_ok, .9)/1000:6.0f}s   max {max(lat_ok)/1000:6.0f}s")
    if lat_fail:
        print(f"latency on failure   p50 {pctile(lat_fail, .5)/1000:6.0f}s   "
              f"p90 {pctile(lat_fail, .9)/1000:6.0f}s   max {max(lat_fail)/1000:6.0f}s")
    if thought:
        print(f"thinking tokens      p50 {pctile(thought, .5):6.0f}    "
              f"p90 {pctile(thought, .9):6.0f}    max {max(thought):6.0f}")
    if lat_ok:
        print("\nwhat a client timeout would have cost, over the successful calls alone:")
        for wall in (60, 120, 240, 480, 900):
            lost = sum(1 for x in lat_ok if x > wall * 1000)
            print(f"  cap {wall:>4}s: {lost:5d}/{len(lat_ok)} of them cut off ({lost/len(lat_ok)*100:5.1f} %)")

    known = [s for s in sessions if s.get("knownWinnable")]
    if known:
        won = sum(1 for s in known if s.get("outcome") == "won")
        prog = [s.get("finalProgress") or 0 for s in known]
        print(f"\nplay on {len(known)} deck(s) known to be winnable: {won} won, "
              f"median progress {statistics.median(prog):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
