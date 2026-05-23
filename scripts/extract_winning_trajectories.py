"""Option #1: extract all winning trajectories from the decisions corpus.

Each session whose peak completionProgress >= --min-progress is treated as a
win and all its successful decision rows are emitted in turn order. Output
JSONL is the LLM-labeller training set for the win-filter variant.

Run:
  .venv/bin/python scripts/extract_winning_trajectories.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", default="data/dataset/decisions.jsonl")
    parser.add_argument("--min-progress", type=float, default=95.0,
                        help="peak completionProgress threshold (default 95%)")
    parser.add_argument("--out", default="data/dataset/demos/option1_win_filtered.jsonl")
    parser.add_argument("--training-eligible-only", action="store_true",
                        help="emit only rows with trainingEligible=true")
    args = parser.parse_args()

    sessions = defaultdict(list)
    with open(args.decisions) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("sessionId") or ""
            if sid:
                sessions[sid].append(r)

    win_sessions = []
    for sid, rows in sessions.items():
        peak = max(r.get("completionProgress", 0) or 0 for r in rows)
        if peak >= args.min_progress:
            rows.sort(key=lambda r: r.get("turnIndex", 0))
            win_sessions.append((sid, peak, rows))
    win_sessions.sort(key=lambda x: x[0])

    print(f"## Option #1 — Win-filtered trajectories")
    print(f"input:  {args.decisions} ({sum(len(r) for r in sessions.values())} rows, {len(sessions)} sessions)")
    print(f"wins:   {len(win_sessions)} sessions with peak completionProgress >= {args.min_progress}%")
    for sid, peak, rows in win_sessions:
        elig = sum(1 for r in rows if r.get("trainingEligible"))
        print(f"  - {sid[-12:]}  rows={len(rows)}  peak={peak:.0f}%  eligible={elig}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as f:
        for sid, peak, rows in win_sessions:
            for r in rows:
                if args.training_eligible_only and not r.get("trainingEligible"):
                    continue
                f.write(json.dumps(r) + "\n")
                written += 1
    print(f"\nwrote {written} per-turn records -> {out_path}")


if __name__ == "__main__":
    main()
