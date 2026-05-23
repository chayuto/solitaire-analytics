"""Option #2: per-move quality filter on the decisions corpus.

For every successful decision row, compute the per-move effect by joining
turnIndex T with turnIndex T+1 within the same session. A move is "quality"
if it strictly increased foundationCards OR strictly decreased faceDownTotal
(both classic Klondike progress signals).

Emits each quality move as its own row with computed deltas attached. Bad
moves are dropped — the session may have been a loss overall, but its good
individual moves are kept.

Run:
  .venv/bin/python scripts/filter_per_move_quality.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", default="data/dataset/decisions.jsonl")
    parser.add_argument("--out", default="data/dataset/demos/option2_quality_filtered.jsonl")
    parser.add_argument("--training-eligible-only", action="store_true")
    args = parser.parse_args()

    sessions = defaultdict(list)
    with open(args.decisions) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("sessionId") or ""
            if sid:
                sessions[sid].append(r)

    print(f"## Option #2 — Per-move quality filter")
    print(f"input: {args.decisions}  ({sum(len(r) for r in sessions.values())} rows, {len(sessions)} sessions)")

    total_examined = 0
    kept_foundation = 0
    kept_reveal = 0
    kept = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for sid, rows in sessions.items():
            rows.sort(key=lambda r: r.get("turnIndex", 0))
            # Pair each successful turn with the NEXT successful turn in the
            # same session. The corpus has gaps from provider errors, so
            # consecutive turnIndex isn't guaranteed; what we care about is
            # the state delta caused by whatever moves happened between
            # observed turns, which we attribute to the earlier row.
            for r, nxt in zip(rows, rows[1:]):
                total_examined += 1
                f_now = r.get("foundationCards", 0) or 0
                f_nxt = nxt.get("foundationCards", 0) or 0
                d_now = r.get("faceDownTotal", 0) or 0
                d_nxt = nxt.get("faceDownTotal", 0) or 0
                foundation_delta = f_nxt - f_now
                facedown_delta = d_nxt - d_now
                if args.training_eligible_only and not r.get("trainingEligible"):
                    continue
                quality = foundation_delta > 0 or facedown_delta < 0
                if not quality:
                    continue
                kept += 1
                if foundation_delta > 0:
                    kept_foundation += 1
                if facedown_delta < 0:
                    kept_reveal += 1
                enriched = dict(r)
                enriched["__derived"] = {
                    "foundationDelta": foundation_delta,
                    "faceDownDelta": facedown_delta,
                    "qualityReasons": [
                        *(("foundation_progress",) if foundation_delta > 0 else ()),
                        *(("reveal",) if facedown_delta < 0 else ()),
                    ],
                }
                f.write(json.dumps(enriched) + "\n")

    print(f"examined (T -> T+1 pairs in same session): {total_examined}")
    print(f"kept (any quality signal):                  {kept}  ({100*kept/max(total_examined,1):.1f}%)")
    print(f"  of which advanced foundation:             {kept_foundation}")
    print(f"  of which revealed face-down:              {kept_reveal}")
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
