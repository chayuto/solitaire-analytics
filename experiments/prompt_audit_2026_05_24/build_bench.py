"""Extract 20 board states from interactions.jsonl, balanced across categories.

Composition target:
  6 oscillation contexts (reversal in legalMoves)
  6 midgame (face-down 10-15)
  4 early-game (face-down 18+)
  4 endgame (face-down <=4)

Output: bench.json — list of {state_id, category, source, full_prompt, current_game_json}
This is the FIXED bench; changing it invalidates between-arm comparisons.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# Reuse parsing/heuristic from scripts/compare_prompt_formats.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from compare_prompt_formats import parse_current_game, looks_like_oscillation  # noqa: E402


def classify(cg: dict) -> str | None:
    """Bucket a state into one of: oscillation, midgame, early, endgame."""
    fd = (cg.get("metrics") or {}).get("faceDownTotal", 0) or 0
    legal = cg.get("legalMoves") or []
    if len(legal) < 2:
        return None  # not informative
    if looks_like_oscillation(cg):
        return "oscillation"
    if fd <= 4:
        return "endgame"
    if fd >= 18:
        return "early"
    if 10 <= fd <= 15:
        return "midgame"
    return None


def has_min_history(cg: dict, min_recent: int = 3) -> bool:
    return len(cg.get("recentMoves") or []) >= min_recent


def state_hash(cg: dict) -> str:
    payload = json.dumps(
        {
            "f": cg.get("foundations"),
            "t": cg.get("tableau"),
            "w": cg.get("discardTop"),
            "s": cg.get("drawPileCount"),
            "lm": [m.get("describe") for m in (cg.get("legalMoves") or [])],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactions",
        default="/Users/chayut/repos/solitaire-analytics/data/store/interactions.jsonl",
    )
    parser.add_argument(
        "--out",
        default="/Users/chayut/repos/solitaire-analytics/experiments/prompt_audit_2026_05_24/bench.json",
    )
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    targets = {"oscillation": 6, "midgame": 6, "early": 4, "endgame": 4}
    buckets: dict[str, list] = {k: [] for k in targets}
    seen_hashes: set[str] = set()

    in_path = Path(args.interactions)
    with in_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("outcome") != "success":
                continue
            prompt = r.get("prompt") or ""
            cg = parse_current_game(prompt)
            if not cg:
                continue
            if not has_min_history(cg, min_recent=3):
                continue
            cat = classify(cg)
            if not cat or cat not in targets:
                continue
            h = state_hash(cg)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            buckets[cat].append(
                {
                    "interaction_id": r.get("id"),
                    "session_id": r.get("requestId", "").split("-")[0],
                    "turn_index": r.get("turnIndex"),
                    "full_prompt": prompt,
                    "current_game": cg,
                    "state_hash": h,
                }
            )

    rng = random.Random(args.seed)
    picked = []
    for cat, n in targets.items():
        pool = buckets[cat]
        if len(pool) < n:
            print(f"WARNING: only {len(pool)} candidates for '{cat}' (wanted {n})")
            sample = pool
        else:
            sample = rng.sample(pool, n)
        for s in sample:
            s["category"] = cat
            s["state_id"] = f"{cat}-{s['state_hash']}"
            picked.append(s)

    bench = {
        "version": 1,
        "seed": args.seed,
        "source": str(in_path),
        "composition": {k: sum(1 for s in picked if s["category"] == k) for k in targets},
        "states": picked,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(bench, indent=2))
    print(f"wrote {out_path}")
    print(f"composition: {bench['composition']}")
    print(f"total states: {len(picked)}")
    print(f"pool sizes by category: {dict((k, len(v)) for k, v in buckets.items())}")


if __name__ == "__main__":
    main()
