#!/usr/bin/env python3
"""Generate fresh, solver-confirmed-winnable Klondike decks for the
generalization test, guaranteed unseen by teacher and student.

Motivation: the won-only gate beats base on decks held out BY SEED, but those
decks come from the harvester win pool, so training and eval share a
distribution. This builds decks from FRESH random seeds dealt locally, none in
any training corpus or the existing benchmark, each confirmed winnable under
perfect play by the sound engine best-first solver. Running base vs gate on
these tests whether the gate learned Klondike or the harvester deck
distribution.

Solver choice (empirical probe, 2026-06-14): pyksolve's load_pysol is broken
(non-discriminating, calls a clearly-dead position winnable), so it is NOT
used. The repo engine `solve_winnable` is sound; on fresh full deals it returns
SOLVED quickly when winnable (under ~72k nodes, a few seconds in the probe) and
otherwise burns the node cap. We therefore fast-reject at a modest node cap:
any deal not SOLVED under the cap is discarded.

KNOWN BIAS (documented for the study): keeping only deals SOLVED under the cap
selects easy-to-moderate winnable deals; harder-but-winnable deals that the
solver cannot crack under the cap are excluded. The generalization set is thus
"fresh winnable deals of easy-to-moderate difficulty," not the full winnable
distribution. The paired base-vs-gate delta on identical decks remains valid.

Output format matches data/benchmarks/winnable_decks.json so the harness loads
it unchanged (with --deck-path).

Run:
  .venv/bin/python gemma4_finetune/build_generalization_decks.py \
      --n 12 --node-cap 200000 --seed-start 9000001
  # -> data/benchmarks/generalization_decks.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(REPO / ".claude" / "skills" / "solitaire-analyst" / "scripts"))

import play_deck_with_student as H  # RANK/SUIT maps, deck_to_state
from solitaire_analytics.game import deal_klondike  # noqa: E402
from winnability_solver import solve_winnable  # noqa: E402

RANK_INT_TO_STR = {v: k for k, v in H.RANK_STR_TO_INT.items()}
OUT = REPO / "data" / "benchmarks" / "generalization_decks.json"


def used_seeds() -> set[int]:
    """Seeds already in the benchmark or any training corpus (avoid overlap)."""
    seen: set[int] = set()
    bm = REPO / "data" / "benchmarks" / "winnable_decks.json"
    if bm.exists():
        for d in json.loads(bm.read_text()).get("decks", []):
            if d.get("seed"):
                seen.add(int(d["seed"]))
    for f in glob.glob(str(REPO / "data" / "dataset" / "training*.jsonl")):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("seed"):
                seen.add(int(r["seed"]))
    return seen


def serialize(state) -> dict:
    """GameState -> winnable_decks.json deck record (tableau + stock).

    deck_to_state reads tableau bottom-to-top with per-card face_up, and
    REVERSES the stock list while forcing face_up=True, so we reverse here to
    round-trip."""
    def card(c):
        return {"rank": RANK_INT_TO_STR[c.rank], "suit": c.suit.value,
                "face_up": c.face_up}
    tableau = [[card(c) for c in col] for col in state.tableau]
    stock = [{"rank": RANK_INT_TO_STR[c.rank], "suit": c.suit.value}
             for c in reversed(state.stock)]
    return {"tableau": tableau, "stock": stock}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="how many winnable decks to keep")
    ap.add_argument("--node-cap", type=int, default=200_000,
                    help="engine solver node cap; SOLVED under this = keep, else discard")
    ap.add_argument("--seed-start", type=int, default=9_000_001)
    ap.add_argument("--max-tries", type=int, default=120)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    skip = used_seeds()
    decks = []
    tried = solved = dead = unknown = 0
    seed = args.seed_start
    while len(decks) < args.n and tried < args.max_tries:
        if seed in skip:
            seed += 1
            continue
        tried += 1
        g = deal_klondike(seed)
        rec = serialize(g)
        rec["seed"] = seed
        rec["source_file"] = "generated:build_generalization_decks.py"
        rec["draw_count"] = 1
        # solve the EXACT state the harness will play
        gprime = H.deck_to_state(rec)
        verdict, nodes = solve_winnable(gprime, node_cap=args.node_cap)
        if verdict == "SOLVED":
            rec["solver_nodes"] = nodes
            decks.append(rec)
            solved += 1
            print(f"  seed {seed}: SOLVED nodes={nodes}  (kept {len(decks)}/{args.n})", flush=True)
        elif verdict == "UNSOLVABLE":
            dead += 1
        else:
            unknown += 1
        seed += 1

    Path(args.out).write_text(json.dumps({"decks": decks}, indent=1))
    print(f"\ntried {tried} fresh seeds: {solved} winnable kept, {dead} dead, "
          f"{unknown} unknown (too hard at cap {args.node_cap})")
    print(f"-> {args.out}  ({len(decks)} decks)")
    return 0 if len(decks) == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
