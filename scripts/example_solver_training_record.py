"""Demo: produce full-game training records, two ways.

Two labeller modes are demonstrated end-to-end as the experiment for the
Gemma-4 capability discussion (option #2: solver-as-labeller, plus its
LLM-as-labeller foil):

  --mode solver        deal a fresh seed, run ParallelSolver to find a
                       winning move sequence, emit per-turn records.
                       Output has authentic ground-truth moves but NO
                       natural-language reasoning.

  --mode replay        take an already-won session from the corpus
                       (default: 0154e1), emit per-turn records from
                       data/dataset/decisions.jsonl. Output has authentic
                       reasoning trails but is limited to the small number
                       of winning sessions actually in the corpus.

Both modes emit JSONL in the same shape downstream consumers can ingest.

Run:
  .venv/bin/python scripts/example_solver_training_record.py --mode replay
  .venv/bin/python scripts/example_solver_training_record.py --mode solver
"""

import argparse
import json
import time
from pathlib import Path

from solitaire_analytics import (
    GameState,
    ParallelSolver,
    deal_klondike,
    generate_moves,
)
from solitaire_analytics.engine import apply_move


def turn_record(turn_index: int, state: GameState, chosen, alternatives) -> dict:
    """One per-turn training record, shaped to mirror the harvester schema."""
    legal_moves = [
        {"index": i, "describe": str(m), "to_dict": m.to_dict()}
        for i, m in enumerate(alternatives)
    ]
    chosen_index = next(
        i for i, m in enumerate(alternatives) if m == chosen
    )
    new_state = apply_move(state, chosen)
    return {
        "turnIndex": turn_index,
        "currentGame": state.to_dict(),
        "metrics": {
            "foundationCards": sum(len(f) for f in state.foundations),
            "faceDownTotal": state.count_face_down_cards(),
            "stockRemaining": len(state.stock),
            "wasteSize": len(state.waste),
        },
        "legalMoves": legal_moves,
        "chosenMoveIndex": chosen_index,
        "chosenMove": chosen.to_dict(),
        "labeller": "ParallelSolver",
        "moveEffect": {
            "foundationDelta": (
                sum(len(f) for f in new_state.foundations)
                - sum(len(f) for f in state.foundations)
            ),
            "faceDownDelta": (
                new_state.count_face_down_cards() - state.count_face_down_cards()
            ),
        },
    }


def solve_until_winnable(seeds, beam_width: int, timeout: float):
    """Try seeds until one solves within the timeout."""
    solver = ParallelSolver(
        max_depth=200, beam_width=beam_width, timeout=timeout, n_jobs=-1
    )
    for seed in seeds:
        print(f"  trying seed={seed} ...", flush=True)
        state = deal_klondike(seed=seed)
        t0 = time.time()
        result = solver.solve(state)
        elapsed = time.time() - t0
        status = "WON" if result.success else "no solve in budget"
        print(
            f"    {status} in {elapsed:.1f}s "
            f"({result.states_explored} states, {len(result.moves)} moves)"
        )
        if result.success:
            return seed, state, result
    return None, None, None


def run_solver_mode(args):
    print("## Solver-as-labeller demo")
    print(f"beam_width={args.beam_width}, per-seed timeout={args.timeout}s")
    seed, initial_state, result = solve_until_winnable(
        args.seeds, args.beam_width, args.timeout
    )
    if not result:
        raise SystemExit(
            "No seed solved within budget. The in-repo ParallelSolver does not "
            "reliably solve fresh Klondike deals — even at beam=20k, t=180s, "
            "2.8M states explored. This is itself a finding: option #2 "
            "(solver-as-labeller) is currently BLOCKED on either a stronger "
            "solver (e.g. solvitaire) or running on partial mid-game states "
            "rather than fresh deals."
        )

    print(f"\n## Generating training records from seed={seed}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = initial_state
    written = 0
    with out_path.open("w") as f:
        for turn_index, chosen in enumerate(result.moves):
            alternatives = generate_moves(state)
            assert chosen in alternatives, f"solver move missing from legal set at turn {turn_index}"
            rec = turn_record(turn_index, state, chosen, alternatives)
            f.write(json.dumps(rec) + "\n")
            written += 1
            state = apply_move(state, chosen)

    print(f"  wrote {written} turn records -> {out_path}")
    print(f"  final state: foundationCards={sum(len(f) for f in state.foundations)}, "
          f"is_won={state.is_won()}")
    summarise_moves(result.moves)


def summarise_moves(moves):
    print("\n## Quick stats on the generated set")
    print(f"  total turns: {len(moves)}")
    print(f"  foundation moves: {sum(1 for m in moves if 'foundation' in str(m).lower())}")
    print(f"  draws: {sum(1 for m in moves if str(m) == 'Draw from stock')}")
    print(f"  tableau-to-tableau: {sum(1 for m in moves if 'tableau' in str(m).lower() and 'to tableau' in str(m).lower())}")


def run_replay_mode(args):
    print(f"## LLM-as-labeller demo (replay session containing '{args.session}')")
    rows = []
    with open(args.decisions) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("sessionId") or ""
            if args.session in sid:
                rows.append(r)
    if not rows:
        raise SystemExit(
            f"No rows in {args.decisions} matched session substring '{args.session}'."
        )
    rows.sort(key=lambda r: r.get("turnIndex", 0))
    print(f"  matched {len(rows)} successful turns across "
          f"turnIndex {rows[0]['turnIndex']}..{rows[-1]['turnIndex']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            slim = {
                "turnIndex": r["turnIndex"],
                "sessionId": r["sessionId"],
                "appCommit": r.get("appCommit"),
                "model": r.get("model"),
                "labeller": "LLM",
                "metrics": {
                    "foundationCards": r.get("foundationCards"),
                    "faceDownTotal": r.get("faceDownTotal"),
                    "completionProgress": r.get("completionProgress"),
                    "progressScore": r.get("progressScore"),
                    "turnsSinceProgress": r.get("turnsSinceProgress"),
                },
                "chosenMoveDescribe": r.get("chosenMoveDescribe"),
                "chosenMoveType": r.get("chosenMoveType"),
                "nLegalMoves": r.get("nLegalMoves"),
                "boardAnalysis": r.get("boardAnalysis"),
                "reasoning": r.get("reasoning"),
                "confidence": r.get("confidence"),
                "trainingEligible": r.get("trainingEligible"),
            }
            f.write(json.dumps(slim) + "\n")
    print(f"  wrote {len(rows)} per-turn records -> {out_path}")
    final = rows[-1]
    print(f"  final state: foundationCards={final.get('foundationCards')}, "
          f"completionProgress={final.get('completionProgress')}%")
    eligible = sum(1 for r in rows if r.get("trainingEligible"))
    print(f"  training-eligible rows: {eligible}/{len(rows)} "
          f"({100*eligible/len(rows):.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["solver", "replay"], default="replay",
                        help="solver: deal+solve fresh; replay: read corpus win")
    # solver-mode flags
    parser.add_argument("--seeds", type=int, nargs="*",
                        default=[42, 7, 1, 100, 2026, 99, 13, 21, 314])
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    # replay-mode flags
    parser.add_argument("--session", default="0154e1",
                        help="short id substring of a won session")
    parser.add_argument("--decisions", default="data/dataset/decisions.jsonl")
    # shared
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = (
            f"data/dataset/demos/{args.mode}_training_example.jsonl"
        )

    if args.mode == "solver":
        run_solver_mode(args)
    else:
        run_replay_mode(args)


if __name__ == "__main__":
    main()
