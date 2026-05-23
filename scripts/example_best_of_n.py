"""Option #3: best-of-N at decode time, harness demo.

Demonstrates the test-time-compute scaffolding pattern that wraps a weak
proposer with a deterministic scorer. The proposer (an LLM in production)
suggests N candidate moves per turn. The scorer (engine + heuristic, no
LLM) picks the best. Output is a per-turn record with the chosen move,
the alternatives considered, and their scores — same shape downstream
training expects.

This demo uses three proposers as stand-ins for an LLM (so it runs
offline without API access) and plays one full game per proposer for
comparison:

  first_legal   — always returns the first legal move (single candidate)
  random_n      — samples N moves uniformly at random from legal moves
                  with replacement (simulates a noisy LLM)
  exhaustive    — returns every legal move (best-of-all-legal)

The exhaustive proposer is the ceiling for a 1-ply scorer; the other two
let us measure how much best-of-N adds over a noisy proposer. In
production, swap `random_n` for an actual LLM that samples N completions
and parses move-+-rationale from each.

Run:
  .venv/bin/python scripts/example_best_of_n.py
"""

import argparse
import json
import random
from pathlib import Path
from typing import Callable, List, Tuple

from solitaire_analytics import GameState, deal_klondike, generate_moves
from solitaire_analytics.engine import apply_move
from solitaire_analytics.models import Move
from solitaire_analytics.solvers.parallel_solver import ParallelSolver


Proposer = Callable[[GameState, int], List[Move]]


def first_legal_proposer(state: GameState, n: int) -> List[Move]:
    moves = generate_moves(state)
    return moves[:1]


def random_n_proposer(state: GameState, n: int, rng: random.Random) -> List[Move]:
    moves = generate_moves(state)
    if not moves:
        return []
    return [rng.choice(moves) for _ in range(n)]


def exhaustive_proposer(state: GameState, n: int) -> List[Move]:
    return generate_moves(state)


def score_move(state: GameState, move: Move, scorer: ParallelSolver) -> float:
    """Score = heuristic on the resulting state. Illegal moves get -inf."""
    new_state = apply_move(state, move)
    if new_state is None:
        return float("-inf")
    return scorer._heuristic_score(new_state)


def best_of_n_step(
    state: GameState, proposer: Proposer, n: int, scorer: ParallelSolver
) -> Tuple[Move, List[Tuple[Move, float]]]:
    candidates = proposer(state, n)
    if not candidates:
        return None, []
    # Dedupe while preserving order
    seen = set()
    unique = []
    for m in candidates:
        key = (m.move_type, m.source_pile, m.dest_pile, m.num_cards)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    scored = [(m, score_move(state, m, scorer)) for m in unique]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored


def play_game(
    seed: int,
    proposer_name: str,
    proposer: Proposer,
    n: int,
    max_turns: int,
    scorer: ParallelSolver,
    out_path: Path,
) -> dict:
    state = deal_klondike(seed=seed)
    initial_facedown = state.count_face_down_cards()
    turn_records = []
    won = False
    stuck = False

    for turn in range(max_turns):
        if state.is_won():
            won = True
            break
        chosen, scored = best_of_n_step(state, proposer, n, scorer)
        if chosen is None:
            stuck = True
            break
        record = {
            "turnIndex": turn,
            "proposer": proposer_name,
            "n_candidates_considered": len(scored),
            "metrics": {
                "foundationCards": sum(len(f) for f in state.foundations),
                "faceDownTotal": state.count_face_down_cards(),
            },
            "candidates": [
                {"describe": str(m), "score": float(s)}
                for m, s in scored[:5]  # top-5 for inspection
            ],
            "chosenMove": chosen.to_dict(),
        }
        turn_records.append(record)
        new_state = apply_move(state, chosen)
        if new_state is None or new_state == state:
            stuck = True
            break
        state = new_state

    with out_path.open("w") as f:
        for r in turn_records:
            f.write(json.dumps(r) + "\n")

    return {
        "proposer": proposer_name,
        "seed": seed,
        "turns_played": len(turn_records),
        "final_foundationCards": sum(len(f) for f in state.foundations),
        "final_faceDownTotal": state.count_face_down_cards(),
        "facedown_revealed": initial_facedown - state.count_face_down_cards(),
        "won": won,
        "stuck": stuck,
        "out_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/dataset/demos")
    args = parser.parse_args()

    rng = random.Random(args.rng_seed)
    scorer = ParallelSolver()  # we only use its _heuristic_score
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proposers = [
        ("first_legal", first_legal_proposer),
        ("random_n", lambda s, k: random_n_proposer(s, k, rng)),
        ("exhaustive", exhaustive_proposer),
    ]

    print(f"## Option #3 — Best-of-N harness demo")
    print(f"seed={args.seed}, n={args.n}, max_turns={args.max_turns}\n")

    summaries = []
    for name, proposer in proposers:
        out_path = out_dir / f"option3_best_of_n_{name}.jsonl"
        summary = play_game(args.seed, name, proposer, args.n, args.max_turns, scorer, out_path)
        summaries.append(summary)

    print(f"{'proposer':<14}{'turns':>6}{'foundation':>12}{'faceDown':>10}{'revealed':>10}{'won':>5}{'stuck':>7}")
    for s in summaries:
        print(f"{s['proposer']:<14}"
              f"{s['turns_played']:>6}"
              f"{s['final_foundationCards']:>12}"
              f"{s['final_faceDownTotal']:>10}"
              f"{s['facedown_revealed']:>10}"
              f"{'Y' if s['won'] else '':>5}"
              f"{'Y' if s['stuck'] else '':>7}")

    print(f"\nartifacts written to {out_dir}/option3_best_of_n_*.jsonl")


if __name__ == "__main__":
    main()
