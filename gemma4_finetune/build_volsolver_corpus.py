#!/usr/bin/env python3
"""Mix SOLVER-GROUNDED play-matched rows into the volume corpus (2026-06-21).

Form 3 of step 4 (docs/reports/20260621_data_volume_and_strategy_text_eval.md):
dataset_volsolver = volume train + N solver-grounded rows (the output of
build_solver_grounded_corpus.py). valid/test are byte-identical to volume, so
volsolver-vs-volume isolates exactly the solver rows, and volsolver-vs-volstrategy
compares solver-grounded form 3 against hand-authored declarative form 1 on the
same base. The solver rows are already {prompt, completion} in the v1.6 play
format, so mixing is a direct concatenation (the trainer shuffles batches).

  build_volsolver_corpus.py [--rows solver_grounded_rows.jsonl] [--n 0=all]
      [--src dataset_volume] [--out dataset_volsolver]
"""
import argparse
import json
from pathlib import Path

THIS = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(THIS / "solver_grounded_rows.jsonl"),
                    help="solver-grounded {prompt,completion} rows to mix in")
    ap.add_argument("--n", type=int, default=0,
                    help="how many solver rows to use (0 = all)")
    ap.add_argument("--src", default=str(THIS / "dataset_volume"),
                    help="volume split to mix into")
    ap.add_argument("--out", default=str(THIS / "dataset_volsolver"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # valid/test verbatim from volume so the comparison isolates the train mix
    for name in ("valid", "test"):
        (out / f"{name}.jsonl").write_text((src / f"{name}.jsonl").read_text())

    base = [json.loads(l) for l in (src / "train.jsonl").read_text().splitlines() if l.strip()]
    solver = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    if args.n and args.n < len(solver):
        solver = solver[:args.n]

    # sanity: every solver row must carry the two expected keys
    bad = sum(1 for r in solver if "prompt" not in r or "completion" not in r)
    if bad:
        raise SystemExit(f"{bad} solver rows missing prompt/completion")

    train = base + solver
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))

    print(f"base volume train rows:      {len(base)}")
    print(f"solver-grounded rows added:  {len(solver)}  "
          f"({100 * len(solver) / len(train):.1f}% of train)")
    print(f"-> {out} train rows: {len(train)}  (valid/test byte-identical to {src.name})")


if __name__ == "__main__":
    main()
