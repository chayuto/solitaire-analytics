#!/usr/bin/env python3
"""Window A: scaled full-game tournament of ORPO checkpoints vs the untuned base.

Plays each FROZEN winnable deck with each model, one SUBPROCESS PER GAME so the
model is loaded, plays, writes summary.json, and exits, releasing all memory.
Inference peaks ~3.3 GB and never accumulates across games (the training OOM that
lagged the laptop does not apply to inference, but subprocess isolation keeps it
that way regardless). Runs games strictly sequentially: play is GPU-bound on the
unified-memory M-series, so parallel processes would only time-slice the GPU for
no wall-clock gain and more memory pressure.

This is the play-graded eval that GATES the ORPO track. Validation loss and the
20-state single-turn bench both mispredict play (the bench even ranked untuned
E2B above the 31B teacher), so the decision metric is full-game foundation
progress, paired per deck against the untuned base.

Idempotent / resumable: a game whose summary.json already exists is skipped, so a
re-launch (or resume after an interrupt) continues where it left off. The deck
set is whatever winnable_decks.json holds at launch (freeze-at-kickoff).

Models benched (all on the SAME Gemma 4 E2B int4 base the adapters were trained
on; the base arm is no-adapter):
  base      -- untuned Gemma 4 E2B int4
  v7-300    -- move-contrast ORPO, iter 300 (current play-best)
  v7b-600   -- v7b lr 1e-4, iter 600 (mid checkpoint)
  v7b-1000  -- v7b final (val-best; tests the overcook hypothesis)

Usage:
  .venv/bin/python gemma4_finetune/tournament_A.py --max-turns 80
  .venv/bin/python gemma4_finetune/tournament_A.py --smoke   # 1 model, 1 deck, cap 6
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
DECKS = REPO / "data" / "benchmarks" / "winnable_decks.json"
HARNESS = THIS / "play_deck_with_student.py"
GEMMA4 = "mlx-community/Gemma4-E2B-IT-Text-int4"
OUT = THIS / "play_runs" / "tourA"

MODELS = [
    ("base",     GEMMA4, None),
    ("v7-300",   GEMMA4, str(THIS / "adapters_orpo_v7_at300")),
    ("v7b-600",  GEMMA4, str(THIS / "adapters_orpo_v7b_at600")),
    ("v7b-1000", GEMMA4, str(THIS / "adapters_orpo_v7b")),
]

PER_GAME_TIMEOUT = 3000  # 50 min ceiling per game (80 turns * ~14s + load + slack)


def max_fc_from_turns(game_dir: Path):
    """Max foundation-cards reached across the game (fc can be flat or rise)."""
    f = game_dir / "turns.jsonl"
    if not f.exists():
        return None
    best = 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fc = r.get("fc")
        if isinstance(fc, int):
            best = max(best, fc)
    return best


def loop_onset_turn(game_dir: Path):
    """First turn index where fc stops rising for the rest of the game (rough
    'when did it stop progressing' marker). None if it never plateaus."""
    f = game_dir / "turns.jsonl"
    if not f.exists():
        return None
    fcs = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("fc"), int):
            fcs.append(r["fc"])
    if not fcs:
        return None
    final = fcs[-1]
    for i, v in enumerate(fcs):
        if v == final and all(x == final for x in fcs[i:]):
            return i
    return None


def run_game(label, model_id, adapter, seed, max_turns, max_tokens):
    game_dir = OUT / label / f"seed{seed}"
    summ = game_dir / "summary.json"
    if summ.exists():
        # Resume: skip only a genuinely COMPLETED game. A transient-failure stub
        # (RUNNER_ERROR / timeout) or a corrupt/partial summary is re-run, so an
        # interrupt or a one-off subprocess crash recovers on the next launch.
        try:
            s = json.loads(summ.read_text())
        except (json.JSONDecodeError, ValueError):
            s = {}
        if s.get("outcome") and s.get("outcome") != "RUNNER_ERROR" and "turns_played" in s:
            s.setdefault("max_fc", max_fc_from_turns(game_dir))
            return s, True
        # else fall through and replay (partial / corrupt / error stub)
    game_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(HARNESS),
        "--deck-seed", str(seed),
        "--model-id", model_id,
        "--out-dir", str(game_dir),
        "--max-turns", str(max_turns),
        "--max-tokens", str(max_tokens),
    ]
    if adapter:
        cmd += ["--adapter-path", adapter]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=PER_GAME_TIMEOUT)
        err_tail = proc.stderr[-800:]
    except subprocess.TimeoutExpired:
        err_tail = "TIMEOUT"
    if not summ.exists():
        stub = {"deck_seed": seed, "outcome": "RUNNER_ERROR",
                "runner_seconds": round(time.time() - t0, 1),
                "stderr_tail": err_tail}
        summ.write_text(json.dumps(stub, indent=1))
        return stub, False
    s = json.loads(summ.read_text())
    s["max_fc"] = max_fc_from_turns(game_dir)
    s["loop_onset_turn"] = loop_onset_turn(game_dir)
    s["runner_seconds"] = round(time.time() - t0, 1)
    summ.write_text(json.dumps(s, indent=1))
    return s, False


def aggregate(models):
    rows = []
    for label, _, _ in models:
        for d in sorted((OUT / label).glob("seed*")):
            f = d / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                s["model"] = label
                rows.append(s)
    (OUT / "leaderboard.json").write_text(json.dumps(rows, indent=1))

    # per-model + per-deck paired-vs-base text table
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r.get("deck_seed")] = r
    base = by_model.get("base", {})
    lines = ["# Window A tournament leaderboard", ""]
    for label, _, _ in models:
        gm = by_model.get(label, {})
        if not gm:
            continue
        fcs = [g.get("max_fc") or 0 for g in gm.values()]
        outs = {}
        for g in gm.values():
            outs_key = g.get("outcome", "?")
            outs[outs_key] = outs.get(outs_key, 0) + 1
        wins = outs.get("won", 0)
        # paired delta vs base
        deltas = [
            (g.get("max_fc") or 0) - (base[s].get("max_fc") or 0)
            for s, g in gm.items() if s in base and label != "base"
        ]
        line = (f"{label:10s} n={len(gm):2d} won={wins} "
                f"meanFC={statistics.mean(fcs):4.1f} medFC={statistics.median(fcs):4.1f} "
                f"maxFC={max(fcs):2d} outcomes={outs}")
        if deltas:
            line += (f"  vs-base: meanDelta={statistics.mean(deltas):+.1f} "
                     f"wins>{sum(1 for d in deltas if d > 0)} ties={sum(1 for d in deltas if d == 0)} "
                     f"loses{sum(1 for d in deltas if d < 0)}")
        lines.append(line)
    (OUT / "leaderboard.txt").write_text("\n".join(lines) + "\n")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true",
                    help="1 model (v7-300), 1 deck, cap 6 -- validates orchestration")
    args = ap.parse_args()

    decks = json.loads(DECKS.read_text())["decks"]
    # The harness selects a deck by --deck-seed, so only seeded decks are playable
    # (one benchmark deck is unseeded). Skip the unseeded one.
    seeds = [int(d["seed"]) for d in decks if d.get("seed")]
    models = MODELS
    if args.smoke:
        models = [("v7-300", GEMMA4, str(THIS / "adapters_orpo_v7_at300"))]
        seeds = seeds[:1]
        args.max_turns = 6

    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / "config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        if prev.get("max_turns") != args.max_turns:
            print(f"WARNING: resuming with --max-turns {args.max_turns} but prior run "
                  f"used {prev.get('max_turns')}. Completed games keep their old cap; "
                  f"this mixes caps. Re-run with --max-turns {prev.get('max_turns')} "
                  f"to stay consistent, or clear {OUT} to start fresh.", flush=True)
    (cfg_path).write_text(json.dumps({
        "models": [[m[0], m[2]] for m in models],
        "n_decks": len(seeds), "seeds": seeds,
        "max_turns": args.max_turns, "max_tokens": args.max_tokens,
    }, indent=1))

    total = len(models) * len(seeds)
    done = 0
    t0 = time.time()
    print(f"Window A: {len(models)} models x {len(seeds)} decks = {total} games, "
          f"cap {args.max_turns} turns. Sequential, subprocess-per-game.", flush=True)
    for label, mid, adapter in models:
        for seed in seeds:
            s, skipped = run_game(label, mid, adapter, seed,
                                  args.max_turns, args.max_tokens)
            done += 1
            fc = s.get("max_fc", s.get("final_foundation_cards"))
            tag = "skip" if skipped else s.get("outcome", "?")
            print(f"[{done:3d}/{total}] {label:10s} seed{seed:<11d} {tag:14s} "
                  f"fc={fc} turns={s.get('turns_played')} "
                  f"({(time.time() - t0) / 60:.1f}min)", flush=True)
            aggregate(models)
    print("\n=== DONE ===", flush=True)
    for line in aggregate(models):
        print(line, flush=True)
    print(f"\nleaderboard -> {OUT/'leaderboard.txt'} and {OUT/'leaderboard.json'}",
          flush=True)


if __name__ == "__main__":
    main()
