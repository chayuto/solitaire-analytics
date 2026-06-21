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
    # Won-only SFT gate (2026-06-13): trained on 18 won games with the 13
    # rescue benchmark decks HELD OUT (lora_config_gate.yaml); eval on those
    # held-out decks vs the base numbers in play_runs/tourA_v16_rescue.
    ("wononly-gate", GEMMA4, str(THIS / "adapters_gate")),
    # Filter-vs-volume ablation (2026-06-13): matched 2500-row natural-mix
    # (38% won) corpus vs the gate's 100%-won, same holdout/recipe
    # (lora_config_allsucc.yaml). gate >> allsucc => the won-filter is the
    # lever; gate ~= allsucc => it was volume, not the filter.
    ("allsucc", GEMMA4, str(THIS / "adapters_allsucc")),
    # Volume-scaling arm (2026-06-14): the ENTIRE non-eval success pool (6859
    # rows / 77 games, 36% won) at fixed iters, same holdout/recipe
    # (lora_config_volume.yaml). volume >> allsucc => more data still helps;
    # volume ~= allsucc => the matched 2500 already saturated.
    ("volume", GEMMA4, str(THIS / "adapters_volume")),
    # Checkpoint-selection for the publish candidate (2026-06-16): volume is the
    # strongest adapter (5/13 in-dist, +12.9 generalization) but carries a JSON-
    # discipline regression (34 temp-parse-rescues vs base ~12 on the 13 decks).
    # Eval the intermediate checkpoints to find the one that keeps the wins with
    # the least format damage. Each dir = <iter>_adapters.safetensors renamed to
    # adapters.safetensors + the shared adapter_config.json. "volume" == iter1000.
    ("volume-250", GEMMA4, str(THIS / "adapters_volume_ckpt0250")),
    ("volume-500", GEMMA4, str(THIS / "adapters_volume_ckpt0500")),
    ("volume-750", GEMMA4, str(THIS / "adapters_volume_ckpt0750")),
    # Loop-compression spike (2026-06-16): the volume corpus minus exact-state
    # doom-loop cycle bodies (draw-safe, 5.5% cut, mostly lost-game cycles),
    # volume-identical hypers. loopcompress vs volume on the 13 held-out decks
    # isolates whether removing exact loops helps imitation.
    ("loopcompress", GEMMA4, str(THIS / "adapters_loopcompress")),
    # Close-out augmentation (2026-06-17): the loopcompress split with TRAIN-ONLY
    # 2x oversample of won-game close-out rows (won-session & faceDown<=2), to fix
    # the replay-verified false-resign failure (student reaches winnable fd=0
    # endgames with legal foundation plays available and resigns). Same holdout/
    # recipe (lora_config_closeout.yaml). closeout converts the 4 false-resign
    # decks => loop-compress + close-out is the recipe; closeout ~= loopcompress
    # => SFT reweighting cannot close the gap (escalate to a resign penalty).
    ("closeout", GEMMA4, str(THIS / "adapters_closeout")),
    # Gentle close-out on VOLUME (2026-06-19): closeout fixed the false-resign but
    # regressed mid-game reach and did not beat volume. This arm is a GENTLE (1
    # extra copy) close-out oversample on the VOLUME corpus (the lead student, 0
    # resign, best reach), targeting volume's near-win fd0/1 cap-stalls without the
    # mid-game cost (lora_config_volcloseout.yaml). GATE on meanFC vs volume.
    ("volcloseout", GEMMA4, str(THIS / "adapters_volcloseout")),
    # Volume on MORE data (2026-06-20): the `volume` recipe retrained on the
    # current 31B pool after the harvest grew +4 wins (#bd2080/#211fcc/#fbfdf8/
    # #adc679 = +560 win rows, 33 -> 37 winning sessions). dataset_volume_v0620
    # (8115 train vs old 5663), identical hypers. Isolates the +data effect from
    # the gentle-closeout recipe; compare wins vs volume's stable 5/13.
    ("volume_v0620", GEMMA4, str(THIS / "adapters_volume_v0620")),
    # Recipe + data (2026-06-20): the gentle close-out oversample applied to the
    # +data pool (dataset_volcloseout_v0620, 9792 train). The ship-candidate.
    # Compare vs volume_v0620 (recipe effect on +data) and vs old volcloseout
    # (does the recipe survive +data). lora_config_volcloseout_v0620.yaml.
    ("volcloseout_v0620", GEMMA4, str(THIS / "adapters_volcloseout_v0620")),
    # Strategy-text probe (2026-06-20): volume + 27 hand-authored draw-1 strategy
    # Q&A rows (x12, 5.4% of train). Tests whether declarative strategy in the
    # WEIGHTS improves play (the v1.6 prompt already gives it in-context and the
    # model under-applies it). Compare wins + JSON parse-rescue rate vs volume.
    ("volstrategy", GEMMA4, str(THIS / "adapters_volstrategy")),
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


def run_game(label, model_id, adapter, seed, max_turns, max_tokens,
             prompt_version="v1.6", max_parse_failures=3, parse_retry_temp=0.3,
             max_illegal_moves=3, deck_path=None):
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
        "--prompt-version", prompt_version,
        "--max-parse-failures", str(max_parse_failures),
        "--parse-retry-temp", str(parse_retry_temp),
        "--max-illegal-moves", str(max_illegal_moves),
    ]
    if deck_path:
        cmd += ["--deck-path", deck_path]
    if adapter:
        cmd += ["--adapter-path", adapter]
    t0 = time.time()
    # PER_GAME_TIMEOUT was sized for cap 80 (~14 s/call); scale for bigger caps
    # (measured base arm: 18.9 s/call mean, so 25 s/turn + load gives slack).
    timeout = max(PER_GAME_TIMEOUT, max_turns * 25 + 300)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
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
    ap.add_argument("--arms", default="base,v7-300,v7b-600,v7b-1000",
                    help="comma-separated subset of model arms to run")
    ap.add_argument("--deck-path", default=str(DECKS),
                    help="Deck JSON for seed enumeration and play (default: the "
                         "winnable benchmark; use generalization_decks.json for "
                         "the fresh-deck generalization test).")
    ap.add_argument("--out-name", default="tourA",
                    help="results dir name under play_runs/ (separate dirs keep "
                         "prompt-version runs from resume-skipping each other)")
    ap.add_argument("--prompt-version", choices=["v1.6", "v1.0"], default="v1.6",
                    help="forwarded to the play harness (v1.6 = corpus-faithful)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated subset of benchmark deck seeds to play, "
                         "in the given (priority) order; default = all seeded decks")
    ap.add_argument("--max-parse-failures", type=int, default=3,
                    help="forwarded to the play harness")
    ap.add_argument("--parse-retry-temp", type=float, default=0.3,
                    help="forwarded to the play harness")
    ap.add_argument("--max-illegal-moves", type=int, default=3,
                    help="forwarded to the play harness")
    ap.add_argument("--smoke", action="store_true",
                    help="1 model (v7-300), 1 deck, cap 6 -- validates orchestration")
    args = ap.parse_args()

    global OUT
    OUT = THIS / "play_runs" / args.out_name

    deck_path = Path(args.deck_path)
    decks = json.loads(deck_path.read_text())["decks"]
    # The harness selects a deck by --deck-seed, so only seeded decks are playable
    # (one benchmark deck is unseeded). Skip the unseeded one.
    seeds = [int(d["seed"]) for d in decks if d.get("seed")]
    if args.seeds:
        wanted_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        unknown = [s for s in wanted_seeds if s not in seeds]
        if unknown:
            raise SystemExit(f"--seeds not in {deck_path.name}: {unknown}")
        seeds = wanted_seeds  # preserve the given (priority) order
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    models = [m for m in MODELS if m[0] in wanted]
    if not models:
        raise SystemExit(f"no known arms in --arms {args.arms!r}; "
                         f"known: {[m[0] for m in MODELS]}")
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
        "prompt_version": args.prompt_version,
        "max_parse_failures": args.max_parse_failures,
        "parse_retry_temp": args.parse_retry_temp,
    }, indent=1))

    total = len(models) * len(seeds)
    done = 0
    t0 = time.time()
    print(f"Window A: {len(models)} models x {len(seeds)} decks = {total} games, "
          f"cap {args.max_turns} turns. Sequential, subprocess-per-game.", flush=True)
    for label, mid, adapter in models:
        for seed in seeds:
            s, skipped = run_game(label, mid, adapter, seed,
                                  args.max_turns, args.max_tokens,
                                  prompt_version=args.prompt_version,
                                  max_parse_failures=args.max_parse_failures,
                                  parse_retry_temp=args.parse_retry_temp,
                                  max_illegal_moves=args.max_illegal_moves,
                                  deck_path=(args.deck_path if Path(args.deck_path) != DECKS else None))
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
