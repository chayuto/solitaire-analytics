#!/usr/bin/env python3
"""Extract a GRPO / RLOO advantage-weighted corpus from best-of-N self-play.

Unlike extract_star_corpus.py (which keeps ONLY winning trajectories for SFT),
this keeps EVERY rollout and tags each decision turn with the trajectory's
group-relative ADVANTAGE, so the trainer can push the policy toward the
deep-reach / winning rollouts AND away from the shallow-stall ones on the same
deck. That negative gradient is the lever SFT-on-self-wins never had.

Reward and baseline (outcome-supervised, trajectory-level):
  R_i        = final_foundation_cards_i / 52            (a win is fc=52 -> R=1.0)
  per deck   : mean, std over that deck's N rollouts
  advantage  : A_i = (R_i - mean) / (std + eps),  SHARED by every turn of i
  a deck whose rollouts all tie (std ~ 0, e.g. 9000237 5/5 wins) carries no
  signal and is SKIPPED entirely.

The rollouts here were sampled by the VOLUME policy (temp 0.7), so training the
volume adapter on them for one epoch is on-policy iteration-0.

Reconstruction fidelity, resign handling, and the strict-JSON completion filter
are inherited verbatim from extract_star_corpus.extract_one:
  - the prompt text is not stored, so each turn is re-rendered by drift-gated
    replay through the v1.6 renderer (re-rendered prompt_chars must match the
    recorded value or the sample is skipped);
  - extract_one BREAKS at the resign row, so a resigned rollout contributes its
    pre-resign reach turns (good, high-reward reach) but NOT the resign move
    itself -- which matters because a false-resign-at-fc50 rollout (seed9000207)
    is a HIGH-reward trajectory whose resign action we must not reinforce.

Decks are located by seed across every data/benchmarks/*.json (same approach as
adjudicate_final_position.py), so gen-decks and winrate-decks rollouts can be
extracted together in one pass.

Usage:
  .venv/bin/python gemma4_finetune/extract_grpo_corpus.py \
      --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
      --runs-root gemma4_finetune/play_runs/bestofN_winrate_losses \
      --out-corpus gemma4_finetune/grpo_corpus_iter1.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

import play_deck_with_student as H  # noqa: E402
from extract_star_corpus import extract_one, DriftError  # noqa: E402


def build_deck_index() -> dict:
    """seed -> deck record, unioned across all benchmark deck files. First file
    wins on the (rare) chance a seed appears twice; we assert board identity when
    that happens so a silent collision can't corrupt replay."""
    index: dict = {}
    for p in sorted(glob.glob(str(REPO / "data" / "benchmarks" / "*.json"))):
        try:
            decks = json.loads(Path(p).read_text()).get("decks", [])
        except Exception:
            continue
        for d in decks:
            seed = d.get("seed")
            if seed is None:
                continue
            if seed in index and index[seed]["tableau"] != d["tableau"]:
                raise SystemExit(f"seed {seed} appears in two benchmark files "
                                 f"with DIFFERENT boards -- cannot disambiguate")
            index.setdefault(seed, d)
    return index


def collect_samples(runs_roots: list[Path]) -> dict:
    """seed -> list of (sample_dir, summary) across all run roots."""
    bydeck: dict = defaultdict(list)
    for root in runs_roots:
        for sp in sorted(glob.glob(str(root / "seed*" / "s*" / "summary.json"))):
            summ = json.loads(Path(sp).read_text())
            bydeck[summ["deck_seed"]].append((Path(sp).parent, summ))
    return bydeck


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", action="append", required=True,
                    help="best-of-N output root (repeatable)")
    ap.add_argument("--out-corpus", required=True)
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="std floor; a deck with std below this is dropped (no signal)")
    ap.add_argument("--min-abs-adv", type=float, default=0.0,
                    help="drop turns whose |advantage| is below this (0 = keep all)")
    args = ap.parse_args()

    header = H.PROMPT_HEADER_V16_PATH.read_text()
    deck_index = build_deck_index()
    bydeck = collect_samples([Path(r) for r in args.runs_root])

    rows: list[dict] = []
    per_deck_report: list[dict] = []
    skipped_drift = 0
    skipped_nosignal = 0

    for seed in sorted(bydeck):
        samples = bydeck[seed]
        rewards = [s.get("final_foundation_cards", 0) / 52.0 for _, s in samples]
        mean_r = statistics.fmean(rewards)
        std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        if std_r < args.eps:
            skipped_nosignal += len(samples)
            per_deck_report.append({"seed": seed, "n": len(samples),
                                    "mean_R": round(mean_r, 3), "std_R": 0.0,
                                    "status": "SKIP(no signal)", "rows": 0})
            print(f"  seed{seed}: n={len(samples)} mean_R={mean_r:.3f} std=0 "
                  f"-> SKIP (no signal)", flush=True)
            continue

        deck = deck_index.get(seed)
        if deck is None:
            raise SystemExit(f"seed {seed} not found in any data/benchmarks/*.json")

        deck_rows = 0
        for (sample_dir, summ), R in zip(samples, rewards):
            adv = (R - mean_r) / (std_r + args.eps)
            if abs(adv) < args.min_abs_adv:
                continue
            try:
                got = extract_one(sample_dir, deck, header)
            except DriftError as e:
                skipped_drift += 1
                print(f"    DRIFT skip {sample_dir.name}: {e}", flush=True)
                continue
            for r in got:
                r["advantage"] = adv
                r["_R"] = R
            rows.extend(got)
            deck_rows += len(got)
        per_deck_report.append({"seed": seed, "n": len(samples),
                                "mean_R": round(mean_r, 3), "std_R": round(std_r, 3),
                                "status": "used", "rows": deck_rows})
        advs = [round((R - mean_r) / (std_r + args.eps), 2) for R in rewards]
        print(f"  seed{seed}: n={len(samples)} mean_R={mean_r:.3f} std={std_r:.3f} "
              f"adv={sorted(advs)} -> {deck_rows} rows", flush=True)

    Path(args.out_corpus).write_text("".join(json.dumps(r) + "\n" for r in rows))

    advs_all = [r["advantage"] for r in rows]
    pos = sum(1 for a in advs_all if a > 0)
    neg = sum(1 for a in advs_all if a < 0)
    used_seeds = sum(1 for d in per_deck_report if d["status"] == "used")
    print(f"\n[grpo-extract] {len(rows)} turn-rows from {used_seeds} decks "
          f"({pos} positive-advantage, {neg} negative-advantage) -> {args.out_corpus}")
    if advs_all:
        print(f"  advantage range [{min(advs_all):+.2f}, {max(advs_all):+.2f}], "
              f"mean {statistics.fmean(advs_all):+.3f}")
    print(f"  skipped: {skipped_nosignal} rollouts in no-signal decks, "
          f"{skipped_drift} rollouts on drift")
    # sidecar manifest for the run record
    Path(args.out_corpus + ".manifest.json").write_text(
        json.dumps({"per_deck": per_deck_report, "n_rows": len(rows),
                    "pos": pos, "neg": neg,
                    "skipped_drift": skipped_drift,
                    "skipped_nosignal": skipped_nosignal}, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
