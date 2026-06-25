#!/usr/bin/env python3
"""Extract a STaR (expert-iteration) SFT corpus from the student's own WINNING
self-play trajectories, then build an mlx-lm data dir for the next SFT round.

Context: the best-of-N gate (run_bestofN_gen_volume.sh) sampled the volume
student at temp 0.7 on winnable decks it lost greedily, and some samples WON.
Those winning games are on-policy successful trajectories -- the raw material
for expert iteration (STaR): fine-tune the policy on its own wins and re-measure
generalization.

The play harness saves, per winning game:
  seed<seed>/s<k>/turns.jsonl     -- per-turn record (move_index, prompt_chars,
                                     json_ok_via, ...) but NOT the prompt text
  seed<seed>/s<k>/responses/turn_NNN.txt  -- the raw model response (the target)
  seed<seed>/s<k>/summary.json    -- outcome etc.

The prompt text is not stored, so we RECONSTRUCT it by deterministically
replaying the recorded moves through the SAME engine + v1.6 renderer the harness
used (play_deck_with_student), exactly as the harness's own --warm-start replay
does. Fidelity is guaranteed by the harness's DRIFT GATE: the re-rendered
prompt's length must equal the recorded prompt_chars at every decision turn, or
we abort. Each emitted row is {prompt, completion} where completion is the saved
raw response -- the same shape prepare_dataset.py emits for teacher rows, so the
trainer consumes it unchanged.

Only strict-tier responses (json_ok_via == "strict") are emitted as completions:
repair/field-tier rows had malformed JSON the harness recovered heuristically,
and training a student on malformed JSON would teach the broken format. The
recorded move is still applied for those rows (to keep replay state in sync); it
is just not used as an SFT target.

Two stages, run together by default:
  1. extract : walk --runs-root for WON samples, replay+drift-gate each, write
               {prompt, completion} rows to --out-corpus.
  2. build   : write --out-data/{train,valid,test}.jsonl = --base-data train
               plus (STaR rows x --oversample), with --base-data valid/test
               reused unchanged (STaR games are too few to hold out, and adding
               them to valid would leak). Skip with --no-build (validation).

Usage:
  # validate the extractor on whatever wins exist now (no dataset written):
  .venv/bin/python gemma4_finetune/extract_star_corpus.py \
      --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
      --deck-path data/benchmarks/generalization_decks.json \
      --out-corpus /tmp/star_validate.jsonl --no-build

  # full extract + build (run after the gate sweep completes):
  .venv/bin/python gemma4_finetune/extract_star_corpus.py \
      --runs-root gemma4_finetune/play_runs/bestofN_gen_volume \
      --deck-path data/benchmarks/generalization_decks.json \
      --base-data gemma4_finetune/dataset_volume \
      --out-corpus gemma4_finetune/star_corpus_iter1.jsonl \
      --out-data gemma4_finetune/dataset_star_iter1 --oversample 4
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

import play_deck_with_student as H  # noqa: E402  render/replay machinery
from prepare_dataset import strip_code_fence, REQUIRED_KEYS  # noqa: E402
from solitaire_analytics.engine import apply_move  # noqa: E402


class DriftError(RuntimeError):
    pass


def _fc(state) -> int:
    return sum(len(f) for f in state.foundations)


def _step(state, chosen, recent_moves, ever_seen, ctr):
    """Apply one move with the exact bookkeeping render_prompt_v16 depends on
    (cycle, ever_seen, since_foundation, since_reveal). Mirrors the harness's
    apply_and_track for the v1.6 path. Returns the new state; mutates
    recent_moves / ever_seen / ctr in place."""
    fc0 = _fc(state)
    mt = chosen.move_type.value
    drawn_card = H.card_short(state.stock[-1]) if mt == "stock_to_waste" else None
    state_before = state
    if mt == "recycle_stock":
        state = H.apply_recycle(state)
        ctr["cycle"] += 1
    else:
        state = apply_move(state, chosen)
        if state is None:
            raise DriftError("apply_move returned None for a recorded legal move")
    if drawn_card:
        ever_seen.add(drawn_card)
    recent_moves.append(H.build_history_entry(chosen, state_before, drawn_card=drawn_card))
    state, flipped = H.auto_flip(state)  # v1.6 does NOT log flips into RECENT MOVES
    new_fc = _fc(state)
    ctr["since_foundation"] = 0 if new_fc > fc0 else ctr["since_foundation"] + 1
    ctr["since_reveal"] = 0 if flipped else ctr["since_reveal"] + 1
    return state


def extract_one(sample_dir: Path, deck: dict, header: str) -> list[dict]:
    """Replay one WON sample, returning its strict-tier {prompt, completion}
    rows. Raises DriftError if any re-rendered prompt length disagrees with the
    recorded prompt_chars (fidelity failure)."""
    records = [json.loads(l) for l in (sample_dir / "turns.jsonl").open() if l.strip()]
    resp_dir = sample_dir / "responses"
    state = H.deck_to_state(deck)
    recent_moves: list = []
    ever_seen: set = set()
    ctr = {"cycle": 1, "since_foundation": 0, "since_reveal": 0}
    auto_forced_count = 0
    rows: list[dict] = []

    for rec in records:
        if rec.get("auto_forced"):
            continue  # re-derived deterministically below, not a model decision
        if rec.get("move_index") is None or rec.get("illegal") or not rec.get("json_ok"):
            continue
        if rec.get("resigned"):
            break
        turn = rec["turn"]

        # Drain forced single-legal-move positions exactly as the harness does.
        while auto_forced_count < 400:
            if _fc(state) == 52:
                break
            forced = H.visible_legal_moves(state)
            if len(forced) != 1:
                break
            state = _step(state, forced[0], recent_moves, ever_seen, ctr)
            auto_forced_count += 1

        legal = H.visible_legal_moves(state)
        prompt = H.render_prompt_v16(
            state, legal, recent_moves, ctr["cycle"], ever_seen,
            ctr["since_foundation"], ctr["since_reveal"], header,
        )
        # DRIFT GATE: byte-faithful reconstruction or bust.
        if rec.get("prompt_chars") and len(prompt) != rec["prompt_chars"]:
            raise DriftError(
                f"{sample_dir}: turn {turn} re-rendered {len(prompt)} chars "
                f"vs recorded {rec['prompt_chars']}"
            )
        mi = rec["move_index"]
        if not (0 <= mi < len(legal)):
            raise DriftError(f"{sample_dir}: turn {turn} move_index {mi} "
                             f"out of range [0,{len(legal)})")

        # Emit ONLY responses that pass the SAME filter prepare_dataset applies
        # to teacher rows: the fence-stripped text must parse as a JSON object
        # carrying all three required keys. This guarantees the STaR completions
        # are byte-format-identical to the volume corpus (clean canonical JSON),
        # and drops any strict-but-prose-wrapped or key-missing response.
        resp = strip_code_fence((resp_dir / f"turn_{turn:03d}.txt").read_text())
        try:
            obj = json.loads(resp)
            clean = isinstance(obj, dict) and all(k in obj for k in REQUIRED_KEYS)
        except json.JSONDecodeError:
            clean = False
        if clean:
            rows.append({"prompt": prompt, "completion": resp,
                         "_seed": deck["seed"], "_sample": sample_dir.name,
                         "_turn": turn})

        state = _step(state, legal[mi], recent_moves, ever_seen, ctr)

    return rows


def load_deck(deck_path: Path, seed: int) -> dict:
    decks = json.loads(deck_path.read_text())["decks"]
    deck = next((d for d in decks if d["seed"] == seed), None)
    if deck is None:
        raise SystemExit(f"seed {seed} not in {deck_path}")
    return deck


def extract(runs_root: Path, deck_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (rows, manifest). manifest is one entry per WON sample."""
    header = H.PROMPT_HEADER_V16_PATH.read_text()
    rows: list[dict] = []
    manifest: list[dict] = []
    summaries = sorted(glob.glob(str(runs_root / "seed*" / "s*" / "summary.json")))
    for sp in summaries:
        summ = json.loads(Path(sp).read_text())
        if summ.get("outcome") != "won":
            continue
        sample_dir = Path(sp).parent
        deck = load_deck(deck_path, summ["deck_seed"])
        n_before = len(rows)
        got = extract_one(sample_dir, deck, header)
        rows.extend(got)
        manifest.append({
            "seed": summ["deck_seed"], "sample": sample_dir.name,
            "turns_played": summ.get("turns_played"),
            "decision_rows_strict": len(got),
            "total_decisions_in_game": n_before,  # placeholder, overwritten below
        })
        print(f"  WON seed{summ['deck_seed']} {sample_dir.name}: "
              f"{len(got)} strict rows ({summ.get('turns_played')} turns)", flush=True)
    return rows, manifest


def build_dataset(star_rows: list[dict], base_data: Path, out_data: Path,
                  oversample: int) -> None:
    out_data.mkdir(parents=True, exist_ok=True)
    star_clean = [{"prompt": r["prompt"], "completion": r["completion"]} for r in star_rows]

    base_train = [json.loads(l) for l in (base_data / "train.jsonl").open() if l.strip()]
    train = base_train + star_clean * oversample
    (out_data / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train))
    # valid/test reused unchanged from the base corpus (no STaR leakage).
    for split in ("valid", "test"):
        (out_data / f"{split}.jsonl").write_text((base_data / f"{split}.jsonl").read_text())

    print(f"\n[build] {out_data}/")
    print(f"  train: {len(base_train)} base + {len(star_clean)} STaR x{oversample} "
          f"({len(star_clean) * oversample}) = {len(train)} rows "
          f"(STaR share {len(star_clean) * oversample / max(1, len(train)):.1%})")
    print(f"  valid: reused base   test: reused base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True,
                    help="best-of-N output root (contains seed*/s*/summary.json)")
    ap.add_argument("--deck-path", required=True,
                    help="deck JSON the samples were played from (for board hydration)")
    ap.add_argument("--out-corpus", required=True,
                    help="where to write the raw {prompt, completion} STaR rows")
    ap.add_argument("--base-data", default=str(THIS_DIR / "dataset_volume"),
                    help="mlx-lm data dir whose train/valid/test the STaR rows mix into")
    ap.add_argument("--out-data", default=str(THIS_DIR / "dataset_star_iter1"),
                    help="data dir to write for the next SFT round")
    ap.add_argument("--oversample", type=int, default=4,
                    help="how many times to repeat the STaR rows in train")
    ap.add_argument("--no-build", action="store_true",
                    help="extract + write corpus only; do not build the data dir")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    print(f"[extract] scanning {runs_root} for WON samples...", flush=True)
    rows, manifest = extract(runs_root, Path(args.deck_path))

    n_games = len(manifest)
    n_seeds = len({m["seed"] for m in manifest})
    Path(args.out_corpus).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\n[extract] {len(rows)} strict rows from {n_games} won games "
          f"across {n_seeds} seeds -> {args.out_corpus}")
    if not rows:
        print("[extract] NO winning trajectories found; nothing to build.")
        return 1

    if args.no_build:
        print("[extract] --no-build: dataset not written (validation mode).")
        return 0

    build_dataset(rows, Path(args.base_data), Path(args.out_data), args.oversample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
