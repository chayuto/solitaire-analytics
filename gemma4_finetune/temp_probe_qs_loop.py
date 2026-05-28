#!/usr/bin/env python3
"""Temperature micro-probe on the Gemma 4 QS col5/col7 doom-loop.

Pre-registered in docs/reports/20260528_compute_window_plan_v4A_and_temp_probe.md
section 4. Question: is the deterministic (temp 0.0) QS oscillation a greedy-
decoding fixed point that sampling escapes, or an attractor that survives
temperature?

Method:
  1. Hydrate seed 3263196305 and replay GREEDILY (temp 0.0) up to TARGET_TURN
     using play_deck_with_student.py's own loop functions. Self-check: each
     replayed greedy move_index must match the recorded untuned run in
     play_runs/gemma4_untuned_seed3263196305_run1/turns.jsonl (proves the
     reconstructed prompt is faithful).
  2. At TARGET_TURN, render the exact prompt and sample N times at temps
     0.4 / 0.7 / 1.0 (plus a temp-0.0 sanity sample). Tally the move_index
     distribution and compute escape probability (mass off the greedy pick).

Escape = a sampled move_index != the greedy/recorded move_index at TARGET_TURN.
MP2 (primary prediction): escape probability > 0.20 at temp 0.7.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

MODEL_ID = "mlx-community/Gemma4-E2B-IT-Text-int4"
DECK_SEED = 3263196305
TARGET_TURN = 12          # solidly inside the QS loop (loop onset ~turn 6)
N_SAMPLES = 20
TEMPS = [0.4, 0.7, 1.0]
MAX_TOKENS = 2048
RECORDED_RUN = REPO / "gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/turns.jsonl"
OUT_PATH = THIS_DIR / "play_runs" / "temp_probe_qs_loop_result.json"


def load_recorded_moves():
    """move_index by turn from the original untuned run, for the self-check."""
    rec = {}
    for line in RECORDED_RUN.read_text().splitlines():
        t = json.loads(line)
        if t.get("json_ok") and "move_index" in t and not t.get("illegal"):
            rec[t["turn"]] = t["move_index"]
    return rec


def main():
    import gemma4_text_patch  # noqa: F401  (Gemma 4 sanitize patch)
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from solitaire_analytics.engine import apply_move

    import play_deck_with_student as R

    recorded = load_recorded_moves()
    decks = json.loads(R.DECK_PATH.read_text())["decks"]
    deck = next(d for d in decks if d["seed"] == DECK_SEED)
    state = R.deck_to_state(deck)

    print(f"Loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  load {time.time()-t0:.1f}s", flush=True)

    greedy = make_sampler(temp=0.0)

    def call(prompt, sampler):
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return generate(model, tokenizer, prompt=wrapped,
                        max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)

    # --- Phase 1: greedy replay up to TARGET_TURN (self-checking) -----------
    recent_moves, prior_decisions, seen_in_waste = [], [], []
    match, mismatch = 0, 0
    prompt = None
    legal = None
    for turn in range(TARGET_TURN + 1):
        legal = R.visible_legal_moves(state)
        if sum(len(f) for f in state.foundations) == 52 or not legal:
            print(f"  unexpected terminal at turn {turn}", flush=True)
            break
        recycle = R.compute_recycle_available(state, len(seen_in_waste))
        prompt = R.render_prompt(state, legal, recent_moves, prior_decisions,
                                 seen_in_waste, recycle)
        if turn == TARGET_TURN:
            break  # stop BEFORE applying; this is the board we probe
        resp = call(prompt, greedy)
        mi, conf, ba, sp, ok = R.extract_decision(resp)
        rec_mi = recorded.get(turn)
        tag = ""
        if rec_mi is not None:
            if mi == rec_mi:
                match += 1
            else:
                mismatch += 1; tag = f"  <-- MISMATCH (recorded {rec_mi})"
        print(f"  replay turn {turn:>2}: greedy mi={mi} (legal={len(legal)}){tag}", flush=True)
        if not ok or mi is None or not (0 <= mi < len(legal)):
            print(f"  replay broke at turn {turn}; aborting", flush=True)
            return
        chosen = legal[mi]
        drawn = R.card_short(state.stock[-1]) if chosen.move_type.value == "stock_to_waste" else None
        state_before = state
        state = apply_move(state, chosen)
        if chosen.move_type.value == "stock_to_waste" and drawn:
            if len(state_before.stock) == 0:
                seen_in_waste = [drawn]
            elif drawn not in seen_in_waste:
                seen_in_waste.append(drawn)
        recent_moves.append(R.build_history_entry(chosen, state_before, drawn_card=drawn))
        state, flipped = R.auto_flip(state)
        for f in flipped:
            recent_moves.append({"engine_move_type": "flip_card", "card": f["card"], "from_col": f["column"]})
        prior_decisions.append({"move_text": R.describe_move(chosen, state_before), "why": sp or ""})
        prior_decisions = prior_decisions[-5:]

    print(f"\nReplay self-check: {match} match / {mismatch} mismatch vs recorded run", flush=True)
    legal_desc = [f"[{i}] {R.describe_move(m, state)}" for i, m in enumerate(legal)]
    print(f"TARGET_TURN={TARGET_TURN} legal moves:")
    for d in legal_desc:
        print("   ", d, flush=True)

    # --- Phase 2: temp-0 sanity + sampling at each temp ---------------------
    results = {
        "model_id": MODEL_ID, "deck_seed": DECK_SEED, "target_turn": TARGET_TURN,
        "n_samples": N_SAMPLES, "max_tokens": MAX_TOKENS,
        "replay_self_check": {"match": match, "mismatch": mismatch},
        "legal_moves": legal_desc, "by_temp": {},
    }

    sanity = call(prompt, greedy)
    smi, *_ , sok = R.extract_decision(sanity)
    greedy_mi = smi
    results["temp_0_sanity"] = {"move_index": smi, "recorded_target": recorded.get(TARGET_TURN)}
    print(f"\ntemp 0.0 sanity: greedy mi={smi}  (recorded target-turn mi={recorded.get(TARGET_TURN)})", flush=True)

    for temp in TEMPS:
        sampler = make_sampler(temp=temp)
        dist, parse_fail = Counter(), 0
        tt0 = time.time()
        for k in range(N_SAMPLES):
            mi, *_ , ok = R.extract_decision(call(prompt, sampler))
            if not ok or mi is None:
                parse_fail += 1
            else:
                dist[mi] += 1
        good = sum(dist.values())
        escape = sum(v for k, v in dist.items() if k != greedy_mi)
        esc_p = escape / good if good else None
        results["by_temp"][str(temp)] = {
            "distribution": dict(dist), "parse_failures": parse_fail,
            "escape_prob": esc_p, "seconds": round(time.time() - tt0, 1),
        }
        print(f"temp {temp}: dist={dict(dist)} parse_fail={parse_fail} "
              f"escape_prob={esc_p}  ({time.time()-tt0:.0f}s)", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
