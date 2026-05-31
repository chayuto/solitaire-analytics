#!/usr/bin/env python3
"""Stall-field A/B micro-probe on the Gemma 4 QS col5/col7 doom-loop.

Tests the harvester recommendation in
docs/reports/20260531_harvester_recommendation_doomloop_temporal_state.md:
does rendering TEMPORAL stall state (STALL / REPEAT lines, from stall_field.py)
break the deterministic QS loop that the temperature probe showed survives
sampling (20/20 loop move at temp 0.4/0.7/1.0)?

This is the cheapest, most reliable bench we own for the loop: the loop is a
fixed attractor at seed 3263196305 turn 12, so a single board fully exercises it.

Method (mirrors temp_probe_qs_loop.py, which self-checked 12/12 faithful):
  1. Hydrate seed 3263196305, replay GREEDILY to TARGET_TURN, self-checking each
     move_index against the recorded untuned run. While replaying, track per-turn
     board_signature and (foundation_count, face_down_count) so we can compute,
     at TARGET_TURN, the two stall STATE values a player would perceive:
        no_progress_moves   = turns since foundation OR face-down count changed
        position_seen_before = how many earlier turns had THIS exact position
  2. Render the baseline prompt. Build the +stall prompt by inserting the
     stall_field lines immediately after the PROGRESS line (probe-local string
     insertion; the shared renderer and harvester path are untouched).
  3. For each arm (baseline, +stall): temp-0 sanity (1) + N samples at 0.7 and
     1.0. Escape = move_index != the loop move (the baseline greedy pick).

Decision signal: baseline escape ~0 (reproduces the temp probe) AND +stall
escape materially > 0  =>  the stall STATE is perception-addressable on the
student; validates the v1.4 P0 and finds a student-inference lever at once.
Both ~0  =>  the model ignores the new signal as it ignores its own reasoning
(the obedience-trap prediction); kill the proposal cheaply.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))   # play_deck_with_student, stall_field, gemma4_text_patch
sys.path.insert(0, str(REPO))       # solitaire_analytics from source (venv-independent)

MODEL_ID = "mlx-community/Gemma4-E2B-IT-Text-int4"
DECK_SEED = 3263196305
TARGET_TURN = 12          # solidly inside the QS loop (loop onset ~turn 6)
N_SAMPLES = 20
TEMPS = [0.7, 1.0]
MAX_TOKENS = 2048
RECORDED_RUN = REPO / "gemma4_finetune/play_runs/gemma4_untuned_seed3263196305_run1/turns.jsonl"
OUT_PATH = THIS_DIR / "play_runs" / "stall_field_probe_result.json"


def load_recorded_moves():
    """move_index by turn from the original untuned run, for the self-check."""
    rec = {}
    for line in RECORDED_RUN.read_text().splitlines():
        t = json.loads(line)
        if t.get("json_ok") and "move_index" in t and not t.get("illegal"):
            rec[t["turn"]] = t["move_index"]
    return rec


def foundation_count(state) -> int:
    return sum(len(f) for f in state.foundations)


def face_down_count(state) -> int:
    return sum(1 for col in state.tableau for c in col if not c.face_up)


def main():
    import gemma4_text_patch  # noqa: F401  (Gemma 4 sanitize patch)
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from solitaire_analytics.engine import apply_move

    import play_deck_with_student as R
    import stall_field as SF

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
    # temporal trackers
    sig_counts = Counter()                 # board_signature -> times seen so far
    last_change_turn = 0                   # last turn foundation/facedown changed
    prev_fc, prev_fd = foundation_count(state), face_down_count(state)
    no_progress_at_target = None
    repeat_at_target = None

    for turn in range(TARGET_TURN + 1):
        legal = R.visible_legal_moves(state)
        if foundation_count(state) == 52 or not legal:
            print(f"  unexpected terminal at turn {turn}", flush=True)
            break

        # temporal state AS OF this board (before applying this turn's move)
        fc, fd = foundation_count(state), face_down_count(state)
        if turn > 0 and (fc != prev_fc or fd != prev_fd):
            last_change_turn = turn
        prev_fc, prev_fd = fc, fd
        sig = SF.board_signature(state)
        seen_before = sig_counts[sig]      # occurrences strictly earlier
        no_progress = turn - last_change_turn

        recycle = R.compute_recycle_available(state, len(seen_in_waste))
        prompt = R.render_prompt(state, legal, recent_moves, prior_decisions,
                                 seen_in_waste, recycle)
        if turn == TARGET_TURN:
            no_progress_at_target = no_progress
            repeat_at_target = seen_before
            break  # stop BEFORE applying; this is the board we probe

        sig_counts[sig] += 1
        resp = call(prompt, greedy)
        mi, conf, ba, sp, ok = R.extract_decision(resp)
        rec_mi = recorded.get(turn)
        tag = ""
        if rec_mi is not None:
            if mi == rec_mi:
                match += 1
            else:
                mismatch += 1; tag = f"  <-- MISMATCH (recorded {rec_mi})"
        print(f"  replay turn {turn:>2}: greedy mi={mi} fc={fc} fd={fd} "
              f"noprog={no_progress} seen={seen_before} (legal={len(legal)}){tag}", flush=True)
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
    print(f"TARGET_TURN={TARGET_TURN}: no_progress_moves={no_progress_at_target} "
          f"position_seen_before={repeat_at_target}")
    print("legal moves:")
    for d in legal_desc:
        print("   ", d, flush=True)

    stall = SF.stall_lines(no_progress_at_target or 0, repeat_at_target or 0)
    print("\nstall STATE lines to inject:")
    for s in stall:
        print("   ", s, flush=True)
    if not stall:
        print("   (none -- no_progress and repeat both 0; the probe would be a no-op. "
              "Check TARGET_TURN / loop onset.)", flush=True)

    # Baseline was rendered above with stall_info=None. Render the +stall arm
    # through the SAME render_prompt path the full-game runner uses, so the probe
    # tests exactly the bytes the runner would send (the two arms differ only by
    # the STALL/REPEAT block).
    baseline_prompt = prompt
    stall_prompt = R.render_prompt(
        state, legal, recent_moves, prior_decisions, seen_in_waste, recycle,
        stall_info={"no_progress_moves": no_progress_at_target or 0,
                    "position_seen_before": repeat_at_target or 0},
    )

    # --- Phase 2: sample both arms ------------------------------------------
    def loop_pick():
        mi, *_ , ok = R.extract_decision(call(baseline_prompt, greedy))
        return mi

    loop_mi = loop_pick()  # baseline greedy pick == the loop move
    print(f"\nbaseline greedy (loop) move_index = {loop_mi} "
          f"(recorded target-turn mi = {recorded.get(TARGET_TURN)})", flush=True)

    def run_arm(name, the_prompt):
        arm = {"temp_0_sanity": None, "by_temp": {}}
        smi, *_ , sok = R.extract_decision(call(the_prompt, greedy))
        arm["temp_0_sanity"] = smi
        print(f"[{name}] temp 0.0 sanity: mi={smi}", flush=True)
        for temp in TEMPS:
            sampler = make_sampler(temp=temp)
            dist, parse_fail = Counter(), 0
            tt0 = time.time()
            for k in range(N_SAMPLES):
                mi, *_ , ok = R.extract_decision(call(the_prompt, sampler))
                if not ok or mi is None:
                    parse_fail += 1
                else:
                    dist[mi] += 1
            good = sum(dist.values())
            escape = sum(v for k, v in dist.items() if k != loop_mi)
            esc_p = escape / good if good else None
            arm["by_temp"][str(temp)] = {
                "distribution": dict(dist), "parse_failures": parse_fail,
                "escape_prob": esc_p, "seconds": round(time.time() - tt0, 1),
            }
            print(f"[{name}] temp {temp}: dist={dict(dist)} parse_fail={parse_fail} "
                  f"escape_prob={esc_p}  ({time.time()-tt0:.0f}s)", flush=True)
        return arm

    print("\n===== ARM A: baseline (control) =====", flush=True)
    arm_baseline = run_arm("baseline", baseline_prompt)
    print("\n===== ARM B: +stall field =====", flush=True)
    arm_stall = run_arm("stall", stall_prompt)

    results = {
        "model_id": MODEL_ID, "deck_seed": DECK_SEED, "target_turn": TARGET_TURN,
        "n_samples": N_SAMPLES, "temps": TEMPS, "max_tokens": MAX_TOKENS,
        "replay_self_check": {"match": match, "mismatch": mismatch},
        "loop_move_index": loop_mi,
        "temporal_state": {"no_progress_moves": no_progress_at_target,
                           "position_seen_before": repeat_at_target,
                           "stall_lines": stall},
        "legal_moves": legal_desc,
        "arm_baseline": arm_baseline,
        "arm_stall": arm_stall,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}", flush=True)

    # headline
    def esc(arm, t): return arm["by_temp"][str(t)]["escape_prob"]
    print("\n========== HEADLINE ==========", flush=True)
    for t in TEMPS:
        print(f"  temp {t}: baseline escape={esc(arm_baseline,t)}  "
              f"+stall escape={esc(arm_stall,t)}", flush=True)


if __name__ == "__main__":
    main()
