"""Run the 5-arm sweep on Haiku 4.5.

For each (arm, state, run): render the prompt, call Haiku, parse the chosen
move_index, score it. Save raw + scored to raw/<arm>/<state>/run<N>.json.

Usage:
  .venv/bin/python experiments/prompt_audit_2026_05_24/run_haiku_sweep.py
  .venv/bin/python ... --arms A1 A3 --states 0 1 2 --runs 2  # subset for debugging
  .venv/bin/python ... --dry-run                              # render only, no API calls

Environment: requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make experiment + scripts importable
EXP_DIR = Path(__file__).parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_arms import ARMS, render  # noqa: E402
from ab_test_prompt_formats import classify_pick, TIER_RANK  # noqa: E402

import anthropic  # noqa: E402


MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096
THINKING_BUDGET = 2048

JSON_BLOCK_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def extract_move_index(response_text: str) -> int | None:
    """Find move_index in the model's text response.

    Tries multiple shapes in order:
      1. {"move_index": N, ...}                              (direct, A2/A3 schema)
      2. {... "final_decision": {"move_index": N, ...} ...}  (nested, C0/A1/A4 schema)
    Returns None if nothing parseable.
    """
    # Try the LAST JSON block first; sometimes models prefix prose.
    candidates = JSON_BLOCK_RE.findall(response_text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        # Direct schema
        if isinstance(obj, dict) and "move_index" in obj:
            v = obj["move_index"]
            if isinstance(v, int):
                return v
        # Nested schema
        if isinstance(obj, dict) and isinstance(obj.get("final_decision"), dict):
            v = obj["final_decision"].get("move_index")
            if isinstance(v, int):
                return v
    return None


def detect_chose_reversal(cg: dict, chosen_idx: int | None) -> bool:
    """Did the chosen move reverse the most recent tableau move?"""
    if chosen_idx is None:
        return False
    lm = cg.get("legalMoves") or []
    if chosen_idx < 0 or chosen_idx >= len(lm):
        return False
    move = lm[chosen_idx]
    desc = (move.get("describe") or "").lower()
    rm = cg.get("recentMoves") or []
    for entry in reversed(rm):
        m = re.match(r"move\s+[A-Z0-9]{2}\s+col\s+(\d+)\s+->\s+col\s+(\d+)",
                     entry.strip(), re.IGNORECASE)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        needles = (f"from column {dst} to column {src}", f"col {dst} to col {src}")
        return any(n in desc for n in needles)
    return False


def call_haiku(client: anthropic.Anthropic, prompt: str) -> dict:
    """Single Haiku call with thinking. Returns response shape we need."""
    t0 = time.time()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    # Walk content blocks: thinking blocks vs text blocks
    text_parts, thinking_parts = [], []
    for block in resp.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
    return {
        "text": "\n".join(text_parts),
        "thinking": "\n".join(thinking_parts),
        "stop_reason": resp.stop_reason,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                resp.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(
                resp.usage, "cache_read_input_tokens", 0),
        },
        "elapsed_ms": elapsed_ms,
    }


def score_run(state: dict, arm: str, prompt: str, api_resp: dict) -> dict:
    """Build the scored record for one (arm, state, run)."""
    cg = state["current_game"]
    idx = extract_move_index(api_resp["text"])
    tier = classify_pick(cg, idx) if idx is not None else "illegal"
    tier_score = TIER_RANK.get(tier, 0)
    json_valid = idx is not None
    chose_reversal = detect_chose_reversal(cg, idx)
    chosen_desc = ""
    if idx is not None and 0 <= idx < len(cg.get("legalMoves", [])):
        chosen_desc = cg["legalMoves"][idx].get("describe", "")
    return {
        "arm": arm,
        "state_id": state["state_id"],
        "category": state["category"],
        "prompt_chars": len(prompt),
        "chosen_index": idx,
        "chosen_describe": chosen_desc,
        "tier": tier,
        "tier_score": tier_score,
        "json_valid": json_valid,
        "chose_reversal": chose_reversal,
        "stop_reason": api_resp["stop_reason"],
        "elapsed_ms": api_resp["elapsed_ms"],
        "output_tokens": api_resp["usage"]["output_tokens"],
        "input_tokens": api_resp["usage"]["input_tokens"],
        "response_text": api_resp["text"],
        "thinking_text": api_resp["thinking"][:4000],  # truncate long thinking
    }


def one_call(client: anthropic.Anthropic, state: dict, arm: str, run_n: int,
             out_dir: Path, dry_run: bool) -> dict | None:
    state_dir = out_dir / arm / state["state_id"]
    state_dir.mkdir(parents=True, exist_ok=True)
    out_file = state_dir / f"run{run_n}.json"

    if out_file.exists():
        return json.loads(out_file.read_text())  # resume safety

    try:
        prompt = render(arm, state["full_prompt"], state["current_game"])
    except Exception as e:
        record = {"arm": arm, "state_id": state["state_id"], "error": f"render: {e}"}
        out_file.write_text(json.dumps(record, indent=2))
        return record

    if dry_run:
        return {"arm": arm, "state_id": state["state_id"],
                "dry_run": True, "prompt_chars": len(prompt)}

    # Retry on transient errors
    last_err = None
    for attempt in range(3):
        try:
            api_resp = call_haiku(client, prompt)
            record = score_run(state, arm, prompt, api_resp)
            out_file.write_text(json.dumps(record, indent=2))
            return record
        except anthropic.RateLimitError as e:
            wait = 2 ** (attempt + 2)
            print(f"  rate-limit on {arm}/{state['state_id']}/run{run_n}, sleeping {wait}s")
            time.sleep(wait)
            last_err = e
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            wait = 2 * (attempt + 1)
            print(f"  api error on {arm}/{state['state_id']}/run{run_n}: {e}; sleeping {wait}s")
            time.sleep(wait)
            last_err = e
    record = {"arm": arm, "state_id": state["state_id"], "run": run_n,
              "error": f"3 attempts failed: {last_err}"}
    out_file.write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=str(EXP_DIR / "bench.json"))
    parser.add_argument("--out", default=str(EXP_DIR / "raw"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--states", nargs="+", type=int, default=None,
                        help="indices into bench.states; default all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    bench = json.loads(Path(args.bench).read_text())
    states = bench["states"]
    if args.states is not None:
        states = [states[i] for i in args.states]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    jobs = [(s, arm, n) for s in states for arm in args.arms
            for n in range(1, args.runs + 1)]
    print(f"sweep: {len(args.arms)} arms × {len(states)} states × {args.runs} runs = "
          f"{len(jobs)} calls (workers={args.workers})")

    t0 = time.time()
    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(one_call, client, s, arm, n, out_dir, args.dry_run): (s, arm, n)
                   for s, arm, n in jobs}
        for fut in as_completed(futures):
            s, arm, n = futures[fut]
            try:
                rec = fut.result()
                completed += 1
                if rec and "error" in rec:
                    failures += 1
                if completed % 20 == 0:
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(jobs) - completed) / max(rate, 1e-6)
                    print(f"  {completed}/{len(jobs)}  ({rate:.1f}/s, "
                          f"eta {eta:.0f}s, {failures} failures)")
            except Exception as e:
                print(f"  job crashed: {arm}/{s['state_id']}/run{n}: {e}")
                failures += 1

    elapsed = time.time() - t0
    print(f"\ndone: {completed}/{len(jobs)} in {elapsed:.0f}s "
          f"({completed/elapsed:.1f}/s avg), {failures} failures")
    print(f"raw outputs: {out_dir}")


if __name__ == "__main__":
    main()
