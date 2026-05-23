"""A/B test: does hybrid prompt format change the model's move choice?

For each sampled corpus snapshot we build TWO prompts that share the same
preamble (rules, response-format instructions) and the same tail ("now reply
with JSON"), but differ in the middle:

  A = original CURRENT GAME (JSON) block as the harvester ships today
  B = hybrid ASCII tableau + numbered RECENT MOVES + numbered LEGAL MOVES
      (from scripts/compare_prompt_formats.py render_hybrid)

We call the model twice per snapshot (same temperature) and parse
final_decision.move_index from each response. Aggregates:
  - agreement rate (do A and B pick the same move?)
  - on oscillation samples: did each format pick the reversal?
  - on any sample: did each format pick a "quality" move (foundation_to_*
    or a tableau move that uncovers a face-down card)?

Requires GEMINI_API_KEY in environment.

Run:
  GEMINI_API_KEY=... .venv/bin/python scripts/ab_test_prompt_formats.py --n 10
  GEMINI_API_KEY=... .venv/bin/python scripts/ab_test_prompt_formats.py \\
      --n 20 --filter oscillation --model gemma-3-27b-it
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from compare_prompt_formats import (  # noqa: E402
    CURRENT_GAME_RE,
    looks_like_oscillation,
    parse_current_game,
    render_hybrid,
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def call_gemini(model: str, prompt: str, api_key: str, temperature: float = 0.3,
                timeout: int = 240, retries: int = 4) -> dict:
    url = f"{API_BASE}/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            # Budget for thinking tokens too; Gemma's thought stream can run
            # 500-2000 tokens before the actual answer. 2048 is enough headroom
            # for our short "{move_index: N, reason: ...}" answer.
            "maxOutputTokens": 2048,
        },
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=body, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            # Retry transient server errors (500, 503) — Gemma 4 31B in
            # particular is unstable on long prompts, same pattern the
            # harvester sees.
            if resp.status_code in (500, 503) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            # Honour 429 Retry-After (free-tier RPM limit on Gemini models)
            if resp.status_code == 429 and attempt < retries:
                wait = 30.0
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", resp.text)
                if m:
                    wait = float(m.group(1)) + 1.0
                time.sleep(wait)
                continue
            return {"_error": f"HTTP {resp.status_code}", "_body": resp.text[:500]}
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return {"_error": f"timeout after {retries+1} attempts: {last_err}"}


def extract_text(resp: dict) -> str:
    """Return the model's user-visible answer, skipping thinking parts.

    Gemma 4 / Gemini 2.5+ may emit multiple parts in candidates[0].content.parts,
    with the chain-of-thought tagged "thought": true. The final answer is in
    a separate untagged part. We concatenate all non-thought text.
    """
    if "_error" in resp:
        return ""
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    visible = [p.get("text", "") for p in parts if not p.get("thought")]
    return "".join(visible).strip()


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_move_index(text: str) -> int | None:
    """Try to extract final_decision.move_index from a model response."""
    if not text:
        return None
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    m = JSON_OBJ_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    # Support both the simplified ask ({"move_index": N}) and the harvester
    # shape ({"final_decision": {"move_index": N}}).
    candidates = [obj.get("move_index"), (obj.get("final_decision") or {}).get("move_index")]
    for idx in candidates:
        if isinstance(idx, int):
            return idx
        if isinstance(idx, str) and idx.isdigit():
            return int(idx)
    return None


SIMPLE_TAIL = (
    "\n\nPick the best legal move. Respond with ONLY a JSON object of the form "
    '{"move_index": N, "reason": "<one short sentence>"} where N is the index '
    "from the legal moves list. No other text."
)


def build_prompts(original_prompt: str, cg: dict) -> tuple[str, str]:
    """Build A/B prompts that share preamble (rules) but differ in board format.

    We trim the harvester's long response-format instructions and replace them
    with a single-line JSON ask, so the model emits a parseable answer in a few
    hundred tokens instead of multi-thousand-token chain-of-thought. The format
    delta we're measuring is in the CURRENT GAME block, not the response shape.
    """
    # Keep only the rules preamble — strip everything from "RESPONSE FORMAT"
    # (or fall back to keeping everything before CURRENT GAME).
    m = CURRENT_GAME_RE.search(original_prompt)
    if not m:
        raise ValueError("no CURRENT GAME block")
    pre_full = original_prompt[: m.start()]
    # Cut at the response-format section if we can find it.
    cut_idx = pre_full.find("RESPONSE FORMAT")
    if cut_idx < 0:
        cut_idx = pre_full.find("Respond with")
    pre = pre_full[:cut_idx].rstrip() + "\n\n" if cut_idx > 0 else pre_full
    json_block = "CURRENT GAME (JSON):\n" + json.dumps(cg, indent=2)
    hybrid_block = "CURRENT GAME (board view):\n" + render_hybrid(cg)
    return pre + json_block + SIMPLE_TAIL, pre + hybrid_block + SIMPLE_TAIL


def is_reversal_pick(cg: dict, idx: int) -> bool:
    """True if the chosen move undoes the most recent tableau move."""
    if idx is None:
        return False
    rm = cg.get("recentMoves") or []
    lm = cg.get("legalMoves") or []
    if idx < 0 or idx >= len(lm):
        return False
    for entry in reversed(rm):
        m = re.match(r"^move\s+([A-Z0-9]{2})\s+col\s+(\d+)\s+->\s+col\s+(\d+)",
                     entry.strip(), re.IGNORECASE)
        if not m:
            continue
        _, src, dst = m.group(1), m.group(2), m.group(3)
        desc = (lm[idx].get("describe") or "").lower()
        return (f"from column {dst} to column {src}" in desc
                or f"col {dst} to col {src}" in desc)
    return False


def classify_pick(cg: dict, idx: int) -> str:
    """Tag the chosen move with a quality tier we can aggregate over.

    Tiers (best to worst):
      foundation   — any move sending a card to the foundations
      reveal       — tableau→tableau that empties source's face-up stack and
                     flips a face-down beneath
      waste_play   — waste→tableau (places a waste card productively)
      shuffle      — tableau→tableau that reveals nothing
      draw         — draw from stock to waste
      recycle      — recycle waste back to stock
      illegal      — parser returned None or out-of-range index
    """
    if idx is None:
        return "illegal"
    lm = cg.get("legalMoves") or []
    if idx < 0 or idx >= len(lm):
        return "illegal"
    move = lm[idx]
    mtype = move.get("type") or ""
    if "foundation" in mtype:
        return "foundation"
    if mtype == "draw_card":
        return "draw"
    if mtype == "recycle_stock" or "recycle" in mtype:
        return "recycle"
    if mtype == "discard_to_tableau":
        return "waste_play"
    if mtype == "tableau_to_tableau":
        m = re.search(r"from column (\d+) to column (\d+)", move.get("describe") or "")
        if not m:
            return "shuffle"
        src = int(m.group(1))
        pm = re.search(r"plus (\d+) more", move.get("describe") or "")
        n_moved = (int(pm.group(1)) + 1) if pm else 1
        for col in cg.get("tableau") or []:
            if col.get("column") == src:
                fd = col.get("faceDownCount", 0) or 0
                fu = col.get("faceUp") or []
                if fd > 0 and n_moved >= len(fu):
                    return "reveal"
                return "shuffle"
    return "shuffle"


# rank for "is A better than B" comparisons
TIER_RANK = {
    "foundation": 6,
    "reveal": 5,
    "waste_play": 4,
    "shuffle": 2,
    "draw": 1,
    "recycle": 1,
    "illegal": 0,
}


def is_quality_pick(cg: dict, idx: int) -> bool:
    return classify_pick(cg, idx) in ("foundation", "reveal", "waste_play")


def sample_snapshots(path: Path, n: int, filter_mode: str, seed: int) -> list:
    rng = random.Random(seed)
    pool = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("outcome") != "success":
                continue
            prompt = r.get("prompt") or ""
            cg = parse_current_game(prompt)
            if not cg:
                continue
            if filter_mode == "oscillation" and not looks_like_oscillation(cg):
                continue
            pool.append((r.get("id"), r.get("sessionId"), r.get("turnIndex"),
                         prompt, cg))
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/store/interactions.jsonl")
    parser.add_argument("--out-dir", default="data/dataset/demos/ab_prompt_format")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="gemma-4-31b-it",
                        help="MUST be the harvester model to measure real failures. "
                             "Do not substitute stronger Gemini models — Gemma-specific "
                             "failure modes (oscillation, decode-incoherence) won't "
                             "reproduce on them. Tolerate the 500/429 friction.")
    parser.add_argument("--retries", type=int, default=4,
                        help="retry budget per call (Gemma is unstable; bump higher)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--filter", choices=("any", "oscillation"), default="oscillation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="seconds between API calls — default 4s keeps us "
                             "under gemini-2.5-flash's 20 req/min free-tier limit")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY env var not set")

    samples = sample_snapshots(Path(args.interactions), args.n, args.filter, args.seed)
    if not samples:
        raise SystemExit(f"no samples for filter={args.filter}")
    print(f"# A/B prompt format test")
    print(f"model={args.model}  temp={args.temperature}  filter={args.filter}  "
          f"samples={len(samples)}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (iid, sid, turn, prompt, cg) in enumerate(samples):
        sub = out_dir / f"sample-{i:02d}"
        sub.mkdir(exist_ok=True)
        prompt_a, prompt_b = build_prompts(prompt, cg)
        (sub / "prompt_A_json.txt").write_text(prompt_a)
        (sub / "prompt_B_hybrid.txt").write_text(prompt_b)

        resp_a = call_gemini(args.model, prompt_a, api_key, args.temperature,
                             retries=args.retries)
        time.sleep(args.sleep)
        resp_b = call_gemini(args.model, prompt_b, api_key, args.temperature,
                             retries=args.retries)
        time.sleep(args.sleep)

        text_a = extract_text(resp_a)
        text_b = extract_text(resp_b)
        (sub / "response_A.txt").write_text(text_a or json.dumps(resp_a, indent=2))
        (sub / "response_B.txt").write_text(text_b or json.dumps(resp_b, indent=2))

        idx_a = parse_move_index(text_a)
        idx_b = parse_move_index(text_b)
        rev_a = is_reversal_pick(cg, idx_a)
        rev_b = is_reversal_pick(cg, idx_b)
        tier_a = classify_pick(cg, idx_a)
        tier_b = classify_pick(cg, idx_b)

        lm = cg.get("legalMoves") or []
        desc_a = lm[idx_a]["describe"] if idx_a is not None and 0 <= idx_a < len(lm) else "?"
        desc_b = lm[idx_b]["describe"] if idx_b is not None and 0 <= idx_b < len(lm) else "?"

        row = {
            "sample": i, "sessionId": sid, "turnIndex": turn,
            "n_legal": len(lm),
            "A_idx": idx_a, "A_desc": desc_a, "A_tier": tier_a, "A_reversal": rev_a,
            "B_idx": idx_b, "B_desc": desc_b, "B_tier": tier_b, "B_reversal": rev_b,
            "agreement": idx_a is not None and idx_a == idx_b,
            "tier_delta": TIER_RANK[tier_b] - TIER_RANK[tier_a],  # +ve = B better
        }
        results.append(row)
        (sub / "result.json").write_text(json.dumps(row, indent=2))

        print(f"sample {i:>2}  A=[{idx_a}]{tier_a:<11}{'rev' if rev_a else '   '}"
              f"  |  B=[{idx_b}]{tier_b:<11}{'rev' if rev_b else '   '}"
              f"  |  agree={row['agreement']}  delta={row['tier_delta']:+d}")

    # aggregate
    n = len(results)
    parsed_a = sum(1 for r in results if r["A_idx"] is not None)
    parsed_b = sum(1 for r in results if r["B_idx"] is not None)
    agree = sum(1 for r in results if r["agreement"])
    rev_a = sum(1 for r in results if r["A_reversal"])
    rev_b = sum(1 for r in results if r["B_reversal"])

    from collections import Counter
    tiers_a = Counter(r["A_tier"] for r in results)
    tiers_b = Counter(r["B_tier"] for r in results)
    b_better = sum(1 for r in results if r["tier_delta"] > 0)
    b_worse = sum(1 for r in results if r["tier_delta"] < 0)
    b_tied = sum(1 for r in results if r["tier_delta"] == 0)

    summary = {
        "model": args.model,
        "temperature": args.temperature,
        "filter": args.filter,
        "n_samples": n,
        "A_format": "original CURRENT GAME (JSON)",
        "B_format": "hybrid ASCII board + numbered moves",
        "A_parsed_responses": parsed_a,
        "B_parsed_responses": parsed_b,
        "agreement_count": agree,
        "agreement_rate": agree / n if n else 0,
        "A_reversal_picks": rev_a,
        "B_reversal_picks": rev_b,
        "A_tiers": dict(tiers_a),
        "B_tiers": dict(tiers_b),
        "B_strictly_better_than_A": b_better,
        "B_strictly_worse_than_A": b_worse,
        "tied": b_tied,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    tier_order = ["foundation", "reveal", "waste_play", "shuffle", "draw", "recycle", "illegal"]
    print()
    print("## summary")
    print(f"parsed responses:  A={parsed_a}/{n}   B={parsed_b}/{n}")
    print(f"agreement:         {agree}/{n}  ({100*agree/max(n,1):.0f}%)")
    print(f"reversal picks:    A={rev_a}/{n}   B={rev_b}/{n}")
    print(f"tier distribution:")
    print(f"  {'tier':<12} {'A':>4} {'B':>4}  {'delta':>6}")
    for t in tier_order:
        a, b = tiers_a.get(t, 0), tiers_b.get(t, 0)
        if a or b:
            print(f"  {t:<12} {a:>4} {b:>4}  {b-a:>+6}")
    print(f"B vs A: better={b_better}  worse={b_worse}  tied={b_tied}")
    print(f"\nartifacts -> {out_dir}/")


if __name__ == "__main__":
    main()
