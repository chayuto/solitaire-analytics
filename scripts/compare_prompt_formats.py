"""Side-by-side comparison: current JSON prompt vs proposed hybrid format.

Loads real corpus snapshots from data/store/interactions.jsonl, then for each
sampled snapshot renders two prompts of the same underlying state:

  original.txt — the CURRENT GAME (JSON) block as the harvester ships today
  hybrid.txt   — ASCII tableau + numbered RECENT MOVES + numbered LEGAL MOVES

Writes one subdirectory per sample under --out-dir, plus a summary.md with
token estimates so we can see the size delta at a glance. The artifacts are
inspection-only; if we like what we see, the next step is to actually send
both formats to a model and measure decision-quality delta.

Run:
  .venv/bin/python scripts/compare_prompt_formats.py
  .venv/bin/python scripts/compare_prompt_formats.py --n 10 --filter oscillation
"""

import argparse
import json
import random
import re
from pathlib import Path

CURRENT_GAME_RE = re.compile(r"CURRENT GAME \(JSON\):\s*(\{.*\})", re.DOTALL)
TABLEAU_MOVE = re.compile(
    r"^move\s+([A-Z0-9]{2})\s+col\s+(\d+)\s+->\s+col\s+(\d+)", re.IGNORECASE
)


def parse_current_game(prompt: str) -> dict | None:
    m = CURRENT_GAME_RE.search(prompt)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def render_tableau_ascii(tableau: list) -> str:
    lines = []
    for col in tableau:
        n = col.get("column")
        fd = col.get("faceDownCount", 0) or 0
        fu = col.get("faceUp") or []
        face_down = " ".join(["??"] * fd)
        face_up = " ".join(fu)
        parts = [p for p in (face_down, face_up) if p]
        content = " ".join(parts) if parts else "<empty>"
        lines.append(f"  col{n}: {content}")
    return "\n".join(lines)


def render_hybrid(cg: dict) -> str:
    parts = []
    parts.append(
        "NOTATION: rank+suit (A 2-9 T J Q K; H D C S). "
        "?? = face-down. In each column the top of the stack is the rightmost card."
    )
    parts.append("")

    f = cg.get("foundations") or {}
    fline = "   ".join(
        f"{suit[0].upper()}: {f.get(suit) or '--'}"
        for suit in ("hearts", "diamonds", "clubs", "spades")
    )
    parts.append(f"FOUNDATIONS: {fline}")
    parts.append(
        f"STOCK: {cg.get('drawPileCount', 0)} cards   "
        f"WASTE top: {cg.get('discardTop') or '--'}   "
        f"recycle stock: {'yes' if cg.get('canRecycleStock') else 'no'}"
    )
    parts.append("")

    parts.append("TABLEAU:")
    parts.append(render_tableau_ascii(cg.get("tableau") or []))
    parts.append("")

    rm = cg.get("recentMoves") or []
    if rm:
        parts.append(
            "RECENT MOVES (oldest -> newest; review before picking, "
            "do not undo your own work):"
        )
        for i, m in enumerate(rm, 1):
            parts.append(f"  {i:>2}. {m}")
        parts.append("")

    seen = cg.get("seenDrawPileCards") or []
    if seen:
        # Keep as one line — needed for stock-cycle / late-game decisions.
        parts.append(f"SEEN IN WASTE THIS CYCLE: {' '.join(seen)}")
        parts.append("")

    lm = cg.get("legalMoves") or []
    parts.append("LEGAL MOVES (respond with the index of your chosen move):")
    for m in lm:
        parts.append(f"  [{m.get('index')}] {m.get('type', ''):<24} {m.get('describe', '')}")
    parts.append("")

    metrics = cg.get("metrics") or {}
    parts.append(
        f"PROGRESS: foundation={metrics.get('foundationCards', 0)}/52, "
        f"face-down remaining={metrics.get('faceDownTotal', 0)}, "
        f"completion={metrics.get('completionProgress', 0)}%"
    )
    return "\n".join(parts)


def render_original(cg: dict) -> str:
    return "CURRENT GAME (JSON):\n" + json.dumps(cg, indent=2)


def looks_like_oscillation(cg: dict) -> bool:
    """Heuristic: the most recent tableau move's reverse is in legalMoves.

    Note format mismatch: recentMoves uses "col N -> col M" while legalMoves
    describe uses "from column N to column M". Check both shapes.
    """
    rm = cg.get("recentMoves") or []
    lm = cg.get("legalMoves") or []
    for entry in reversed(rm):
        m = TABLEAU_MOVE.match(entry.strip())
        if not m:
            continue
        _, src, dst = m.group(1), m.group(2), m.group(3)
        needles = (
            f"from column {dst} to column {src}",
            f"col {dst} to col {src}",
        )
        for opt in lm:
            desc = (opt.get("describe") or "").lower()
            if any(n in desc for n in needles):
                return True
        return False
    return False


def sample_snapshots(path: Path, n: int, filter_mode: str, seed: int) -> list:
    rng = random.Random(seed)
    pool = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("outcome") != "success":
                continue
            cg = parse_current_game(r.get("prompt") or "")
            if not cg:
                continue
            if filter_mode == "oscillation" and not looks_like_oscillation(cg):
                continue
            if filter_mode == "midgame":
                fd = (cg.get("metrics") or {}).get("faceDownTotal", 0) or 0
                if not (10 <= fd <= 18):
                    continue
            pool.append((r.get("id"), r.get("sessionId"), r.get("turnIndex"), cg))

    if not pool:
        return []
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/store/interactions.jsonl")
    parser.add_argument("--out-dir", default="data/dataset/demos/prompt_format_compare")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--filter",
        choices=("any", "oscillation", "midgame"),
        default="any",
        help="any = uniform sample; oscillation = turns where reversal was offered; "
        "midgame = 10-18 face-down cards remaining",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.interactions)
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}")

    samples = sample_snapshots(in_path, args.n, args.filter, args.seed)
    if not samples:
        raise SystemExit(f"no samples matched filter={args.filter}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"## Prompt format comparison — {len(samples)} samples (filter={args.filter})")
    print(f"out: {out_dir}\n")
    print(f"{'sample':<8}{'session':<10}{'turn':>5}{'orig_chars':>12}"
          f"{'hybrid_chars':>14}{'delta':>9}{'ratio':>9}")

    summary_rows = []
    for i, (iid, sid, turn, cg) in enumerate(samples):
        sub = out_dir / f"sample-{i:02d}"
        sub.mkdir(exist_ok=True)
        original = render_original(cg)
        hybrid = render_hybrid(cg)
        (sub / "original.txt").write_text(original)
        (sub / "hybrid.txt").write_text(hybrid)
        (sub / "meta.json").write_text(
            json.dumps({"interactionId": iid, "sessionId": sid, "turnIndex": turn}, indent=2)
        )
        o_chars, h_chars = len(original), len(hybrid)
        delta = h_chars - o_chars
        ratio = h_chars / o_chars if o_chars else 0
        summary_rows.append((i, sid, turn, o_chars, h_chars, delta, ratio))
        sid_short = (sid or "")[:8]
        print(f"{i:<8}{sid_short:<10}{turn:>5}{o_chars:>12}"
              f"{h_chars:>14}{delta:>+9}{ratio:>9.2f}x")

    # markdown summary
    md = ["# Prompt format comparison", ""]
    md.append(f"Samples: {len(samples)}  ·  filter: {args.filter}  ·  seed: {args.seed}")
    md.append("")
    md.append("| sample | session | turn | original chars | hybrid chars | delta | ratio |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for i, sid, turn, oc, hc, d, r in summary_rows:
        md.append(f"| {i} | `{(sid or '')[:8]}` | {turn} | {oc} | {hc} | {d:+d} | {r:.2f}x |")
    md.append("")
    md.append("Open `sample-NN/original.txt` and `sample-NN/hybrid.txt` side-by-side "
              "to compare. Token estimate = chars / 4.")
    (out_dir / "summary.md").write_text("\n".join(md))
    print(f"\nwrote -> {out_dir}/summary.md")


if __name__ == "__main__":
    main()
