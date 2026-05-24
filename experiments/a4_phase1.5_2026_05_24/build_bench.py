"""Build N=20 bench from post-cutover (de7dc06 / 0462323c... template) sessions.

Composition (rebalanced from plan — endgame is empty in post-cutover data
because both seed-2114734045 and seed-2325626768 games stalled at ≤21% progress):

  5 early       (face-down ≥ 18; mixed foundation availability)
  8 midgame     (face-down 10–17; mixed foundation availability)
  0 endgame     (not available in post-cutover data)
  7 oscillation (high draw-ratio in RECENT MOVES OR tableau reversal; 4 of 7 with
                 tableau_to_foundation move available — H2 test ground)

Output: bench.json with full prompt + extracted metadata per state.

A state is identified by its (sessionId, turnIndex). We de-duplicate by requiring
at least 4 turns between any two selected states from the same session, so the
bench doesn't collapse to a few near-duplicate boards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

POST_CUTOVER_FILES = [
    "/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-359e1f-1779581014927.json",
    "/Users/chayut/repos/solitaire-analytics/data/raw/solitaire-ai-log-6459d9-1779581010470.json",
]

EXP_DIR = Path(__file__).parent
OUT = EXP_DIR / "bench.json"


def parse_state(interaction: dict) -> dict | None:
    p = interaction.get("prompt")
    if not p or interaction.get("outcome") != "success":
        return None
    fd_m = re.search(r"face-down remaining=(\d+)", p)
    prog_m = re.search(r"completion=(\d+)%", p)
    if not (fd_m and prog_m):
        return None
    fd = int(fd_m.group(1))
    prog = int(prog_m.group(1))
    # RECENT MOVES data section (NOT the format-doc mention)
    rm_match = re.search(r"RECENT MOVES \(oldest[^\n]*\n((?:\s+\d+\.[^\n]+\n)+)", p)
    n_recent = n_unprod = 0
    reversal = False
    if rm_match:
        body = rm_match.group(1)
        lines = [l for l in body.split("\n") if l.strip()]
        n_recent = len(lines)
        n_unprod = sum(
            1 for l in lines if re.search(r"^\s*\d+\.\s*(draw|recycle)\b", l, re.IGNORECASE)
        )
        swaps = re.findall(r"col\s+(\d+)\s*->\s*col\s+(\d+)", body, re.IGNORECASE)
        seen: set[tuple[str, str]] = set()
        for a, b in swaps[-8:]:
            if (b, a) in seen:
                reversal = True
                break
            seen.add((a, b))
    stall = n_unprod / n_recent if n_recent else 0.0
    return {
        "iid": interaction["id"],
        "session_id": interaction["sessionId"],
        "session_tail": interaction["sessionId"][-12:],
        "turn": interaction["turnIndex"],
        "fd": fd,
        "prog": prog,
        "n_recent": n_recent,
        "stall_ratio": stall,
        "reversal": reversal,
        "has_foundation": "tableau_to_foundation" in p or "discard_to_foundation" in p,
        "prompt": p,
    }


def classify(s: dict) -> str:
    is_osc = (s["n_recent"] >= 6 and s["stall_ratio"] >= 0.7) or s["reversal"]
    if is_osc:
        return "oscillation"
    if s["fd"] >= 18:
        return "early"
    if s["fd"] >= 10:
        return "midgame"
    return "endgame"


def select_with_spacing(candidates: list[dict], n: int, min_gap: int = 4) -> list[dict]:
    """Pick n states with at least min_gap turns between any two from the same session.

    Greedy by turn order. Returns up to n states.
    """
    picked: list[dict] = []
    by_session: dict[str, list[int]] = {}
    for c in sorted(candidates, key=lambda r: (r["session_tail"], r["turn"])):
        sess = c["session_tail"]
        existing = by_session.get(sess, [])
        if any(abs(t - c["turn"]) < min_gap for t in existing):
            continue
        picked.append(c)
        by_session.setdefault(sess, []).append(c["turn"])
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    all_states: list[dict] = []
    for f in POST_CUTOVER_FILES:
        d = json.loads(Path(f).read_text())
        for i in d["interactions"]:
            s = parse_state(i)
            if s:
                all_states.append(s)

    by_cat: dict[str, list[dict]] = {"early": [], "midgame": [], "endgame": [], "oscillation": []}
    for s in all_states:
        by_cat[classify(s)].append(s)

    # Quotas
    quotas = {"early": 5, "midgame": 8, "endgame": 0, "oscillation": 7}

    # For oscillation, prioritize the 4 with-foundation states (H2 test)
    osc_with_found = [s for s in by_cat["oscillation"] if s["has_foundation"]]
    osc_without = [s for s in by_cat["oscillation"] if not s["has_foundation"]]
    osc_pick = select_with_spacing(osc_with_found, 4) + select_with_spacing(osc_without, 3)
    # Trim if oversubscribed
    osc_pick = osc_pick[: quotas["oscillation"]]

    # Other categories: mix of foundation-available and not, with spacing
    selected: dict[str, list[dict]] = {}
    selected["oscillation"] = osc_pick
    for cat in ("early", "midgame"):
        cands = by_cat[cat]
        # Pick spaced sample
        selected[cat] = select_with_spacing(cands, quotas[cat])

    states: list[dict] = []
    for cat in ("early", "midgame", "endgame", "oscillation"):
        for s in selected.get(cat, []):
            state_id = f"{cat}-{s['iid'][-12:]}"
            states.append(
                {
                    "state_id": state_id,
                    "category": cat,
                    "session_id": s["session_id"],
                    "turn": s["turn"],
                    "face_down": s["fd"],
                    "progress_pct": s["prog"],
                    "has_foundation_move": s["has_foundation"],
                    "stall_ratio": round(s["stall_ratio"], 2),
                    "reversal": s["reversal"],
                    "full_prompt": s["prompt"],
                }
            )

    bench = {
        "version": "a4_phase1.5_v1",
        "built_at": "2026-05-24",
        "source_files": POST_CUTOVER_FILES,
        "prompt_template_hash": "0462323c366204b491790a90930fffa2916117b4bb210f390b26c33ddd0cdb9c",
        "quotas": quotas,
        "n_states": len(states),
        "states": states,
    }
    OUT.write_text(json.dumps(bench, indent=2))
    print(f"wrote {OUT}")
    print(f"composition:")
    for cat in ("early", "midgame", "endgame", "oscillation"):
        cat_states = [s for s in states if s["category"] == cat]
        n_found = sum(1 for s in cat_states if s["has_foundation_move"])
        print(f"  {cat:12s} {len(cat_states)} (with foundation: {n_found})")
    print(f"total: {len(states)} states")


if __name__ == "__main__":
    main()
