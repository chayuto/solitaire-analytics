"""Pre-render all (arm, state) prompts to disk for subagent consumption.

Writes:
  prompts/<arm>/<state_id>/prompt.txt
  prompts/<arm>/<state_id>/meta.json
  prompts/manifest.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
from render_arms import ARMS, render  # noqa: E402


def main() -> None:
    bench = json.loads((EXP_DIR / "bench.json").read_text())
    out_dir = EXP_DIR / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for state in bench["states"]:
        for arm in ARMS:
            prompt = render(arm, state["full_prompt"])
            sub = out_dir / arm / state["state_id"]
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "prompt.txt").write_text(prompt)
            (sub / "meta.json").write_text(json.dumps({
                "arm": arm,
                "state_id": state["state_id"],
                "category": state["category"],
                "has_foundation_move": state["has_foundation_move"],
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            }, indent=2))
            manifest_rows.append({
                "arm": arm,
                "state_id": state["state_id"],
                "category": state["category"],
                "prompt_path": str(sub / "prompt.txt"),
                "prompt_chars": len(prompt),
            })
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n"
    )
    print(f"wrote {len(manifest_rows)} prompts to {out_dir}")


if __name__ == "__main__":
    main()
