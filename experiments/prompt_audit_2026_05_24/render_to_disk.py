"""Pre-render all (arm, state) prompts to disk so subagents can consume them.

Writes:
  prompts/<arm>/<state_id>/prompt.txt   (just the prompt body, ready to feed)
  prompts/<arm>/<state_id>/meta.json    (state_id, category, arm, hashes)
  prompts/manifest.jsonl                (one line per prompt; used by the runner)

Note: prompt content is identical across runs (no randomness in the prompt itself);
randomness comes from the model's sampling. So we render ONCE per (arm, state),
not per run. The runner spawns M subagents against the same prompt file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
from render_arms import ARMS, render  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=str(EXP_DIR / "bench.json"))
    parser.add_argument("--out", default=str(EXP_DIR / "prompts"))
    args = parser.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    rows = []
    for state in bench["states"]:
        for arm in ARMS:
            prompt = render(arm, state["full_prompt"], state["current_game"])
            sub = out_dir / arm / state["state_id"]
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "prompt.txt").write_text(prompt)
            (sub / "meta.json").write_text(json.dumps({
                "arm": arm,
                "state_id": state["state_id"],
                "category": state["category"],
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            }, indent=2))
            rows.append({
                "arm": arm,
                "state_id": state["state_id"],
                "category": state["category"],
                "prompt_path": str(sub / "prompt.txt"),
                "prompt_chars": len(prompt),
            })

    manifest_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} prompts to {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
