#!/usr/bin/env python3
"""Integrity gate for the Parquet sharding migration.

Compares each HF config's Parquet shards against a reference, by interaction id,
field for field, after normalising away the typed-empty (``[]``/``{}``) vs null
differences the two formats represent differently. Exits non-zero on any drift.

Usage:
    # local shards vs the prior monolithic JSONL (pre-push gate)
    python scripts/validate_shards.py --ref jsonl

    # hub shards vs local shards (post-push gate)
    python scripts/validate_shards.py --ref local --src hub
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUB = REPO / "data" / "publish"
REPO_ID = "chayuto/klondike-llm-decisions"
CONFIGS = [
    "client_v1_full_corpus_raw",
    "client_v1_teacher_clean_raw",
    "client_v1_teacher_clean_lean",
    "client_v1_26b_raw",
    "client_v1_26b_lean",
]


def clean(v):
    """Canonical form: drop None and empty containers recursively so a source
    row (missing keys) and a null-filled Parquet row reduce to the same thing."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            cx = clean(x)
            if cx is not None and cx != {} and cx != []:
                out[k] = cx
        return out
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def rows_from_jsonl(name: str) -> dict[str, dict]:
    p = PUB / f"{name}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def rows_from_parquet_local(name: str) -> dict[str, dict]:
    import pyarrow.parquet as pq
    out = {}
    for part in sorted((PUB / name).glob("*.parquet")):
        for r in pq.read_table(part).to_pylist():
            out[r["id"]] = r
    return out


def rows_from_hub(name: str) -> dict[str, dict]:
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, name, split="train",
                      download_mode="force_redownload")
    return {r["id"]: r for r in ds}


def compare(name: str, ref: dict[str, dict], got: dict[str, dict]) -> int:
    problems = 0
    if not ref:
        print(f"  {name}: no reference rows (skipped)")
        return 0
    if set(ref) != set(got):
        only_ref = len(set(ref) - set(got))
        only_got = len(set(got) - set(ref))
        print(f"  {name}: ID SET MISMATCH ref={len(ref)} got={len(got)} "
              f"(only_ref={only_ref} only_got={only_got})")
        problems += 1
    field_mismatch = 0
    samples = []
    for iid in ref:
        if iid not in got:
            continue
        a, b = clean(ref[iid]), clean(got[iid])
        if a != b:
            field_mismatch += 1
            if len(samples) < 3:
                diff = sorted(set(a) ^ set(b)) or [
                    k for k in set(a) & set(b) if a[k] != b[k]]
                samples.append(f"id …{iid[-6:]} keys={diff[:5]}")
    if field_mismatch:
        print(f"  {name}: {field_mismatch} row(s) differ field-for-field; "
              f"examples: {samples}")
        problems += 1
    if not problems:
        print(f"  {name}: OK ({len(got)} rows, id-set + every field match)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", choices=["jsonl", "local"], default="jsonl",
                    help="reference: prior monolithic jsonl, or local parquet")
    ap.add_argument("--src", choices=["local", "hub"], default="local",
                    help="what to validate: local parquet, or the Hub")
    args = ap.parse_args()

    ref_fn = {"jsonl": rows_from_jsonl, "local": rows_from_parquet_local}[args.ref]
    got_fn = {"local": rows_from_parquet_local, "hub": rows_from_hub}[args.src]

    print(f"validate: ref={args.ref}  src={args.src}")
    total = 0
    for name in CONFIGS:
        total += compare(name, ref_fn(name), got_fn(name))
    if total:
        print(f"\nINTEGRITY FAILED: {total} problem(s).")
        return 1
    print("\nINTEGRITY OK: every config matches by id and by field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
