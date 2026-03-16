#!/usr/bin/env python3
"""
Pass@k generation checklist.

Examples:
  python ./src/data_management/make_passatk_checklist.py cot0
  python ./src/data_management/make_passk_checklist.py cot1

This checks outputs written by ./src/pass_at_k.py, e.g.
  ./data/sample_generations/together-qwen/har/pass-at-k/vis-ust_cot0_n20.jsonl

We verify (per dataset / embedding_types / n / cot):
  - file exists
  - how many lines are readable JSON
  - which idx are present
  - for each idx, how many attempts are present
  - completeness: expected_rows = expected_examples * n
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Use the same style you used in variant checklist:
# change this if you want to point to your /raid tree
ROOT = Path("/raid/hdd249/data/sample_generations")

MODELS = ["qwen"]  # expand later if needed
DATASETS = ["ctu", "emg", "had", "har", "rwc", "tee"]

# Checklist rows (label -> embedding_types string as written by pass_at_k.py)
EMBEDDINGS = {
    "lets-ust": "lets,ust",
    "vis-ust": "vis,ust",
    "vis-lets-ust": "lets,vis,ust",
}

# Default n values you commonly run; you can add more.
N_VALUES = [20]

# Matches pass_at_k.py output naming:
#   {embedding_types.replace(',','-')}_cot{CoT}_n{n}.jsonl
PASSK_RE = re.compile(r"^(?P<emb>.+)_cot(?P<cot>[01])_n(?P<n>\d+)\.jsonl$")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def scan_passk_file(path: Path) -> Tuple[int, int, Dict[int, Set[int]]]:
    """
    Returns:
      lines_total: total file lines (including bad JSON)
      lines_json: number of successfully parsed JSON objects
      idx_to_attempts: idx -> set(attempt)
    """
    lines_total = 0
    lines_json = 0
    idx_to_attempts: Dict[int, Set[int]] = {}

    if not path.exists():
        return 0, 0, idx_to_attempts

    with path.open("r") as f:
        for line in f:
            lines_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # allow partial trailing line if job died mid-write
                continue

            lines_json += 1
            idx = _safe_int(obj.get("idx"))
            attempt = _safe_int(obj.get("attempt"))

            if idx is None or attempt is None:
                continue
            idx_to_attempts.setdefault(idx, set()).add(attempt)

    return lines_total, lines_json, idx_to_attempts


def summarize(idx_to_attempts: Dict[int, Set[int]], expected_examples: int, n: int) -> Tuple[int, int, int, int]:
    """
    Returns:
      present_idxs: number of distinct idx observed
      complete_idxs: number of idx that have all attempts 0..n-1
      rows_present: sum over idx of attempts present (unique)
      rows_expected: expected_examples * n
    """
    rows_present = sum(len(s) for s in idx_to_attempts.values())
    rows_expected = expected_examples * n

    present_idxs = len(idx_to_attempts)
    complete_idxs = 0
    needed = set(range(n))
    for idx, atts in idx_to_attempts.items():
        if atts.issuperset(needed):
            complete_idxs += 1

    return present_idxs, complete_idxs, rows_present, rows_expected


def main(cot_arg: str) -> None:
    if cot_arg not in {"cot0", "cot1"}:
        raise ValueError("cot must be cot0 or cot1")

    cot_int = 0 if cot_arg == "cot0" else 1

    for model in MODELS:
        print(model.upper())

        for dataset in DATASETS:
            print(f"  {dataset}")

            base = ROOT / f"together-{model}" / dataset / "pass-at-k"
            # Your pass_at_k.py truncates test to 500 max.
            # This checklist assumes expected_examples=500.
            # If you ever change that in pass_at_k.py, update here too.
            expected_examples = 500

            for label, emb_str in EMBEDDINGS.items():
                emb_file_prefix = emb_str.replace(",", "-")

                for n in N_VALUES:
                    out_file = base / f"{emb_file_prefix}_cot{cot_int}_n{n}.jsonl"

                    if not out_file.exists():
                        print(f"    {label:<14} n={n:<3} ✗  (missing)  {out_file.name}")
                        continue

                    lines_total, lines_json, idx_to_attempts = scan_passk_file(out_file)
                    present_idxs, complete_idxs, rows_present, rows_expected = summarize(
                        idx_to_attempts, expected_examples=expected_examples, n=n
                    )

                    # A run is "complete" if we have exactly expected rows.
                    # (We allow extra lines? Usually no, but use >= to be robust.)
                    complete = (rows_present >= rows_expected) and (complete_idxs >= expected_examples)
                    status = "✓" if complete else "✗"

                    # Show: idx coverage, fully-complete idx count, row coverage, json lines, total lines
                    print(
                        f"    {label:<14} n={n:<3} {status}  "
                        f"idx {present_idxs:>3}/{expected_examples} | "
                        f"full {complete_idxs:>3}/{expected_examples} | "
                        f"rows {rows_present:>6}/{rows_expected:<6} | "
                        f"json_lines {lines_json:>6} | lines {lines_total:>6}"
                    )

            print()

        print()

    print("Legend: ✓ = all idx present AND each idx has all attempts 0..n-1")
    print(f"Mode checked: {cot_arg}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: must pass cot0 or cot1")
        sys.exit(1)
    main(sys.argv[1])
