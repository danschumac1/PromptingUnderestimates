#!/usr/bin/env python3
"""
Usage:
  python ./src/data_management/make_variant_checklist.py cot0
  python ./src/data_management/make_variant_checklist.py cot1
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/raid/hdd249/data/sample_generations")

MODELS = ["qwen"]  # expand later if needed
DATASETS = ["ctu", "emg", "had", "har", "rwc", "tee"]

# Logical checklist rows → filename suffixes
EMBEDDINGS = {
    "lets-ust": "lets,ust",
    "vis-ust": "vis,ust",
    "vis-lets-ust": "lets,vis,ust",
}

VARIANT_RE = re.compile(
    r"sys(?P<num>\d{3})_gq(?P=num)_(?P<suffix>.+)\.jsonl$"
)

def normalize_suffix(s: str) -> tuple[tuple[str, ...], str]:
    tokens_part, cot_part = s.rsplit("_", 1)
    toks = tuple(sorted(t.strip() for t in tokens_part.split(",") if t.strip()))
    return toks, cot_part

def count_variants_and_lines(dirpath: Path, suffix_base: str, cot: str) -> tuple[int, int]:
    nums = set()
    total_lines = 0

    if not dirpath.exists():
        return 0, 0

    wanted_toks = tuple(sorted(t.strip() for t in suffix_base.split(",") if t.strip()))

    for p in dirpath.iterdir():
        m = VARIANT_RE.match(p.name)
        if not m:
            continue

        file_toks, file_cot = normalize_suffix(m.group("suffix"))
        if file_cot != cot or file_toks != wanted_toks:
            continue

        nums.add(m.group("num"))
        try:
            with p.open("r") as f:
                total_lines += sum(1 for _ in f)
        except OSError:
            pass

    return len(nums), total_lines


def main(cot: str) -> None:
    assert cot in {"cot0", "cot1"}

    for model in MODELS:
        print(model.upper())
        for dataset in DATASETS:
            print(f"  {dataset}")
            base = ROOT / f"together-{model}" / dataset / "prompt_variants"

            for label, suffix_base in EMBEDDINGS.items():
                suffix = f"{suffix_base}_{cot}"
                n, lines = count_variants_and_lines(base, suffix_base, cot)
                status = "✓" if n == 10 else "✗"
                print(f"    {label:<14} {n:>2}/10 {status}  {lines}")
        print()

    print("Legend: ✓ = 10 variants present, ✗ = incomplete")
    print(f"Mode checked: {cot}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: must pass cot0 or cot1")
        sys.exit(1)

    main(sys.argv[1])
