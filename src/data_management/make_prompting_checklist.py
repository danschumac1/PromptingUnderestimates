#!/usr/bin/env python3
"""
Usage:
  python ./src/data_management/make_prompting_checklist.py Direct > check_direct.log
  python ./src/data_management/make_prompting_checklist.py CoT >  check_cot.log

"""

import sys
from pathlib import Path

ROOT = Path("data/sample_generations")

MODELS = ["llama", "mistral", "qwen"]
DATASETS = ["ctu", "emg", "har", "tee", "rwc", "had"]
EMBEDDINGS = [
    "ts-ust",
    "lets-ust",
    "vis-ust",
    "ts-vis-ust",
    "vis-lets-ust",
]


def main(mode: str) -> None:
    assert mode in {"Direct", "CoT"}

    for model in MODELS:
        print(model)
        for dataset in DATASETS:
            print(f"  {dataset}")
            for emb in EMBEDDINGS:
                path = (
                    ROOT
                    / model
                    / dataset
                    / "visual_prompting"
                    / f"{emb}_0-shot_{mode}.jsonl"
                )
                status = "✓" if path.exists() else "✗"
                print(f"    {emb:<14} {status}")
        print()

    print(f"Legend: ✓ = exists, ✗ = missing ({mode})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: must pass Direct or CoT")
        sys.exit(1)

    main(sys.argv[1])
