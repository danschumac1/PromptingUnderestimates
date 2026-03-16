#!/usr/bin/env python3
"""
Usage:
  python ./src/data_management/make_embed_checklist.py
"""

from pathlib import Path
import numpy as np

ROOT = Path("/raid/hdd249/data")  # change if needed

MODELS = [
    "random"
    # "llama", 
    # "mistral", 
    # "qwen"
    ]

DATASETS = [
    # "tee", 
    # "emg", 
    # "ctu", 
    # "har", 
    "had", 
    "rwc"
    ]

EMBED_TYPES = [
    "lets-ust", 
    "vis-ust", 
    "vis-lets-ust"
    ]

MODE = "Direct"  # you can parameterize later if you want
SHOT = "0-shot"


def _npz_n_rows(npz_path: Path) -> tuple[int, str]:
    """
    Returns (n_rows, shape_str). If unreadable, returns (-1, reason).
    Heuristic:
      - if key 'X' exists, use that
      - else if a single array, use it
      - else use first array key in sorted order
    """
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            keys = sorted(z.files)

            if not keys:
                return (-1, "no arrays in npz")

            key = "X" if "X" in z.files else keys[0]
            arr = z[key]
            # typical embeddings: (N, D) or (N, ...)
            n = int(arr.shape[0]) if hasattr(arr, "shape") and len(arr.shape) >= 1 else -1
            return (n, f"key={key} shape={arr.shape} keys={keys}")
    except Exception as e:
        return (-1, f"{type(e).__name__}: {e}")


def main() -> None:
    for model in MODELS:
        print(model)
        for dataset in DATASETS:
            print(f"  {dataset}")
            for emb in EMBED_TYPES:
                base = (
                    ROOT
                    / "sample_features"
                    / model
                    / dataset
                    / f"{emb}_{SHOT}_{MODE}"
                )

                # check both (you can remove train if you only care about test)
                paths = {
                    "train": [base / "train_embeddings.npz", base / "train__embeddings.npz"],
                    "test": [base / "test_embeddings.npz", base / "test__embeddings.npz"],
                }

                for split_name, path_list in paths.items():
                    path = None
                    for p in path_list:
                        if p.exists():
                            path = p
                            break
                    if path is None:
                        status = "✗"
                    else:
                        n, info = _npz_n_rows(path)
                        if n >= 0:
                            status = f"✓ (n={n}) {info}"
                        else:
                            status = f"⚠ unreadable ({info})"

                    print(f"    {emb:<12} {split_name:<5} {status}")
        print()

    print("Legend: ✓ = exists, ✗ = missing, ⚠ = exists but unreadable")


if __name__ == "__main__":
    main()
