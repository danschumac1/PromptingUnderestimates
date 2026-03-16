"""
python ./src/data_management/_patch_data_for_instruct_time.py --dataset har

Top-to-bottom, end-to-end script:
- Loads ./data/samples/{dataset} via load_train_test
- Builds samples_{train,test}.pkl under ./data/instruct_time/{dataset}/
- Each sample is a tuple: (prompt, ts(T,C) float32, y_id float)
- DOES NOT leak the answer into the prompt
- Optional: drop NaNs, enforce (T,C) orientation, and optionally enforce a fixed (T,C) shape
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append("./src")
from utils.loaders import load_train_test


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        required=True,
        help="Dataset name (must match ./data/samples/{dataset})",
    )
    # data hygiene / shape controls
    p.add_argument("--drop_nans", action="store_true", help="Drop samples that contain any NaN")
    p.add_argument("--force_tc", action="store_true", help="Ensure time series is oriented as (T,C)")
    p.add_argument("--T", type=int, default=None, help="Optional expected T (time length). If set, enforce it.")
    p.add_argument("--C", type=int, default=None, help="Optional expected C (channels). If set, enforce it.")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def to_2d_ts(x: np.ndarray) -> np.ndarray:
    """
    Convert common shapes to 2D:
      (1, T, C) -> (T, C)
      (T,)      -> (T, 1)
    Leaves (T, C) as-is.
    """
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 1:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(f"Expected 2D (T,C) or convertible input, got shape={x.shape}")

    return x


def ensure_tc(
    ts: np.ndarray,
    *,
    force_tc: bool,
    T_expected: Optional[int] = None,
    C_expected: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Make best-effort to ensure (T,C) and optionally enforce exact (T,C).
    Returns None if it cannot satisfy constraints.
    """
    ts = to_2d_ts(ts).astype(np.float32)

    # If we have both expected dims, we can disambiguate orientation.
    if force_tc and (T_expected is not None) and (C_expected is not None):
        if ts.shape == (C_expected, T_expected):
            ts = ts.T  # (C,T) -> (T,C)

    # If enforcing exact dims, check them
    if (T_expected is not None) and (C_expected is not None):
        if ts.shape != (T_expected, C_expected):
            return None

    return ts


def _lookup_label(label_maps: dict, y_id: int) -> Tuple[str, str]:
    """
    Handle either int or str keys in label_maps.
    Expects:
      label_maps["id_to_letter"], label_maps["id_to_name"]
    """
    id_to_letter = label_maps["id_to_letter"]
    id_to_name = label_maps["id_to_name"]

    key = y_id if (y_id in id_to_letter) else str(y_id)
    return id_to_letter[key], id_to_name[key]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        n_shots=0,
    )

    splits = {"train": train, "test": test}
    outdata: Dict[str, List[Tuple[str, np.ndarray, float]]] = {"train": [], "test": []}

    out_dir = f"./data/instruct_time/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    kept = {"train": 0, "test": 0}
    dropped_nan = {"train": 0, "test": 0}
    dropped_shape = {"train": 0, "test": 0}

    for split_name, split in splits.items():
        for row in split:
            # robust scalar extraction
            y_id = int(np.ravel(row.y)[0])

            # label text (OPTIONAL; do not leak into prompt)
            letter, label_name = _lookup_label(train.label_maps, y_id)

            # ✅ Prompt should be the task/question/options ONLY (no ground-truth answer)
            prompt = train.general_question

            # ts: 2D float32
            ts = ensure_tc(
                row.X,
                force_tc=args.force_tc,
                T_expected=args.T,
                C_expected=args.C,
            )

            if ts is None:
                dropped_shape[split_name] += 1
                continue

            if args.drop_nans and np.isnan(ts).any():
                dropped_nan[split_name] += 1
                continue

            outdata[split_name].append((prompt, ts, float(y_id)))
            kept[split_name] += 1

        out_path = os.path.join(out_dir, f"samples_{split_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(outdata[split_name], f)

        print(f"[{split_name}] saved: {out_path}")
        print(
            f"[{split_name}] kept={kept[split_name]} "
            f"dropped_nan={dropped_nan[split_name]} "
            f"dropped_shape={dropped_shape[split_name]}"
        )

    # quick sanity peek
    if outdata["train"]:
        p0, ts0, y0 = outdata["train"][0]
        print("\n[example train item]")
        print("prompt (first 200 chars):", p0[:200].replace("\n", " "))
        print("ts shape:", ts0.shape, "dtype:", ts0.dtype, "label:", y0)


if __name__ == "__main__":
    main()



# from dataclasses import dataclass
# import argparse
# import os
# import pickle
# import sys
# from typing import Dict, List

# import numpy as np

# sys.path.append("./src")
# from utils.loaders import load_train_test


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset", choices=["ctu", "emg", "had", "har", "rwc", "tee"], required=True)
#     return p.parse_args()


# def to_2d_ts(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x)

#     # common cases: (1, T, C) -> (T, C)
#     if x.ndim == 3 and x.shape[0] == 1:
#         x = x[0]

#     # (T,) -> (T, 1)
#     if x.ndim == 1:
#         x = x[:, None]

#     if x.ndim != 2:
#         raise ValueError(f"Expected 2D (T,C), got shape={x.shape}")

#     return x


# def _lookup_label(label_maps: dict, y_id: int):
#     """Handle either int or str keys in label_maps."""
#     id_to_letter = label_maps["id_to_letter"]
#     id_to_name = label_maps["id_to_name"]

#     if y_id in id_to_letter:
#         key = y_id
#     else:
#         key = str(y_id)

#     return id_to_letter[key], id_to_name[key]


# def main():
#     args = parse_args()

#     train, test = load_train_test(
#         f"./data/samples/{args.dataset}",
#         n_shots=0,
#     )

#     data = {"train": train, "test": test}
#     outdata: Dict[str, List[tuple]] = {"train": [], "test": []}

#     out_dir = f"./data/instruct_time/{args.dataset}"
#     os.makedirs(out_dir, exist_ok=True)

#     for split_name, split in data.items():
#         for row in split:
#             y_id = int(np.ravel(row.y)[0])  # robust scalar extraction

#             letter, label_name = _lookup_label(train.label_maps, y_id)

#             prompt = train.general_question + f"\n\nThe answer is [{letter}] {label_name}"
#             ts = to_2d_ts(row.X)

#             outdata[split_name].append(((prompt, ts, float(y_id))))

#         out_path = os.path.join(out_dir, f"samples_{split_name}.pkl")
#         with open(out_path, "wb") as f:
#             pickle.dump(outdata[split_name], f)

#         print(f"Saved to {out_path}")
#     # print(f"EXAMPLE PROMPT: {outdata['train'][0].prompt}")


# if __name__ == "__main__":
#     main()



