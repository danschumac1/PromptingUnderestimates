#!/usr/bin/env python
"""
Build SensorLLM Stage1 artifacts (TRAIN + TEST only).

Reads:
  load_train_test(<samples_dir>, split_id)

Writes (always):
  <out_root>/<dataset>/train_data_stage1.pkl
  <out_root>/<dataset>/train_labels_stage1.pkl
  <out_root>/<dataset>/train_qa_stage1.json

  <out_root>/<dataset>/test_data_stage1.pkl
  <out_root>/<dataset>/test_labels_stage1.pkl
  <out_root>/<dataset>/test_qa_stage1.json

Notes:
- No "full" outputs.
- Uses LEGEND_MAPPINGS for per-channel QA when available.
"""

import os
import json
import pickle
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np

import sys
sys.path.append("./src")

from utils.constants import LABEL_MAPPING, LEGEND_MAPPINGS
from utils.loaders import load_train_test, Split
from utils.sllm_utils import (
    QA_summary,
    analyze_trend,
    dscb_simple_trend,
    merge_adjacent_rows,
    select_random_pair,
)


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        required=True,
    )
    p.add_argument(
        "--samples_root",
        type=str,
        default="./data/samples",
        help="Root containing ./<dataset>/ splits",
    )
    p.add_argument(
        "--split_id",
        type=int,
        default=0,
        help="Which split id to read from load_train_test(samples_dir, split_id). Usually 0.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/raid/hdd249/Classification_v2/data/SLLM",
        help="Root for outputs: ./<dataset>/...",
    )
    p.add_argument("--sr", type=int, default=50, help="Sample rate for trend analysis")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_write_pkl(path: str, obj: Any, overwrite: bool) -> None:
    _ensure_dir(os.path.dirname(path))
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Wrote:", path)


def _safe_write_json(path: str, obj: Any, overwrite: bool) -> None:
    _ensure_dir(os.path.dirname(path))
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("Wrote:", path)


# -----------------------------------------------------------------------------
# Stage1 formatting
# -----------------------------------------------------------------------------
def split_to_stage1_lists(
    split: Split,
    activity_map: Dict[int, str],
    subject: str = "NA",
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Returns:
      segments: List[np.ndarray] where each element is shape (T, D)
      labels:   List[dict] same length as segments
    """
    segments: List[np.ndarray] = []
    labels: List[dict] = []

    for i in range(len(split)):
        X_i = split.X[i]          # (T, D) numpy array
        y_i = int(split.y[i])     # scalar int label
        idx_i = int(split.idx[i]) # original index (traceability)

        if not isinstance(X_i, np.ndarray) or X_i.ndim != 2:
            raise ValueError(
                f"Expected X_i as 2D np.ndarray, got {type(X_i)} shape={getattr(X_i,'shape',None)}"
            )

        segments.append(X_i)

        labels.append(
            {
                "subject": subject,
                "activity_name": activity_map[y_i],
                "activity": y_i,                    # 0-based id
                "segments": [0, int(X_i.shape[0])], # full window
                "idx": idx_i,                       # traceability
            }
        )

    return segments, labels


def build_qa_json(dataset: str, segments: List[np.ndarray], sr: int = 50) -> Dict[str, Any]:
    """
    segments: List[np.ndarray], each shape (T, D)
    """
    legend = LEGEND_MAPPINGS.get(dataset.upper())
    qa_dict: Dict[str, Any] = {"author": "", "version": "", "date": "...", "dataset": []}

    for i, d in enumerate(segments):
        if not isinstance(d, np.ndarray) or d.ndim != 2:
            raise ValueError(f"Expected 2D array (T,D), got {type(d)} shape={getattr(d,'shape',None)}")

        T, D = d.shape

        # Decide channels to analyze
        if legend is None:
            reading_names = ["ts"]
            reading_list = [d.mean(axis=1)]
        else:
            if len(legend) != D:
                raise ValueError(f"Legend length {len(legend)} != D={D} for dataset={dataset}")
            reading_names = list(legend)
            reading_list = [d[:, j] for j in range(D)]

        row: Dict[str, Any] = {
            "index": i,
            "summaries": {},
            "qa_pairs": {name: [] for name in reading_names},
        }

        for name, r in zip(reading_names, reading_list):
            sensor_name = f"normalized {name}"

            t_df = analyze_trend(r, sr)
            trend_df = merge_adjacent_rows(t_df)
            trend_pair_list = select_random_pair()

            row["summaries"][name] = QA_summary(
                r,
                trend_df,
                sensor_name,
                trend_pair_list,
                whether_gpt=False,
                model_type="3.5",
            )

            row["qa_pairs"][name].append(
                dscb_simple_trend(
                    trend_df,
                    sensor_name,
                    trend_pair_list,
                    whether_gpt=False,
                    model_type="4",
                )
            )

        qa_dict["dataset"].append(row)

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(segments)} processed")

    return qa_dict


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    dataset = args.dataset

    samples_dir = os.path.join(args.samples_root, dataset)
    out_dir = os.path.join(args.out_root, dataset)
    _ensure_dir(out_dir)

    train, test = load_train_test(samples_dir, args.split_id)
    activity_map = LABEL_MAPPING[dataset.upper()]

    train_segments, train_labels = split_to_stage1_lists(train, activity_map, subject="NA")
    test_segments, test_labels = split_to_stage1_lists(test, activity_map, subject="NA")

    # Sanity checks
    assert len(train_segments) == len(train_labels)
    assert len(test_segments) == len(test_labels)

    print("TRAIN:", len(train_segments), "TEST:", len(test_segments))
    print("Example TRAIN label:", train_labels[0])
    print("Example TRAIN segment shape:", train_segments[0].shape)

    # ---- Write TRAIN stage1 ----
    train_data_pkl = os.path.join(out_dir, "train_data_stage1.pkl")
    train_labels_pkl = os.path.join(out_dir, "train_labels_stage1.pkl")
    train_qa_json = os.path.join(out_dir, "train_qa_stage1.json")

    _safe_write_pkl(train_data_pkl, train_segments, overwrite=args.overwrite)
    _safe_write_pkl(train_labels_pkl, train_labels, overwrite=args.overwrite)
    _safe_write_json(train_qa_json, build_qa_json(dataset, train_segments, sr=args.sr), overwrite=args.overwrite)

    # ---- Write TEST stage1 ----
    test_data_pkl = os.path.join(out_dir, "test_data_stage1.pkl")
    test_labels_pkl = os.path.join(out_dir, "test_labels_stage1.pkl")
    test_qa_json = os.path.join(out_dir, "test_qa_stage1.json")

    _safe_write_pkl(test_data_pkl, test_segments, overwrite=args.overwrite)
    _safe_write_pkl(test_labels_pkl, test_labels, overwrite=args.overwrite)
    _safe_write_json(test_qa_json, build_qa_json(dataset, test_segments, sr=args.sr), overwrite=args.overwrite)

    print("\nDone.")
    print("TRAIN stage1:")
    print(" ", train_data_pkl)
    print(" ", train_labels_pkl)
    print(" ", train_qa_json)
    print("TEST stage1:")
    print(" ", test_data_pkl)
    print(" ", test_labels_pkl)
    print(" ", test_qa_json)


if __name__ == "__main__":
    main()
