#!/usr/bin/env python
"""
Build SensorLLM Stage2 artifacts from TRAIN/TEST Stage1 artifacts (no "full").

Reads (required):
  <root>/<dataset>/train_data_stage1.pkl
  <root>/<dataset>/train_labels_stage1.pkl
  <root>/<dataset>/train_qa_stage1.json

  <root>/<dataset>/test_data_stage1.pkl
  <root>/<dataset>/test_labels_stage1.pkl
  <root>/<dataset>/test_qa_stage1.json

Writes:
  <root>/<dataset>/train_data_stage2.pkl
  <root>/<dataset>/train_qa_stage2.json          (for stage2 CasualLM)
  <root>/<dataset>/train_qa_cls_stage2.json      (for stage2 SequenceClassification)

  <root>/<dataset>/test_data_stage2.pkl
  <root>/<dataset>/test_qa_stage2.json
  <root>/<dataset>/test_qa_cls_stage2.json

  <root>/<dataset>/label2id.json
  <root>/<dataset>/id2label.json
"""

import os
import json
import pickle
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import sys
sys.path.append("./src")

from utils.constants import LABEL_MAPPING, LEGEND_MAPPINGS


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["ctu", "emg", "had", "har", "rwc", "tee"], required=True)
    p.add_argument(
        "--root",
        type=str,
        default="/raid/hdd249/Classification_v2/data/SLLM",
        help="Root folder that contains ./<dataset>/train/test stage1 files",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing stage2 outputs")
    return p.parse_args()


# -----------------------------
# IO helpers
# -----------------------------
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: Any, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("Wrote:", path)


def _read_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _write_pkl(path: str, obj: Any, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Wrote:", path)


# -----------------------------
# Formatting helpers
# -----------------------------
def build_options_block(id_to_name: Dict[int, str]) -> str:
    # Stage2 regex expects lines like: "<num>. <Words>."
    lines = [f"{i+1}. {id_to_name[i]}." for i in sorted(id_to_name.keys())]
    return "\n".join(lines)


def safe_get_text(maybe_dict_or_str: Any, fallback: str = "") -> str:
    # Stage1 QA_summary/trend entries are often dicts like {"Q":..., "A":...}
    if isinstance(maybe_dict_or_str, dict):
        return str(maybe_dict_or_str.get("A", fallback))
    if isinstance(maybe_dict_or_str, str):
        return maybe_dict_or_str
    return fallback


def combine_channel_texts(per_channel: Dict[str, Any], legend: Optional[List[str]]) -> str:
    if legend is None:
        return safe_get_text(per_channel.get("ts", ""), "")
    parts: List[str] = []
    for name in legend:
        if name in per_channel:
            parts.append(f"{name}: {safe_get_text(per_channel[name], '')}".strip())
        else:
            parts.append(f"{name}: ")
    return "\n".join(parts).strip()


def validate_segments(segments: List[np.ndarray], legend: Optional[List[str]]) -> Tuple[int, int]:
    if len(segments) == 0:
        raise ValueError("No segments found.")
    first = segments[0]
    if not isinstance(first, np.ndarray) or first.ndim != 2:
        raise ValueError(f"Expected segment as 2D np.ndarray, got {type(first)} shape={getattr(first,'shape',None)}")
    T, D = first.shape
    for j, seg in enumerate(segments[:10]):
        if not isinstance(seg, np.ndarray) or seg.ndim != 2:
            raise ValueError(f"Bad segment at {j}: {type(seg)} shape={getattr(seg,'shape',None)}")
        if seg.shape[1] != D:
            raise ValueError(f"Inconsistent D at segment {j}: {seg.shape} vs (*,{D})")
    if legend is not None and len(legend) != D:
        raise ValueError(f"Legend length {len(legend)} != D={D}")
    return T, D


# -----------------------------
# Stage2 builders
# -----------------------------
def build_stage2_text_qa(stage1_labels: List[dict], id_to_name: Dict[int, str]) -> List[dict]:
    options_block = build_options_block(id_to_name)

    out: List[dict] = []
    for i, lab in enumerate(stage1_labels):
        activity_id = int(lab["activity"])          # 0-based
        activity_name = str(lab["activity_name"])

        final_line = f"{activity_id + 1}. {activity_name}."
        cot = "I will compare overall temporal patterns and channel dynamics to match the correct activity."

        q = (
            "You are given a multichannel sensor time-series window.\n"
            "Choose the correct activity from the options below:\n\n"
            f"{options_block}\n\n"
            "Explain briefly, then provide the final answer as one option line exactly."
        )

        a = f"{cot}\n\n{final_line}"
        out.append({"index": i, "qa_pair": {"Q": q, "A": a}})

    return out


def build_stage2_cls_qa(stage1_qa: Dict[str, Any], stage1_labels: List[dict], legend: Optional[List[str]]) -> List[dict]:
    rows = stage1_qa.get("dataset", [])
    if len(rows) != len(stage1_labels):
        raise ValueError(f"Stage1 QA rows ({len(rows)}) != Stage1 labels ({len(stage1_labels)})")

    out: List[dict] = []
    for i, (row, lab) in enumerate(zip(rows, stage1_labels)):
        summaries = row.get("summaries", {})
        qa_pairs = row.get("qa_pairs", {})

        smry_text = combine_channel_texts(summaries, legend)

        trend_per_channel: Dict[str, Any] = {}
        if isinstance(qa_pairs, dict):
            for ch, lst in qa_pairs.items():
                if isinstance(lst, list) and len(lst) > 0:
                    trend_per_channel[ch] = lst[0]
        trend_text = combine_channel_texts(trend_per_channel, legend)

        activity_name = str(lab["activity_name"])
        q = "Predict the activity label for this window. Return only the activity name."

        qa_pair = {
            "Q": q,
            "smry": smry_text,
            "trend_text": trend_text,
            "corr_text": "",
            "info_text": "",
            "A": activity_name,
        }
        out.append({"index": i, "qa_pair": qa_pair})

    return out


def build_stage2_for_split(
    *,
    split_name: str,
    out_root: str,
    id_to_name: Dict[int, str],
    legend: Optional[List[str]],
    overwrite: bool,
) -> None:
    """
    split_name in {"train", "test"}
    """
    data_stage1_pkl = os.path.join(out_root, f"{split_name}_data_stage1.pkl")
    labels_stage1_pkl = os.path.join(out_root, f"{split_name}_labels_stage1.pkl")
    qa_stage1_json = os.path.join(out_root, f"{split_name}_qa_stage1.json")

    for p in [data_stage1_pkl, labels_stage1_pkl, qa_stage1_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required stage1 file: {p}")

    segments = _read_pkl(data_stage1_pkl)
    labels = _read_pkl(labels_stage1_pkl)
    qa_stage1 = _read_json(qa_stage1_json)

    if not isinstance(segments, list):
        raise TypeError(f"Expected {split_name} segments to be a list of np.ndarray.")
    if not isinstance(labels, list):
        raise TypeError(f"Expected {split_name} labels to be a list of dicts.")
    if len(segments) != len(labels):
        raise ValueError(f"{split_name}: segments != labels: {len(segments)} vs {len(labels)}")

    validate_segments(segments, legend)

    # outputs
    data_stage2_pkl = os.path.join(out_root, f"{split_name}_data_stage2.pkl")
    qa_stage2_json = os.path.join(out_root, f"{split_name}_qa_stage2.json")
    qa_cls_stage2_json = os.path.join(out_root, f"{split_name}_qa_cls_stage2.json")

    # Stage2 loader converts list -> np.array(dtype=float64), so ensure float64 now.
    segments64 = [np.asarray(x, dtype=np.float64) for x in segments]
    _write_pkl(data_stage2_pkl, segments64, overwrite=overwrite)

    # CasualLM stage2
    text_rows = build_stage2_text_qa(labels, id_to_name)
    _write_json(qa_stage2_json, {"author": "", "version": "", "date": "...", "dataset": text_rows}, overwrite=overwrite)

    # SequenceClassification stage2
    cls_rows = build_stage2_cls_qa(qa_stage1, labels, legend)
    _write_json(
        qa_cls_stage2_json,
        {"author": "", "version": "", "date": "...", "dataset": cls_rows},
        overwrite=overwrite,
    )

    # sanity
    assert len(segments64) == len(text_rows)
    assert len(segments64) == len(cls_rows)

    print(f"\n{split_name.upper()} Stage2 done.")
    print(" ", data_stage2_pkl)
    print(" ", qa_stage2_json)
    print(" ", qa_cls_stage2_json)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    dataset = args.dataset
    out_root = os.path.join(args.root, dataset)

    id_to_name = LABEL_MAPPING[dataset.upper()]  # {int: str}
    legend = LEGEND_MAPPINGS.get(dataset.upper())

    # label maps (used by CLS stage2)
    label2id = {str(name): int(i) for i, name in id_to_name.items()}
    id2label = {str(i): str(name) for i, name in id_to_name.items()}

    _write_json(os.path.join(out_root, "label2id.json"), label2id, overwrite=args.overwrite)
    _write_json(os.path.join(out_root, "id2label.json"), id2label, overwrite=args.overwrite)

    # build both splits
    build_stage2_for_split(
        split_name="train",
        out_root=out_root,
        id_to_name=id_to_name,
        legend=legend,
        overwrite=args.overwrite,
    )
    build_stage2_for_split(
        split_name="test",
        out_root=out_root,
        id_to_name=id_to_name,
        legend=legend,
        overwrite=args.overwrite,
    )

    print("\nAll done.")
    print("Label maps:")
    print(" ", os.path.join(out_root, "label2id.json"))
    print(" ", os.path.join(out_root, "id2label.json"))


if __name__ == "__main__":
    main()
