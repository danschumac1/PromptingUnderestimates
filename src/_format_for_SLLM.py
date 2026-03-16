# ./src/_format_for_SLLM.py
"""
Convert your cleaned dataset format (train.npz/test.npz + artifacts)
into SensorLLM's expected "data.pkl + qa.json" format.

Writes:
  {out_root}/{sllm_dataset_name}/
    train_data.pkl
    eval_data.pkl
    train_qa.json
    eval_qa.json
    meta.json

How to run:
  python ./src/_format_for_SLLM.py --dataset ctu
  python ./src/_format_for_SLLM.py --dataset har --normalize
  python ./src/_format_for_SLLM.py --dataset had --out_root ./data/sllm

Notes:
- Uses resources/ts_backbone.yaml for:
    - id2label
    - channel_num
    - sample_rate (for meta only)
    - ts start/end tokens (for SensorLLM token utils patching)
- SensorLLM *code patch still required* for datasets not in:
    ["usc-had", "uci", "capture24", "mhealth", "pamap", "pamap50"]
  (see meta.json for the exact tokens/channel keys we export).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import yaml  # pip install pyyaml if needed

from utils.loaders import load_train_test


# ---------------------------------------------------------------------
# SensorLLM dataset-name mapping (optional but convenient)
# ---------------------------------------------------------------------

# Channel keys SensorLLM stage1 expects
CHANNEL_KEYS = {
    "univariate": ["ts"],
    "har": ["x_acc", "y_acc", "z_acc"],
    "had": ["x_acc", "y_acc", "z_acc", "x_g", "y_g", "z_g"],
    # (optional) keep if you still use it elsewhere:
    "uci": ["x_acc", "y_acc", "z_acc"],
}


# The JSON keys used by SensorLLM stage1 code for QA grouping
CHANNEL_FRIENDLY = {
    "x_acc": "x-axis accelerometer",
    "y_acc": "y-axis accelerometer",
    "z_acc": "z-axis accelerometer",
    "x_g": "x-axis gyroscope",
    "y_g": "y-axis gyroscope",
    "z_g": "z-axis gyroscope",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing YAML at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure: expected dict at root in {path}")
    return cfg


def id2label_from_yaml(cfg: Dict[str, Any], dataset: str) -> Dict[int, str]:
    ds = cfg.get(dataset)
    if not isinstance(ds, dict):
        raise ValueError(f"Dataset '{dataset}' not found in YAML config")
    id2label = ds.get("id2label")
    if not isinstance(id2label, dict):
        raise ValueError(f"Dataset '{dataset}' missing 'id2label' in YAML")
    # YAML keys may come in as int already, but normalize defensively
    return {int(k): str(v) for k, v in id2label.items()}


def make_id2letter(id2label: Dict[int, str]) -> Dict[int, str]:
    # A,B,C,... in sorted label-id order
    keys = sorted(id2label.keys())
    return {k: chr(ord("A") + i) for i, k in enumerate(keys)}


def format_options(id2letter: Dict[int, str], id2label: Dict[int, str]) -> str:
    return ", ".join(f"[{id2letter[k]}] {id2label[k]}" for k in sorted(id2label.keys()))


def make_cls_qa(label: int, id2letter: Dict[int, str], id2label: Dict[int, str]) -> Dict[str, Any]:
    """
    Build a SensorLLM-style QA dict.
    stage1 preprocess_time_series2 expects:
      {"Q": "...", "A": "...", "type": ..., "summary": {"A": "..."}}
    """
    options = format_options(id2letter, id2label)

    q = (
        "Classify the time series into one of the following classes:\n"
        f"{options}\n\n"
        'Answer using exactly this format: "The answer is [X] [CLASS_NAME]".'
    )

    letter = id2letter.get(int(label), "?")
    name = id2label.get(int(label), str(label))
    a = f"The answer is [{letter}] {name}"

    return {
        "Q": q,
        "A": a,
        "type": "cls",
        "summary": {"A": f"Summary: predicted class is [{letter}] {name}."},
    }


def split_to_pickle_list(X: np.ndarray) -> List[np.ndarray]:
    """
    SensorLLM expects pickled list-like indexable data_file where each item is array-like (L, C).
    Their code does:
        data = data_file[int(data_idx)]
        torch.from_numpy(data[:, i])
    So we store each sample as (T, D) float64.
    """
    if X.ndim == 2:
        # (N,T) -> (N,T,1)
        X = X[:, :, None]
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim in {{2,3}}; got {X.ndim} with shape {X.shape}")

    out: List[np.ndarray] = []
    for i in range(X.shape[0]):
        out.append(np.asarray(X[i], dtype=np.float64))  # (T,D)
    return out


def dataset_channel_keys(sllm_dataset_name: str, channel_num: int) -> list[str]:
    if sllm_dataset_name in CHANNEL_KEYS:
        keys = CHANNEL_KEYS[sllm_dataset_name]
    elif channel_num == 1:
        keys = CHANNEL_KEYS["univariate"]
    else:
        raise ValueError(
            f"No CHANNEL_KEYS entry for dataset {sllm_dataset_name!r} and channel_num={channel_num}."
        )

    if len(keys) != channel_num:
        raise ValueError(
            f"Channel mismatch: sllm_dataset_name={sllm_dataset_name!r} "
            f"channel_num(from yaml)={channel_num} but keys={keys} (len={len(keys)})."
        )
    return keys


def build_stage1_qa_json(
    X: np.ndarray,
    y: np.ndarray,
    channel_keys: List[str],
    id2letter: Dict[int, str],
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """
    Build SensorLLM stage1 JSON schema:
    {
      "dataset": [
        {
          "index": "0",
          "qa_pairs": {
            "x-axis accelerometer": [ {Q,A,type,summary}, ... ],
            ...
          },
          "summaries": {
            "x-axis accelerometer": {"A": "..."},
            ...
          }
        },
        ...
      ]
    }

    We generate ONE QA per channel per sample (so N * C items after flattening).
    """
    n = int(X.shape[0])
    entries: List[Dict[str, Any]] = []

    for i in range(n):
        qa_pairs: Dict[str, List[Dict[str, Any]]] = {}
        summaries: Dict[str, Dict[str, str]] = {}

        for ck in channel_keys:
            friendly = CHANNEL_FRIENDLY.get(ck, ck)
            qa = make_cls_qa(int(y[i]), id2letter, id2label)
            qa_pairs.setdefault(friendly, []).append(qa)
            summaries[friendly] = {"A": qa["summary"]["A"]}

        entries.append({"index": str(i), "qa_pairs": qa_pairs, "summaries": summaries})

    return {"dataset": entries}


# ---------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------
def save_for_sllm(
    dataset: str,
    in_root: str,
    out_root: str,
    ts_backbone_yaml: str,
    n_shots: int,
    normalize: bool,
) -> None:
    """
    Export one dataset.
    """
    input_folder = os.path.join(in_root, dataset)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # Your loader requires n_shots and other flags
    train, test = load_train_test(
        input_folder=input_folder,
        n_shots=n_shots,
        mmap=False,
        attach_artifacts=True,  # ok if some are missing; we use YAML for labels
        normalize=normalize,
    )

    cfg = load_yaml(ts_backbone_yaml)

    # Determine the SensorLLM dataset name (folder name + token family behavior)
    sllm_name = dataset

    # We will look up label/tokens using the SAME name we output under
    yaml_dataset_name = sllm_name

    ds_cfg = cfg.get(yaml_dataset_name)
    if not isinstance(ds_cfg, dict):
        raise ValueError(f"Dataset '{yaml_dataset_name}' not found in YAML: {ts_backbone_yaml}")

    id2label = id2label_from_yaml(cfg, yaml_dataset_name)
    id2letter = make_id2letter(id2label)

    channel_num = int(ds_cfg.get("channel_num", -1))
    if channel_num <= 0:
        raise ValueError(f"Invalid channel_num for '{yaml_dataset_name}' in YAML: {channel_num}")

    channel_keys = dataset_channel_keys(sllm_name, channel_num)

    out_dir = os.path.join(out_root, sllm_name)
    ensure_dir(out_dir)

    # ----------------------------------------------------------
    # Write pickles (train/eval)
    # ----------------------------------------------------------
    train_list = split_to_pickle_list(train.X)
    eval_list = split_to_pickle_list(test.X)

    with open(os.path.join(out_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_list, f)
    with open(os.path.join(out_dir, "eval_data.pkl"), "wb") as f:
        pickle.dump(eval_list, f)

    # ----------------------------------------------------------
    # Write stage1 QA JSON (train/eval)
    # ----------------------------------------------------------
    train_qa = build_stage1_qa_json(train.X, train.y, channel_keys, id2letter, id2label)
    eval_qa = build_stage1_qa_json(test.X, test.y, channel_keys, id2letter, id2label)

    with open(os.path.join(out_dir, "train_qa.json"), "w", encoding="utf-8") as f:
        json.dump(train_qa, f, indent=2)
    with open(os.path.join(out_dir, "eval_qa.json"), "w", encoding="utf-8") as f:
        json.dump(eval_qa, f, indent=2)

    # ----------------------------------------------------------
    # Meta for debugging + patching SensorLLM token utils
    # ----------------------------------------------------------
    meta = {
        "source_dataset_arg": dataset,
        "sensorllm_dataset_name": sllm_name,
        "yaml_dataset_name": yaml_dataset_name,
        "input_folder": input_folder,
        "output_folder": out_dir,
        "normalize": normalize,
        "n_shots_loaded": n_shots,
        "train_shape": list(train.X.shape),
        "eval_shape": list(test.X.shape),
        "channel_num": channel_num,
        "channel_keys_used": channel_keys,
        "num_labels": int(ds_cfg.get("num_labels", len(id2label))),
        "id2label": id2label,
        "ts_backbone_info": {
            "ts_backbone_type": cfg.get("ts_backbone_type"),
            "default_ts_token": cfg.get("default_ts_token"),
            "chronos_model": cfg.get("chronos_model"),
        },
        "dataset_yaml_fields": {
            # helpful for patching SensorLLM token utils
            "sample_rate": ds_cfg.get("sample_rate", ds_cfg.get("raw_sample_rate_hz")),
            "default_ts_start_token": ds_cfg.get("default_ts_start_token"),
            "default_ts_end_token": ds_cfg.get("default_ts_end_token"),
            "default_x_acc_start_token": ds_cfg.get("default_x_acc_start_token"),
            "default_x_acc_end_token": ds_cfg.get("default_x_acc_end_token"),
            "default_y_acc_start_token": ds_cfg.get("default_y_acc_start_token"),
            "default_y_acc_end_token": ds_cfg.get("default_y_acc_end_token"),
            "default_z_acc_start_token": ds_cfg.get("default_z_acc_start_token"),
            "default_z_acc_end_token": ds_cfg.get("default_z_acc_end_token"),
            "default_x_gyro_start_token": ds_cfg.get("default_x_gyro_start_token"),
            "default_x_gyro_end_token": ds_cfg.get("default_x_gyro_end_token"),
            "default_y_gyro_start_token": ds_cfg.get("default_y_gyro_start_token"),
            "default_y_gyro_end_token": ds_cfg.get("default_y_gyro_end_token"),
            "default_z_gyro_start_token": ds_cfg.get("default_z_gyro_start_token"),
            "default_z_gyro_end_token": ds_cfg.get("default_z_gyro_end_token"),
        },
        "sensorllm_patch_needed": {
            "univariate_needed": sllm_name not in ("usc-had", "uci"),
            "har_alias_used": (dataset == "har"),
            "had_alias_used": (dataset == "had"),
            "suggested_get_token_dict_branch": (
                f"elif dataset in {[dataset]!r}:" if sllm_name not in ("usc-had", "uci") else "none"
            ),
        },
    }

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Exported dataset={dataset!r} -> {out_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export datasets to SensorLLM format.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ctu", "emg", "har", "rwc", "tee", "had", "ecg"],
        help="Which dataset to export (your internal dataset name).",
    )
    parser.add_argument(
        "--in_root",
        type=str,
        default="./data/samples",
        help="Root folder containing your cleaned datasets (each dataset is a subfolder).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./data/sllm_samples",
        help="Where to write SensorLLM-formatted outputs.",
    )
    parser.add_argument(
        "--ts_backbone_yaml",
        type=str,
        default="./resources/ts_backbone.yaml",
        help="Path to SensorLLM ts_backbone.yaml (used for labels + channel metadata).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply the same z-normalization option used by your loader.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=0,
        help="Passed through to load_train_test. Does not affect SensorLLM export directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[RUN] dataset={args.dataset} in_root={args.in_root} out_root={args.out_root}")
    save_for_sllm(
        dataset=args.dataset,
        in_root=args.in_root,
        out_root=args.out_root,
        ts_backbone_yaml=args.ts_backbone_yaml,
        n_shots=args.n_shots,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
