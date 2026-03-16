import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from sktime.datasets import load_from_tsfile

import sys; sys.path.append("./src")
from utils.constants import LABEL_MAPPING
from utils.preprocessing import _letters, _sort_key_for_label_id, build_question_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .ts time-series dataset to clean NumPy format.")
    parser.add_argument("--dataset", choices=["ctu", "emg", "har", "tee"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_sample_length", type=int, default=1000)
    parser.add_argument("--test_sample_length", type=int, default=-1)
    return parser.parse_args()


def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def ts_dataframe_to_numpy(ts_dataframe: Any) -> np.ndarray:
    num_channels: int = ts_dataframe.shape[1]
    list_of_samples: List[np.ndarray] = []
    for _, row in ts_dataframe.iterrows():
        channels: List[np.ndarray] = [row.iloc[i].to_numpy() for i in range(num_channels)]
        sample: np.ndarray = np.stack(channels, axis=1)  # (T, D)
        list_of_samples.append(sample)
    return np.stack(list_of_samples, axis=0)


def build_label_letter_mappings(dataset_name: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str]]:
    dataset_key = dataset_name.strip().upper()
    class_id_to_name = LABEL_MAPPING[dataset_key]

    sorted_items = sorted(class_id_to_name.items(), key=lambda item: _sort_key_for_label_id(item[0]))
    class_id_to_letter = {cid: _letters(i + 1) for i, (cid, _) in enumerate(sorted_items)}
    letter_to_class_id = {ltr: cid for cid, ltr in class_id_to_letter.items()}
    return class_id_to_letter, letter_to_class_id, class_id_to_name


def save_split_folder(
    output_directory: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_shots: Dict[int, List[int]],
    general_question: str,
    label_maps: Dict[str, Dict[str, str]],
    dataset_statistics: List[str],
) -> None:
    ensure_directory_exists(output_directory)

    np.savez(os.path.join(output_directory, "train.npz"), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(output_directory, "test.npz"),  X_test=X_test,  y_test=y_test)

    class_shots_json = {str(int(k)): [int(i) for i in v] for k, v in class_shots.items()}
    with open(os.path.join(output_directory, "class_shots.json"), "w") as f:
        json.dump(class_shots_json, f, indent=2)

    with open(os.path.join(output_directory, "general_question.txt"), "w") as f:
        f.write(general_question)

    with open(os.path.join(output_directory, "label_maps.json"), "w") as f:
        json.dump(label_maps, f, indent=2)

    with open(os.path.join(output_directory, "data_statistics.txt"), "w") as f:
        for line in dataset_statistics:
            f.write(str(line) + "\n")


def _slice_n(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Safe slicing: n=-1 => full; n>len => full; else first n."""
    if n == -1 or n >= len(y):
        return X, y
    if n < 0:
        raise ValueError(f"n must be -1 or nonnegative; got {n}")
    return X[:n], y[:n]


def _few_shots(y: np.ndarray, rng: np.random.RandomState, k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        sel = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
        out[int(label)] = sel.tolist()
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    dataset_name: str = args.dataset
    train_ts_path = f"data/raw_data/{dataset_name}/{dataset_name.upper()}_TRAIN.ts"
    test_ts_path  = f"data/raw_data/{dataset_name}/{dataset_name.upper()}_TEST.ts"

    train_ts_df, train_labels = load_from_tsfile(train_ts_path)
    test_ts_df,  test_labels  = load_from_tsfile(test_ts_path)

    train_labels = train_labels.astype(int)
    test_labels  = test_labels.astype(int)

    # Optional shift to 0-based (keep only if your LABEL_MAPPING matches!)
    if train_labels.min() != 0:
        shift = int(train_labels.min())
        train_labels -= shift
        test_labels  -= shift

    X_train_full = ts_dataframe_to_numpy(train_ts_df)
    X_test_full  = ts_dataframe_to_numpy(test_ts_df)

    # Shuffle full datasets
    perm_train = rng.permutation(len(train_labels))
    perm_test  = rng.permutation(len(test_labels))

    X_train_full = X_train_full[perm_train]
    train_labels = train_labels[perm_train]
    X_test_full  = X_test_full[perm_test]
    test_labels  = test_labels[perm_test]

    # Sample splits (safe)
    X_train_sample, y_train_sample = _slice_n(X_train_full, train_labels, args.train_sample_length)
    X_test_sample,  y_test_sample  = _slice_n(X_test_full,  test_labels,  args.test_sample_length)

    # Few-shot indices MUST match the split they’re saved with
    few_shot_full = _few_shots(train_labels, rng, k=5)
    few_shot_sample = _few_shots(y_train_sample, rng, k=5)

    general_question: str = build_question_text(dataset_name).strip()
    class_id_to_letter, letter_to_class_id, class_id_to_name = build_label_letter_mappings(dataset_name)

    # Make JSON consistent: store everything as strings
    label_maps: Dict[str, Dict[str, str]] = {
        "letter_to_id": {ltr: str(int(cid)) for ltr, cid in letter_to_class_id.items()},
        "id_to_letter": {str(int(cid)): ltr for cid, ltr in class_id_to_letter.items()},
        "id_to_name":   {str(int(cid)): str(name) for cid, name in class_id_to_name.items()},
    }

    dataset_statistics: List[str] = [
        f"Dataset: {dataset_name}",
        f"Seed: {args.seed}",
        f"Train full: {len(train_labels)}",
        f"Test full:  {len(test_labels)}",
        f"Train sample: {len(y_train_sample)}",
        f"Test sample:  {len(y_test_sample)}",
        f"Train full shape: {X_train_full.shape}",
        f"Test full shape:  {X_test_full.shape}",
        f"Unique train labels: {np.unique(train_labels)}",
        f"Unique test labels:  {np.unique(test_labels)}",
    ]

    # Save FULL split
    full_output_dir = f"data/datasets/{dataset_name}"
    save_split_folder(
        output_directory=full_output_dir,
        X_train=X_train_full,
        y_train=train_labels,
        X_test=X_test_full,
        y_test=test_labels,
        class_shots=few_shot_full,
        general_question=general_question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics + [f"Few-shot indices (full): {few_shot_full}"],
    )

    # Save SAMPLE split
    sample_output_dir = f"data/samples/{dataset_name}"
    save_split_folder(
        output_directory=sample_output_dir,
        X_train=X_train_sample,
        y_train=y_train_sample,
        X_test=X_test_sample,
        y_test=y_test_sample,
        class_shots=few_shot_sample,
        general_question=general_question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics + [f"Few-shot indices (sample): {few_shot_sample}"],
    )

    print(f"\n✔ Saved FULL split → {full_output_dir}")
    print(f"✔ Saved SAMPLE split → {sample_output_dir}\n")


if __name__ == "__main__":
    main()
