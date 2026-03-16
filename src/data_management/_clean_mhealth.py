"""
python ./src/_scratch_pad.py

PURPOSE: cleaning raw data of mhealth
- Read mHealth_subject*.log
- Window into fixed-length samples
- Subject-stratified train/dev/test split
- Save to NPZ in a format consistent with the rest of your pipeline
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np


ACTIVITY_NAMES = {
    0: "null",
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_of_arms",
    8: "knees_bending_crouching",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Clean mHealth .log dataset into NPZ splits")

    p.add_argument("--raw_dir", type=str, default="data/raw_data/mhealth")
    p.add_argument("--out_dir", type=str, default="data/datasets/mhealth")

    p.add_argument("--sample_rate_hz", type=int, default=50)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--label_mode", type=str, default="majority", choices=["majority", "center"])
    p.add_argument("--majority_frac", type=float, default=0.6)

    p.add_argument("--drop_null", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_ecg", type=int, default=0, choices=[0, 1])
    p.add_argument("--standardize", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_sample_n", type=int, default=500)
    p.add_argument("--dev_sample_n", type=int, default=100)
    p.add_argument("--test_sample_n", type=int, default=100)


    # NEW: dev + test subject counts
    p.add_argument("--dev_subjects", type=int, default=2,
                   help="Number of subjects to hold out for dev/val (subject-wise)")
    p.add_argument("--test_subjects", type=int, default=2,
                   help="Number of subjects to hold out for test (subject-wise)")

    return p.parse_args()


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified sampling without replacement.
    If dataset is smaller than n_samples, return full dataset.
    """
    if len(y) <= n_samples:
        return X, y

    sampled_indices = []
    labels = np.unique(y)

    # proportional allocation
    for label in labels:
        idxs = np.where(y == label)[0]
        k = max(1, int(round(n_samples * len(idxs) / len(y))))
        k = min(k, len(idxs))
        sampled_indices.extend(rng.choice(idxs, size=k, replace=False))

    # trim if we overshot
    sampled_indices = np.array(sampled_indices)
    if len(sampled_indices) > n_samples:
        sampled_indices = rng.choice(sampled_indices, size=n_samples, replace=False)

    return X[sampled_indices], y[sampled_indices]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_subject_files(raw_dir: str) -> List[Tuple[int, str]]:
    files = []
    for fn in os.listdir(raw_dir):
        if fn.startswith("mHealth_subject") and fn.endswith(".log"):
            stem = fn.replace("mHealth_subject", "").replace(".log", "")
            try:
                sid = int(stem)
            except ValueError:
                continue
            files.append((sid, os.path.join(raw_dir, fn)))
    files.sort(key=lambda x: x[0])
    return files


def load_log(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data


def window_subject(
    data: np.ndarray,
    window_size: int,
    stride: int,
    label_mode: str,
    majority_frac: float,
    drop_ecg: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    X_raw = data[:, :-1]
    y_raw = data[:, -1].astype(int)

    if drop_ecg:
        keep = [i for i in range(X_raw.shape[1]) if i not in (3, 4)]
        X_raw = X_raw[:, keep]

    n = len(y_raw)
    windows_X = []
    windows_y = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        x_win = X_raw[start:end]
        y_win = y_raw[start:end]

        if label_mode == "center":
            label = int(y_win[window_size // 2])
        else:
            vals, counts = np.unique(y_win, return_counts=True)
            winner_idx = int(np.argmax(counts))
            label = int(vals[winner_idx])
            frac = float(counts[winner_idx]) / float(window_size)
            if frac < majority_frac:
                continue

        windows_X.append(x_win)
        windows_y.append(label)

    if not windows_X:
        return np.empty((0, window_size, X_raw.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(windows_X, axis=0).astype(np.float32)
    y = np.asarray(windows_y, dtype=np.int64)
    return X, y


def compute_channel_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, None, :]) / std[None, None, :]


def label_hist(y: np.ndarray) -> Dict[int, int]:
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    subject_files = discover_subject_files(args.raw_dir)
    if not subject_files:
        raise FileNotFoundError(f"No mHealth_subject*.log files found in: {args.raw_dir}")

    subject_ids = [sid for sid, _ in subject_files]
    n_subj = len(subject_ids)

    if args.dev_subjects + args.test_subjects >= n_subj:
        raise ValueError(
            f"dev_subjects + test_subjects must be < #subjects ({n_subj}). "
            f"Got dev={args.dev_subjects}, test={args.test_subjects}."
        )

    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    test_subject_set = set(shuffled[: args.test_subjects])
    dev_subject_set = set(shuffled[args.test_subjects : args.test_subjects + args.dev_subjects])
    train_subject_set = set(shuffled[args.test_subjects + args.dev_subjects :])

    X_train_list, y_train_list = [], []
    X_dev_list, y_dev_list = [], []
    X_test_list, y_test_list = [], []

    for sid, path in subject_files:
        data = load_log(path)

        if data.shape[1] != 24:
            raise ValueError(
                f"{os.path.basename(path)} has {data.shape[1]} columns; expected 24."
            )

        X_sub, y_sub = window_subject(
            data=data,
            window_size=args.window_size,
            stride=args.stride,
            label_mode=args.label_mode,
            majority_frac=args.majority_frac,
            drop_ecg=bool(args.drop_ecg),
        )

        if args.drop_null:
            keep = y_sub != 0
            X_sub = X_sub[keep]
            y_sub = y_sub[keep]

        if sid in train_subject_set:
            X_train_list.append(X_sub); y_train_list.append(y_sub)
        elif sid in dev_subject_set:
            X_dev_list.append(X_sub); y_dev_list.append(y_sub)
        else:
            X_test_list.append(X_sub); y_test_list.append(y_sub)

    def concat_or_fail(xs, ys, name: str):
        if not xs or sum(x.shape[0] for x in xs) == 0:
            raise RuntimeError(f"No {name} windows produced. Try lowering --majority_frac or adjusting window/stride.")
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = concat_or_fail(X_train_list, y_train_list, "train")
    X_dev, y_dev     = concat_or_fail(X_dev_list, y_dev_list, "dev")
    X_test, y_test   = concat_or_fail(X_test_list, y_test_list, "test")

    # Shuffle within each split
    def shuffle_split(X, y):
        p = rng.permutation(len(y))
        return X[p], y[p]

    X_train, y_train = shuffle_split(X_train, y_train)
    X_dev, y_dev     = shuffle_split(X_dev, y_dev)
    X_test, y_test   = shuffle_split(X_test, y_test)

    # Standardize using TRAIN stats only (then apply to dev/test)
    standardize_info = None
    if args.standardize:
        mean, std = compute_channel_stats(X_train)
        X_train = apply_standardize(X_train, mean, std)
        X_dev   = apply_standardize(X_dev, mean, std)
        X_test  = apply_standardize(X_test, mean, std)
        standardize_info = {"mean": mean.tolist(), "std": std.tolist()}

    # -------------------------
    # SAMPLE splits
    # -------------------------
    rng_sample = np.random.RandomState(args.seed + 1)

    X_train_sample, y_train_sample = stratified_sample(
        X_train, y_train, args.train_sample_n, rng_sample
    )
    X_dev_sample, y_dev_sample = stratified_sample(
        X_dev, y_dev, args.dev_sample_n, rng_sample
    )
    X_test_sample, y_test_sample = stratified_sample(
        X_test, y_test, args.test_sample_n, rng_sample
    )

    # -------------------------
    # Save FULL splits
    # -------------------------
    ensure_dir(args.out_dir)
    np.savez(os.path.join(args.out_dir, "train.npz"), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(args.out_dir, "dev.npz"),   X_dev=X_dev,     y_dev=y_dev)
    np.savez(os.path.join(args.out_dir, "test.npz"),  X_test=X_test,   y_test=y_test)

    # -------------------------
    # Save SAMPLE splits
    # -------------------------
    sample_dir = os.path.join(args.out_dir, "sample")
    ensure_dir(sample_dir)

    np.savez(os.path.join(sample_dir, "train.npz"),
             X_train=X_train_sample, y_train=y_train_sample)
    np.savez(os.path.join(sample_dir, "dev.npz"),
             X_dev=X_dev_sample, y_dev=y_dev_sample)
    np.savez(os.path.join(sample_dir, "test.npz"),
             X_test=X_test_sample, y_test=y_test_sample)

    # -------------------------
    # Metadata (now safe to reference *_sample vars)
    # -------------------------
    meta = {
        "dataset": "mhealth",
        "raw_dir": args.raw_dir,
        "out_dir": args.out_dir,
        "sample_rate_hz": args.sample_rate_hz,
        "window_size": args.window_size,
        "stride": args.stride,
        "label_mode": args.label_mode,
        "majority_frac": args.majority_frac,
        "drop_null": bool(args.drop_null),
        "drop_ecg": bool(args.drop_ecg),
        "standardize": bool(args.standardize),
        "seed": args.seed,
        "train_subjects": sorted(list(train_subject_set)),
        "dev_subjects": sorted(list(dev_subject_set)),
        "test_subjects": sorted(list(test_subject_set)),
        "activity_names": ACTIVITY_NAMES,
        "standardize_info": standardize_info,
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_dev": list(X_dev.shape),
            "y_dev": list(y_dev.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "label_hist": {
            "train": label_hist(y_train),
            "dev": label_hist(y_dev),
            "test": label_hist(y_test),
        },
        "sample_sizes": {
            "train": int(len(y_train_sample)),
            "dev": int(len(y_dev_sample)),
            "test": int(len(y_test_sample)),
        },
        "sample_label_hist": {
            "train": label_hist(y_train_sample),
            "dev": label_hist(y_dev_sample),
            "test": label_hist(y_test_sample),
        },
        "sample_shapes": {
            "X_train_sample": list(X_train_sample.shape),
            "X_dev_sample": list(X_dev_sample.shape),
            "X_test_sample": list(X_test_sample.shape),
        }
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # -------------------------
    # Stats file (f is in-scope here)
    # -------------------------
    with open(os.path.join(args.out_dir, "data_statistics.txt"), "w") as f:
        f.write("Dataset: mhealth\n")
        f.write(f"Train subjects: {meta['train_subjects']}\n")
        f.write(f"Dev subjects:   {meta['dev_subjects']}\n")
        f.write(f"Test subjects:  {meta['test_subjects']}\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_dev shape:   {X_dev.shape}\n")
        f.write(f"X_test shape:  {X_test.shape}\n")
        f.write(f"Train labels: {np.unique(y_train)}\n")
        f.write(f"Dev labels:   {np.unique(y_dev)}\n")
        f.write(f"Test labels:  {np.unique(y_test)}\n")
        f.write(f"Label hist train: {meta['label_hist']['train']}\n")
        f.write(f"Label hist dev:   {meta['label_hist']['dev']}\n")
        f.write(f"Label hist test:  {meta['label_hist']['test']}\n")

        f.write("\n--- SAMPLE SPLITS ---\n")
        f.write(f"Train sample: {X_train_sample.shape}\n")
        f.write(f"Dev sample:   {X_dev_sample.shape}\n")
        f.write(f"Test sample:  {X_test_sample.shape}\n")
        f.write(f"Sample label hist train: {label_hist(y_train_sample)}\n")
        f.write(f"Sample label hist dev:   {label_hist(y_dev_sample)}\n")
        f.write(f"Sample label hist test:  {label_hist(y_test_sample)}\n")

    print("\n✔ mHealth cleaned and saved (train/dev/test + sample)")
    print(f"  → {args.out_dir}/train.npz")
    print(f"  → {args.out_dir}/dev.npz")
    print(f"  → {args.out_dir}/test.npz")
    print(f"  → {sample_dir}/train.npz")
    print(f"  → {sample_dir}/dev.npz")
    print(f"  → {sample_dir}/test.npz")
    print(f"  → {args.out_dir}/metadata.json")
    print(f"  → {args.out_dir}/data_statistics.txt\n")


if __name__ == "__main__":
    main()
