"""
PURPOSE: cleaning raw data of USC-HAD
- Read Subject*/a*t*.mat files
- Window into fixed-length samples
- Subject-wise train/test split (NO dev split)
- Save to NPZ in a format consistent with the rest of your pipeline
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


ACTIVITY_NAMES = {
    1: "walking_forward",
    2: "walking_left",
    3: "walking_right",
    4: "walking_upstairs",
    5: "walking_downstairs",
    6: "running_forward",
    7: "jumping_up",
    8: "sitting",
    9: "standing",
    10: "sleeping",
    11: "elevator_up",
    12: "elevator_down",
}

ACTIVITY_NAME_TO_NUMBER = {
    "walking forward": 1, "walking-forward": 1, "walking_forward": 1,
    "walking left": 2, "walking-left": 2, "walking_left": 2,
    "walking right": 3, "walking-right": 3, "walking_right": 3,
    "walking upstairs": 4, "walking-upstairs": 4, "walking_upstairs": 4,
    "walking downstairs": 5, "walking-downstairs": 5, "walking_downstairs": 5,
    "running forward": 6, "running-forward": 6, "running_forward": 6,
    "jumping up": 7, "jumping-up": 7, "jumping_up": 7,
    "sitting": 8,
    "standing": 9,
    "sleeping": 10,
    "elevator up": 11, "elevator-up": 11, "elevator_up": 11,
    "elevator down": 12, "elevator-down": 12, "elevator_down": 12,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Clean USC-HAD .mat dataset into NPZ splits (train/test only)")

    p.add_argument("--raw_dir", type=str, default="data/raw_data/USC-HAD")
    p.add_argument("--out_dir", type=str, default="data/datasets/had")

    p.add_argument("--sample_rate_hz", type=int, default=100)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)

    # kept for compatibility w/ other scripts even if not used here
    p.add_argument("--label_mode", type=str, default="majority", choices=["majority", "center"])
    p.add_argument("--majority_frac", type=float, default=0.6)

    p.add_argument("--standardize", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_sample_n", type=int, default=1000)
    p.add_argument("--test_sample_n", type=int, default=-1)

    # Subject-wise split
    p.add_argument("--test_subjects", type=int, default=2,
                   help="Number of subjects to hold out for test (subject-wise)")

    return p.parse_args()


def sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random sampling without replacement. Ignores labels."""
    if n_samples == -1 or len(y) <= n_samples:
        return X, y
    sampled_indices = rng.choice(len(y), size=n_samples, replace=False)
    return X[sampled_indices], y[sampled_indices]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_subject_files(raw_dir: str) -> List[Tuple[int, List[str]]]:
    """Returns list of (subject_id, [mat_file_paths])"""
    subjects: List[Tuple[int, List[str]]] = []
    for entry in os.listdir(raw_dir):
        subject_path = os.path.join(raw_dir, entry)
        if not os.path.isdir(subject_path):
            continue

        if not entry.startswith("Subject"):
            continue

        try:
            sid = int(entry.replace("Subject", ""))
        except ValueError:
            continue

        mat_files = [
            os.path.join(subject_path, fn)
            for fn in os.listdir(subject_path)
            if fn.endswith(".mat") and fn.startswith("a")
        ]
        if mat_files:
            mat_files.sort()
            subjects.append((sid, mat_files))

    subjects.sort(key=lambda x: x[0])
    return subjects


def load_mat_file(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a single .mat file and return (sensor_readings, activity_number)

    sensor_readings: (T, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    activity_number: int (1-12)
    """
    try:
        mat_data = loadmat(path)

        if "sensor_readings" not in mat_data:
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            raise KeyError(f"'sensor_readings' not found. Available keys: {keys}")

        sensor_readings = mat_data["sensor_readings"]

        activity_number = None
        possible_fields = ["activity_number", "activityNumber", "activity", "label"]

        for field in possible_fields:
            if field not in mat_data:
                continue
            activity_data = mat_data[field]
            # numeric
            try:
                if hasattr(activity_data, "ndim") and activity_data.ndim == 2:
                    activity_number = int(activity_data[0, 0])
                elif hasattr(activity_data, "ndim") and activity_data.ndim == 1:
                    activity_number = int(activity_data[0])
                else:
                    activity_number = int(activity_data)
                break
            except (ValueError, TypeError):
                pass

            # string
            try:
                if hasattr(activity_data, "ndim") and activity_data.ndim >= 1:
                    activity_str = str(activity_data[0]).lower().strip()
                else:
                    activity_str = str(activity_data).lower().strip()
                activity_number = ACTIVITY_NAME_TO_NUMBER.get(activity_str)
                if activity_number is not None:
                    break
            except Exception:
                continue

        # fallback: parse filename a11t4.mat -> activity 11
        if activity_number is None:
            import re
            filename = os.path.basename(path)
            match = re.match(r"a(\d+)t(\d+)\.mat", filename)
            if match:
                activity_number = int(match.group(1))
            else:
                keys = [k for k in mat_data.keys() if not k.startswith("__")]
                raise ValueError(f"Could not infer activity number. Available keys: {keys}")

        # enforce 2D
        sensor_readings = np.asarray(sensor_readings)
        if sensor_readings.ndim == 1:
            sensor_readings = sensor_readings.reshape(-1, 6)

        return sensor_readings, int(activity_number)

    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}") from e


def window_trial(
    data: np.ndarray,
    label: int,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window a single trial into fixed-length segments (label is constant for all windows)."""
    n = len(data)
    windows_X, windows_y = [], []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows_X.append(data[start:end])
        windows_y.append(label)

    if not windows_X:
        return (
            np.empty((0, window_size, data.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.stack(windows_X, axis=0).astype(np.float32)
    y = np.asarray(windows_y, dtype=np.int64)
    return X, y


def process_subject(
    mat_files: List[str],
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process all .mat files for a single subject."""
    X_list, y_list = [], []

    for mat_file in mat_files:
        sensor_readings, activity_number = load_mat_file(mat_file)
        X_win, y_win = window_trial(sensor_readings, activity_number, window_size, stride)
        if len(y_win) > 0:
            X_list.append(X_win)
            y_list.append(y_win)

    if not X_list:
        return np.empty((0, window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def compute_channel_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
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
        raise FileNotFoundError(f"No Subject folders found in: {args.raw_dir}")

    subject_ids = [sid for sid, _ in subject_files]
    n_subj = len(subject_ids)

    if args.test_subjects >= n_subj:
        raise ValueError(f"test_subjects must be < #subjects ({n_subj}). Got test={args.test_subjects}.")

    # Subject-wise split (train/test only)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    test_subject_set = set(shuffled[: args.test_subjects])
    train_subject_set = set(shuffled[args.test_subjects :])

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print("Processing subjects...")
    for sid, mat_files in subject_files:
        print(f"  Subject {sid}: {len(mat_files)} trials")

        X_sub, y_sub = process_subject(
            mat_files=mat_files,
            window_size=args.window_size,
            stride=args.stride,
        )

        if sid in train_subject_set:
            X_train_list.append(X_sub)
            y_train_list.append(y_sub)
        else:
            X_test_list.append(X_sub)
            y_test_list.append(y_sub)

    def concat_or_fail(xs, ys, name: str):
        if not xs or sum(x.shape[0] for x in xs) == 0:
            raise RuntimeError(f"No {name} windows produced. Try adjusting window/stride.")
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = concat_or_fail(X_train_list, y_train_list, "train")
    X_test, y_test = concat_or_fail(X_test_list, y_test_list, "test")

    # Shuffle within each split
    def shuffle_split(X, y):
        p = rng.permutation(len(y))
        return X[p], y[p]

    X_train, y_train = shuffle_split(X_train, y_train)
    X_test, y_test = shuffle_split(X_test, y_test)

    # Standardize using TRAIN stats only
    standardize_info = None
    if args.standardize:
        mean, std = compute_channel_stats(X_train)
        X_train = apply_standardize(X_train, mean, std)
        X_test = apply_standardize(X_test, mean, std)
        standardize_info = {"mean": mean.tolist(), "std": std.tolist()}

    # Sample splits
    rng_sample = np.random.RandomState(args.seed + 1)
    X_train_sample, y_train_sample = sample(X_train, y_train, args.train_sample_n, rng_sample)
    X_test_sample, y_test_sample = sample(X_test, y_test, args.test_sample_n, rng_sample)

    # Save FULL splits
    ensure_dir(args.out_dir)
    np.savez(os.path.join(args.out_dir, "train.npz"), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(args.out_dir, "test.npz"), X_test=X_test, y_test=y_test)

    # Save SAMPLE splits
    sample_dir = args.out_dir.replace("datasets", "samples")
    ensure_dir(sample_dir)
    np.savez(os.path.join(sample_dir, "train.npz"), X_train=X_train_sample, y_train=y_train_sample)
    np.savez(os.path.join(sample_dir, "test.npz"), X_test=X_test_sample, y_test=y_test_sample)

    # Metadata
    meta = {
        "dataset": "usc_had",
        "raw_dir": args.raw_dir,
        "out_dir": args.out_dir,
        "sample_rate_hz": args.sample_rate_hz,
        "window_size": args.window_size,
        "stride": args.stride,
        "label_mode": args.label_mode,
        "majority_frac": args.majority_frac,
        "standardize": bool(args.standardize),
        "seed": args.seed,
        "train_subjects": sorted(list(train_subject_set)),
        "test_subjects": sorted(list(test_subject_set)),
        "activity_names": ACTIVITY_NAMES,
        "standardize_info": standardize_info,
        "sensor_info": {
            "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
            "units": ["g", "g", "g", "dps", "dps", "dps"],
            "acc_range": "+-6g",
            "gyro_range": "+-500dps",
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "label_hist": {
            "train": label_hist(y_train),
            "test": label_hist(y_test),
        },
        "sample_sizes": {
            "train": int(len(y_train_sample)),
            "test": int(len(y_test_sample)),
        },
        "sample_label_hist": {
            "train": label_hist(y_train_sample),
            "test": label_hist(y_test_sample),
        },
        "sample_shapes": {
            "X_train_sample": list(X_train_sample.shape),
            "X_test_sample": list(X_test_sample.shape),
        },
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Statistics file
    with open(os.path.join(args.out_dir, "data_statistics.txt"), "w") as f:
        f.write("Dataset: USC-HAD\n")
        f.write(f"Total subjects: {n_subj}\n")
        f.write(f"Train subjects: {meta['train_subjects']}\n")
        f.write(f"Test subjects:  {meta['test_subjects']}\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape:  {X_test.shape}\n")
        f.write(f"Train labels: {np.unique(y_train)}\n")
        f.write(f"Test labels:  {np.unique(y_test)}\n")
        f.write(f"Label hist train: {meta['label_hist']['train']}\n")
        f.write(f"Label hist test:  {meta['label_hist']['test']}\n")

        f.write("\n--- SAMPLE SPLITS ---\n")
        f.write(f"Train sample: {X_train_sample.shape}\n")
        f.write(f"Test sample:  {X_test_sample.shape}\n")
        f.write(f"Sample label hist train: {label_hist(y_train_sample)}\n")
        f.write(f"Sample label hist test:  {label_hist(y_test_sample)}\n")

    print("\n✔ USC-HAD cleaned and saved (train/test + sample)")
    print(f"  → {args.out_dir}/train.npz")
    print(f"  → {args.out_dir}/test.npz")
    print(f"  → {sample_dir}/train.npz")
    print(f"  → {sample_dir}/test.npz")
    print(f"  → {args.out_dir}/metadata.json")
    print(f"  → {args.out_dir}/data_statistics.txt\n")


if __name__ == "__main__":
    main()
