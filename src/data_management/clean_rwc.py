"""
python ./src/data_management/clean_rwc.py
Single-stage: clean + split + enrich artifacts + save (NPZ) using save_split_folder().
"""

import csv
import os
import json
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf

import sys; sys.path.append("./src")
from utils.preprocessing import _letters, _sort_key_for_label_id, build_question_text
from utils.constants import LABEL_MAPPING

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RAW_ROOT   = "./data/raw_data/rwc"
TRAIN_DIR  = os.path.join(RAW_ROOT, "train")
TRAIN_CSV  = os.path.join(RAW_ROOT, "train.csv")

OUT_DIR    = "./data/datasets/rwc"
SAMP_DIR   = "./data/samples/rwc"

TARGET_LEN = 4000
DTYPE      = np.float32

TRAIN_FRAC = 0.85
SEED       = 1337
SHUFFLE_WITHIN_SPLITS = True

# sample sizes
SAMP_TRAIN_N = 1000
SAMP_TEST_N  = -1

# few-shot config (stored as indices into *train split*)
SHOTS_PER_CLASS = 10


# ---------------------------------------------------------------------
# YOUR save function (import it if it already exists elsewhere)
# ---------------------------------------------------------------------
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
    os.makedirs(output_directory, exist_ok=True)

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


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def stratified_train_test_split(y: np.ndarray, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        k = int(np.floor(train_frac * n))
        if n >= 2:
            k = min(max(1, k), n - 1)
        if n == 1:
            k = 1
        train_idx.append(cls_idx[:k])
        test_idx.append(cls_idx[k:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _to_mono(wave: np.ndarray) -> np.ndarray:
    return wave if wave.ndim == 1 else wave.mean(axis=1)


def _pad_trim_center(wave: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    n = wave.shape[0]
    if n == target_len:
        return wave
    if n > target_len:
        start = (n - target_len) // 2
        return wave[start:start + target_len]
    pad_total = target_len - n
    left = pad_total // 2
    right = pad_total - left
    return np.pad(wave, (left, right), mode="constant", constant_values=pad_value)


def read_aiff(path: str, dtype: type = DTYPE) -> np.ndarray:
    wave, _sr = sf.read(path, always_2d=False)
    wave = np.asarray(wave, dtype=dtype)
    wave = _to_mono(wave)
    return wave


def load_aiff_row(path: str, target_len: int, dtype: type = DTYPE) -> np.ndarray:
    wave = read_aiff(path, dtype=dtype)
    wave = _pad_trim_center(wave, target_len, pad_value=0.0).astype(dtype, copy=False)
    return wave[:, None]  # (L, 1)


def list_aiff_files(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".aiff", ".aif", ".aifc")):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_labels_csv(csv_path: str) -> Dict[str, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")

    mapping: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        sniff = f.read(2048)
        f.seek(0)
        has_header = any(h in sniff.lower() for h in ("filename", "file", "label"))
        reader = csv.reader(f)
        if has_header:
            _ = next(reader, None)

        for row in reader:
            if not row or len(row) < 2:
                continue
            filename_str = row[0].strip()
            label_str = row[1].strip()
            s = stem(filename_str)

            raw_label = int(float(label_str))
            if raw_label in (0, 1):
                y = raw_label
            elif raw_label in (1, 2):
                y = raw_label - 1
            else:
                raise ValueError(f"Unexpected label {raw_label} for {filename_str}; expected 0/1 or 1/2.")
            mapping[s] = y

    if not mapping:
        raise RuntimeError(f"No valid (filename,label) pairs read from {csv_path}")
    return mapping


def build_letter_maps(dataset: str):
    key = dataset.strip().upper()
    id_to_name = LABEL_MAPPING[key]  # e.g. {0:"...", 1:"..."}
    items = sorted(id_to_name.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    id_to_letter = {int(cid): _letters(i + 1) for i, (cid, _) in enumerate(items)}
    letter_to_id = {letter: cid for cid, letter in id_to_letter.items()}
    return id_to_letter, letter_to_id, id_to_name


def class_dist_str(y: np.ndarray) -> str:
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    return json.dumps(dist, sort_keys=True)


def make_class_shots(y_train: np.ndarray, shots_per_class: int, seed: int) -> Dict[int, List[int]]:
    """
    Returns: {class_id: [train_index0, train_index1, ...]}
    Indices refer to positions in X_train / y_train.
    """
    rng = np.random.default_rng(seed)
    class_shots: Dict[int, List[int]] = {}
    for cls in np.unique(y_train):
        idxs = np.where(y_train == cls)[0]
        rng.shuffle(idxs)
        k = min(int(shots_per_class), len(idxs))
        class_shots[int(cls)] = [int(i) for i in idxs[:k]]
    return class_shots


def take_sample_split(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    n_train: int, n_test: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_n = min(int(n_train), len(y_train))
    te_n = min(int(n_test),  len(y_test))

    tr_idx = rng.permutation(len(y_train))[:tr_n]
    if n_test != -1:
        te_idx = rng.permutation(len(y_test))[:te_n]
    else: 
        te_idx = rng.permutation(len(y_test))

    return X_train[tr_idx], y_train[tr_idx], X_test[te_idx], y_test[te_idx]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    dataset = "rwc"

    # --- Load labels ---
    label_map = load_labels_csv(TRAIN_CSV)  # stem -> {0,1}

    # --- Gather files ---
    train_files = list_aiff_files(TRAIN_DIR)
    if not train_files:
        raise RuntimeError(f"No AIFF files found in {TRAIN_DIR}")

    # --- Load waveforms matched to labels ---
    rows: List[np.ndarray] = []
    labels: List[int] = []
    skipped_missing_label = 0

    for path in train_files:
        s = stem(path)
        y = label_map.get(s)
        if y is None:
            skipped_missing_label += 1
            continue
        try:
            row = load_aiff_row(path, target_len=TARGET_LEN, dtype=DTYPE)  # (1, L)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {path} ({e})")
            continue
        rows.append(row)
        labels.append(int(y))

    if not rows:
        raise RuntimeError("No labeled samples loaded from TRAIN.")
    if skipped_missing_label:
        print(f"[INFO] Skipped {skipped_missing_label} files not present in train.csv.")

    X_all = np.stack(rows, axis=0).astype(DTYPE)  # (N, L, 1)
    y_all = np.asarray(labels, dtype=np.int64)    # (N,)
    assert X_all.shape[1] == TARGET_LEN
    assert X_all.shape[2] == 1

    # --- Stratified split ---
    train_idx, test_idx = stratified_train_test_split(y_all, TRAIN_FRAC, SEED)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

    # Ensure (N, 1, L)
    if X_train.ndim == 2:
        X_train = X_train[:, None, :]
    if X_test.ndim == 2:
        X_test = X_test[:, None, :]

    if SHUFFLE_WITHIN_SPLITS:
        rng = np.random.default_rng(SEED)
        p = rng.permutation(len(y_train))
        X_train, y_train = X_train[p], y_train[p]
        p = rng.permutation(len(y_test))
        X_test, y_test = X_test[p], y_test[p]

    # --- Build “enrich” artifacts ---
    id_to_letter, letter_to_id, id_to_name = build_letter_maps(dataset)
    question = build_question_text(dataset).strip()

    label_maps = {
        "letter_to_id": {str(k): int(v) for k, v in letter_to_id.items()},     # keys are letters already
        "id_to_letter": {str(int(k)): v for k, v in id_to_letter.items()},
        "id_to_name":   {str(int(k)): v for k, v in id_to_name.items()},
    }

    class_shots = make_class_shots(y_train, SHOTS_PER_CLASS, SEED)

    dataset_statistics = [
        f"dataset={dataset}",
        f"seed={SEED}",
        f"target_len={TARGET_LEN}",
        f"train_frac={TRAIN_FRAC}",
        f"X_train_shape={tuple(X_train.shape)}",
        f"y_train_shape={tuple(y_train.shape)}",
        f"X_test_shape={tuple(X_test.shape)}",
        f"y_test_shape={tuple(y_test.shape)}",
        f"train_class_dist={class_dist_str(y_train)}",
        f"test_class_dist={class_dist_str(y_test)}",
    ]

    # --- Save FULL split ---
    save_split_folder(
        output_directory=OUT_DIR,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        class_shots=class_shots,
        general_question=question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics,
    )
    print(f"[OK] wrote full split → {OUT_DIR}")

    # --- Save SAMPLE split ---
    X_tr_s, y_tr_s, X_te_s, y_te_s = take_sample_split(
        X_train, y_train, X_test, y_test,
        n_train=SAMP_TRAIN_N,
        n_test=SAMP_TEST_N,
        seed=SEED,
    )

    class_shots_samp = make_class_shots(y_tr_s, min(SHOTS_PER_CLASS, 10), SEED)

    dataset_statistics_samp = dataset_statistics + [
        f"sample_train_n={len(y_tr_s)}",
        f"sample_test_n={len(y_te_s)}",
        f"sample_train_class_dist={class_dist_str(y_tr_s)}",
        f"sample_test_class_dist={class_dist_str(y_te_s)}",
    ]

    save_split_folder(
        output_directory=SAMP_DIR,
        X_train=X_tr_s, y_train=y_tr_s,
        X_test=X_te_s,  y_test=y_te_s,
        class_shots=class_shots_samp,
        general_question=question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics_samp,
    )
    print(f"[OK] wrote sample split → {SAMP_DIR}")


if __name__ == "__main__":
    main()
