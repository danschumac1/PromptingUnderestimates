"""
Rebuild sktime .ts files from your cleaned npz splits.

Example:
python ./src/data_management/prep_moment.py 
"""

PERCISION = 6

DATASETS = [
    "ctu",
    "emg",
    "had",
    "har",
    "rwc",
    "tee",
]
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# --------------------------
# IO helpers
# --------------------------
def _load_npz_split(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    # your naming
    if "X_train" in keys and "y_train" in keys:
        return d["X_train"], d["y_train"]
    if "X_test" in keys and "y_test" in keys:
        return d["X_test"], d["y_test"]

    # fallback common naming
    if "X" in keys and "y" in keys:
        return d["X"], d["y"]

    raise KeyError(f"Could not find X/y arrays in {npz_path}. Keys={sorted(keys)}")


def _ensure_n_t_c(X: np.ndarray) -> np.ndarray:
    """
    Accept:
      - (N, T) -> (N, T, 1)
      - (N, T, C) -> unchanged
      - (T, C) or (T,) (single sample) -> add N=1
    """
    X = np.asarray(X)

    if X.ndim == 1:          # (T,)
        X = X[None, :, None] # (1,T,1)
    elif X.ndim == 2:
        # ambiguous: could be (N,T) or (T,C)
        # In your pipeline, stacked samples => (N,T,...) so treat as (N,T)
        # If you truly have single sample multivariate, pass a 3D array.
        X = X[:, :, None]    # (N,T,1)
    elif X.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected X with ndim 1/2/3, got shape={X.shape}")

    return X


def _format_channel(values: np.ndarray, precision: int) -> str:
    # values: (T,)
    # avoid scientific notation by default; keep it stable
    # trim trailing zeros but keep precision
    fmt = f"{{:.{precision}f}}"
    return ",".join(fmt.format(float(v)).rstrip("0").rstrip(".") if precision > 0 else str(float(v))
                    for v in values)


def _format_case(x_tc: np.ndarray, y: int, precision: int) -> str:
    # x_tc: (T,C)
    T, C = x_tc.shape
    chans = [_format_channel(x_tc[:, c], precision) for c in range(C)]
    return " : ".join(chans) + f" : {int(y)}"


def _infer_class_values(y: np.ndarray) -> List[int]:
    y = np.asarray(y).astype(int)
    return sorted(set(int(v) for v in np.unique(y)))


def _read_label_maps(label_maps_path: Path) -> Optional[List[int]]:
    if not label_maps_path.exists():
        return None
    try:
        m = json.loads(label_maps_path.read_text())
        # stored as strings in your pipeline
        ids = list(m.get("id_to_letter", {}).keys())
        if ids:
            return sorted(int(i) for i in ids)
    except Exception:
        return None
    return None


# --------------------------
# .ts writer
# --------------------------
def write_ts_file(
    out_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    problem_name: str,
    class_values: List[int],
    precision: int = 6,
) -> None:
    """
    Writes a sktime-compatible .ts:
      - @timestamps false
      - @univariate true/false
      - @equalLength true
      - @seriesLength T
      - @classLabel true <values...>
    """
    X = _ensure_n_t_c(X)
    y = np.asarray(y).astype(int)

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)} len(y)={len(y)}")

    N, T, C = X.shape
    univariate = (C == 1)

    lines: List[str] = []
    lines.append(f"@problemName {problem_name}")
    lines.append("@timestamps false")
    lines.append(f"@univariate {'true' if univariate else 'false'}")
    lines.append("@equalLength true")
    lines.append(f"@seriesLength {T}")
    lines.append("@missing false")
    # sktime expects: @classLabel true <class1> <class2> ...
    lines.append("@classLabel true " + " ".join(str(int(v)) for v in class_values))
    lines.append("@data")

    for i in range(N):
        lines.append(_format_case(X[i], y[i], precision))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
      # decimal places for float formatting
    for dataset in DATASETS:
        split_dir = Path(f"/raid/hdd249/data/datasets/{dataset}")
        out_dir = Path(f"/raid/hdd249/data/moment/{dataset}")

        X_train, y_train = _load_npz_split(split_dir / "train.npz")
        X_test,  y_test  = _load_npz_split(split_dir / "test.npz")

        # Determine class values for header
        class_values = _read_label_maps(split_dir / "label_maps.json")
        if class_values is None:
            class_values = sorted(set(_infer_class_values(y_train) + _infer_class_values(y_test)))

        problem_name = dataset.strip().upper()

        write_ts_file(
            out_dir / f"{problem_name}_TRAIN.ts",
            X_train,
            y_train,
            problem_name=problem_name,
            class_values=class_values,
            precision=PERCISION,
        )
        write_ts_file(
            out_dir / f"{problem_name}_TEST.ts",
            X_test,
            y_test,
            problem_name=problem_name,
            class_values=class_values,
            precision=PERCISION,
        )

        print(f"✅ Wrote: {out_dir / f'{problem_name}_TRAIN.ts'}")
        print(f"✅ Wrote: {out_dir / f'{problem_name}_TEST.ts'}")
        print(f"   Shapes: train={_ensure_n_t_c(X_train).shape}, test={_ensure_n_t_c(X_test).shape}")
        print(f"   Classes: {class_values}")


if __name__ == "__main__":
    main()
