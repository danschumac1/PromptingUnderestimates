"""

How to run:
  # BERT embeddings live at:
  #   /raid/hdd249/Classification_v2/data/sample_features/bert/{dataset}/{model_tag}/train.npz
  #   /raid/hdd249/Classification_v2/data/sample_features/bert/{dataset}/{model_tag}/test.npz
  python ./src/logistic_regression_bert.py --dataset ctu --model_name bert-large-uncased

What this does
--------------
- Loads 0-shot train/test samples from ./data/samples/{dataset} (for idx/labels)
- Loads precomputed BERT embeddings from .npz (X, y, meta)
- Runs GridSearchCV over LogisticRegression hyperparameters (C, max_iter)
- Evaluates on test split (accuracy + macro-F1)
- Writes:
    /raid/hdd249/Classification_v2/data/sample_generations/bert/{dataset}/logistic_regression/{model_tag}/preds.jsonl
    /raid/hdd249/Classification_v2/data/sample_generations/bert/{dataset}/logistic_regression/{model_tag}/best_params.jsonl

Notes
-----
- This script assumes embeddings were produced by ./src/bert.py and stored as:
    {DATA_DIR}/sample_features/bert/{dataset}/{model_tag}/train.npz
    {DATA_DIR}/sample_features/bert/{dataset}/{model_tag}/test.npz
- We only do a single run (no per-layer loop) since BERT embeddings here are (N,H) once.
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from utils.file_io import append_jsonl
from utils.loaders import load_train_test
from eval import accuracy_score, f1_score


# ----------------------------
# DEFAULT GRID SEARCH CONFIG
# ----------------------------
DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_MAX_ITER_VALUES = [1000]
DEFAULT_CV_FOLDS = 5

# Root data dir (absolute)
DATA_DIR = "/raid/hdd249/Classification_v2/data"


def extract_scalar(x):
    return x.item() if hasattr(x, "item") else x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logistic regression + GridSearchCV on BERT embeddings.")

    p.add_argument("--dataset", type=str, required=True, choices=["ctu", "emg", "had", "har", "rwc", "tee"])
    p.add_argument("--model_name", type=str, default="bert-large-uncased")

    p.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    p.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)

    p.add_argument("--c_values", type=str, default=None, help="Comma-separated list, e.g. 0.1,1,10")
    p.add_argument("--max_iter_values", type=str, default=None, help="Comma-separated list, e.g. 1000,2000")

    return p.parse_args()


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def build_embed_paths(data_dir: str, dataset: str, model_name: str) -> Dict[str, str]:
    """
    Returns dict with:
      train_npz_path, test_npz_path, model_tag
    """
    model_tag = sanitize_model_name(model_name)
    root = os.path.join(data_dir, "sample_features", "bert", dataset, model_tag)
    return {
        "model_tag": model_tag,
        "train_npz_path": os.path.join(root, "train.npz"),
        "test_npz_path": os.path.join(root, "test.npz"),
    }


def build_out_root(data_dir: str, dataset: str, model_tag: str) -> str:
    """
    Output layout:
      {data_dir}/sample_generations/bert/{dataset}/logistic_regression/{model_tag}/
    """
    return os.path.join(
        data_dir,
        "sample_generations",
        "bert",
        dataset,
        "logistic_regression",
        model_tag,
    )


def load_npz_xy_meta(npz_path: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    d = np.load(npz_path, allow_pickle=True)
    if "X" not in d.files or "y" not in d.files:
        raise ValueError(f"{npz_path} missing required keys. Found: {d.files}. Expected at least X and y.")
    X = d["X"]
    y = d["y"]
    meta = {}
    if "meta" in d.files:
        # meta saved as object array containing dict
        meta_obj = d["meta"]
        try:
            meta = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
        except Exception:
            meta = {}
    return X, y, meta


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    args = parse_args()

    c_values = [float(x) for x in args.c_values.split(",")] if args.c_values else DEFAULT_C_VALUES
    max_iter_values = [int(x) for x in args.max_iter_values.split(",")] if args.max_iter_values else DEFAULT_MAX_ITER_VALUES

    print("Grid Search Configuration:")
    print(f"  C values: {c_values}")
    print(f"  max_iter values: {max_iter_values}")
    print(f"  CV folds: {args.cv}")

    # ------------------------------------------------------------------
    # LOAD ORIGINAL DATA (for idx/y alignment)
    # ------------------------------------------------------------------
    train, test = load_train_test(
        os.path.join(DATA_DIR, "samples", args.dataset),
        0,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    y_train_from_data = np.array([extract_scalar(s.y) for s in train]).ravel()
    y_true = [extract_scalar(s.y) for s in test]

    # ------------------------------------------------------------------
    # LOAD EMBEDDINGS
    # ------------------------------------------------------------------
    paths = build_embed_paths(DATA_DIR, args.dataset, args.model_name)
    train_npz_path = paths["train_npz_path"]
    test_npz_path = paths["test_npz_path"]
    model_tag = paths["model_tag"]

    print("\nUsing embeddings:")
    print(f"  train: {train_npz_path}")
    print(f"  test : {test_npz_path}")

    if not os.path.exists(train_npz_path):
        raise FileNotFoundError(f"Missing train embeddings: {train_npz_path}")
    if not os.path.exists(test_npz_path):
        raise FileNotFoundError(f"Missing test embeddings: {test_npz_path}")

    X_train, y_train_from_npz, train_meta = load_npz_xy_meta(train_npz_path)
    X_test, y_test_from_npz, test_meta = load_npz_xy_meta(test_npz_path)

    # Sanity checks
    assert X_train.shape[0] == len(train), f"Train size mismatch: X_train has {X_train.shape[0]} rows, train has {len(train)}"
    assert X_test.shape[0] == len(test), f"Test size mismatch: X_test has {X_test.shape[0]} rows, test has {len(test)}"
    assert X_train.shape[1] == X_test.shape[1], "Train/test embedding dims must match"

    # Prefer labels from the Split object to match your existing eval code,
    # but verify they agree with NPZ labels when possible.
    if y_train_from_npz.shape[0] == y_train_from_data.shape[0]:
        try:
            if not np.all(y_train_from_npz.astype(np.int64) == y_train_from_data.astype(np.int64)):
                print("[WARN] y_train from NPZ differs from y_train in samples. Using samples' y to match pipeline.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # OUTPUT ROOTS
    # ------------------------------------------------------------------
    out_root = build_out_root(DATA_DIR, args.dataset, model_tag)
    os.makedirs(out_root, exist_ok=True)

    preds_path = os.path.join(out_root, "preds.jsonl")
    best_params_path = os.path.join(out_root, "best_params.jsonl")

    # clear outputs
    with open(preds_path, "w") as f:
        pass
    with open(best_params_path, "w") as f:
        pass

    print("\nOutputs will be written to:")
    print(f"  preds: {preds_path}")
    print(f"  best_params: {best_params_path}")

    # ------------------------------------------------------------------
    # GRID SEARCH + FIT
    # ------------------------------------------------------------------
    param_grid = {"C": c_values, "max_iter": max_iter_values}
    base_clf = LogisticRegression(solver="lbfgs", n_jobs=-1)

    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=args.cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train_from_data)

    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)
    best_clf = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # EVAL
    # ------------------------------------------------------------------
    predictions = best_clf.predict(X_test)
    y_pred = [extract_scalar(p) for p in predictions]

    acc = float(accuracy_score(y_true=y_true, y_pred=y_pred))
    f1 = float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

    print("\nResults:")
    print(f"  best_params: {best_params}")
    print(f"  best_cv_f1_macro: {best_cv_score:.4f}")
    print(f"  test_acc: {acc:.4f}")
    print(f"  test_f1_macro: {f1:.4f}")

    # ------------------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------------------
    for row, pred in zip(test, predictions):
        append_jsonl(preds_path, {
            "idx": extract_scalar(row.idx),
            "gt": extract_scalar(row.y),
            "pred": extract_scalar(pred),
        })

    # ------------------------------------------------------------------
    # SAVE PARAMS / SUMMARY
    # ------------------------------------------------------------------
    append_jsonl(best_params_path, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model_name": args.model_name,
        "model_tag": model_tag,
        "cv_folds": int(args.cv),
        "grid": {"C": c_values, "max_iter": max_iter_values},
        "best_params": best_params,
        "best_cv_f1_macro": best_cv_score,
        "test_accuracy": acc,
        "test_f1_macro": f1,
        "pred_path": preds_path,
        "embeddings": {
            "train_npz": train_npz_path,
            "test_npz": test_npz_path,
            "train_meta": train_meta,
            "test_meta": test_meta,
        },
    })

    print(f"\n✅ Done. Saved predictions under: {out_root}")
