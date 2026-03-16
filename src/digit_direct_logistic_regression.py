'''
How to run:
   python ./src/digit_direct_logistic_regression.py
'''


import argparse
import os
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.file_io import append_jsonl
from utils.loaders import load_train_test
from eval import accuracy_score, f1_score


def extract_scalar(x):
    return x.item() if hasattr(x, "item") else x


# ----------------------------
# DEFAULT GRID SEARCH CONFIG
# ----------------------------
DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_MAX_ITER_VALUES = [1000]
DEFAULT_CV_FOLDS = 5

# Embedding types you actually have on disk
EMBEDDING_TYPES = [
    "ts-ust",
    "vis-ust",
    "lets-ust",
    "ts-vis-ust",
    "vis-lets-ust",
    "slike",
]

# Root data dir (absolute)
DATA_DIR = "/raid/hdd249/data"




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logistic regression with GridSearchCV on per-layer embeddings."
    )
    p.add_argument("--dataset", type=str, required=True)
    # Only relevant for non-slike embeddings

    p.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    p.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)
    p.add_argument("--c_values", type=str, default=None)
    p.add_argument("--max_iter_values", type=str, default=None)
    p.add_argument("--shot", type=int, default=0)
    return p.parse_args()


def main():
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    args = parse_args()

    c_values =  DEFAULT_C_VALUES
    max_iter_values = DEFAULT_MAX_ITER_VALUES

    print("Grid Search Configuration:")
    print(f"  C values: {c_values}")
    print(f"  max_iter values: {max_iter_values}")
    print(f"  CV folds: {args.cv}")

    # ------------------------------------------------------------------
    # LOAD ORIGINAL DATA
    # ------------------------------------------------------------------
    train, test = load_train_test(
        os.path.join(DATA_DIR, "samples", args.dataset),
        0,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # ------------------------------------------------------------------
    # OUTPUT ROOTS (FIXED PATHING)
    # ------------------------------------------------------------------
    out_root = f"/raid/hdd249/data/generation/digit_direct_logistic_regression/{args.dataset}/"
    os.makedirs(out_root, exist_ok=True)


    print("\nOutputs will be written to:")
    print(f"  out_root: {out_root}")

    # ------------------------------------------------------------------
    # TRAIN/EVAL PER LAYER
    # ------------------------------------------------------------------
    # 1. Ensure 2D shape for sklearn (Flattening)
    X_train = np.array([s.X.ravel() for s in train])
    X_test = np.array([s.X.ravel() for s in test])
    
    # Split.y is (N,). Ensure it's a flat 1D array for sklearn
    y_train = np.array([extract_scalar(val) for val in train.y]).ravel()
    y_true = [extract_scalar(val) for val in test.y]
    # 2. Add class_weight to handle potential imbalance
    param_grid = {"C": c_values, "max_iter": max_iter_values}
    base_clf = LogisticRegression(solver="lbfgs", n_jobs=-1, class_weight="balanced")

    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=args.cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)
    best_clf = grid_search.best_estimator_

    predictions = best_clf.predict(X_test)
    y_pred = [extract_scalar(p) for p in predictions]

    acc = float(accuracy_score(y_true=y_true, y_pred=y_pred))
    f1 = float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

    pred_path = os.path.join(out_root, "preds.jsonl")
    with open(pred_path, "w") as f:
        pass

    for row, pred in zip(test, predictions):
        append_jsonl(pred_path, {
            "idx": extract_scalar(row.idx),
            "gt": extract_scalar(row.y),
            "pred": extract_scalar(pred),
            "correct": extract_scalar(pred) == extract_scalar(row.y),
        })

    # total acc and f1 
    metrics_path = os.path.join(out_root, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        pass
    append_jsonl(metrics_path, {
        "accuracy": acc,
        "f1_macro": f1,
        "best_cv_f1_macro": best_cv_score,
    })

    print(f"\n📊 Evaluation Metrics for {args.dataset.upper()}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")

    print(f"\n✅ Done. Saved predictions under: {out_root}")


if __name__ == "__main__":
    main()