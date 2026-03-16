"""
python ./src/moment_logistic_regresion.py \
    --dataset had
"""

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

# Root data dir (absolute)
DATA_DIR = "/raid/hdd249/data"
OUT_ROOT = "/raid/hdd249/data/sample_generations/moment/DATASET/logistic_regression"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logistic regression with GridSearchCV on MOMENT embeddings (single layer saved)."
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    return p.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    args = parse_args()
    print("Grid Search Configuration:")
    print(f"  C values: {DEFAULT_C_VALUES}")
    print(f"  max_iter values: {DEFAULT_MAX_ITER_VALUES}")
    print(f"  CV folds: {DEFAULT_CV_FOLDS}")

    # Fixed layer tag for MOMENT embeddings (single saved layer)
    layer_key = 999

    # ------------------------------------------------------------------
    # LOAD ORIGINAL DATA (for labels + idx)
    # ------------------------------------------------------------------
    train, test = load_train_test(
        os.path.join(DATA_DIR, "datasets", args.dataset),
        0,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # ------------------------------------------------------------------
    # LOAD EMBEDDINGS (.NPY)
    # ------------------------------------------------------------------
    train_emb_path = f"/raid/hdd249/data/sample_features/moment/{args.dataset}/train__embeddings.npy"
    test_emb_path = f"/raid/hdd249/data/sample_features/moment/{args.dataset}/test__embeddings.npy"
    print("\nUsing embeddings:")
    print(f"  train: {train_emb_path}")
    print(f"  test : {test_emb_path}")

    if not os.path.exists(train_emb_path):
        raise FileNotFoundError(f"Missing train embeddings: {train_emb_path}")
    if not os.path.exists(test_emb_path):
        raise FileNotFoundError(f"Missing test embeddings: {test_emb_path}")

    train_embeddings = np.load(train_emb_path)
    test_embeddings = np.load(test_emb_path)

    print(
        "LOADED embeddings from NPY files.",
        "Shapes:",
        " train:", train_embeddings.shape,
        " test:", test_embeddings.shape,
    )

    # ------------------------------------------------------------------
    # OUTPUT ROOTS
    # ------------------------------------------------------------------
    out_root = OUT_ROOT.replace("DATASET", args.dataset)
    os.makedirs(out_root, exist_ok=True)

    best_params_dir = os.path.join(out_root, "best_params")
    os.makedirs(best_params_dir, exist_ok=True)

    print("\nOutputs will be written to:")
    print(f"  out_root: {out_root}")
    print(f"  best_params_dir: {best_params_dir}")

    # ------------------------------------------------------------------
    # LABELS
    # ------------------------------------------------------------------
    y_train = np.array([extract_scalar(s.y) for s in train]).ravel()
    y_true = [extract_scalar(s.y) for s in test]

    # ------------------------------------------------------------------
    # SANITY CHECKS
    # ------------------------------------------------------------------
    assert train_embeddings.ndim == 2, f"Expected train embeddings 2D (N,D), got {train_embeddings.shape}"
    assert test_embeddings.ndim == 2, f"Expected test embeddings 2D (N,D), got {test_embeddings.shape}"
    assert train_embeddings.shape[0] == len(train), "mismatch in number of train embeddings and samples"
    assert test_embeddings.shape[0] == len(test), "mismatch in number of test embeddings and samples"
    assert train_embeddings.shape[1] == test_embeddings.shape[1], "train/test embeddings must have same feature dimension"

    # ------------------------------------------------------------------
    # GRID SEARCH (single layer)
    # ------------------------------------------------------------------
    param_grid = {"C": DEFAULT_C_VALUES, "max_iter": DEFAULT_MAX_ITER_VALUES}
    base_clf = LogisticRegression(solver="lbfgs", n_jobs=-1)

    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=DEFAULT_CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(train_embeddings, y_train)

    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)
    best_clf = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # TEST EVAL
    # ------------------------------------------------------------------
    predictions = best_clf.predict(test_embeddings)
    y_pred = [extract_scalar(p) for p in predictions]

    acc = float(accuracy_score(y_true=y_true, y_pred=y_pred))
    f1 = float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

    print(
        f"\nLayer {layer_key} | "
        f"best_params={best_params} | best_cv_f1_macro={best_cv_score:.4f} | "
        f"test_acc={acc:.4f} test_f1_macro={f1:.4f}"
    )

    # ------------------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------------------
    pred_path = os.path.join(out_root, f"layer{layer_key}.jsonl")
    with open(pred_path, "w") as f:
        pass

    for row, pred in tqdm(list(zip(test, predictions)), desc="Saving preds", ncols=100):
        append_jsonl(
            pred_path,
            {
                "idx": extract_scalar(row.idx),
                "gt": extract_scalar(row.y),
                "pred": extract_scalar(pred),
                "layer": layer_key,
            },
        )

    # ------------------------------------------------------------------
    # SAVE PARAMS / SUMMARY
    # ------------------------------------------------------------------
    params_path = os.path.join(best_params_dir, f"layer{layer_key}.jsonl")
    with open(params_path, "w") as f:
        pass

    append_jsonl(
        params_path,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "normalize": int(args.normalize),
            "layer": layer_key,
            "cv_folds": DEFAULT_CV_FOLDS,
            "grid": {
                "C": DEFAULT_C_VALUES,
                "max_iter": DEFAULT_MAX_ITER_VALUES,
            },
            "best_params": best_params,
            "best_cv_f1_macro": best_cv_score,
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "pred_path": pred_path,
        },
    )

    print(f"\n✅ Done. Saved predictions under: {pred_path}")
    print(f"✅ Saved params under: {params_path}")
