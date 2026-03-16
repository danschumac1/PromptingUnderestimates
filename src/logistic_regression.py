"""

How to run (examples):
  # combo embeddings (folder contains train_embeddings.npz / test_embeddings.npz)
  python ./src/logistic_regression.py --dataset ctu --model_stem llama --embedding_types ts-ust --CoT_string Direct

  # slike baseline (files train_slike.npz / test_slike.npz live at dataset root)
  python ./src/logistic_regression.py --dataset ctu --model_stem llama --embedding_types slike

Uses GridSearchCV to find best C and max_iter values per layer.
Saves:
  /raid/hdd249/Classification_v2/data/sample_generations/{model}/{dataset}/logistic_regression/{tag}/layer{L}.jsonl
  /raid/hdd249/Classification_v2/data/sample_generations/{model}/{dataset}/logistic_regression/{tag}/best_params/layer{L}.jsonl
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
# DATA_DIR = "./data/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logistic regression with GridSearchCV on per-layer embeddings."
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_stem", required=True, type=str, choices=[
        "llama", 
        "mistral", 
        "qwen", 
        "random_llama", 
        "random_mistral", 
        "random_qwen", 
    ])
    p.add_argument("--embedding_types", type=str, required=True, choices=EMBEDDING_TYPES)

    # Only relevant for non-slike embeddings
    p.add_argument("--CoT_string", type=str, default=None, choices=["CoT", "Direct"])

    p.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    p.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)

    p.add_argument("--c_values", type=str, default=None)
    p.add_argument("--max_iter_values", type=str, default=None)
    p.add_argument("--shot", type=int, default=0)

    return p.parse_args()


def _layer_sort_key(k: str):
    """
    Robust layer key sorting.
    - If k is numeric (e.g., "0", "31"), sort by int(k)
    - Otherwise fallback to string sort
    """
    return (0, int(k)) if k.lstrip("-").isdigit() else (1, k)


def build_embed_paths(
        data_dir: str, model: str, dataset: str, embed_type: str, cot: str | None, shot: int = 0):
    """
    Returns: (train_npz_path, test_npz_path, out_tag)

    Disk layout (per your example tree):
      - combo embeds:
          {data_dir}/sample_features/{model}/{dataset}/{embed_type}_0-shot_{cot}/train__embeddings.npz
          {data_dir}/sample_features/{model}/{dataset}/{embed_type}_0-shot_{cot}/test__embeddings.npz

      - slike baseline:
          {data_dir}/sample_features/{model}/{dataset}/train_slike.npz
          {data_dir}/sample_features/{model}/{dataset}/test_slike.npz
    """
    root = os.path.join(data_dir, "sample_features", model, dataset)

    if embed_type == "slike":
        train_path = os.path.join(root, "train_slike.npz")
        test_path = os.path.join(root, "test_slike.npz")
        out_tag = "slike"
    else:
        if cot is None:
            raise ValueError("--CoT_string is required for non-slike embedding_types (CoT or Direct).")
        folder = f"{embed_type}_{shot}-shot_{cot}"
        train_path = os.path.join(root, folder, "train__embeddings.npz")
        test_path = os.path.join(root, folder, "test__embeddings.npz")
        out_tag = folder

    return train_path, test_path, out_tag


def build_out_root(data_dir: str, model: str, dataset: str, out_tag: str) -> str:
    """
    Output layout:
      {data_dir}/sample_generations/{model}/{dataset}/logistic_regression/{out_tag}/
    """
    return os.path.join(
        data_dir,
        "sample_generations",
        model,
        dataset,
        "logistic_regression",
        out_tag,
    )


if __name__ == "__main__":
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
    # LOAD EMBEDDINGS (FIXED PATHING)
    # ------------------------------------------------------------------
    train_npz_path, test_npz_path, out_tag = build_embed_paths(
        DATA_DIR,
        args.model_stem,
        args.dataset,
        args.embedding_types,
        args.CoT_string,
        shot=args.shot,
    )

    print("\nUsing embeddings:")
    print(f"  train: {train_npz_path}")
    print(f"  test : {test_npz_path}")

    if not os.path.exists(train_npz_path):
        raise FileNotFoundError(f"Missing train embeddings: {train_npz_path}")
    if not os.path.exists(test_npz_path):
        raise FileNotFoundError(f"Missing test embeddings: {test_npz_path}")

    train_data = np.load(train_npz_path)  # NpzFile
    test_data = np.load(test_npz_path)

    train_keys = list(train_data.files)
    test_keys = list(test_data.files)

    print("\nTrain layers:", train_keys[:5], "...", f"(n={len(train_keys)})")
    print("Test layers :", test_keys[:5],  "...", f"(n={len(test_keys)})")
    assert set(train_keys) == set(test_keys), "train/test NPZ layer keys differ"

    layer_keys = sorted(train_keys, key=_layer_sort_key)

    # ------------------------------------------------------------------
    # OUTPUT ROOTS (FIXED PATHING)
    # ------------------------------------------------------------------
    out_root = build_out_root(DATA_DIR, args.model_stem, args.dataset, out_tag)
    os.makedirs(out_root, exist_ok=True)

    best_params_dir = os.path.join(out_root, "best_params")
    os.makedirs(best_params_dir, exist_ok=True)

    print("\nOutputs will be written to:")
    print(f"  out_root: {out_root}")
    print(f"  best_params_dir: {best_params_dir}")

    # ------------------------------------------------------------------
    # TRAIN/EVAL PER LAYER
    # ------------------------------------------------------------------
    y_train = np.array([extract_scalar(s.y) for s in train]).ravel()
    y_true = [extract_scalar(s.y) for s in test]

    param_grid = {"C": c_values, "max_iter": max_iter_values}
    base_clf = LogisticRegression(solver="lbfgs", n_jobs=-1)

    for layer_key in tqdm(layer_keys, desc="GridSearchCV per layer"):
        train_embeddings = train_data[layer_key]
        test_embeddings = test_data[layer_key]

        # sanity checks
        assert train_embeddings.shape[0] == len(train), "mismatch in number of train embeddings and samples"
        assert test_embeddings.shape[0] == len(test), "mismatch in number of test embeddings and samples"
        assert train_embeddings.shape[1] == test_embeddings.shape[1], "train/test embeddings must have same feature dimension"

        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=args.cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        grid_search.fit(train_embeddings, y_train)

        best_params = grid_search.best_params_
        best_cv_score = float(grid_search.best_score_)
        best_clf = grid_search.best_estimator_

        predictions = best_clf.predict(test_embeddings)
        y_pred = [extract_scalar(p) for p in predictions]

        acc = float(accuracy_score(y_true=y_true, y_pred=y_pred))
        f1 = float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

        # print(
        #     f"\nLayer {layer_key} | "
        #     f"best_params={best_params} | best_cv_f1={best_cv_score:.4f} | "
        #     f"test_acc={acc:.4f} test_f1={f1:.4f}"
        # )

        # --------------------------------------------------------------
        # SAVE PREDICTIONS
        # --------------------------------------------------------------
        pred_path = os.path.join(out_root, f"layer{layer_key}.jsonl")
        with open(pred_path, "w") as f:
            pass

        for row, pred in zip(test, predictions):
            append_jsonl(pred_path, {
                "idx": extract_scalar(row.idx),
                "gt": extract_scalar(row.y),
                "pred": extract_scalar(pred),
                "layer": layer_key,
            })

        # --------------------------------------------------------------
        # SAVE PARAMS
        # --------------------------------------------------------------
        params_path = os.path.join(best_params_dir, f"layer{layer_key}.jsonl")
        with open(params_path, "w") as f:
            pass

        append_jsonl(params_path, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model_stem": args.model_stem,
            "embedding_types": args.embedding_types,
            "cot": args.CoT_string,
            "layer": layer_key,
            "cv_folds": args.cv,
            "grid": {
                "C": c_values,
                "max_iter": max_iter_values,
            },
            "best_params": best_params,
            "best_cv_f1_macro": best_cv_score,
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "pred_path": pred_path,
        })

    print(f"\n✅ Done. Saved predictions under: {out_root}")
    print(f"✅ Saved params under: {best_params_dir}")
