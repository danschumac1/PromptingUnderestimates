'''
How to run:
   python ./src/knn.py \
    --dataset har \
    --model_stem qwen \
    --embedding_types ts \
    --layer -1

Uses GridSearchCV to find best K value.
'''


import argparse
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from utils.file_io import append_jsonl, load_embeddings
from utils.loaders import load_train_test
from eval import accuracy_score, f1_score


def extract_scalar(x):
    return x.item() if hasattr(x, "item") else x


# Default grid search parameters
DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
DEFAULT_CV_FOLDS = 5

EMBEDDING_TYPES = [
    "ts",           # ✅
    "vis",          # ✅
    "lets",         # ✅
    "ts_vis",       # ✅
    "ts_ust",       # ✅
    "vis_lets",     # 🏃
    "vis_ust",      # ✅
    "lets_ust",     # 🏃
    "ts_vis_ust",   # ✅
    "vis_lets_ust", # ❌
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN classification with GridSearchCV on per-layer embeddings."
    )
    # REQUIRED arguments
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset to run KNN on",
    )
    p.add_argument(
        "--model_stem", type=str, choices=["llama", "mistral", "qwen"], required=True,
        help="Which model features ran on (points to path)",
    )
    p.add_argument(
        "--embedding_types", type=str, required=True,
        choices=EMBEDDING_TYPES,
        help="Type of embedding to use",
    )
    # LAYER SELECTION
    p.add_argument(
        "--layer", type=int, default=-1,
        help="Which layer's embeddings to use. Use -1 for last layer, or 0..N for specific layer.",
    )
    # OPTIONAL: normalize
    p.add_argument(
        "--normalize", type=int, default=1, choices=[0, 1],
        help="Whether to normalize data in load_train_test",
    )
    # OPTIONAL: cross-validation folds
    p.add_argument(
        "--cv", type=int, default=DEFAULT_CV_FOLDS,
        help=f"Number of cross-validation folds (default: {DEFAULT_CV_FOLDS})",
    )
    # OPTIONAL: custom K values (comma-separated)
    p.add_argument(
        "--k_values", type=str, default=None,
        help="Comma-separated K values to search (default: 1,3,5,7,9,11,15,21)",
    )
    # OPTIONAL: distance metric
    p.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance metric for KNN (default: cosine)",
    )
    return p.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    args = parse_args()

    # Parse K values (use defaults if not provided)
    k_values = (
        [int(x) for x in args.k_values.split(",")]
        if args.k_values else DEFAULT_K_VALUES
    )

    print(f"Grid Search Configuration:")
    print(f"  K values: {k_values}")
    print(f"  CV folds: {args.cv}")
    print(f"  Layer: {args.layer}")
    print(f"  Metric: {args.metric}")

    # LOAD ORIGINAL DATA
    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # LOAD EMBEDDINGS (from .npz, selecting specific layer)
    embed_path = (
        f"./data/sample_features/{args.model_stem}/{args.dataset}/"
        f"{args.embedding_types}_0-shot/"
    )

    train_embeddings, test_embeddings = load_embeddings(embed_path, layer=args.layer)

    print(f"Loaded embeddings from layer {args.layer}:")
    print(f"  Train shape: {train_embeddings.shape}")
    print(f"  Test shape: {test_embeddings.shape}")

    # make sure same number of embeddings as samples
    assert train_embeddings.shape[0] == len(train), \
        "mismatch in number of train embeddings and samples"
    assert test_embeddings.shape[0] == len(test), \
        "mismatch in number of test embeddings and samples"
    assert train_embeddings.shape[1] == test_embeddings.shape[1], \
        "train and test embeddings must have the same feature dimension"

    # Output path includes layer info
    layer_str = f"layer{args.layer}"
    outdir = (
        f"./data/sample_generations/{args.model_stem}/{args.dataset}/"
        f"knn/{args.embedding_types}/{layer_str}/"
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "predictions.jsonl")
    results_path = os.path.join(outdir, "best_params.txt")

    # clear / create file
    with open(outpath, "w") as f:
        pass

    # ------------------------------------------------------------------
    # TRAIN KNN WITH GRID SEARCH CV
    # ------------------------------------------------------------------
    y_train = np.array([sample.y for sample in train]).ravel()

    param_grid = {
        "n_neighbors": k_values,
    }

    base_knn = KNeighborsClassifier(metric=args.metric, n_jobs=-1)

    grid_search = GridSearchCV(
        base_knn,
        param_grid,
        cv=args.cv,
        scoring="f1_macro",  # optimize for macro F1
        n_jobs=-1,
        verbose=1,
        refit=True,  # refit best model on full training data
    )

    print(f"\nRunning GridSearchCV with {len(k_values)} K values...")
    grid_search.fit(train_embeddings, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_knn = grid_search.best_estimator_

    print(f"\n✅ GridSearchCV Complete!")
    print(f"  Best K: {best_params['n_neighbors']}")
    print(f"  Best CV F1 (macro): {best_cv_score:.4f}")

    # ------------------------------------------------------------------
    # PREDICT WITH BEST MODEL
    # ------------------------------------------------------------------
    predictions = best_knn.predict(test_embeddings)

    # ------------------------------------------------------------------
    # EVAL + SAVE
    # ------------------------------------------------------------------
    y_true = [extract_scalar(sample.y) for sample in test]
    y_pred = [extract_scalar(pred) for pred in predictions]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    print(f"\n📊 Test Set Results (embedding={args.embedding_types}, layer={args.layer}):")
    print(f"  Best K: {best_params['n_neighbors']}")
    print(f"  Metric: {args.metric}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    # Save predictions
    for row, pred in zip(test, predictions):
        line = {
            "idx": extract_scalar(row.idx),
            "gt": extract_scalar(row.y),
            "pred": extract_scalar(pred),
        }
        append_jsonl(outpath, line)

    # Save best parameters and results summary
    with open(results_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_stem}\n")
        f.write(f"Embedding: {args.embedding_types}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"CV folds: {args.cv}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"\nGrid Search:\n")
        f.write(f"  K values: {k_values}\n")
        f.write(f"\nBest Parameters:\n")
        f.write(f"  K: {best_params['n_neighbors']}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Best CV F1 (macro): {best_cv_score:.4f}\n")
        f.write(f"  Test Accuracy: {acc:.4f}\n")
        f.write(f"  Test F1 (macro): {f1:.4f}\n")

    print(f"\n💾 Saved predictions to {outpath}")
    print(f"💾 Saved best params to {results_path}")
