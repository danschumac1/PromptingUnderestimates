#!/usr/bin/env python3
"""
Make two t-SNE panels side-by-side:

Left : mistral / had / test  (NPZ in: /raid/hdd249/data/sample_features/mistral/had/{EMB}_{SHOT}_{METHOD}/test__embeddings.npz)
Right: moment  / had / test  (NPY in: /raid/hdd249/data/sample_features/moment/had/test__embeddings.npy)

Run:
python ./src/visualization/tsne_mistral_vs_moment_had_test.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import sys; sys.path.append("./src")
from utils.file_io import load_embeddings
from utils.loaders import load_train_test


# -------------------------
# CONFIG
# -------------------------
DATASET = "had"
TRAIN_TEST = "test"

LEFT_MODEL = "llama"
LEFT_EMBED_TYPES = "vis-ust"
LEFT_SHOT_STR = "0-shot"
LEFT_METHOD = "Direct"

RIGHT_MODEL = "moment"  # MOMENT uses .npy directly in sample_features/moment/{dataset}/
RIGHT_EMBED_TYPES = "vis-ust"   # kept for naming only
RIGHT_SHOT_STR = "0-shot"
RIGHT_METHOD = "Direct"

# t-SNE params (keep identical across panels)
TSNE_KWARGS = dict(
    n_components=2,
    learning_rate="auto",
    init="random",
    perplexity=3,
    random_state=42,
)


# -------------------------
# Helpers
# -------------------------
def emb_paths(model: str, dataset: str, embed_types: str, shot_str: str, method: str, train_test: str):
    """
    Returns (embeddings_dir, emb_file_path) and validates the expected layout.
    - moment: /raid/hdd249/data/sample_features/moment/{dataset}/{train_test}__embeddings.npy
    - others: /raid/hdd249/data/sample_features/{model}/{dataset}/{embed_types}_{shot_str}_{method}/{train_test}__embeddings.npz
    """
    if model == "moment":
        embeddings_dir = f"/raid/hdd249/data/sample_features/moment/{dataset}"
        emb_file = f"{embeddings_dir}/{train_test}__embeddings.npy"
    else:
        embeddings_dir = f"/raid/hdd249/data/sample_features/{model}/{dataset}/{embed_types}_{shot_str}_{method}"
        emb_file = f"{embeddings_dir}/{train_test}__embeddings.npz"
    return embeddings_dir, emb_file


def load_xy_for_panel(model: str, dataset: str, embed_types: str, shot_str: str, method: str, train_test: str):
    """
    Loads X (embeddings) and y (labels) for the requested split.
    Uses your existing:
      - load_embeddings(embeddings_dir) -> (X_train, X_test)
      - load_train_test(input_data_path, n_shots=0) -> (train, test)
    """
    input_data_path = f"/raid/hdd249/data/samples/{dataset}/"
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Missing data path: {input_data_path}")

    embeddings_dir, emb_file = emb_paths(model, dataset, embed_types, shot_str, method, train_test)
    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"Missing embeddings file: {emb_file}")

    X_train, X_test = load_embeddings(embeddings_dir)

    train, test = load_train_test(input_data_path, n_shots=0)
    y_train = [train[i].y for i in range(len(train))]
    y_test  = [test[i].y  for i in range(len(test))]

    if train_test == "train":
        return X_train, y_train
    if train_test == "test":
        return X_test, y_test

    raise ValueError(f"train_test must be 'train' or 'test', got: {train_test}")


def style_axes(ax):
    # match your "plot_random_llama_subplots_by_embedding" vibe: no grid, big ticks, despine
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=14)

    # Clean spines (like sns.despine)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # square panel
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass


# -------------------------
# Main
# -------------------------
def main() -> None:
    # load data
    X_left, y_left = load_xy_for_panel(
        model=LEFT_MODEL,
        dataset=DATASET,
        embed_types=LEFT_EMBED_TYPES,
        shot_str=LEFT_SHOT_STR,
        method=LEFT_METHOD,
        train_test=TRAIN_TEST,
    )

    X_right, y_right = load_xy_for_panel(
        model=RIGHT_MODEL,
        dataset=DATASET,
        embed_types=RIGHT_EMBED_TYPES,
        shot_str=RIGHT_SHOT_STR,
        method=RIGHT_METHOD,
        train_test=TRAIN_TEST,
    )

    # run t-SNE separately (same params)
    Z_left = TSNE(**TSNE_KWARGS).fit_transform(X_left)
    Z_right = TSNE(**TSNE_KWARGS).fit_transform(X_right)

    print("[INFO] left  t-SNE shape:", Z_left.shape)
    print("[INFO] right t-SNE shape:", Z_right.shape)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    # left panel
    axes[0].scatter(
        Z_left[:, 0],
        Z_left[:, 1],
        c=y_left,
        s=22,
        alpha=0.9,
        linewidths=0,
    )
    axes[0].set_title(f"{LEFT_MODEL.capitalize()}", fontsize=16, pad=10)
    style_axes(axes[0])

    # right panel
    axes[1].scatter(
        Z_right[:, 0],
        Z_right[:, 1],
        c=y_right,
        s=22,
        alpha=0.9,
        linewidths=0,
    )
    axes[1].set_title(f"{RIGHT_MODEL.capitalize()}", fontsize=16, pad=10)
    style_axes(axes[1])

    # shared labels (match the other script: big, shared)
    # fig.supxlabel("Dim 1", fontsize=18, y=0.02)
    # fig.supylabel("Dim 2", fontsize=18, x=0.05)

    # layout (reserve top band similar to your other script)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_dir = "./images/tsne"
    os.makedirs(out_dir, exist_ok=True)

    out_base = f"{DATASET}__{TRAIN_TEST}__{LEFT_MODEL}_vs_{RIGHT_MODEL}"
    out_png = os.path.join(out_dir, f"{out_base}.png")
    out_pdf = os.path.join(out_dir, f"{out_base}.pdf")

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


if __name__ == "__main__":
    main()
