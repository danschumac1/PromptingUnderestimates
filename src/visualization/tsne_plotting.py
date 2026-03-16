'''
python ./src/visualization/tsne_plotting.py
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys; sys.path.append('./src')
from utils.file_io import load_embeddings
from utils.loaders import load_train_test

DATASET= "rwc"
MODEL = "moment"
EMBED_TYPES = "vis-ust"
SHOT_STR = "0-shot"
METHOD = "Direct"
TRAIN_TEST = "test"

def main():
    input_data_path = f"/raid/hdd249/data/samples/{DATASET}/"
    if MODEL == "moment":
        embeddings_path = f"/raid/hdd249/data/sample_features/moment/{DATASET}"
        emb_file = f"{embeddings_path}/{TRAIN_TEST}__embeddings.npy"
    else:
        embeddings_path = (
            f"/raid/hdd249/data/sample_features/{MODEL}/{DATASET}/"
            f"{EMBED_TYPES}_{SHOT_STR}_{METHOD}"
        )
        emb_file = f"{embeddings_path}/{TRAIN_TEST}__embeddings.npz"

    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"Missing embeddings file: {emb_file}")

    X_train, X_test = load_embeddings(embeddings_path)

    train, test = load_train_test(input_data_path, n_shots=0)
    y_train = [train[i].y for i in range(len(train))]; y_test = [test[i].y for i in range(len(test))]

    if TRAIN_TEST == "train":
        X = X_train
        y = y_train
    elif TRAIN_TEST == "test":
        X = X_test
        y = y_test
    # print(len(y_train), len(y_test))

    # Run t-SNE
    X_embedded = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        perplexity=3,
        random_state=42,
    ).fit_transform(X)

    print("t-SNE shape:", X_embedded.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y,
        s=22,
        alpha=0.9,
        linewidths=0,
    )

    # Styling
    ax.set_title("")                      # remove title
    ax.set_xlabel("Dim 1", fontsize=18)   # big X label
    ax.set_ylabel("Dim 2", fontsize=18)   # big Y label
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(False)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Square panel
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    plt.tight_layout()
    out_png = f"./images/tsne/{DATASET}_{MODEL}_{EMBED_TYPES}_{SHOT_STR}_{METHOD}.png"
    out_pdf = out_png.replace("png","pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.show()

    print(f"t-SNE plot saved to {out_png} and {out_pdf} ")

if __name__ == "__main__":
    main()
