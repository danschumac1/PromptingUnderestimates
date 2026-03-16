# ./src/method/classification/random_baseline.py
"""

Run:
  python ./src/random_baseline.py \
    --input_folder ./data/datasets/har \
    --mode prior 

What it does:
- Loads y_train.npy and y_test.npy (X_* not needed) from --input_folder
- Predicts labels using:
    * uniform : sample uniformly over train label set
    * prior   : sample from empirical train label distribution
    * majority: always predict the majority train label
- Writes JSONL: {"pred": int, "gt": int}
- Prints Accuracy and Macro-F1
"""
import sys, os, json, argparse, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # → adds project root

from collections import Counter
from typing import List, Tuple, Callable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# If your loader lives elsewhere, adjust this import path:
from utils.loaders import load_train_test  # returns (train_split, test_split)


# -------------------- CLI --------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Random baseline for time series classification")
    p.add_argument("--input_folder", type=str, required=True,
                   help="Folder with y_train.npy and y_test.npy (and optionally X_* files)")
    p.add_argument("--mode", type=str, default="prior",
                   choices=["uniform", "prior", "majority"],
                   help="Sampling strategy")
    return p.parse_args(argv)


# -------------------- Sampler builder --------------------
def make_sampler(args, train, rng: random.Random) -> Callable[[], int]:
    """
    Build a label sampler based on args.mode.
    Uses TRAIN split for label set and prior.
    """
    labels: List[int] = sorted(map(int, np.unique(train.y)))
    if not labels:
        raise ValueError("No labels found in train split.")

    mode = args.mode

    if mode == "uniform":
        def sampler() -> int:
            return rng.choice(labels)

    elif mode == "prior":
        counts = Counter(map(int, train.y))
        probs = np.array([counts[l] for l in labels], dtype=float)
        probs /= probs.sum()

        prior_path = os.path.splitext(args.out_file)[0] + ".prior.json"
        os.makedirs(os.path.dirname(prior_path), exist_ok=True)
        with open(prior_path, "w", encoding="utf-8") as f:
            json.dump({str(l): float(p) for l, p in zip(labels, probs)}, f, indent=4)

        def sampler() -> int:
            return int(np.random.choice(labels, p=probs))

    elif mode == "majority":
        majority_label = Counter(map(int, train.y)).most_common(1)[0][0]
        def sampler() -> int:
            return majority_label
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sampler


# -------------------- Main --------------------
def main():
    args = parse_args()

    # Seed RNGs
    rng = random.Random(42)
    np.random.seed(42)

    # I/O paths
    dataset = os.path.basename(os.path.normpath(args.input_folder))
    out_dir = os.path.join("/raid/hdd249/data/sample_generations/no_model", dataset, "random")

    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, f"{args.mode}.jsonl")

    # Load data (X not needed for random baseline)
    train, test = load_train_test(args.input_folder, n_shots=0)

    # Warn if test contains unseen labels
    labels_train = set(map(int, np.unique(train.y)))
    unseen = sorted(set(map(int, np.unique(test.y))) - labels_train)
    if unseen:
        print(f"⚠️  Test contains unseen labels (ignored for sampling): {unseen}")

    # Sampler
    sampler = make_sampler(args, train, rng)

    # Predict
    preds = [int(sampler()) for _ in range(len(test.y))]

    # Save JSONL
    with open(args.out_file, "w", encoding="utf-8") as f:
        for idx, pred, gt in zip(test.idx.tolist(), preds, test.y.tolist()):
            f.write(json.dumps({"idx": idx, "pred": pred, "gt": int(gt)}) + "\n")

    # Metrics
    acc = accuracy_score(test.y, preds)
    f1m = f1_score(test.y, preds, average="macro")
    print(f"Dataset    : {dataset}")
    print(f"Mode       : {args.mode}")
    print(f"Seed       : {42}")
    print(f"Train size : {len(train.y)} | Test size: {len(test.y)}")
    print(f"Labels     : {sorted(map(int, labels_train))}")
    print(f"Accuracy   : {acc:.4f} ({acc:.2%})")
    print(f"Macro F1   : {f1m:.4f} ({f1m:.2%})")
    print(f"Saved JSONL: {args.out_file}")

if __name__ == "__main__":
    main()