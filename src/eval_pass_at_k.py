"""
python src/eval_pass_at_k.py --dataset ctu --embedding_types vis,ust
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils.file_io import load_jsonl, append_tsv


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if c <= 0 or k <= 0:
        return 0.0
    if k > n:
        raise ValueError(f"k={k} cannot be > n={n}")
    if (n - c) < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["ctu", "emg", "had","har","rwc","tee"], required=True)
    p.add_argument("--embedding_types", type=str, choices=["vis-ust", "lets-ust", "lets-vis-ust"], required=True)
    p.add_argument("--n", type=int, default=20, help="number of samples per idx in the jsonl")
    p.add_argument("--out_tsv", type=str, default="./data/results/pass_at_k.tsv")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = Path(
        f"/raid/hdd249/data/sample_generations/together-qwen/{args.dataset}/pass-at-k/"
        f"{args.embedding_types.replace(',', '-')}_cot0_n{args.n}.jsonl"
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    ks = [1, 5, 10, 15, 20]

    data = load_jsonl(str(in_path))
    df = pd.DataFrame(data)

    # basic sanity checks
    for col in ["idx", "correct"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in jsonl rows. Found: {list(df.columns)}")

    # compute pass@k per idx, then average across idx
    per_idx = []
    for idx, g in df.groupby("idx"):
        correct = g["correct"].astype(int).to_numpy()
        n_i = len(correct)

        # If your file guarantees exactly args.n per idx, you can enforce it:
        if n_i != args.n:
            raise ValueError(f"idx={idx} has n={n_i} samples, expected args.n={args.n}")

        c_i = int(correct.sum())
        row = {"idx": idx, "n": n_i, "c": c_i}
        for k in ks:
            row[f"pass@{k}"] = pass_at_k_estimator(n=n_i, c=c_i, k=k)
        per_idx.append(row)

    per_df = pd.DataFrame(per_idx)

    out_row = {
        "dataset": args.dataset,
        "embed_types": args.embedding_types,
        "pass@1": float(per_df["pass@1"].mean()),
        "pass@5": float(per_df["pass@5"].mean()),
        "pass@10": float(per_df["pass@10"].mean()),
        "pass@15": float(per_df["pass@15"].mean()),
        "pass@20": float(per_df["pass@20"].mean()),
        "n_idxs": int(len(per_df)),
        "n_per_idx": int(args.n),
    }

    # append ONE line to TSV
    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    append_tsv(args.out_tsv, out_row)

    # also print a single human-readable line
    print(
        f"{out_row['dataset']}\t{out_row['embed_types']}\t"
        f"{out_row['pass@1']:.6f}\t"
        f"{out_row['pass@5']:.6f}\t{out_row['pass@10']:.6f}\t"
        f"{out_row['pass@15']:.6f}\t{out_row['pass@20']:.6f}"
    )


if __name__ == "__main__":
    main()
