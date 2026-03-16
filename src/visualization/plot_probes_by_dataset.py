#!/usr/bin/env python3
"""
python ./src/visualization/plot_probes_by_dataset.py \
  --in_path "./data/plot_ready.tsv" \
  --out_path "./images/visualizations/probes/qwen_vis-lets-ust_by_dataset.png" \
  --model "qwen" \
  --embedding_type "vis-lets-ust" \
  --metric "macro_f1"

One plot:
- filter to 1 model + 1 embedding_type (+ method=logistic_regression)
- X: layer
- Y: metric
- Lines (legend/colors): dataset
- Aggregation: mean over duplicates at (dataset, layer)
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_path", type=str, default="./images/visualizations/probes/qwen_vis-lets-ust_by_dataset.png")
    p.add_argument("--model", type=str, default="qwen")
    p.add_argument("--embedding_type", type=str, default="vis-lets-ust")
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--method", type=str, default="logistic_regression")
    return p.parse_args()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_path, sep="\t")
    print(f"[INFO] loaded {len(df)} rows from {args.in_path}")

    required = {"model", "dataset", "method", "embedding_type", "layer", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["layer", args.metric]).copy()
    df["layer"] = df["layer"].astype(int)

    # filters
    df = df[df["method"] == args.method].copy()
    df = df[df["model"].astype(str) == str(args.model)].copy()
    df = df[df["embedding_type"].astype(str) == str(args.embedding_type)].copy()

    print(f"[INFO] after filters: rows={len(df)}")
    if df.empty:
        raise ValueError(
            f"No rows after filtering method={args.method} model={args.model} embedding_type={args.embedding_type}"
        )

    # aggregate (in case of duplicates)
    agg = (
        df.groupby(["dataset", "layer"], as_index=False)[[args.metric]]
          .mean()
          .sort_values(["dataset", "layer"])
    )

    # plot
    out_dir = os.path.dirname(args.out_path) or "."
    safe_mkdir(out_dir)

    plt.figure(figsize=(10, 6))

    for dataset in sorted(agg["dataset"].astype(str).unique()):
        sub = agg[agg["dataset"].astype(str) == str(dataset)].sort_values("layer")
        plt.plot(sub["layer"], sub[args.metric], marker="o", label=str(dataset))

    plt.title(f"{args.model.upper()} • {args.embedding_type} • {args.metric} (hue=dataset)")
    plt.xlabel("Layer")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(title="dataset", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {args.out_path}")


if __name__ == "__main__":
    main()
