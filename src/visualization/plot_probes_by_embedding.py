#!/usr/bin/env python3
"""
python ./src/visualization/plot_probes_by_embedding.py \
  --in_path "./data/plot_ready.tsv" \
  --out_path "./images/visualizations/probes/har_by_embedding.png" \
  --dataset trHARteHAD \
  --metric macro_f1 \
  --model llama

Probe plot for ONE dataset:
- X: layer
- Y: metric (macro_f1 or accuracy)
- Hue: embedding_type
- Filters:
    --dataset <name> (required for the view you want)
    --model <optional> (restrict to one model)
    --method (default logistic_regression)
- Aggregation:
    If --model is NOT provided, averages over models at each (embedding_type, layer).
    If --model IS provided, no cross-model averaging happens (still averages duplicates if any).
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_path", type=str, default="./images/visualizations/probes/by_embedding.png")

    # knobs
    p.add_argument("--dataset", type=str, required=True, help="dataset to plot (e.g., har, had, emg)")
    p.add_argument("--model", type=str, default=None, help="optional: restrict to one model (e.g., qwen)")
    p.add_argument("--method", type=str, default="logistic_regression")
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])

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

    # ---- types ----
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["layer", args.metric]).copy()
    df["layer"] = df["layer"].astype(int)

    # ---- filters ----
    df = df[df["dataset"].astype(str) == str(args.dataset)].copy()
    df = df[df["method"].astype(str) == str(args.method)].copy()

    if args.model is not None:
        df = df[df["model"].astype(str) == str(args.model)].copy()

    if df.empty:
        raise ValueError(
            f"No rows left after filtering dataset={args.dataset} method={args.method} model={args.model}"
        )

    print(f"[INFO] after filtering: {len(df)} rows")

    # ---- aggregate ----
    # If model not fixed, average over models too.
    group_cols = ["embedding_type", "layer"]
    if args.model is None:
        group_cols = ["embedding_type", "layer"]  # still; model is averaged implicitly by mean()
    else:
        group_cols = ["embedding_type", "layer"]

    agg = (
        df.groupby(group_cols, as_index=False)[args.metric]
          .mean()
          .sort_values(["embedding_type", "layer"])
    )

    # ---- plot ----
    out_dir = os.path.dirname(args.out_path) or "."
    safe_mkdir(out_dir)

    plt.figure(figsize=(10, 6))

    for emb in sorted(agg["embedding_type"].astype(str).unique()):
        sub = agg[agg["embedding_type"].astype(str) == str(emb)].sort_values("layer")
        plt.plot(sub["layer"], sub[args.metric], marker="o", label=str(emb))

    title = f"{args.dataset.upper()} • {args.metric} (hue=embedding_type)"
    if args.model:
        title = f"{args.model.upper()} • " + title

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(title="embedding_type", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {args.out_path}")


if __name__ == "__main__":
    main()
