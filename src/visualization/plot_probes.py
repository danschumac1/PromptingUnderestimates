#!/usr/bin/env python3
"""
python ./src/visualization/plot_probes.py \
    --in_path "./data/plot_ready.tsv" \
    --out_dir "./images/visualizations/probes/by_model"

For each model:
- X: layer
- Y: macro_f1
- Lines (colors): embedding_type
- Averaged over datasets
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_dir", type=str, default="./images/visualizations/probes/by_model")
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

    # Clean types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["layer", args.metric]).copy()
    df["layer"] = df["layer"].astype(int)

    # Filter to LR
    df = df[df["method"] == args.method].copy()
    if df.empty:
        raise ValueError(f"No rows for method={args.method}")

    safe_mkdir(args.out_dir)

    # Loop over models → one plot per model
    for model in sorted(df["model"].astype(str).unique()):
        sub = df[df["model"].astype(str) == str(model)].copy()
        if sub.empty:
            continue

        # Average over datasets
        agg = (
            sub.groupby(["embedding_type", "layer"], as_index=False)[[args.metric]]
               .mean()
               .sort_values(["embedding_type", "layer"])
        )

        if agg.empty:
            print(f"[SKIP] model={model} has no aggregated rows")
            continue

        plt.figure(figsize=(10, 6))

        for emb in sorted(agg["embedding_type"].astype(str).unique()):
            emb_sub = agg[agg["embedding_type"].astype(str) == str(emb)]
            plt.plot(
                emb_sub["layer"],
                emb_sub[args.metric],
                marker="o",
                label=str(emb),
            )

        plt.title(f"{model.upper()} • AVG over datasets • {args.metric}")
        plt.xlabel("Layer")
        plt.ylabel(args.metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend(title="embedding_type", fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"{model}_{args.metric}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
