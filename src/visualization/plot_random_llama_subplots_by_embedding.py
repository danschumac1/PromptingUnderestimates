#!/usr/bin/env python3
"""
python ./src/visualization/plot_random_llama_subplots_by_embedding.py \
  --in_path "./data/plot_ready.tsv" \
  --out_path "./images/visualizations/probes/random_qwen__d_vs_dv.pdf" \
  --dataset all \
  --metric macro_f1
"""

from __future__ import annotations

import argparse
import os
from matplotlib.lines import Line2D

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_path", type=str, default="./images/visualizations/probes/random_llama__d_vs_dv.pdf")
    p.add_argument("--dataset", type=str, default="all", help='Use "all" to average, or dataset like ctu/emg/had/har')
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--method", type=str, default="logistic_regression")
    p.add_argument("--model_a", type=str, default="qwen")
    p.add_argument("--model_b", type=str, default="random_qwen")

    # Optional: force exact embedding_type strings (recommended for determinism)
    p.add_argument("--emb_d", type=str, default=None, help='Exact embedding_type for d, e.g. "lets-ust"')
    p.add_argument("--emb_dv", type=str, default=None, help='Exact embedding_type for d+v, e.g. "vis-lets-ust"')

    return p.parse_args()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bucket(emb: str) -> str:
    s = str(emb).lower()
    has_lets = "lets" in s
    has_vis = "vis" in s
    if has_lets and has_vis:
        return "dv"
    if has_lets and not has_vis:
        return "d"
    return ""


def pick_representative_embedding(df: pd.DataFrame, model_a: str, model_b: str, which_bucket: str) -> str:
    sub = df[df["emb_bucket"] == which_bucket].copy()
    if sub.empty:
        raise ValueError(f"No rows in bucket='{which_bucket}' after filtering")

    a_vals = set(sub[sub["model"].astype(str) == model_a]["embedding_type"].astype(str).unique())
    b_vals = set(sub[sub["model"].astype(str) == model_b]["embedding_type"].astype(str).unique())
    common = sorted(a_vals & b_vals)
    if not common:
        raise ValueError(f"No common embedding_type values for bucket='{which_bucket}' between {model_a} and {model_b}")

    counts = sub[sub["embedding_type"].astype(str).isin(common)]["embedding_type"].astype(str).value_counts()
    return str(counts.index[0])


def aggregate(df: pd.DataFrame, metric: str, dataset: str) -> pd.DataFrame:
    sub = df.copy()
    if dataset != "all":
        sub = sub[sub["dataset"].astype(str) == dataset].copy()

    # avg over datasets happens naturally when dataset=="all" because dataset not in groupby
    return (
        sub.groupby(["model", "embedding_type", "layer"], as_index=False)[metric]
           .mean()
    )


def main() -> None:
    args = parse_args()

    # seaborn theme (no grid)
    sns.set_theme(style="white", context="paper")

    # palette (match the other plot style)
    palette = {
        args.model_b: "#003f5c",  # llama (blue)
        args.model_a: "#ffa600",  # random (orange)
    }

    df = pd.read_csv(args.in_path, sep="\t")
    print(f"[INFO] loaded {len(df)} rows from {args.in_path}")

    required = {"model", "dataset", "method", "embedding_type", "layer", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    # filters
    df = df[df["method"].astype(str) == str(args.method)].copy()
    df = df[df["model"].astype(str).isin([args.model_a, args.model_b])].copy()

    if args.dataset != "all":
        df = df[df["dataset"].astype(str) == str(args.dataset)].copy()

    df = df.dropna(subset=[args.metric, "embedding_type"]).copy()
    df["emb_bucket"] = df["embedding_type"].apply(bucket)
    df = df[df["emb_bucket"].isin(["d", "dv"])].copy()

    if df.empty:
        raise ValueError("No rows after filtering to (random,llama) + (d,dv) buckets")

    # choose embedding string to represent each bucket
    emb_d = args.emb_d if args.emb_d else pick_representative_embedding(df, args.model_a, args.model_b, "d")
    emb_dv = args.emb_dv if args.emb_dv else pick_representative_embedding(df, args.model_a, args.model_b, "dv")

    print(f"[INFO] using d  embedding_type:  {emb_d}")
    print(f"[INFO] using d+v embedding_type: {emb_dv}")

    # aggregate and restrict
    agg = aggregate(df.dropna(subset=["layer"]).copy(), args.metric, args.dataset)
    agg = agg.dropna(subset=["layer"]).copy()
    agg["layer"] = agg["layer"].astype(int)
    agg = agg[agg["embedding_type"].astype(str).isin([emb_d, emb_dv])].copy()

    agg_d = agg[agg["embedding_type"].astype(str) == str(emb_d)].copy()
    agg_dv = agg[agg["embedding_type"].astype(str) == str(emb_dv)].copy()

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Left: d
    sns.lineplot(
        data=agg_d,
        x="layer",
        y=args.metric,
        hue="model",
        hue_order=[args.model_a, args.model_b],
        palette=palette,
        ax=axes[0],
        linewidth=2.2,
        marker=None,
        legend=False,
    )
    axes[0].set_title(r"$d$", fontsize=16, pad=10)

    # Right: d+v
    sns.lineplot(
        data=agg_dv,
        x="layer",
        y=args.metric,
        hue="model",
        hue_order=[args.model_a, args.model_b],
        palette=palette,
        ax=axes[1],
        linewidth=2.2,
        marker=None,
        legend=False,
    )
    axes[1].set_title(r"$d{+}v$", fontsize=16, pad=10)

    # clean axes: no grid, no per-axis labels
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=14)
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        sns.despine(ax=ax)

    # shared labels
    fig.supxlabel("Layer", fontsize=18, y=0.02)
    fig.supylabel("Macro F1" if args.metric == "macro_f1" else "Accuracy", fontsize=18, x=0.09)

    # one shared legend (match other plot spacing)
    handles = [
        Line2D([0], [0], color=palette[args.model_a], lw=2.2),
        Line2D([0], [0], color=palette[args.model_b], lw=2.2),
    ]

    labels = ["Qwen", "Qwen Random"]
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])

    # --- Legend ABOVE titles, inside reserved band ---
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.15),
        ncol=1,
        frameon=False,
        fontsize=14,
    )

    pdf_path = args.out_path
    png_path = pdf_path.replace("pdf", "png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


    print(f"[SAVED] {pdf_path}\n{png_path}")


if __name__ == "__main__":
    main()
