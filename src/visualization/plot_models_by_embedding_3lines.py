#!/usr/bin/env python3
"""
python ./src/visualization/plot_models_by_embedding_3lines.py \
  --in_path "./data/plot_ready.tsv" \
  --out_dir "./images/visualizations/probes/by_model_3emb/" \
  --models "mistral,qwen" \
  --dataset all \
  --metric macro_f1 \
  --embeddings "lets-ust,vis-ust,vis-lets-ust"

What it does (same style as your random-vs-llama subplot figure):
- Subplots = one per model (e.g., mistral, qwen)
- Within each subplot: 3 lines (one per embedding)
- X: layer
- Y: metric
- Dataset knob:
    --dataset all => average over datasets
    --dataset had => filter to had only
- Single shared legend (d / v / d+v)
- Single shared axis labels
- No markers, no grid
- Vector output (PDF)

Notes:
- Expects plot_ready schema with columns:
  model, dataset, method, embedding_type, layer, macro_f1/accuracy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_dir", type=str, default="./images/visualizations/probes/by_model_3emb/")
    p.add_argument("--dataset", type=str, default="all", help='Use "all" to average, or dataset like ctu/emg/had/har')
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--method", type=str, default="logistic_regression")
    p.add_argument("--models", type=str, required=True, help="comma-separated models, e.g. mistral,qwen")
    p.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="comma-separated EXACTLY 3 embeddings, e.g. lets-ust,vis-ust,vis-lets-ust",
    )
    p.add_argument("--out_name", type=str, default=None, help="optional override for output filename (no extension)")
    return p.parse_args()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# -----------------------------
# Label mapping (your request)
# -----------------------------
def embed_label_map(embeddings: List[str]) -> Dict[str, str]:
    """
    Convert your embedding strings into legend labels:
      lets-*         -> d
      vis-*          -> v
      vis-lets-*     -> d+v
    Also strips '-ust' in labels (we don't show it).
    """
    mapping = {}
    for e in embeddings:
        s = e.lower()
        has_lets = "lets" in s
        has_vis = "vis" in s
        if has_lets and has_vis:
            mapping[e] = r"$d{+}v$"
        elif has_lets:
            mapping[e] = r"$d$"
        elif has_vis:
            mapping[e] = r"$v$"
        else:
            mapping[e] = e  # fallback
    return mapping


# -----------------------------
# Data handling
# -----------------------------
def aggregate(df: pd.DataFrame, metric: str, dataset: str) -> pd.DataFrame:
    sub = df.copy()
    if dataset != "all":
        sub = sub[sub["dataset"].astype(str) == dataset].copy()

    # If dataset == "all", averaging over datasets happens naturally by not grouping on dataset
    agg = (
        sub.groupby(["model", "embedding_type", "layer"], as_index=False)[metric]
        .mean()
    )
    return agg


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    models = parse_csv_list(args.models)
    if not models:
        raise ValueError("--models produced an empty list")

    embeddings = parse_csv_list(args.embeddings)
    if len(embeddings) != 3:
        raise ValueError(f"--embeddings must have exactly 3 items; got {len(embeddings)}: {embeddings}")

    # seaborn theme: same as your last script (clean, no grid)
    sns.set_theme(style="white", context="paper")

    # Use your palette for the 3 embeddings (consistent across subplots)
    # Order corresponds to embeddings list order from CLI.
    colors = ["#003f5c", "#ffa600", "#bc5090",]
    emb_palette = {embeddings[i]: colors[i] for i in range(3)}

    # read
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
    df = df[df["model"].astype(str).isin(models)].copy()
    df = df[df["embedding_type"].astype(str).isin(embeddings)].copy()
    df = df.dropna(subset=["layer", args.metric]).copy()
    df["layer"] = df["layer"].astype(int)

    if args.dataset != "all":
        df = df[df["dataset"].astype(str) == str(args.dataset)].copy()

    if df.empty:
        raise ValueError("No rows after filtering (method/models/embeddings/dataset).")

    # aggregate (avg over datasets if dataset==all)
    agg = aggregate(df, args.metric, args.dataset)
    agg["layer"] = agg["layer"].astype(int)

    # prep labels
    emb_to_label = embed_label_map(embeddings)

    # -----------------------------
    # Figure layout: one subplot per model
    # Aim for square-ish panels like your previous plot.
    # -----------------------------
    n = len(models)
    fig_w = max(8, 5 * n)
    fig_h = 5
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)

    if n == 1:
        axes = [axes]

    # plot each model
    for ax, model in zip(axes, models):
        sub = agg[agg["model"].astype(str) == str(model)].copy()
        if sub.empty:
            ax.set_title(f"{model.capitalize()} (no data)", fontsize=16)
            ax.axis("off")
            continue

        # draw 3 lines (embedding hue)
        sns.lineplot(
            data=sub,
            x="layer",
            y=args.metric,
            hue="embedding_type",
            hue_order=embeddings,
            palette=emb_palette,
            ax=ax,
            linewidth=2.2,
            marker=None,
            legend=False,  # shared legend only
        )

        # title: model name only (per your preference)
        ax.set_title(model.capitalize(), fontsize=16)

        # remove per-axis labels (we'll use shared ones)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=14)
        sns.despine(ax=ax)

        # square-ish panel
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    # shared axis labels
    fig.supxlabel("Layer", fontsize=18, y=0.02)
    fig.supylabel("Macro F1" if args.metric == "macro_f1" else "Accuracy", fontsize=18, x=0.09)

    # one shared legend: embedding labels only (d, v, d+v)
    legend_handles = [Line2D([0], [0], color=emb_palette[e], lw=2.2) for e in embeddings]
    legend_labels  = [emb_to_label[e] for e in embeddings]

  
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.92, 0.2),
        ncol=1,
        frameon=False,
        fontsize=14,
    )


    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.90])

    # output path (dynamic)
    safe_mkdir(args.out_dir)
    if args.out_name:
        fname = args.out_name
    else:
        ds = args.dataset
        ms = "-".join(models)
        em = "d_v_dv"
        fname = f"by_model__{ms}__{ds}__{em}__{args.metric}"

    pdf_path = Path(args.out_dir) / f"{fname}.pdf"
    png_path = Path(args.out_dir) / f"{fname}.png"


    plt.savefig(pdf_path, bbox_inches="tight")  # vector pdf
    plt.savefig(png_path, bbox_inches="tight")  # vector pdf

    plt.close()

    print(f"[SAVED] {pdf_path}\n{png_path}")


if __name__ == "__main__":
    main()
