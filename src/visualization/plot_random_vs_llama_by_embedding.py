#!/usr/bin/env python3
"""
python ./src/visualization/plot_random_vs_llama_by_embedding.py \
  --in_path "./data/plot_ready.tsv" \
  --out_path "./images/visualizations/probes/random_vs_llama.png" \
  --dataset all \
  --metric macro_f1

OR single dataset:
python ./src/visualization/plot_randomQwen_vs_Qwen_by_embedding.py \
  --in_path "./data/plot_ready.tsv" \
  --out_path "./images/visualizations/probes/random_vs_llama_ctu.png" \
  --dataset all \
  --metric macro_f1 \
  --embeddings vis-lets-ust,lets-ust,vis-ust

What it does:
- Compares TWO models: random vs llama
- Produces 6 lines total:
    - 3 lines for random (shades of red)  = 3 embedding types
    - 3 lines for llama  (shades of blue) = 3 embedding types
- X: layer
- Y: metric
- Dataset option:
    --dataset all   => average over datasets
    --dataset ctu   => filter to dataset==ctu

Notes:
- Expects method=logistic_regression rows for llama.
- For random, there are often no "layers". We handle this in a practical way:
    - If random rows have a layer column populated, we plot them normally.
    - If random has no layers (all NaN), we plot a flat line at y=mean(metric) across the llama layer range.
      (This still gives you 6 lines on the same axes.)
- Embedding selection:
    - By default, uses the top 3 embedding types (most frequent) that exist for BOTH models after filtering.
    - You can override with: --embeddings vis-lets-ust,lets-ust,vis-ust
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/plot_ready.tsv")
    p.add_argument("--out_path", type=str, default="./images/visualizations/probes/random_vs_llama.png")
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--method", type=str, default="logistic_regression")

    # dataset knob: "all" or a dataset name
    p.add_argument("--dataset", type=str, default="all", help='Use "all" to average, or a dataset like ctu/emg/had/har')

    # pick exactly 3 embeddings (optional override)
    p.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Optional comma-separated list of EXACTLY 3 embedding types (e.g., vis-lets-ust,lets-ust,vis-ust)",
    )

    # models are fixed per your request
    p.add_argument("--model_a", type=str, default="random")
    p.add_argument("--model_b", type=str, default="llama")

    return p.parse_args()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_embeddings_arg(s: str | None) -> List[str] | None:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    if len(items) != 3:
        raise ValueError(f"--embeddings must be exactly 3 comma-separated items; got {len(items)}: {items}")
    return items


def choose_top3_common_embeddings(df: pd.DataFrame, model_a: str, model_b: str) -> List[str]:
    """
    Choose 3 embedding types that exist for BOTH models.
    Preference: most frequent (by row count) in the filtered data.
    """
    a = df[df["model"].astype(str) == model_a]
    b = df[df["model"].astype(str) == model_b]

    common = sorted(set(a["embedding_type"].astype(str).unique()) & set(b["embedding_type"].astype(str).unique()))
    if not common:
        raise ValueError(f"No common embedding_type values between models {model_a} and {model_b} after filtering")

    counts = df[df["embedding_type"].astype(str).isin(common)]["embedding_type"].astype(str).value_counts()
    top3 = list(counts.index[:3])

    if len(top3) < 3:
        raise ValueError(f"Need 3 common embedding types, but only found {len(top3)}: {top3}")

    return top3


def base_color(model: str, model_a: str, model_b: str) -> str:
    # fixed mapping: random=red family, llama=blue family
    if model == model_a:
        return "Reds"
    if model == model_b:
        return "Blues"
    return "Greys"


def get_shades(cmap_name: str, k: int = 3):
    """
    Return k distinct shades from a matplotlib colormap.
    We avoid specifying literal colors; using a colormap is stable and clean.
    """
    cmap = plt.get_cmap(cmap_name)
    # pick mid-range to avoid too-light/too-dark extremes
    xs = np.linspace(0.35, 0.85, k)
    return [cmap(x) for x in xs]


def aggregate_for_plot(df: pd.DataFrame, metric: str, dataset: str) -> pd.DataFrame:
    """
    Returns a dataframe with columns: model, embedding_type, layer, metric
    If dataset == "all": average over datasets
    Else: filter to that dataset (no dataset aggregation)
    """
    sub = df.copy()

    if dataset != "all":
        sub = sub[sub["dataset"].astype(str) == dataset].copy()

    # group keys: dataset included only when dataset=="all" (so we can average over it)
    if dataset == "all":
        # average over datasets => group by (model, embedding_type, layer)
        agg = (
            sub.groupby(["model", "embedding_type", "layer"], as_index=False)[metric]
               .mean()
        )
    else:
        agg = (
            sub.groupby(["model", "embedding_type", "layer"], as_index=False)[metric]
               .mean()
        )

    return agg


def main() -> None:
    args = parse_args()
    embeddings_override = parse_embeddings_arg(args.embeddings)

    df = pd.read_csv(args.in_path, sep="\t")
    print(f"[INFO] loaded {len(df)} rows from {args.in_path}")

    required = {"model", "dataset", "method", "embedding_type", "layer", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric]).copy()

    # method filter
    df = df[df["method"].astype(str) == str(args.method)].copy()

    # models filter
    df = df[df["model"].astype(str).isin([args.model_a, args.model_b])].copy()

    if df.empty:
        raise ValueError("No rows after filtering to method + (random,llama)")

    # If layer is missing for some rows, we keep them for now; we'll handle in plotting
    # But for llama we usually need layers.
    # We'll aggregate on layer for the layered plots.

    # choose embeddings
    if embeddings_override is not None:
        chosen_embeddings = embeddings_override
    else:
        # only consider rows with non-null embedding_type
        df = df[df["embedding_type"].notna()].copy()
        chosen_embeddings = choose_top3_common_embeddings(df, args.model_a, args.model_b)

    print(f"[INFO] embeddings: {chosen_embeddings}")

    # restrict to chosen embeddings
    df = df[df["embedding_type"].astype(str).isin(chosen_embeddings)].copy()
    if df.empty:
        raise ValueError("No rows after restricting to chosen embeddings")

    # aggregate
    agg = aggregate_for_plot(df.dropna(subset=["layer"], how="any"), args.metric, args.dataset)

    # Determine x-range (layers) from layered data (prefer llama)
    llama_layers = sorted(
        agg[agg["model"].astype(str) == args.model_b]["layer"].dropna().astype(int).unique()
    )
    if not llama_layers:
        # fallback: any layers
        llama_layers = sorted(agg["layer"].dropna().astype(int).unique())
    if not llama_layers:
        raise ValueError("No numeric layers found to plot on X-axis")

    x_min, x_max = min(llama_layers), max(llama_layers)

    # Prepare output
    out_dir = os.path.dirname(args.out_path) or "."
    safe_mkdir(out_dir)

    plt.figure(figsize=(11, 6))

    # Color shades
    red_shades = get_shades("Reds", 3)
    blue_shades = get_shades("Blues", 3)
    emb_to_idx = {e: i for i, e in enumerate(chosen_embeddings)}

    # Plot each model x embedding
    for model in [args.model_a, args.model_b]:
        for emb in chosen_embeddings:
            idx = emb_to_idx[emb]
            color = red_shades[idx] if model == args.model_a else blue_shades[idx]

            sub = agg[(agg["model"].astype(str) == model) & (agg["embedding_type"].astype(str) == emb)].copy()
            sub = sub.dropna(subset=["layer"]).copy()
            if not sub.empty:
                sub["layer"] = sub["layer"].astype(int)
                sub = sub.sort_values("layer")
                plt.plot(sub["layer"], sub[args.metric], marker="o", color=color, label=f"{model}:{emb}")
            else:
                # If random has no layer rows after parsing/cleaning, draw flat baseline
                # using mean metric from original df (still filtered) for that model+emb
                flat_src = df[(df["model"].astype(str) == model) & (df["embedding_type"].astype(str) == emb)].copy()
                y = float(flat_src[args.metric].mean()) if len(flat_src) else np.nan
                if np.isnan(y):
                    continue
                xs = list(range(x_min, x_max + 1))
                ys = [y] * len(xs)
                plt.plot(xs, ys, linestyle="--", color=color, label=f"{model}:{emb} (flat)")

    title_ds = "AVG over datasets" if args.dataset == "all" else f"dataset={args.dataset}"
    plt.title(f"{args.model_a.upper()} vs {args.model_b.upper()} • {title_ds} • {args.metric}")
    plt.xlabel("Layer")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="model:embedding",
        fontsize=9,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # INSIDE right edge
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {args.out_path}")


if __name__ == "__main__":
    main()
