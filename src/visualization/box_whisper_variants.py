#!/usr/bin/env python3
"""
Box-and-whisker plot of prompt variant performance.

How to run:
  python ./src/visualization/box_whisper_variants.py \
    --in_path "./data/variant_results.tsv" \
    --out_path "./images/visualizations/variants/variant_boxplot.pdf"

Input TSV must include:
  dataset, embed_types, f1_macro
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# map raw embedding strings -> modality labels
EMB_LABEL_MAP = {
    "lets,ust": r"$d$",
    "lets-vis-ust": r"$d{+}v$",
    "lets,vis,ust": r"$d{+}v$",
    "vis,ust": r"$v$",
}

MODALITY_COLORS = {
    r"$d$":     "#003f5c",  # blue
    r"$v$":     "#bc5090",  # magenta
    r"$d{+}v$": "#ffa600",  # orange
}

BOX_ALPHA = 0.35  # subtle fill


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/variant_results.tsv")
    p.add_argument("--out_path", type=str, default="./images/visualizations/variants/variant_boxplot.pdf")
    p.add_argument("--metric", type=str, default="f1_macro", choices=["f1_macro", "accuracy"])
    p.add_argument("--dataset", type=str, default="all", help='Use "all" or a dataset name')
    return p.parse_args()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_path, sep="\t")
    print(f"[INFO] loaded {len(df)} rows from {args.in_path}")

    required = {"dataset", "embed_types", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # numeric + filter
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["dataset", "embed_types", args.metric]).copy()

    if args.dataset != "all":
        df = df[df["dataset"].astype(str) == str(args.dataset)].copy()

    if df.empty:
        raise ValueError("No rows after filtering")

    # enforce modality order: d, v, d+v
    MODALITY_ORDER = [r"$d$", r"$v$", r"$d{+}v$"]

    # map embed_types -> modality labels
    df["modality"] = df["embed_types"].astype(str).map(EMB_LABEL_MAP)
    df = df.dropna(subset=["modality"]).copy()

    if df.empty:
        raise ValueError("No rows after mapping embed_types to modality (check EMB_LABEL_MAP vs your TSV values).")

    # build data in fixed order
    data = []
    labels = []
    for m in MODALITY_ORDER:
        vals = df.loc[df["modality"] == m, args.metric].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(m)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(linewidth=2.2, color="black"),
        boxprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # color per modality (aligned to labels)
    for box, lab in zip(bp["boxes"], labels):
        box.set_facecolor(MODALITY_COLORS[lab])
        box.set_alpha(BOX_ALPHA)

    # axis labels
    ax.set_xlabel("Modality", fontsize=18)
    ax.set_ylabel("Macro F1" if args.metric == "f1_macro" else "Accuracy", fontsize=18)

    # correct xtick labels (LaTeX)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=16)  # no rotation needed

    ax.tick_params(axis="y", labelsize=14)
    ax.grid(False)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_pdf = args.out_path
    out_png = out_pdf.replace(".pdf", ".png")
    safe_mkdir(os.path.dirname(out_pdf))

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"[SAVED] {out_pdf}\n{out_png}")


if __name__ == "__main__":
    main()
