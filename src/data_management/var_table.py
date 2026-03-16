#!/usr/bin/env python3
"""
Summarize prompt-variant performance into a min/max/mean/median/delta table.

How to run:
  python ./src/data_management/summarize_variant_results.py \
    --in_path "./data/variant_results.tsv" \
    --out_path "./data/variant_summary.tsv"

Input TSV must include:
  dataset, model, embed_types, cot, f1_macro
"""

from __future__ import annotations

import argparse
import os
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="./data/variant_results.tsv")
    p.add_argument("--out_path", type=str, default="./data/variant_summary.tsv")
    p.add_argument(
        "--group_cols",
        type=str,
        default="dataset,model,embed_types,cot",
        help="Comma-separated columns to group by.",
    )
    p.add_argument("--metric", type=str, default="f1_macro", choices=["f1_macro", "acc"])
    return p.parse_args()


def safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_path, sep="\t")
    print(f"[INFO] loaded {len(df)} rows from {args.in_path}")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    required = set(group_cols) | {args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # numeric + clean
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=group_cols + [args.metric]).copy()

    if df.empty:
        raise ValueError("No rows after filtering.")

    # ---- aggregation ----
    summary = (
        df.groupby(group_cols, dropna=False)[args.metric]
        .agg(
            min_f1="min",
            max_f1="max",
            mean_f1="mean",
            median_f1="median",
            n="count",
        )
        .reset_index()
    )

    summary["delta_f1"] = summary["max_f1"] - summary["min_f1"]

    # ordering
    sort_cols = [c for c in ["dataset", "model", "embed_types", "cot"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    # rounding (presentation only)
    for c in ["min_f1", "max_f1", "mean_f1", "median_f1", "delta_f1"]:
        summary[c] = summary[c].astype(float).round(6)

    # save
    out_path = args.out_path
    safe_mkdir(os.path.dirname(out_path))
    summary.to_csv(out_path, sep="\t", index=False)

    print(f"[SAVED] {out_path}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
