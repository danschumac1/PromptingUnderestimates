#!/usr/bin/env python3
"""
python ./src/visualization/plot_probes.py \
  --in_path ./data/plot_ready.tsv \
  --out_path ./images/visualizations/dynamic/har_by_embedding.png \
  --metric macro_f1 \
  --method logistic_regression \
  --filters dataset=trHARteHAD,embedding_type=lets-ust  \
  --hue model

A dynamic plotting script for layer-wise probe curves.

Core:
  X-axis: layer
  Y-axis: metric (default macro_f1)
  Lines: hue column (e.g., embedding_type, model, dataset, style, shots)

Knobs:
  - --filters: comma-separated key=value pairs (value can be pipe-separated list)
      e.g. --filters dataset=har,model=llama|mistral,style=Direct
  - --hue: which column defines the legend/lines (default embedding_type)
  - --avg_over: columns to average over (comma-separated). These columns are NOT kept in the groupby.
      e.g. avg_over=dataset means you average across datasets.
  - --keep: extra columns to keep (comma-separated). These columns ARE kept in the groupby.
      Useful if you want multiple panels-worth of curves in one plot (careful—can get busy).

Optional prompting baselines (dotted):
  - if --add_baselines 1, uses method==visual_prompting rows to compute a dotted horizontal baseline
    per hue category, after applying the same filters (except method) and after picking one row per
    (model,dataset,embedding_type) preferring 0-shot then latest timestamp.

Notes:
  - Expects an input TSV that already has parsed columns:
      model, dataset, method, embedding_type, layer, accuracy, macro_f1, timestamp (optional), style (optional), shots (optional)
  - Uses matplotlib only (no seaborn).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_filters(spec: Optional[str]) -> Dict[str, List[str]]:
    """
    Parse --filters like: "dataset=har,model=llama|mistral,style=Direct"
    Values are split on "|" to allow multiple selections.
    """
    if not spec:
        return {}
    out: Dict[str, List[str]] = {}
    items = [x.strip() for x in spec.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad filter '{item}'. Use key=value or key=v1|v2.")
        k, v = item.split("=", 1)
        k = k.strip()
        vals = [vv.strip() for vv in v.split("|") if vv.strip()]
        if not k or not vals:
            raise ValueError(f"Bad filter '{item}'. Use key=value or key=v1|v2.")
        out[k] = vals
    return out


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply categorical filters (string match)."""
    out = df.copy()
    for col, vals in filters.items():
        if col not in out.columns:
            raise ValueError(f"Filter column '{col}' not in dataframe columns: {sorted(out.columns)}")
        out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out


def _prefer_zero_shot_then_latest(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Pick a single row per (model,dataset,embedding_type) preferring shots==0 then latest timestamp."""
    sub = df_sub.copy()

    if "shots" in sub.columns:
        sub["shots_num"] = pd.to_numeric(sub["shots"], errors="coerce")
        sub0 = sub[sub["shots_num"] == 0].copy()
        if not sub0.empty:
            sub = sub0

    if "timestamp" in sub.columns:
        sub["timestamp_dt"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub = sub.sort_values("timestamp_dt")

    return sub.tail(1)


def compute_prompting_baselines(
    df: pd.DataFrame,
    metric: str,
    hue: str,
    filters: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute dotted baselines from visual_prompting rows.

    Returns a DF with columns: hue, metric (one row per hue value).
    For each (model,dataset,embedding_type): pick 0-shot else latest, then average over datasets.

    If hue != embedding_type, we still compute baselines by embedding_type internally,
    then average/aggregate to the chosen hue. This is best-effort; if it feels weird,
    just disable baselines.
    """
    if df.empty:
        return pd.DataFrame(columns=[hue, metric])

    # Apply same filters, except method (we want visual_prompting)
    f_no_method = {k: v for k, v in filters.items() if k != "method"}
    sub = apply_filters(df, f_no_method)
    sub = sub[sub["method"] == "visual_prompting"].copy()
    if sub.empty:
        return pd.DataFrame(columns=[hue, metric])

    needed = {"model", "dataset", "embedding_type", metric}
    missing = needed - set(sub.columns)
    if missing:
        return pd.DataFrame(columns=[hue, metric])

    picked_rows = []
    for (m, d, e), grp in sub.groupby(["model", "dataset", "embedding_type"]):
        picked_rows.append(_prefer_zero_shot_then_latest(grp))

    picked = pd.concat(picked_rows, ignore_index=True)

    # baseline per embedding_type averaged over datasets
    base = picked.groupby(["model", "embedding_type"], as_index=False)[[metric]].mean()

    # If hue is embedding_type, we're done (but averaged over models too?)
    # For consistency with dynamic plots, collapse over model unless user keeps it via filters.
    # So: average over model here.
    if hue == "embedding_type":
        out = base.groupby(["embedding_type"], as_index=False)[[metric]].mean()
        out = out.rename(columns={"embedding_type": hue})
        return out

    # Otherwise, map baseline rows to hue values using the same columns from picked/base if possible.
    # Best-effort: if hue column exists in 'picked', recompute baseline with that hue directly.
    if hue in picked.columns:
        base2 = picked.groupby([hue], as_index=False)[[metric]].mean()
        return base2

    # If hue isn't available, give up gracefully.
    return pd.DataFrame(columns=[hue, metric])


def build_group_cols(
    hue: str,
    keep: List[str],
    avg_over: List[str],
) -> List[str]:
    """
    Group columns always include:
      - layer
      - hue
      - any keep columns
    Any column listed in avg_over is NOT included (i.e., averaged away).
    """
    cols = ["layer", hue] + [c for c in keep if c not in ("layer", hue)]
    cols = [c for c in cols if c not in set(avg_over)]
    # de-dup while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {sorted(df.columns)}")


# -------------------------
# Plotting
# -------------------------
def plot_lines(
    df: pd.DataFrame,
    metric: str,
    hue: str,
    title: str,
    out_path: str,
    baselines: Optional[pd.DataFrame] = None,
    x_col: str = "layer",
) -> None:
    """
    df is expected to already be aggregated so that each (x_col, hue) has at most one value.
    """
    safe_mkdir(str(os.path.dirname(out_path)) or ".")

    # sort for nice lines
    df = df.sort_values([hue, x_col])

    plt.figure(figsize=(10, 6))

    hue_vals = list(df[hue].dropna().astype(str).unique())
    for hv in hue_vals:
        sub = df[df[hue].astype(str) == str(hv)].copy()
        if sub.empty:
            continue
        plt.plot(sub[x_col], sub[metric], marker="o", label=str(hv))

    # dotted baselines per hue (if provided)
    if baselines is not None and not baselines.empty:
        if hue in baselines.columns and metric in baselines.columns:
            bmap = {
                str(row[hue]): float(row[metric])
                for _, row in baselines.dropna(subset=[hue, metric]).iterrows()
            }
            for hv in hue_vals:
                if str(hv) in bmap:
                    plt.axhline(bmap[str(hv)], linestyle=":", linewidth=2, alpha=0.8)

    plt.title(title)
    plt.xlabel(x_col.title())
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(title=hue, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="./data/logistic_reg_results.tsv")
    ap.add_argument("--out_path", type=str, default="./images/visualizations/dynamic/plot.png")

    ap.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    ap.add_argument("--method", type=str, default="logistic_regression")

    ap.add_argument("--hue", type=str, default="embedding_type")
    ap.add_argument("--filters", type=str, default="", help="comma sep key=value; values sep by '|'")
    ap.add_argument("--avg_over", type=str, default="dataset", help="comma-separated columns to average over")
    ap.add_argument("--keep", type=str, default="", help="comma-separated columns to keep in groupby")

    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "median"])
    ap.add_argument("--add_baselines", type=int, default=0, choices=[0, 1])

    args = ap.parse_args()

    df = pd.read_csv(args.in_path, sep="\t")
    if df.empty:
        raise ValueError(f"No rows in {args.in_path}")

    # basic requirements (plot-ready TSV)
    base_required = ["model", "dataset", "method", "embedding_type", "layer", "accuracy", "macro_f1"]
    validate_columns(df, base_required)
sdf
    # numeric conversions
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["layer", args.metric]).copy()
    df["layer"] = df["layer"].astype(int)

    # apply method filter first
    df = df[df["method"] == args.method].copy()

    filters = parse_filters(args.filters)
    # If user provided method filter in --filters, it must match args.method; enforce for clarity
    if "method" in filters:
        if args.method not in [str(v) for v in filters["method"]]:
            print(f"[WARN] --filters specifies method={filters['method']} but --method is {args.method}; using --method.")
        filters["method"] = [args.method]

    before = len(df)
    df = apply_filters(df, {k: v for k, v in filters.items() if k != "method"})
    after = len(df)
    print(f"[INFO] rows: {before} -> {after} after filters (method={args.method})")

    if df.empty:
        raise ValueError("No rows left after filtering.")

    # build grouping columns
    avg_over = [c.strip() for c in args.avg_over.split(",") if c.strip()]
    keep = [c.strip() for c in args.keep.split(",") if c.strip()]
    validate_columns(df, [args.hue])  # hue must exist
    group_cols = build_group_cols(args.hue, keep=keep, avg_over=avg_over)

    # ensure group cols exist
    validate_columns(df, group_cols)

    # aggregate
    if args.agg == "mean":
        agg_df = df.groupby(group_cols, as_index=False)[[args.metric]].mean()
    elif args.agg == "max":
        agg_df = df.groupby(group_cols, as_index=False)[[args.metric]].max()
    else:  # median
        agg_df = df.groupby(group_cols, as_index=False)[[args.metric]].median()

    # if keep includes extra columns beyond (layer,hue), warn that plot will merge them
    extra_group = [c for c in group_cols if c not in ("layer", args.hue)]
    if extra_group:
        # Collapse extra group cols by string-joining into hue labels to keep single legend dimension.
        # This keeps the "legend knob" behavior predictable.
        def _label_row(r) -> str:
            parts = [f"{args.hue}={r[args.hue]}"]
            for c in extra_group:
                parts.append(f"{c}={r[c]}")
            return " | ".join(parts)

        agg_df = agg_df.copy()
        agg_df["_hue_label"] = agg_df.apply(_label_row, axis=1)
        hue_for_plot = "_hue_label"
    else:
        hue_for_plot = args.hue

    # compute baselines (optional) from original full df (not method-filtered)
    baselines = None
    if args.add_baselines == 1:
        # use the *full* dataframe before method filter, but with same filters
        df_all = pd.read_csv(args.in_path, sep="\t")
        df_all["layer"] = pd.to_numeric(df_all["layer"], errors="coerce")
        df_all[args.metric] = pd.to_numeric(df_all[args.metric], errors="coerce")
        if "timestamp" in df_all.columns:
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")

        baselines = compute_prompting_baselines(
            df=df_all,
            metric=args.metric,
            hue=args.hue if hue_for_plot == args.hue else args.hue,  # baseline computed on base hue
            filters=filters,
        )
        # If we collapsed labels, we can't map baselines reliably, so disable quietly.
        if hue_for_plot != args.hue:
            baselines = None

    # title
    title_bits = [f"{args.metric}", f"x=layer", f"hue={args.hue}", f"method={args.method}"]
    if filters:
        ftxt = "; ".join([f"{k}={('|'.join(v))}" for k, v in filters.items() if k != "method"])
        if ftxt:
            title_bits.append(f"filters: {ftxt}")
    if avg_over:
        title_bits.append(f"avg_over: {','.join(avg_over)}")
    title = " • ".join(title_bits)

    plot_lines(
        df=agg_df.rename(columns={hue_for_plot: "HUE"}).assign(HUE=lambda x: x["HUE"].astype(str))
            .rename(columns={"HUE": hue_for_plot}),
        metric=args.metric,
        hue=hue_for_plot,
        title=title,
        out_path=args.out_path,
        baselines=baselines,
        x_col="layer",
    )


if __name__ == "__main__":
    main()
