"""
python ./src/visualization/plot_probes.py

Average-over-datasets probe plots for logistic regression.

Structure:
  ./images/visualizations/probe_plots_avg/{model}/{metric}.png

- One figure per (model, metric)
- X = layer
- Y = accuracy or macro_f1 (mean over datasets)
- Hue = embedding_type
- Adds dotted horizontal baseline per embedding_type:
  mean visual_prompting score over datasets for that (model, embedding_type),
  when available.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_FILE = "data/logistic_reg_results.tsv"
OUT_ROOT = "./images/visualizations/probe_plots_avg"

MODELS = ["llama", "mistral", "random"
        #   "qwen"
          ]
METRICS = ["macro_f1"]


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _prefer_zero_shot_then_latest(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Pick a single row per (model,dataset,embedding_type) preferring shots==0 then latest timestamp."""
    sub = df_sub.copy()

    # Prefer 0-shot if shots exists
    if "shots" in sub.columns:
        sub["shots_num"] = pd.to_numeric(sub["shots"], errors="coerce")
        sub0 = sub[sub["shots_num"] == 0].copy()
        if not sub0.empty:
            sub = sub0

    # Prefer latest timestamp if exists
    if "timestamp" in sub.columns:
        sub["timestamp_dt"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub = sub.sort_values("timestamp_dt")

    # return last row (most recent)
    return sub.tail(1)


def compute_prompting_baselines(df_prompting: pd.DataFrame) -> pd.DataFrame:
    """
    Returns baselines averaged over datasets:
      columns: model, embedding_type, metric -> baseline value
    """
    if df_prompting.empty:
        return pd.DataFrame(columns=["model", "embedding_type", "accuracy", "macro_f1"])

    # For each (model,dataset,embedding_type), pick single best row (0-shot then latest)
    picked_rows = []
    for (m, d, e), grp in df_prompting.groupby(["model", "dataset", "embedding_type"]):
        picked_rows.append(_prefer_zero_shot_then_latest(grp))

    picked = pd.concat(picked_rows, ignore_index=True)

    # Average over datasets => (model, embedding_type)
    baselines = (
        picked.groupby(["model", "embedding_type"], as_index=False)[["accuracy", "macro_f1"]]
        .mean()
    )

    return baselines


def plot_avg_model_metric(
    df_lr: pd.DataFrame,
    baselines: pd.DataFrame,
    model: str,
    metric: str,
) -> None:
    sub = df_lr[df_lr["model"] == model].copy()
    if sub.empty:
        print(f"[SKIP] No LR data for model={model}")
        return

    # mean over datasets at each (layer, embedding_type)
    agg = (
        sub.groupby(["layer", "embedding_type"], as_index=False)[metric]
        .mean()
        .sort_values(["embedding_type", "layer"])
    )

    if agg.empty:
        print(f"[SKIP] No aggregated rows for model={model}, metric={metric}")
        return

    out_dir = OUT_ROOT
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, f"{model}_{metric}.png")

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=agg,
        x="layer",
        y=metric,
        hue="embedding_type",
        marker="o",
    )

    # Map embedding_type -> color from legend handles
    handles, labels = ax.get_legend_handles_labels()
    emb_to_color = {}
    for h, lab in zip(handles, labels):
        if lab in agg["embedding_type"].unique():
            try:
                emb_to_color[lab] = h.get_color()
            except Exception:
                pass

    # Add dotted baseline per embedding_type (mean prompting over datasets)
    bsub = baselines[baselines["model"] == model]
    for emb_type in sorted(agg["embedding_type"].unique()):
        row = bsub[bsub["embedding_type"] == emb_type]
        if row.empty:
            continue
        y = float(row.iloc[0][metric])
        kwargs = dict(linestyle=":", linewidth=2, alpha=0.9)
        if emb_type in emb_to_color:
            kwargs["color"] = emb_to_color[emb_type]
        plt.axhline(y=y, **kwargs)

    plt.title(f"{model.upper()} • AVG over datasets • {metric} (hue=embedding_type)")
    plt.xlabel("Layer")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {out_path}")


def main():
    safe_mkdir(OUT_ROOT)

    df = pd.read_csv(INPUT_FILE, sep="\t")
    # df = df[df["dataset"] != "tee"]

    required = {"model", "dataset", "method", "embedding_type", "layer", "accuracy", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Split methods
    df_lr = df[df["method"] == "logistic_regression"].copy()
    df_prompting = df[df["method"] == "visual_prompting"].copy()

    # LR requires numeric layer
    df_lr["layer"] = pd.to_numeric(df_lr["layer"], errors="coerce")
    df_lr = df_lr.dropna(subset=["layer"])

    # Baselines from prompting (avg over datasets)
    baselines = compute_prompting_baselines(df_prompting)

    for model in MODELS:
        for metric in METRICS:
            plot_avg_model_metric(df_lr, baselines, model, metric)


if __name__ == "__main__":
    main()
