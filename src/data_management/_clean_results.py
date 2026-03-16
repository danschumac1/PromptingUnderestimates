#!/usr/bin/env python3
"""
python ./src/data_management/_clean_results.py \
  --in_path ./data/___logistic_reg_results.tsv \
  --out_dir feb

Writes:
  ./data/results/<out_name>.tsv
  ./data/results/last_layer.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


LAYER_RE = re.compile(r"^layer(?P<layer>\d+)$")
PROMPT_STEM_RE = re.compile(
    r"^(?P<emb>.+?)_(?P<shots>\d+)-shot(?:_(?P<style>Direct|CoT))?$"
)
EMB_DIR_STYLE_RE = re.compile(r"^(?P<emb>.+?)(?:_(?P<style>Direct|CoT))?$")


def normalize_embedding_type(emb: Optional[str], *, force_ust: bool = True) -> Optional[str]:
    """
    Normalize embedding strings:
      - hyphens -> underscores
      - optionally enforce *_ust suffix (NOT for random variants)
    """
    if emb is None:
        return None
    emb = emb.strip()
    if not emb:
        return None

    emb = emb.replace("-", "_")

    if force_ust:
        if not emb.endswith("_ust"):
            emb = emb + "_ust"

    return emb


def parse_pred_path(pred_path: str) -> Dict[str, Any]:
    """
    Supported patterns:

    Random baselines:
      data/sample_generations/<model>/<dataset>/random/<variant>.jsonl
        -> method=random, variant=<variant>, embedding_type=<variant> (or random_<variant> if you prefer)

    Logistic regression:
      data/sample_generations/<model>/<dataset>/logistic_regression/<embdir>/layer<N>.jsonl
      where <embdir> may end with _Direct or _CoT (style)

    Visual prompting:
      data/sample_generations/<model>/<dataset>/visual_prompting/<stem>.jsonl
      where <stem> is like ts-ust_0-shot_Direct or lets-ust_0-shot_CoT
    """
    s = str(pred_path).strip()
    parts = Path(s).parts

    out: Dict[str, Any] = {
        "model": None,
        "dataset": None,
        "method": None,
        "embedding_type": None,
        "layer": None,
        "shots": None,
        "style": None,     # Direct / CoT
        "variant": None,   # random baseline type: uniform/prior/majority/...
        "pred_file_stem": None,
    }

    try:
        i = parts.index("sample_generations")
    except ValueError:
        return out

    out["model"] = parts[i + 1] if len(parts) > i + 1 else None
    out["dataset"] = parts[i + 2] if len(parts) > i + 2 else None
    out["method"] = parts[i + 3] if len(parts) > i + 3 else None

    method = out["method"]

    # ------------------------------------------------------------------
    # RANDOM BASELINES
    # ------------------------------------------------------------------
    if method == "random":
        # .../random/<variant>.jsonl
        filename = parts[i + 4] if len(parts) > i + 4 else ""
        stem = Path(filename).stem if filename else ""
        out["pred_file_stem"] = stem

        out["variant"] = stem or None
        out["embedding_type"] = stem or None  # or f"random_{stem}" if you prefer
        out["shots"] = None
        out["layer"] = None
        out["style"] = None
        return out

    # ------------------------------------------------------------------
    # LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    if method == "logistic_regression":
        # .../<method>/<embdir>/layer<N>.jsonl
        embdir = parts[i + 4] if len(parts) > i + 4 else None

        style = None
        emb_base = embdir
        if embdir:
            mdir = EMB_DIR_STYLE_RE.match(embdir)
            if mdir:
                emb_base = mdir.group("emb")
                style = mdir.group("style")

        out["style"] = style
        out["embedding_type"] = normalize_embedding_type(emb_base, force_ust=True)

        filename = parts[i + 5] if len(parts) > i + 5 else ""
        stem = Path(filename).stem if filename else ""
        out["pred_file_stem"] = stem

        m = LAYER_RE.match(stem)
        out["layer"] = int(m.group("layer")) if m else None
        out["shots"] = None
        out["variant"] = None
        return out

    # ------------------------------------------------------------------
    # VISUAL PROMPTING
    # ------------------------------------------------------------------
    if method == "visual_prompting":
        filename = parts[i + 4] if len(parts) > i + 4 else ""
        stem = Path(filename).stem if filename else ""
        out["pred_file_stem"] = stem

        m = PROMPT_STEM_RE.match(stem)
        if m:
            emb = m.group("emb")
            out["shots"] = int(m.group("shots"))
            out["style"] = m.group("style")
            out["embedding_type"] = normalize_embedding_type(emb, force_ust=True)
        else:
            out["shots"] = None
            out["style"] = None
            out["embedding_type"] = None

        out["layer"] = None
        out["variant"] = None
        return out

    # ------------------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------------------
    emb = parts[i + 4] if len(parts) > i + 4 else None
    out["embedding_type"] = normalize_embedding_type(emb, force_ust=True)
    out["layer"] = None
    out["shots"] = None
    out["style"] = None
    out["variant"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="./data/raw_results.tsv")
    ap.add_argument("--out_dir", default="./data/results")
    ap.add_argument("--out_name", default="clean_results")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # load + basic cleaning
    # ---------------------------
    df = pd.read_csv(in_path, sep="\t")

    df["accuracy"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
    df["macro_f1"] = pd.to_numeric(df.get("macro_f1"), errors="coerce")
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")

    df["pred_path"] = df.get("pred_path").astype(str)
    df = df.dropna(subset=["accuracy", "macro_f1", "pred_path"]).reset_index(drop=True)

    # ---------------------------
    # parse pred_path -> new cols
    # ---------------------------
    parsed = df["pred_path"].apply(parse_pred_path).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    wanted_front = [
        "model",
        "dataset",
        "method",
        "embedding_type",
        "style",
        "variant",
        "layer",
        "shots",
        "accuracy",
        "macro_f1",
        "pred_path",
        "timestamp",
        "pred_file_stem",
    ]
    cols = [c for c in wanted_front if c in df.columns] + [c for c in df.columns if c not in wanted_front]
    df = df[cols]

    clean_path = out_dir / f"{args.out_name}.tsv"
    df.to_csv(clean_path, sep="\t", index=False)
    print(f"[OK] wrote cleaned: {clean_path} ({len(df)} rows)")

    # ---------------------------
    # last_layer.tsv
    #
    # rule:
    #   - keep ALL visual_prompting rows
    #   - keep ALL random rows
    #   - for logistic_regression: for each (model,dataset,embedding_type,style) keep max(layer)
    # ---------------------------
    is_lr = df["method"] == "logistic_regression"
    is_prompt = df["method"] == "visual_prompting"
    is_random = df["method"] == "random"

    lr = df[is_lr].copy()
    lr["layer"] = pd.to_numeric(lr["layer"], errors="coerce")

    if len(lr) > 0:
        group_cols = ["model", "dataset", "embedding_type", "style"]
        last_layer_map = (
            lr.dropna(subset=["layer"])
              .groupby(group_cols, as_index=False)["layer"]
              .max()
              .rename(columns={"layer": "max_layer"})
        )
        lr_last = lr.merge(last_layer_map, on=group_cols, how="left")
        lr_last = lr_last[lr_last["layer"] == lr_last["max_layer"]].drop(columns=["max_layer"])
    else:
        lr_last = lr

    last_layer_df = pd.concat(
        [df[is_prompt].copy(), df[is_random].copy(), lr_last],
        ignore_index=True,
    )

    last_layer_path = out_dir / "last_layer.tsv"
    last_layer_df.to_csv(last_layer_path, sep="\t", index=False)
    print(f"[OK] wrote last-layer: {last_layer_path} ({len(last_layer_df)} rows)")


if __name__ == "__main__":
    main()
