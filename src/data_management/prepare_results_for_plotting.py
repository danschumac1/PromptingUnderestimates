#!/usr/bin/env python3
"""
python ./src/data_management/prepare_results_for_plotting.py \
  --in_path ./data/logistic_reg_results.tsv \
  --out_path ./data/plot_ready.tsv

Input TSV columns (tab-separated):
  accuracy, macro_f1, pred_path, timestamp

Output TSV columns include (plot-ready):
  model, dataset, method, embedding_type, layer, accuracy, macro_f1, timestamp, pred_path, ...

Assumed pred_path layout:
  .../sample_generations/{model}/{dataset}/{method}/{run_dir_or_file}/...

Examples:
  LR probes:
    .../sample_generations/llama/had/logistic_regression/lets-ust_0-shot_Direct/layer10.jsonl

  Visual prompting:
    .../sample_generations/llama/had/visual_prompting/lets-ust_0-shot_Direct.jsonl

  Random baseline:
    .../sample_generations/no_model/ctu/random/uniform.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ---- regex helpers ----
# LR filename: layer10.jsonl -> stem "layer10"
LR_LAYER_FILE_RE = re.compile(r"^layer(?P<layer>\d+)\.jsonl$")

# prompt stem (file stem for prompting, or run_dir for LR):
# lets-ust_0-shot_Direct   OR  lets-ust_0-shot_CoT   OR  lets-ust_0-shot
SHOT_STYLE_RE = re.compile(
    r"^(?P<emb>.+?)_(?P<shots>\d+)-shot(?:_(?P<style>Direct|CoT))?$"
)

# If LR run_dir is exactly like above; we also allow "emb_only" (no shots) fallbacks
EMB_ONLY_RE = re.compile(r"^(?P<emb>.+?)(?:_(?P<style>Direct|CoT))?$")


def normalize_embedding_type(emb: Optional[str], *, force_ust: bool = False) -> Optional[str]:
    """
    Normalize embedding strings:
      - keep hyphens (since your plot script expects e.g., 'lets-ust')
      - optionally enforce '-ust' suffix (off by default; turn on if you want it)
    """
    if emb is None:
        return None
    emb = str(emb).strip()
    if not emb:
        return None

    if force_ust and not emb.endswith("-ust"):
        emb = emb + "-ust"

    return emb


def parse_pred_path(pred_path: str) -> Dict[str, Any]:
    """
    Extract:
      model, dataset, method, embedding_type, style, shots, layer

    Supports:
      - logistic_regression: .../{method}/{run_dir}/layerN.jsonl
      - visual_prompting   : .../{method}/{stem}.jsonl   (stem contains emb/shots/style)
      - random             : .../{method}/{variant}.jsonl
    """
    out: Dict[str, Any] = {
        "model": None,
        "dataset": None,
        "method": None,
        "embedding_type": None,
        "style": None,
        "shots": None,
        "layer": None,
        "variant": None,
        "pred_file": None,
        "pred_stem": None,
        "run_dir": None,
    }

    s = str(pred_path).strip()
    parts = Path(s).parts

    # find ".../sample_generations/..."
    try:
        i = parts.index("sample_generations")
    except ValueError:
        return out

    # model/dataset/method
    if len(parts) > i + 1:
        out["model"] = parts[i + 1]
    if len(parts) > i + 2:
        out["dataset"] = parts[i + 2]
    if len(parts) > i + 3:
        out["method"] = parts[i + 3]

    method = out["method"]

    # random: .../random/uniform.jsonl
    if method == "random":
        filename = parts[i + 4] if len(parts) > i + 4 else None
        out["pred_file"] = filename
        out["pred_stem"] = Path(filename).stem if filename else None
        out["variant"] = out["pred_stem"]
        out["embedding_type"] = out["variant"]
        return out

    # logistic_regression: .../logistic_regression/<run_dir>/layer10.jsonl
    if method == "logistic_regression":
        run_dir = parts[i + 4] if len(parts) > i + 4 else None
        out["run_dir"] = run_dir

        # parse embedding_type, shots, style from run_dir if possible
        if run_dir:
            m = SHOT_STYLE_RE.match(run_dir)
            if m:
                out["embedding_type"] = normalize_embedding_type(m.group("emb"))
                out["shots"] = int(m.group("shots"))
                out["style"] = m.group("style")
            else:
                m2 = EMB_ONLY_RE.match(run_dir)
                if m2:
                    out["embedding_type"] = normalize_embedding_type(m2.group("emb"))
                    out["style"] = m2.group("style")

        filename = parts[i + 5] if len(parts) > i + 5 else None
        out["pred_file"] = filename
        out["pred_stem"] = Path(filename).stem if filename else None

        # layer from file name
        if filename:
            lm = LR_LAYER_FILE_RE.match(Path(filename).name)
            if lm:
                out["layer"] = int(lm.group("layer"))

        return out

    # visual_prompting: .../visual_prompting/lets-ust_0-shot_Direct.jsonl
    if method == "visual_prompting":
        filename = parts[i + 4] if len(parts) > i + 4 else None
        out["pred_file"] = filename
        out["pred_stem"] = Path(filename).stem if filename else None

        if out["pred_stem"]:
            m = SHOT_STYLE_RE.match(out["pred_stem"])
            if m:
                out["embedding_type"] = normalize_embedding_type(m.group("emb"))
                out["shots"] = int(m.group("shots"))
                out["style"] = m.group("style")
            else:
                # infer missing shots => 0-shot
                m2 = EMB_ONLY_RE.match(out["pred_stem"])
                if m2:
                    out["embedding_type"] = normalize_embedding_type(m2.group("emb"))
                    out["style"] = m2.group("style")
                    out["shots"] = 0

    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t")

    # validate input schema
    required_in = {"accuracy", "macro_f1", "pred_path", "timestamp"}
    missing_in = required_in - set(df.columns)
    if missing_in:
        raise ValueError(f"Missing required input columns: {sorted(missing_in)}")

    # coerce types
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["pred_path"] = df["pred_path"].astype(str)

    df = df.dropna(subset=["accuracy", "macro_f1", "pred_path"]).reset_index(drop=True)

    # parse pred_path
    parsed = df["pred_path"].apply(parse_pred_path).apply(pd.Series)
    df = pd.concat([parsed, df], axis=1)

    # ensure plot_probes.py required columns exist
    # (plot script requires model, dataset, method, embedding_type, layer, accuracy, macro_f1)
    required_out = {"model", "dataset", "method", "embedding_type", "layer", "accuracy", "macro_f1"}
    missing_out = required_out - set(df.columns)
    if missing_out:
        raise ValueError(f"Internal error: missing derived columns: {sorted(missing_out)}")

    # optional ordering: put plot-relevant columns first
    front = [
        "model",
        "dataset",
        "method",
        "embedding_type",
        "style",
        "shots",
        "layer",
        "accuracy",
        "macro_f1",
        "timestamp",
        "pred_path",
        "variant",
        "run_dir",
        "pred_file",
        "pred_stem",
    ]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]

    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] wrote plot-ready TSV: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
