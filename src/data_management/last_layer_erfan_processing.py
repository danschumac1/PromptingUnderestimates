"""
python ./src/data_management/last_layer_erfan_processing.py \
    --in_tsv ./data/raw_results.tsv \
    --out_tsv ./data/last_layer.tsv
"""
import sys; sys.path.append("./src")

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from utils.file_io import load_tsv


FINAL_COLS = [
    "model",
    "dataset",
    "method",
    "modality",
    "style",
    "variant",
    "layer",
    "shots",
    "acc",
    "f1",
    "file_path",
    "timestamp",
]

PRED_PATH_COL = "pred_path"
ACC_COL_IN = "accuracy"
F1_COL_IN = "macro_f1"

_LAYER_RE = re.compile(r"(?:^|/)layer(\d+)\.jsonl$", re.IGNORECASE)
_SHOT_RE = re.compile(r"(\d+)-shot", re.IGNORECASE)


def _parse_layer_num(pred_path: str) -> int:
    m = _LAYER_RE.search(str(pred_path))
    return int(m.group(1)) if m else -1


def _parse_run_token(token: str) -> Tuple[str, str, int]:
    """
    Parse strings like:
      vis-ust_0-shot_Direct
      lets-ust_Direct
      vis-lets-ust_5-shot_CoT

    Returns:
      modality = first underscore chunk (e.g., vis-ust, lets-ust, vis-lets-ust)
      style    = last underscore chunk (e.g., Direct, CoT) or ""
      shots    = parsed k from k-shot else 0
    """
    token = token.replace(".jsonl", "")
    parts = token.split("_")

    modality = parts[0] if parts else token
    style = parts[-1] if len(parts) >= 2 else ""

    m = _SHOT_RE.search(token)
    shots = int(m.group(1)) if m else 0

    return modality, style, shots


def _parse_pred_path(pred_path: str) -> Dict[str, object]:
    """
    Supports BOTH:

    (A) Logistic regression layered:
      .../sample_generations/<model>/<dataset>/<method>/<run_dir>/layerX.jsonl

    (B) Visual prompting (no layer, run info in filename):
      .../sample_generations/<model>/<dataset>/<method>/<run_file>.jsonl
      where run_file like: vis-ust_0-shot_Direct.jsonl
    """
    p = Path(str(pred_path))
    layer_num = _parse_layer_num(str(p))

    parts = p.parts
    try:
        i = parts.index("sample_generations")
    except ValueError:
        return {
            "model": "",
            "dataset": "",
            "method": "",
            "modality": "",
            "style": "",
            "shots": 0,
            "layer": layer_num,
        }

    model = parts[i + 1] if i + 1 < len(parts) else ""
    dataset = parts[i + 2] if i + 2 < len(parts) else ""
    method = parts[i + 3] if i + 3 < len(parts) else ""

    # Decide whether run info is a directory (layered) or filename (non-layered)
    # Layered: .../<run_dir>/layerX.jsonl  -> parent is run_dir
    # Non-layered: .../<method>/<run_file>.jsonl -> file stem has run info
    if layer_num >= 0:
        run_token = p.parent.name  # run_dir like vis-ust_0-shot_Direct
    else:
        run_token = p.stem          # filename like vis-ust_0-shot_Direct

    modality, style, shots = _parse_run_token(run_token)

    return {
        "model": model,
        "dataset": dataset,
        "method": method,
        "modality": modality,
        "style": style,
        "shots": shots,
        "layer": layer_num,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", type=str, default="./data/raw_results.tsv")
    ap.add_argument("--out_tsv", type=str, default="./data/last_layer.tsv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data = load_tsv(args.in_tsv)
    df = pd.DataFrame(data)

    required = [PRED_PATH_COL, ACC_COL_IN, F1_COL_IN, "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in TSV: {missing}")

    parsed = df[PRED_PATH_COL].astype(str).apply(_parse_pred_path).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    df["acc"] = df[ACC_COL_IN]
    df["f1"] = df[F1_COL_IN]
    df["file_path"] = df[PRED_PATH_COL]
    df["variant"] = ""

    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").fillna(-1).astype(int)
    df["shots"] = pd.to_numeric(df["shots"], errors="coerce").fillna(0).astype(int)

    # -------------------------------------------------------------------------
    # Collapse logistic_regression to last layer per group
    # (visual_prompting has layer=-1 so it's unaffected)
    # -------------------------------------------------------------------------
    GROUP_COLS = ["model", "dataset", "method", "modality", "style", "shots", "variant"]

    logReg_df = df[df["method"] == "logistic_regression"].copy()
    non_logReg_df = df[df["method"] != "logistic_regression"].copy()

    if len(logReg_df) > 0:
        max_layer = logReg_df.groupby(GROUP_COLS, dropna=False)["layer"].transform("max")
        last_layer_df = logReg_df[logReg_df["layer"] == max_layer].copy()

        # check ties
        cols_to_compare = ["layer", "acc", "f1"]
        problems = []
        for key, g in last_layer_df.groupby(GROUP_COLS, dropna=False):
            if len(g) <= 1:
                continue
            nunique = g[cols_to_compare].nunique(dropna=False)
            bad_cols = nunique[nunique > 1].index.tolist()
            if bad_cols:
                problems.append((key, len(g), int(g["layer"].iloc[0]), bad_cols))

        if problems:
            for key, nrows, layer, bad_cols in problems[:50]:
                key_dict = dict(zip(GROUP_COLS, key))
                print(f"- {key_dict}  layer={layer}  n={nrows}  differing_cols={bad_cols}")
            raise ValueError(f"Found {len(problems)} groups with non-identical duplicates at last layer.")

        last_layer_final_df = (
            last_layer_df.sort_values(GROUP_COLS + ["layer", "timestamp"], na_position="last")
            .drop_duplicates(subset=GROUP_COLS, keep="first")
            .copy()
        )
    else:
        last_layer_final_df = logReg_df

    stitched_df = pd.concat([non_logReg_df, last_layer_final_df], ignore_index=True)

    for c in FINAL_COLS:
        if c not in stitched_df.columns:
            stitched_df[c] = ""

    stitched_df = stitched_df[FINAL_COLS].copy()
    stitched_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote: {args.out_tsv}  rows={len(stitched_df)}")


if __name__ == "__main__":
    main()
