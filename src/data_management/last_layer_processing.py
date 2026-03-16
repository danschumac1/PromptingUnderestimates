"""
python ./src/data_management/last_layer_processing.py \
    --in_tsv ./data/qwen_random.tsv \
    --out_tsv ./data/last_layer.tsv
"""
import sys; sys.path.append("./src")

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from utils.file_io import load_tsv


# Final required column order
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
_SHOT_RE = re.compile(r"^(\d+)-shot$", re.IGNORECASE)

def _parse_layer_num(pred_path: str) -> int:
    m = _LAYER_RE.search(str(pred_path))
    if not m:
        return -1
    return int(m.group(1))


def _split_modality_style_shots(run_dir: str) -> Tuple[str, int, int]:
    """
    run_dir examples:
      lets-ust_0-shot_Direct
      lets-vis-ust_5-shot_CoT
      lets_10-shot_cot

    Outputs:
      modality: embedding types (everything before the '{k}-shot' token if present)
      style: cot == 1 or 0 (infer from run_dir containing 'cot' case-insensitive)
      shots: integer k from '{k}-shot' if present, else -1
    """
    parts = run_dir.split("_")

    style = 1 if "cot" in run_dir.lower() else 0

    shots = -1
    shot_idx = None
    for i, p in enumerate(parts):
        m = _SHOT_RE.fullmatch(p)
        if m:
            shots = int(m.group(1))
            shot_idx = i
            break

    if shot_idx is None:
        modality = run_dir
    else:
        modality = "_".join(parts[:shot_idx])

    return modality, style, shots


def _parse_pred_path(pred_path: str) -> Dict[str, object]:
    """
    Expected pattern:
      .../sample_generations/<model>/<dataset>/<method>/<run_dir>/layerX.jsonl
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
            "style": 0,
            "shots": -1,
            "layer": layer_num,
        }

    model = parts[i + 1] if i + 1 < len(parts) else ""
    dataset = parts[i + 2] if i + 2 < len(parts) else ""
    method = parts[i + 3] if i + 3 < len(parts) else ""
    run_dir = parts[i + 4] if i + 4 < len(parts) else ""

    modality, style, shots = _split_modality_style_shots(run_dir)

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
    ap.add_argument("--in_tsv", type=str, default="./data/data_dump.tsv")
    ap.add_argument("--out_tsv", type=str, default="./data/data_dump_last_layer.tsv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data = load_tsv(args.in_tsv)
    df = pd.DataFrame(data)

    required = [PRED_PATH_COL, ACC_COL_IN, F1_COL_IN, "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in TSV: {missing}")

    print("rows:", len(df))

    parsed = df[PRED_PATH_COL].astype(str).apply(_parse_pred_path).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Rename/mirror to requested output names
    df["acc"] = df[ACC_COL_IN]
    df["f1"] = df[F1_COL_IN]
    df["file_path"] = df[PRED_PATH_COL]
    df["variant"] = ""  # blank

    # Coerce types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").fillna(-1).astype(int)
    df["shots"] = pd.to_numeric(df["shots"], errors="coerce").fillna(-1).astype(int)
    df["style"] = pd.to_numeric(df["style"], errors="coerce").fillna(0).astype(int)

    # Warn if shots couldn't be inferred
    if (df["shots"] < 0).any():
        bad = df[df["shots"] < 0][["file_path"]].head(10)
        print("\nWARNING: Could not infer shots from run_dir for some rows (showing up to 10):")
        print(bad.to_string(index=False))

    print("methods:", sorted(df["method"].dropna().unique().tolist()))

    # -------------------------------------------------------------------------
    # Collapse logistic_regression to last layer per group
    # -------------------------------------------------------------------------
    GROUP_COLS = ["model", "dataset", "method", "modality", "style", "shots", "variant"]

    logReg_df = df[df["method"] == "logistic_regression"].copy()
    non_logReg_df = df[df["method"] != "logistic_regression"].copy()
    print("logReg rows:", len(logReg_df))
    print("non-logReg rows:", len(non_logReg_df))

    if len(logReg_df) > 0:
        if (logReg_df["layer"] < 0).any():
            bad = logReg_df[logReg_df["layer"] < 0][["file_path"]].head(10)
            print("\nWARNING: Unparseable layer values in logistic_regression (showing up to 10):")
            print(bad.to_string(index=False))

        max_layer = logReg_df.groupby(GROUP_COLS, dropna=False)["layer"].transform("max")
        last_layer_df = logReg_df[logReg_df["layer"] == max_layer].copy()

        # Ensure max-layer ties are identical on identity cols
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
            print("\n⚠️ Non-identical duplicates found at last layer in these groups:")
            for key, nrows, layer, bad_cols in problems[:50]:
                key_dict = dict(zip(GROUP_COLS, key))
                print(f"- {key_dict}  layer={layer}  n={nrows}  differing_cols={bad_cols}")
            raise ValueError(f"Found {len(problems)} groups with non-identical duplicates at last layer.")
        else:
            print("\n✅ All groups: duplicates at last layer match on (layer, acc, f1).")

        last_layer_final_df = (
            last_layer_df.sort_values(GROUP_COLS + ["layer", "timestamp"], na_position="last")
            .drop_duplicates(subset=GROUP_COLS, keep="first")
            .copy()
        )
    else:
        last_layer_final_df = logReg_df

    stitched_df = pd.concat([non_logReg_df, last_layer_final_df], ignore_index=True)

    # Keep exactly required headers and order
    for c in FINAL_COLS:
        if c not in stitched_df.columns:
            stitched_df[c] = ""

    stitched_df = stitched_df[FINAL_COLS].copy()

    stitched_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nWrote: {args.out_tsv}")
    print("rows out:", len(stitched_df))


if __name__ == "__main__":
    main()
