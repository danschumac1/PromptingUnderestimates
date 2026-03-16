#!/usr/bin/env python3
"""
python ./src/data_management/convert_old_results_to_plot_ready.py \
  --in_path ./data/qwen_log.tsv \
  --out_path ./data/plot_ready_qwen.tsv \
  --append 1

This version is robust to "ragged TSV" rows (extra trailing tabs/empty columns),
which is exactly what your sample shows.

Input (old-ish) columns (expected in the HEADER; extra trailing columns are ignored):
  model, dataset, method, embedding_type, style, variant, layer, shots, accuracy, f1, pred_path, timestamp, pred_file_stem

Output (plot_ready) columns:
  model, dataset, method, embedding_type, style, shots, layer, accuracy, macro_f1, timestamp,
  pred_path, variant, run_dir, pred_file, pred_stem

Key behaviors:
- Reads TSV safely even if some rows have more fields than the header (extra tabs).
- Renames f1 -> macro_f1
- If shots missing/blank => shots=0.0
- Derives pred_file / pred_stem from pred_path
- Derives run_dir for logistic_regression:
    "{embedding_type}_{shots}-shot_{style}" (omits missing pieces gracefully)
- If --append 1 and out_path exists: append + dedupe by pred_path (keep latest timestamp)
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd


# -------------------------
# IO: ragged TSV reader
# -------------------------
def read_tsv_ragged(path: Path) -> pd.DataFrame:
    """
    Read a TSV that may have ragged rows due to trailing tabs/extra empty columns.
    We use the header to determine the expected number of fields, then for each line:
      - pad missing fields with "" (rare)
      - trim extra fields beyond header width (common in your sample)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if not lines:
        return pd.DataFrame()

    header = lines[0].split("\t")
    n = len(header)

    fixed_lines: List[str] = [lines[0]]
    repaired = 0
    dropped = 0

    for line in lines[1:]:
        if not line.strip():
            dropped += 1
            continue
        parts = line.split("\t")
        if len(parts) != n:
            repaired += 1
            if len(parts) < n:
                parts = parts + [""] * (n - len(parts))
            else:
                parts = parts[:n]
        fixed_lines.append("\t".join(parts))

    if repaired or dropped:
        print(f"[WARN] ragged TSV: repaired {repaired} lines; dropped {dropped} blank lines")

    return pd.read_csv(StringIO("\n".join(fixed_lines)), sep="\t")


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--append", type=int, default=1, choices=[0, 1])
    p.add_argument("--dedupe_key", type=str, default="pred_path", choices=["pred_path"])
    return p.parse_args()


# -------------------------
# Transform helpers
# -------------------------
def build_run_dir(embedding_type: str, shots: float, style: str, method: str) -> str:
    """Best-effort run_dir creation for LR; returns '' for other methods."""
    if method != "logistic_regression":
        return ""

    emb = (embedding_type or "").strip()
    sty = (style or "").strip()

    try:
        shots_int = int(float(shots))
        shots_part = f"{shots_int}-shot"
    except Exception:
        shots_part = ""

    parts = []
    if emb:
        parts.append(emb)
    if shots_part:
        parts.append(shots_part)
    if sty:
        parts.append(sty)

    return "_".join(parts)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # --- normalize column names ---
    df = df.rename(
        columns={
            "f1": "macro_f1",
            "pred_file_stem": "pred_stem",
        }
    )

    # Ensure presence of expected columns (as strings where relevant)
    for col in ["model", "dataset", "method", "embedding_type", "style", "variant", "pred_path"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # Numeric coercions (these are the ones that were turning into NaN before due to ragged parse)
    for col in ["layer", "shots", "accuracy", "macro_f1"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # shots default => 0.0 if missing
    df["shots"] = df["shots"].fillna(0.0)

    # pred_file / pred_stem
    df["pred_path"] = df["pred_path"].astype(str)
    df["pred_file"] = df["pred_path"].apply(lambda p: Path(p).name if p and p != "nan" else "")
    df["pred_stem"] = df["pred_file"].apply(lambda f: Path(f).stem if f else "")

    # run_dir
    df["run_dir"] = df.apply(
        lambda r: build_run_dir(
            embedding_type=str(r.get("embedding_type") or "").strip(),
            shots=float(r.get("shots") if pd.notna(r.get("shots")) else 0.0),
            style=str(r.get("style") or "").strip(),
            method=str(r.get("method") or "").strip(),
        ),
        axis=1,
    )

    # Drop rows missing core required info
    # (Use stricter required fields; but DON'T drop on variant/run_dir/pred_file stuff)
    required = ["model", "dataset", "method", "embedding_type", "pred_path"]
    for c in required:
        df[c] = df[c].replace({"nan": ""})

    df = df[df["model"].str.strip().ne("")].copy()
    df = df[df["dataset"].str.strip().ne("")].copy()
    df = df[df["method"].str.strip().ne("")].copy()
    df = df[df["embedding_type"].str.strip().ne("")].copy()
    df = df[df["pred_path"].str.strip().ne("")].copy()

    # Now ensure metrics/layer exist
    df = df.dropna(subset=["accuracy", "macro_f1", "layer"]).copy()

    # Cast layer/shots
    df["layer"] = df["layer"].astype(int)
    df["shots"] = df["shots"].astype(float)

    # Output columns in desired order
    out_cols = [
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
    for c in out_cols:
        if c not in df.columns:
            df[c] = "" if c not in ["timestamp"] else pd.NaT

    extras = [c for c in df.columns if c not in out_cols]
    return df[out_cols + extras]


def dedupe_keep_latest(df: pd.DataFrame, key: str = "pred_path") -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", na_position="first")
    return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_new = read_tsv_ragged(in_path)
    if raw_new.empty:
        raise ValueError(f"No rows found in {in_path}")

    new_df = standardize(raw_new)
    print(f"[INFO] standardized new rows: {len(new_df)}")

    if args.append == 1 and out_path.exists():
        raw_existing = read_tsv_ragged(out_path)
        existing_df = standardize(raw_existing) if not raw_existing.empty else pd.DataFrame()
        print(f"[INFO] existing standardized rows: {len(existing_df)}")

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = dedupe_keep_latest(combined, key=args.dedupe_key)
        combined.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] appended + wrote: {out_path} (rows={len(combined)})")
        return

    new_df = dedupe_keep_latest(new_df, key=args.dedupe_key)
    new_df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] wrote: {out_path} (rows={len(new_df)})")


if __name__ == "__main__":
    main()
