"""
Example:
python ./src/eval_prompt_variants.py \
    --dataset tee \
    --model qwen
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


_NUM_RE = re.compile(r"\d+")


def _extract_number(s: str) -> int:
    m = _NUM_RE.search(s)
    if not m:
        raise ValueError(f"No number found in '{s}'")
    return int(m.group())


def collect_data(input_dir: str) -> dict[str, dict[int, dict[int, list[dict[str, Any]]]]]:
    """
    out[embed_types][cot_flag][variant_num] -> list of jsonl rows
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue

            stem = fname[:-5]
            parts = stem.split("_")
            if len(parts) != 4:
                raise ValueError(
                    f"Unexpected filename format: '{fname}'. "
                    "Expected: sys*_gq*_<embed_types>_<cot*>.jsonl"
                )

            sys_num_str, gq_num_str, embed_types, cot_str = parts
            sys_num = _extract_number(sys_num_str)
            gq_num = _extract_number(gq_num_str)
            if sys_num != gq_num:
                raise ValueError(f"sys/gq mismatch in '{fname}'")

            cot_num = _extract_number(cot_str) if any(ch.isdigit() for ch in cot_str) else 0
            cot_flag = 1 if cot_num == 1 else 0

            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as fi:
                rows = [json.loads(line) for line in fi if line.strip()]

            out[embed_types][cot_flag][sys_num].extend(rows)

    return out


def build_eval_dict(
    data: dict[str, dict[int, dict[int, list[dict[str, Any]]]]]
) -> dict[str, dict[int, dict[int, dict[str, float | int]]]]:
    """
    evals[embed_types][cot_flag][variant_num] =
        { "n": int, "acc": float, "f1_macro": float }
    """
    evals = {}

    for embed_types, cot_map in data.items():
        evals[embed_types] = {}
        for cot_flag, variant_map in cot_map.items():
            evals[embed_types][cot_flag] = {}
            for variant_num, rows in variant_map.items():
                if not rows:
                    evals[embed_types][cot_flag][variant_num] = {
                        "n": 0,
                        "acc": 0.0,
                        "f1_macro": 0.0,
                    }
                    continue

                gts = [r["gt"] for r in rows]
                preds = [r["pred"] for r in rows]

                evals[embed_types][cot_flag][variant_num] = {
                    "n": len(rows),
                    "acc": float(accuracy_score(gts, preds)),
                    "f1_macro": float(f1_score(gts, preds, average="macro")),
                }

    return evals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate prompt variants (append TSV).")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, choices=["qwen"], required=True)
    p.add_argument(
        "--out_tsv",
        type=str,
        default="/raid/hdd249/data/prompt_variant_results.tsv",
        help="TSV file to append to (created if missing)",
    )
    return p.parse_args()


def _tsv(fields: list[Any]) -> str:
    return "\t".join(str(f) for f in fields)


def ensure_header(tsv_path: str, header_fields: list[str]) -> None:
    """
    Create TSV and write header if file does not exist or is empty.
    """
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(_tsv(header_fields) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = f"/raid/hdd249/data/sample_generations/together-{args.model}/{args.dataset}/prompt_variants"
    if not os.path.exists(input_dir):
        raise ValueError(f"PATH DOES NOT EXIST: {input_dir}")

    data = collect_data(input_dir)
    evals = build_eval_dict(data)

    header = [
        "dataset",
        "model",
        "embed_types",
        "cot",
        "variant",
        "n",
        "acc",
        "f1_macro",
    ]

    ensure_header(args.out_tsv, header)

    with open(args.out_tsv, "a", encoding="utf-8") as f:
        for embed_types in sorted(evals.keys()):
            for cot_flag in sorted(evals[embed_types].keys()):
                for variant_num in sorted(evals[embed_types][cot_flag].keys()):
                    m = evals[embed_types][cot_flag][variant_num]
                    f.write(
                        _tsv(
                            [
                                args.dataset,
                                args.model,
                                embed_types,
                                cot_flag,
                                variant_num,
                                m["n"],
                                f"{m['acc']:.6f}",
                                f"{m['f1_macro']:.6f}",
                            ]
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
