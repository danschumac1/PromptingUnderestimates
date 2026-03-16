#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List
import sys; sys.path.append("./src")
from utils.file_io import load_json, save_jsonl

DATA_ROOT = Path("/raid/hdd249/Classification_v2/data/SLLM")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        required=True,
    )
    return p.parse_args()


def collect_As(payload: dict) -> Dict[int, List[str]]:
    """
    payload: one loaded QA JSON (train or test)
    returns: { index -> [A, A, ...] }
    """
    out: Dict[int, List[str]] = {}

    for ex in payload["dataset"]:
        idx = int(ex["index"])  # ensure int
        As: List[str] = []

        # ---- summaries ----
        for _, s in ex.get("summaries", {}).items():
            if "A" in s:
                As.append(s["A"])

        # ---- qa_pairs ----
        for _, qa_list in ex.get("qa_pairs", {}).items():
            for qa in qa_list:
                if "A" in qa:
                    As.append(qa["A"])

        if As:
            out[idx] = As

    return out


def dict_to_jsonl_rows(d: Dict[int, List[str]]):
    """{idx: [A,...]} -> [{"index": idx, "As": [...]}, ...] sorted by idx."""
    return [{"index": idx, "As": d[idx]} for idx in sorted(d.keys())]


def main():
    args = parse_args()
    dataset_root = DATA_ROOT / args.dataset

    # ---- load all splits ----
    train_stage1 = load_json(dataset_root / "train_qa_stage1.json")
    train_stage2 = load_json(dataset_root / "train_qa_stage2.json")
    train_cls_stage2 = load_json(dataset_root / "train_qa_cls_stage2.json")

    test_stage1 = load_json(dataset_root / "test_qa_stage1.json")
    test_stage2 = load_json(dataset_root / "test_qa_stage2.json")
    test_cls_stage2 = load_json(dataset_root / "test_qa_cls_stage2.json")

    # ---- collect As ----
    train_As: Dict[int, List[str]] = {}
    test_As: Dict[int, List[str]] = {}

    for payload in [train_stage1, train_stage2, train_cls_stage2]:
        for idx, As in collect_As(payload).items():
            train_As.setdefault(idx, []).extend(As)

    for payload in [test_stage1, test_stage2, test_cls_stage2]:
        for idx, As in collect_As(payload).items():
            test_As.setdefault(idx, []).extend(As)

    # ---- sanity checks ----
    assert train_As, "train_As is empty"
    assert test_As, "test_As is empty"
    assert all(isinstance(v, list) and len(v) > 0 for v in train_As.values())
    assert all(isinstance(v, list) and len(v) > 0 for v in test_As.values())

    total_train_A = sum(len(v) for v in train_As.values())
    total_test_A = sum(len(v) for v in test_As.values())

    print("train indices:", len(train_As), "total As:", total_train_A)
    print("test indices:", len(test_As), "total As:", total_test_A)

    # ---- save as JSONL (one row per index) ----
    save_jsonl(dict_to_jsonl_rows(train_As), str(dataset_root / "train_As.jsonl"))
    save_jsonl(dict_to_jsonl_rows(test_As), str(dataset_root / "test_As.jsonl"))


if __name__ == "__main__":
    main()
