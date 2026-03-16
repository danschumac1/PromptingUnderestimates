"""
python ./src/prompt_w_variants.py \
    --dataset tee \
    --model qwen \
    --batch_size 5 \
    --normalize 1 \
    --CoT 0 \
    --embedding_types lets,vis,ust
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from utils.loaders import Split, load_train_test
from utils.constants import CoT_QUESTION_TAG, NO_CoT_QUESTION_TAG, build_valid_embedding_strings
from utils.build_prompts import build_prompt
from utils.prompters import TogetherPrompter
from utils.prompt_objects import VisionPrompt
from utils.preprocessing import extract_letter_to_idx
from utils.file_io import append_jsonl


# -------------------------
# Model selection (Together hosted)
# -------------------------
TOGETHER_MODEL_MAP = {
    # you can expand this later
    "qwen": "Qwen/Qwen3-VL-32B-Instruct",
}

# -------------------------
# IO helpers
# -------------------------
def read_txt_file(fpath: str) -> str:
    with open(fpath, "r", encoding="utf-8") as f:
        return f.read()


def _numeric_sort_key(name: str):
    """
    sort by leading integer if present (e.g., '10.txt' after '2.txt')
    fallback to name
    """
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 10**9, name)


def load_sysPs_and_GQs(dataset: str) -> Tuple[List[str], List[str]]:
    """
    Reads:
      /raid/hdd249/data/features/prompts/{dataset}/system/*.txt
      /raid/hdd249/data/features/prompts/{dataset}/gq/*.txt

    Returns: (system_variants, gq_variants) where each is a list[str]
    sorted numerically by filename when possible.
    """
    base = Path(f"/raid/hdd249/data/features/prompts/{dataset}")
    sys_dir = base / "system"
    gq_dir = base / "gq"

    if not sys_dir.exists():
        raise FileNotFoundError(f"Missing system prompt dir: {sys_dir}")
    if not gq_dir.exists():
        raise FileNotFoundError(f"Missing gq dir: {gq_dir}")

    sys_files = sorted([p for p in sys_dir.rglob("*.txt")], key=lambda p: _numeric_sort_key(p.name))
    gq_files  = sorted([p for p in gq_dir.rglob("*.txt")], key=lambda p: _numeric_sort_key(p.name))

    sys_vars = [read_txt_file(str(p)) for p in sys_files]
    gq_vars  = [read_txt_file(str(p)) for p in gq_files]

    return sys_vars, gq_vars


# -------------------------
# Prompt builder (explicit GQ)
# -------------------------
def build_classification_query_prompts_from_gq(
    batch_rows: Split,
    *,
    dataset: str,
    model: str,
    general_question: str,
    include_user_text: bool = True,
    include_ts: bool = False,
    include_LETSCLike: bool = False,
    include_vis: bool = True,
    CoT: bool = False,
) -> List[VisionPrompt]:
    """
    Build query VisPrompts (no assistant_text) for a batch of rows,
    using an explicitly provided `general_question`.
    """
    user_text_base = ""
    if include_user_text:
        gq = (general_question or "").strip()
        if not gq:
            raise ValueError("general_question was empty (after strip).")

        user_text_base = gq + "\n\n" + (CoT_QUESTION_TAG if CoT else NO_CoT_QUESTION_TAG)

    prompts: List[VisionPrompt] = []
    for row in batch_rows:
        vp = build_prompt(
            row=row,
            split_name="test",
            dataset=dataset,
            model=model,
            user_text=user_text_base,
            include_ts=include_ts,
            include_LETSCLike=include_LETSCLike,
            include_vis=include_vis,
        )
        prompts.append(vp)

    return prompts


def _load_acc_total_from_jsonl(path: Path) -> tuple[int, int]:
    """Return (acc, total) from an existing jsonl, else (0,0)."""
    if not path.exists():
        return 0, 0
    acc = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            acc += int(obj.get("correct", 0))
    return acc, total



# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True, choices=list(TOGETHER_MODEL_MAP.keys()))
    p.add_argument("--normalize", type=int, choices=[0, 1], default=1)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--CoT", type=int, choices=[0, 1], default=0)

    # NEW: resume flag
    p.add_argument(
        "--resume",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 = overwrite outputs and start fresh. 1 = resume from existing jsonl progress per variant.",
    )

    p.add_argument(
        "--embedding_types",
        type=str,
        default="vis,ust",
        choices=build_valid_embedding_strings(),
        help="Comma-separated: ts,vis,lets,ust (must be a valid combo from build_valid_embedding_strings())",
    )
    p.add_argument(
        "--max_variants",
        type=int,
        default=0,
        help="0 = use all. Otherwise truncate both sys/gq variants to this count.",
    )
    return p.parse_args()


def _load_resume_state(
    path: Path,
    *,
    test_len: int,
    batch_size: int,
) -> tuple[int, int, int, set[int], bool]:
    """
    If path doesn't exist: (acc=0,total=0,batch_start=0,processed=set(),done=False)

    If exists:
      - processed = set of idx already written
      - acc/total recomputed from file
      - batch_start = first incomplete batch index (based on counts per batch)
      - done=True if total == test_len
    """
    if not path.exists():
        return 0, 0, 0, set(), False

    acc = 0
    total = 0
    processed: set[int] = set()
    batch_counts: dict[int, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            total += 1
            acc += int(obj.get("correct", 0))

            idx_val = obj.get("idx", None)
            if idx_val is not None:
                processed.add(int(idx_val))

            b = obj.get("batch_idx", None)
            if b is not None:
                b = int(b)
                batch_counts[b] = batch_counts.get(b, 0) + 1

    done = (total >= test_len)

    # Find first incomplete batch
    num_batches = (test_len + batch_size - 1) // batch_size
    batch_start = 0
    for b in range(num_batches):
        expected = min(batch_size, test_len - b * batch_size)
        seen = batch_counts.get(b, 0)
        if seen < expected:
            batch_start = b
            break
    else:
        # all batches complete
        batch_start = num_batches

    return acc, total, batch_start, processed, done


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # embedding flags
    tokens = args.embedding_types.split(",")
    token_set = set(t.strip() for t in tokens if t.strip())
    include_ts = "ts" in token_set
    include_vis = "vis" in token_set
    include_LETSCLike = "lets" in token_set
    include_user_text = "ust" in token_set

    # basic constraints consistent with your build_prompt assertions
    if not (include_ts or include_vis or include_LETSCLike):
        raise ValueError("Need at least one of ts/vis/lets in --embedding_types.")
    if include_ts and include_LETSCLike:
        raise ValueError("Cannot include both ts and lets at the same time (matches build_prompt asserts).")

    # load variants
    sys_vars, gq_vars = load_sysPs_and_GQs(args.dataset)
    if args.max_variants and args.max_variants > 0:
        sys_vars = sys_vars[: args.max_variants]
        gq_vars = gq_vars[: args.max_variants]

    pairs = list(zip(sys_vars, gq_vars))
    if not pairs:
        raise ValueError("No (system, gq) pairs found. Check your prompt folders.")

    # load data
    _, test = load_train_test(
        os.path.join("/raid/hdd249/data/samples", args.dataset),
        0,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    if len(test) > 500:
        print(f"⚠️  Warning: test set has {len(test)} samples. Truncating to 500 for faster runs.")
        test = test[:500]

    # Together prompter
    together_model_id = TOGETHER_MODEL_MAP[args.model]
    prompter = TogetherPrompter(model_id=together_model_id)

    out_root = Path(f"/raid/hdd249/data/sample_generations/together-{args.model}/{args.dataset}/prompt_variants")
    out_root.mkdir(parents=True, exist_ok=True)

    test_len = len(test)
    num_batches = (test_len + args.batch_size - 1) // args.batch_size

    for v_idx, (sys_m, gq) in enumerate(pairs, start=1):
        prompter.system_prompt = sys_m
        out_file = out_root / f"sys{v_idx:03d}_gq{v_idx:03d}_{args.embedding_types}_cot{args.CoT}.jsonl"

        # -------------------------
        # Resume / overwrite handling
        # -------------------------
        if args.resume == 0:
            # always start fresh
            if out_file.exists():
                out_file.unlink()
            acc, total, batch_start, processed_idxs, done = 0, 0, 0, set(), False
        else:
            # resume from file if possible
            acc, total, batch_start, processed_idxs, done = _load_resume_state(
                out_file,
                test_len=test_len,
                batch_size=args.batch_size,
            )
            if done:
                final_acc = acc / max(total, 1)
                print(f"[variant {v_idx}] ↩️ already complete | acc={final_acc:.4f} | kept -> {out_file}")
                continue

        if batch_start < 0 or batch_start > num_batches:
            raise ValueError(f"Computed batch_start={batch_start} out of range [0,{num_batches}]")

        batch_iter = tqdm(
            range(batch_start, num_batches),
            desc=f"[variant {v_idx}/{len(pairs)}] Prompting | {num_batches} batches",
            ncols=120,
        )

        for b_idx in batch_iter:
            start_idx = b_idx * args.batch_size
            batch_rows = test[start_idx : start_idx + args.batch_size]

            query_prompts = build_classification_query_prompts_from_gq(
                batch_rows=batch_rows,
                dataset=args.dataset,
                model=args.model,
                general_question=gq,
                include_user_text=include_user_text,
                include_ts=include_ts,
                include_LETSCLike=include_LETSCLike,
                include_vis=include_vis,
                CoT=bool(args.CoT),
            )

            outputs: List[str] = prompter.get_completion(query_prompts, batch=True)

            for row, model_output in zip(batch_rows, outputs):
                idx_scalar = int(np.asarray(row.idx).item())

                # If resuming, avoid duplicates
                if args.resume == 1 and idx_scalar in processed_idxs:
                    continue

                y_scalar = int(np.asarray(row.y).item())
                letter, pred = extract_letter_to_idx(model_output, test.label_maps["letter_to_id"])
                pred_scalar = int(pred)

                correct = int(pred_scalar == y_scalar) if pred_scalar >= 0 else 0
                total += 1
                acc += correct
                processed_idxs.add(idx_scalar)

                append_jsonl(str(out_file), {
                    "variant_idx": v_idx,
                    "batch_idx": b_idx,
                    "embedding_types": args.embedding_types,
                    "cot": int(args.CoT),
                    "idx": idx_scalar,
                    "correct": correct,
                    "gt": y_scalar,
                    "pred": pred_scalar,
                    "letter": letter,
                    "model_output": model_output,
                })

            running_acc = acc / max(total, 1)
            batch_iter.set_postfix(acc=f"{running_acc:.4f}", done=f"{total}/{test_len}")

        final_acc = acc / max(total, 1)
        print(f"[variant {v_idx}] ✅ finished | acc={final_acc:.4f} | saved -> {out_file}")


if __name__ == "__main__":
    main()
