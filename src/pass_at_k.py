"""
Example:
python ./src/pass_at_k.py \
  --dataset har \
  --model qwen \
  --embedding_types vis,ust \
  --n 20 \
  --batch_size 5 \
  --resume 1
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import numpy as np
from tqdm import tqdm

from utils.file_io import append_jsonl
from utils.preprocessing import extract_letter_to_idx
from utils.build_prompts import build_classification_query_prompts, build_classification_system_prompt
from utils.constants import build_valid_embedding_strings
from utils.loaders import load_train_test
from utils.prompters import TogetherPrompter


TOGETHER_MODEL_MAP = {
    "qwen": "Qwen/Qwen3-VL-32B-Instruct",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True, choices=list(TOGETHER_MODEL_MAP.keys()))
    p.add_argument("--normalize", type=int, choices=[0, 1], default=1)

    p.add_argument("--n", type=int, default=20, help="number of attempts per example")
    p.add_argument("--batch_size", type=int, default=5, help="attempts per API call (per example)")

    # kept for CLI compatibility
    p.add_argument("--CoT", type=int, choices=[0, 1], default=0)

    p.add_argument(
        "--embedding_types",
        type=str,
        default="vis,ust",
        choices=build_valid_embedding_strings(),
        help="Comma-separated: ts,vis,lets,ust (must be a valid combo from build_valid_embedding_strings())",
    )
    p.add_argument("--max_variants", type=int, default=0, help="unused here (kept for CLI compatibility)")

    # NEW
    p.add_argument(
        "--resume",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, do not clear output; skip completed (idx,attempt) rows and continue.",
    )
    return p.parse_args()


def _load_progress(out_file: Path) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
    """
    Returns:
      done_attempts[idx] = {attempts_completed}
      acc1_correct[idx] = correct value for attempt==0 (if present)
    """
    done_attempts: Dict[int, Set[int]] = {}
    acc1_correct: Dict[int, int] = {}

    if not out_file.exists():
        return done_attempts, acc1_correct

    with out_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # ignore trailing partial line if the job died mid-write
                continue

            idx = int(obj.get("idx"))
            attempt = int(obj.get("attempt"))

            done_attempts.setdefault(idx, set()).add(attempt)

            if attempt == 0 and "correct" in obj:
                acc1_correct[idx] = int(obj["correct"])

    return done_attempts, acc1_correct


def main():
    args = parse_args()

    if args.n < 1:
        raise ValueError("--n must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    # --- embedding flags ---
    token_set = {t.strip() for t in args.embedding_types.split(",") if t.strip()}
    include_ts = "ts" in token_set
    include_vis = "vis" in token_set
    include_LETSCLike = "lets" in token_set
    include_user_text = "ust" in token_set

    if not (include_ts or include_vis or include_LETSCLike):
        raise ValueError("Need at least one of ts/vis/lets in --embedding_types.")
    if include_ts and include_LETSCLike:
        raise ValueError("Cannot include both ts and lets at the same time.")

    # --- load data ---
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


    # --- prompter ---
    together_model_id = TOGETHER_MODEL_MAP[args.model]
    prompter = TogetherPrompter(
        model_id=together_model_id,
        temperature=0.6,
    )
    prompter.system_prompt = build_classification_system_prompt(args.dataset, args.CoT)

    # --- output ---
    out_root = Path(f"/raid/hdd249/data/sample_generations/together-{args.model}/{args.dataset}/pass-at-k")
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"{args.embedding_types.replace(',','-')}_cot{args.CoT}_n{args.n}.jsonl"

    if args.resume and out_file.exists():
        print(f"(RESUME) Using existing output: {out_file}")
    else:
        print(f"(CLEARING OUTPUT) Results will be saved to: {out_file}")
        with open(out_file, "w") as _:
            pass

    # --- resume bookkeeping ---
    done_attempts, acc1_correct = _load_progress(out_file)

    # --- eval bookkeeping (acc@1 over examples processed/known) ---
    total_examples = 0
    acc = 0

    # If resuming, we can pre-count acc@1 for examples where attempt 0 already exists
    # BUT only for examples in THIS test slice/order.
    # We'll update totals as we iterate, counting attempt0 from either cache or new generation.
    for row in tqdm(test, desc="Prompting | per-example (attempt-batched)", ncols=80):
        idx_scalar = int(np.asarray(row.idx).item())
        y_scalar = int(np.asarray(row.y).item())

        # build prompt once for this example
        base_prompt = build_classification_query_prompts(
            batch_rows=row,
            dataset=args.dataset,
            model=args.model,
            include_user_text=include_user_text,
            include_ts=include_ts,
            include_LETSCLike=include_LETSCLike,
            include_vis=include_vis,
            CoT=bool(args.CoT),
        )[0]

        # Determine missing attempts for this idx
        already = done_attempts.get(idx_scalar, set())
        missing = [a for a in range(args.n) if a not in already]

        # If nothing missing, just contribute acc@1 if we have it, then continue
        if len(missing) == 0:
            total_examples += 1
            if idx_scalar in acc1_correct:
                acc += int(acc1_correct[idx_scalar] == 1)
            else:
                # attempt 0 missing shouldn't happen if missing is empty, but be safe
                pass
            running_acc1 = acc / total_examples
            print(f"__running__ acc@1={running_acc1:.4f} | (skipped idx={idx_scalar}, complete)")
            continue

        # Otherwise, run ONLY missing attempts, still in batches
        # To keep batching simple, we batch the missing list in chunks of batch_size.
        for start in range(0, len(missing), args.batch_size):
            chunk = missing[start : start + args.batch_size]
            round_prompts = [base_prompt] * len(chunk)
            round_outputs: list[str] = prompter.get_completion(round_prompts, batch=True)

            for rep, out in enumerate(round_outputs):
                attempt_id = int(chunk[rep])

                letter, pred = extract_letter_to_idx(out, test.label_maps["letter_to_id"])
                pred_scalar = int(pred)

                correct = int(pred_scalar == y_scalar) if pred_scalar >= 0 else 0

                # record progress in-memory (for crash safety mid-example)
                done_attempts.setdefault(idx_scalar, set()).add(attempt_id)
                if attempt_id == 0:
                    acc1_correct[idx_scalar] = correct

                append_jsonl(
                    str(out_file),
                    {
                        "idx": idx_scalar,
                        "attempt": attempt_id,
                        "correct": correct,
                        "gt": y_scalar,
                        "pred": pred_scalar,
                        "embedding_types": args.embedding_types,
                        "cot": int(args.CoT),
                        "n": int(args.n),
                        "letter": letter,
                        "model_output": out,
                    },
                )

        # Now update acc@1 using cached/new attempt0 result (if present)
        total_examples += 1
        if idx_scalar in acc1_correct:
            acc += int(acc1_correct[idx_scalar] == 1)

        running_acc1 = acc / total_examples
        print(f"__running__ acc@1={running_acc1:.4f} |")

    final_acc1 = acc / max(total_examples, 1)
    print(f"finished | examples={total_examples} | acc={final_acc1:.4f} | saved -> {out_file}")


if __name__ == "__main__":
    main()
