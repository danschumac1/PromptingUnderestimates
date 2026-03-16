"""
CUDA_VISIBLE_DEVICES=3 python ./src/prompting.py \
    --dataset har \
    --model qwen \
    --batch_size 3 \
    --n_shots 2 \
    --sample 1 \
    --include_ts 0 \
    --include_vis 1 \
    --include_LETSCLike 0
"""

import argparse
from dataclasses import dataclass
from typing import Any
import numpy as np
from tqdm import tqdm
from utils.constants import build_valid_embedding_strings
from utils.loaders import Split, _load_artifacts_new
from utils.prompters import VisionPrompter
from utils.prompt_objects import VisionPrompt
from utils.setup import setup, standard_args
from utils.build_prompts import (
    build_classification_query_prompts,
    build_classification_system_prompt,
    build_few_shot_classification_examples,
)
from utils.file_io import append_jsonl
from utils.preprocessing import extract_letter_to_idx  # or wherever you keep it



def additional_args(args:argparse.ArgumentParser):
    args.add_argument("--embedding_types", type=str, choices=build_valid_embedding_strings())
    return args.parse_args()



if __name__ == "__main__":
    arg_parser = standard_args()
    args = additional_args(arg_parser)
    tokens = args.embedding_types.split(",")
    token_set = set(tokens)
    args.include_ts = "ts" in token_set
    args.include_vis = "vis" in token_set
    args.include_LETSCLike = "lets" in token_set
    args.include_user_text = "ust" in token_set


    out_file, train, test, logger, prompter = setup(
        script="prompting",
        dataset=args.dataset,
        model=args.model,
        n_shots=args.n_shots,
        include_ts=args.include_ts,
        include_vis=args.include_vis,
        include_LETSCLike=args.include_LETSCLike,
        include_user_text=True,
        sample=args.sample,
        CoT=args.CoT

    )

    if (args.CoT and (args.n_shots > 0)):
       logger.warning( "No CoT with fewshot allowed | OVERWRITING COT TO FALSE")
       args.CoT = 0


    # set the system prompt
    prompter.system_prompt = build_classification_system_prompt(args.dataset)

    # ------------------------------------------------
    # 1) Build few-shot prompts
    # ------------------------------------------------
    few_shot_examples: list[VisionPrompt] = []
    if args.n_shots > 0:
        loc_string = "samples" if args.sample else "datasets"
        class_shots, label_maps, general_question = _load_artifacts_new(f"./data/{loc_string}/{args.dataset}")
        # clip to number of shots per class
        class_shots = {k: v[:args.n_shots] for k, v in class_shots.items()}

        few_shot_examples = build_few_shot_classification_examples(
            class_shots=class_shots,
            dataset=args.dataset,
            model=args.model,
            train=train,
            test=test,
            include_ts=args.include_ts,
            include_LETSCLike=args.include_LETSCLike,
            include_vis=args.include_vis,
            CoT=args.CoT
        )

    # ------------------------------------------------
    # 2) Classification prompting
    # ------------------------------------------------
    acc = 0
    total = 0
    num_batches = (len(test) + args.batch_size - 1) // args.batch_size
    for start_idx in tqdm(
        range(0, len(test), args.batch_size),
        desc=f"Prompting for Classification | {num_batches} Batches | {len(test)} Rows:",
    ):
        batch_rows = test[start_idx : start_idx + args.batch_size]

        query_prompts = build_classification_query_prompts(
            batch_rows=batch_rows,
            dataset=args.dataset,
            model=args.model,
            include_ts=False,
            include_LETSCLike=False,
            include_vis=True,
            CoT=args.CoT
        )

        outputs: list[str] = []

        if not few_shot_examples:
            # zero-shot: direct batched call
            outputs = prompter.get_completion(query_prompts, batch=True)
        else:
            # few-shot: build full conversation per query
            for qp in query_prompts:
                convo_prompts = few_shot_examples + [qp]
                out = prompter.get_completion(convo_prompts, batch=False)
                outputs.append(out)

        # save + running accuracy
        for row, model_output in zip(batch_rows, outputs):
            idx_scalar = int(np.asarray(row.idx).item())
            y_scalar = int(np.asarray(row.y).item())
            letter, pred = extract_letter_to_idx(
                model_output, test.label_maps["letter_to_id"]
            )
            pred_scalar = int(pred)
            correct = int(pred_scalar == y_scalar) if pred_scalar >= 0 else 0

            total += 1
            acc += correct
            running_acc = acc / max(total, 1)

            line = {
                "idx": idx_scalar,
                "correct": correct,
                "gt": y_scalar,
                "pred": pred_scalar,
                "letter": letter,
                "model_output": model_output,
            }
            append_jsonl(out_file, line)

        logger.info(
            f"Running Acc after idx {int(np.asarray(batch_rows[-1].idx).item())}: "
            f"{running_acc:.4f}"
        )

    logger.info(f"✅ Finished. Results saved → {out_file}")
