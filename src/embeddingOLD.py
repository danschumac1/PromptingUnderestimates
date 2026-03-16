"""
CUDA_VISIBLE_DEVICES=2 python ./src/embedding.py \
    --dataset har \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --batch_size 3 \
    --n_shots 3 \
    --sample 1 \
    --include_ts 0 \
    --include_vis 1 \
    --include_LETSCLike 0
"""

import argparse

import numpy as np
from tqdm import tqdm

from utils.file_io import save_embeddings
from utils.setup import setup, standard_args, _build_tag
from utils.build_prompts import (
    build_classification_system_prompt,
    build_classification_query_prompts,
)


def additional_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Extend the standard parser with embedding-specific flags."""
    parser.add_argument("--include_user_text", type=int, choices=[0,1], default=1)
    parser.add_argument("--include_ts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--include_vis", type=int, choices=[0, 1], default=1)
    parser.add_argument("--include_LETSCLike", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------
    # 1) Parse args + setup common objects
    # ------------------------------------------------
    arg_parser = standard_args()
    args = additional_args(arg_parser)

    out_dir, train, test, logger, prompter = setup(
        script="embedding",
        dataset=args.dataset,
        model=args.model,
        n_shots=args.n_shots,
        include_user_text=args.include_user_text,
        include_ts=args.include_ts,
        include_vis=args.include_vis,
        include_LETSCLike=args.include_LETSCLike,
        sample=bool(args.sample),
    )

    # ------------------------------------------------
    # 2) Classification “system” prompt
    # ------------------------------------------------
    prompter.system_prompt = build_classification_system_prompt(args.dataset)

    # Turn 0/1 CLI flags into bools
    include_ts = bool(args.include_ts)
    include_vis = bool(args.include_vis)
    include_LETSCLike = bool(args.include_LETSCLike)
    include_user_text = bool(args.include_user_text)

    # ------------------------------------------------
    # 3) Embed train + test
    # ------------------------------------------------
    embeddings = {}

    for split_name, split in (("train", train), ("test", test)):
        num_batches = (len(split) + args.batch_size - 1) // args.batch_size
        batch_embeds = []

        for start_idx in tqdm(
            range(0, len(split), args.batch_size),
            desc=(
                f"Embedding {split_name} for Classification | "
                f"{num_batches} batches | {len(split)} rows"
            ),
        ):
            batch_rows = split[start_idx : start_idx + args.batch_size]

            # Build VisPrompt objects for this batch
            query_prompts = build_classification_query_prompts(
                batch_rows=batch_rows,
                dataset=args.dataset,
                model=args.model,
                include_user_text=include_user_text,
                include_ts=include_ts,
                include_LETSCLike=include_LETSCLike,
                include_vis=include_vis,
            )

            # get_embedding should return a tensor of shape [B, D] for batch=True
            batch_embed = prompter.get_embedding(query_prompts, batch=True)

            # Make sure we can concatenate later (and free GPU if needed)
            batch_embeds.append(batch_embed.detach().cpu().numpy())

        # Concatenate all batches into a single [N, D] array
        if batch_embeds:
            embeddings[split_name] = np.concatenate(batch_embeds, axis=0)
        else:
            # Degenerate case: empty split
            embeddings[split_name] = np.zeros((0, 0), dtype=float)

        logger.info(
            f"✅ Finished embedding split={split_name}, "
            f"shape={embeddings[split_name].shape}"
        )

    # ------------------------------------------------
    # 4) Save embeddings
    # ------------------------------------------------
    save_embeddings(
        train_embed=embeddings["train"],
        test_embed=embeddings["test"],
        save_path=out_dir,
    )
    logger.info(
        f"✅ All embeddings saved to {out_dir}/"
    )
