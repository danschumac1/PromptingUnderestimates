#!/usr/bin/env python3
"""
Embed concatenated SLIKE answers (train_As.jsonl / test_As.jsonl) and save per-layer embeddings.

Example:
CUDA_VISIBLE_DEVICES=2 python ./src/embed_slike.py \
  --dataset har \
  --model qwen \
  --batch_size 4 \
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from utils.file_io import load_jsonl, save_embeddings
from utils.setup import slike_setup, standard_args

# IMPORTANT: these are your prompt objects
from utils.prompt_objects import LlamaVisPrompt, MistralVisPrompt, QwenVisPrompt  # adjust path if needed

DATA_ROOT = Path("/raid/hdd249/Classification_v2/data/SLLM")
OUT_ROOT = Path("/raid/hdd249/Classification_v2/data/sample_features")


def load_index_texts(jsonl_path: Path, join_with: str = "\n\n") -> List[Tuple[int, str]]:
    """
    Reads rows like {"index": 0, "As": ["...", "..."]} and returns sorted [(idx, joined_text), ...].
    """
    rows = load_jsonl(str(jsonl_path))
    out: List[Tuple[int, str]] = []

    for r in rows:
        idx = int(r["index"])
        As: List[str] = r["As"]
        text = join_with.join(As)
        out.append((idx, text))

    out.sort(key=lambda x: x[0])
    return out


def make_text_prompt(model: str, text: str):
    m = model.strip().lower()

    if m == "qwen":
        return QwenVisPrompt(user_text=text)      # builds Qwen schema

    if m == "mistral":
        return MistralVisPrompt(user_text=text)   # builds Llama/Mistral-style schema

    if m == "llama":
        return LlamaVisPrompt(user_text=text)     # builds Llama schema

    raise ValueError(f"Unknown model={model}. Expected one of: llama, mistral, qwen")


def main():
    arg_parser = standard_args()
    args = arg_parser.parse_args()

    # setup prompter + out_dir
    logger, prompter = slike_setup(
        script="embedding",
        dataset=args.dataset,
        model=args.model,
    )

    # load SLIKE As jsonl
    input_root = DATA_ROOT / args.dataset
    train_pairs = load_index_texts(input_root / "train_As.jsonl", join_with="\n\n")
    test_pairs = load_index_texts(input_root / "test_As.jsonl", join_with="\n\n")

    output_root = OUT_ROOT / args.model / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded train_As rows: {len(train_pairs)}")
    logger.info(f"Loaded test_As rows:  {len(test_pairs)}")

    embeddings: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "test": {}}

    for split_name, pairs in (("train", train_pairs), ("test", test_pairs)):
        num_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
        batch_embeds: Dict[str, List[np.ndarray]] = {}

        for start in tqdm(
            range(0, len(pairs), args.batch_size),
            desc=f"Embedding {split_name} As | {num_batches} batches | {len(pairs)} rows",
        ):
            batch_pairs = pairs[start : start + args.batch_size]
            batch_prompts = [make_text_prompt(args.model, text) for (_, text) in batch_pairs]

            # returns dict[layer_key -> torch.Tensor(B,D)]
            all_layer_embs = prompter.get_all_layer_embeddings(batch_prompts, batch=True)

            for layer_key, emb_tensor in all_layer_embs.items():
                batch_embeds.setdefault(layer_key, []).append(
                    emb_tensor.detach().cpu().numpy()
                )

        # concat batches -> (N,D) per layer
        for layer_key, mats in batch_embeds.items():
            embeddings[split_name][layer_key] = np.concatenate(mats, axis=0)

        num_layers = len(embeddings[split_name])
        sample_shape = next(iter(embeddings[split_name].values())).shape if num_layers else (0, 0)
        logger.info(f"✅ Finished {split_name}: layers={num_layers}, sample_shape={sample_shape}")

    save_embeddings(
        train_embed=embeddings["train"],
        test_embed=embeddings["test"],
        save_path=output_root,
        file_suffix="slike"
    )
    logger.info(f"✅ Saved embeddings to {output_root} (train_slike.npz, test_slike.npz)")


if __name__ == "__main__":
    main()
