"""
python ./src/bert.py --dataset ctu --out_dir ./data/sample_features/bert
"""

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import sys; sys.path.append("./src")
from utils.loaders import load_train_test


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        type=str,
        required=True,
        help="Dataset name (must match ./data/samples/{dataset})",
    )
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1, help="Keep 1 unless you add dynamic padding.")
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or leave empty for auto")
    p.add_argument("--out_dir", type=str, default="./data/sample_features/bert")
    return p.parse_args()


# -----------------------------
# Pooling
# -----------------------------
def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    Returns: (B, H) mean pooled over valid tokens
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                          # (B, 1)
    return summed / counts


# -----------------------------
# Chunked embed
# -----------------------------
@torch.no_grad()
def bert_sentence_embedding_chunked(
    text: str,
    tokenizer,
    model,
    max_length: int = 512,
    stride: int = 64,
    device: str = "cpu",
    weight_exclude_specials: bool = True,
) -> torch.Tensor:
    """
    Single embedding (H,) for `text`.
    - If fits in max_length -> mean pool once
    - Else -> sliding window chunks, embed each, then token-count-weighted mean

    If weight_exclude_specials=True, chunk weights are len(window) (content only).
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = enc["input_ids"][0].tolist()

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Remove global [CLS] ... [SEP] so we can re-add per chunk
    if len(input_ids) >= 2 and input_ids[0] == cls_id and input_ids[-1] == sep_id:
        ids = input_ids[1:-1]
    else:
        ids = input_ids

    content_len = max_length - 2
    step = max(1, content_len - stride)

    # fast path
    if len(ids) + 2 <= max_length:
        chunk = [cls_id] + ids + [sep_id]
        chunk_ids = torch.tensor([chunk], device=device)
        chunk_mask = torch.ones_like(chunk_ids, device=device)

        out = model(input_ids=chunk_ids, attention_mask=chunk_mask)
        emb = mean_pool_last_hidden(out.last_hidden_state, chunk_mask)  # (1, H)
        return emb.squeeze(0).cpu()

    chunk_embs: List[torch.Tensor] = []
    chunk_weights: List[float] = []

    for start in range(0, len(ids), step):
        end = min(start + content_len, len(ids))
        window = ids[start:end]

        chunk = [cls_id] + window + [sep_id]
        chunk_ids = torch.tensor([chunk], device=device)
        chunk_mask = torch.ones_like(chunk_ids, device=device)

        out = model(input_ids=chunk_ids, attention_mask=chunk_mask)
        emb = mean_pool_last_hidden(out.last_hidden_state, chunk_mask)  # (1, H)

        if weight_exclude_specials:
            w = float(len(window))  # content only
        else:
            w = float(chunk_mask.sum().item())  # includes specials

        chunk_embs.append(emb.squeeze(0))
        chunk_weights.append(w)

        if end == len(ids):
            break

    weights = torch.tensor(chunk_weights, device=device, dtype=chunk_embs[0].dtype)  # (C,)
    stacked = torch.stack(chunk_embs, dim=0)                                         # (C, H)

    # Weighted mean across chunks
    denom = weights.sum().clamp(min=1.0)
    final = (stacked * weights.unsqueeze(1)).sum(dim=0) / denom
    return final.cpu()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    device = args.device
    if device is None or device.strip() == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    train, test = load_train_test(
        input_folder=f"./data/samples/{args.dataset}",
        n_shots=0,
    )

    # Try to support both:
    # - train/test as Split objects with .X
    # - train/test iterable of row objects with .X
    def iter_examples(split_obj):
        if hasattr(split_obj, "X"):
            # Split: X is typically (N, T, D) or list-like
            for i in range(len(split_obj.X)):
                yield split_obj.X[i]
        else:
            # iterable rows
            for row in split_obj:
                yield row.X

    def embed_split(split_obj, split_name: str) -> np.ndarray:
        vecs = []
        for x in iter_examples(split_obj):
            # Convert time series array to a JSON string (stable + explicit)
            # If x is np.ndarray: x.tolist() works. If it’s already list-like: fine.
            text = json.dumps(np.asarray(x).tolist())

            emb = bert_sentence_embedding_chunked(
                text=text,
                tokenizer=tokenizer,
                model=model,
                max_length=args.max_length,
                stride=args.stride,
                device=device,
                weight_exclude_specials=True,
            )
            vecs.append(emb)

        embs = torch.stack(vecs, dim=0).numpy()  # (N, H)
        print(f"{split_name}: {embs.shape}")
        return embs

    train_embs = embed_split(train, "train")
    test_embs = embed_split(test, "test")

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train_embeddings.npz")
    test_path = os.path.join(out_dir, "test_embeddings.npz")

    np.savez_compressed(train_path, embeddings=train_embs)
    np.savez_compressed(test_path, embeddings=test_embs)

    print("Saved:")
    print(" ", train_path)
    print(" ", test_path)


if __name__ == "__main__":
    main()
