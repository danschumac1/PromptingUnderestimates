"""
CUDA_VISIBLE_DEVICES=2 python ./src/bert.py --dataset had --model_name bert-large-uncased --auto_chunk

Purpose
-------
Compute fixed-size CLS embeddings for 0-shot train/test time-series samples:
1) Load samples from: ./data/samples/{dataset}
2) Serialize each time series into a text string (json / space / digits)
3) Embed with a HF encoder model using the [CLS] vector
4) Save embeddings to compressed .npz files

Chunking (automatic)
--------------------
We always use CLS pooling. If --auto_chunk is set:
- If any example would exceed the model/tokenizer limit (max_length),
  we chunk the token ids into windows of size max_length and average CLS
  across windows.
- Otherwise we embed normally with truncation (no chunking needed).

Outputs
-------
Writes exactly two files:
  {out_root}/{dataset}/{model_tag}/train.npz
  {out_root}/{dataset}/{model_tag}/test.npz

Each .npz contains:
  X    : embeddings (N, H) float32
  y    : labels     (N,)
  meta : dict with run/config info (including whether chunking was used)
"""

import os
import json
import math
import argparse
from typing import List, Tuple, Any, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from utils.loaders import Split, load_train_test


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        required=True,
        help="Dataset name (must match ./data/samples/{dataset})",
    )

    p.add_argument("--model_name", type=str, default="bert-large-uncased")
    p.add_argument("--batch_size", type=int, default=32, help="Examples per batch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Serialization
    p.add_argument(
        "--ts_format",
        choices=["json", "space", "digits"],
        default="digits",
        help="How to serialize the numeric sequence into text.",
    )
    p.add_argument("--precision", type=int, default=4, help="Used only for ts_format=digits")
    p.add_argument("--step_sep", type=str, default=" ,", help="Separator between timesteps (digits mode)")
    p.add_argument("--channel_sep", type=str, default=" |", help="Separator between channels (digits mode)")

    # Embedding behavior
    p.add_argument("--normalize", action="store_true", help="L2 normalize embeddings")

    # Chunking
    p.add_argument(
        "--auto_chunk",
        action="store_true",
        help="Automatically chunk long sequences based on the tokenizer/model max length.",
    )

    # Output
    p.add_argument(
        "--out_root",
        type=str,
        default="/raid/hdd249/Classification_v2/data/sample_features/bert",
        help="Root output directory (will create dataset/model subdirs).",
    )

    # HF cache (optional)
    p.add_argument(
        "--hf_home",
        type=str,
        default="",
        help="Optional HF cache root (set if shared cache permissions cause issues).",
    )

    return p.parse_args()


# -----------------------------
# Serialization
# -----------------------------

def canonicalize_ts(x: np.ndarray) -> np.ndarray:
    """
    Convert common dataset shapes into either (T,) or (T, C).

    Supported:
      - (T,)                 -> (T,)
      - (T, C)               -> (T, C)
      - (1, T)               -> (T,)
      - (1, T, C)            -> (T, C)
      - (T, C, 1)            -> (T, C)  (squeezes singleton last dim)
      - (1, T, C, 1)         -> (T, C)

    If there are multiple leading segments (B > 1), we raise so it’s explicit.
    """
    x = np.asarray(x)

    # Squeeze *trailing* singleton dims freely (common: last dim=1)
    while x.ndim >= 2 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim == 1:  # (T,)
        return x

    if x.ndim == 2:  # (T, C)
        return x

    if x.ndim == 3:
        # (1, T, C) or (B, T, C)
        if x.shape[0] == 1:
            return x[0]
        raise ValueError(f"Expected leading dim 1 for (B,T,C), got B={x.shape[0]} with shape {x.shape}")

    raise ValueError(f"Unsupported time series shape {x.shape}; expected (T,) or (T,C) after canonicalization.")

def ts_to_text(
    x: np.ndarray,
    fmt: str,
    precision: int = 4,
    step_sep: str = " ,",
    channel_sep: str = " |",
) -> str:
    """
    Serialize a time series for text tokenization.

    fmt:
      - "json"   : json.dumps(list)
      - "space"  : space-separated floats (flatten multivariate)
      - "digits" : fixed-precision digit-spaced encoding

    digits mode:
      - scale = 10^precision, round to int
      - sign is separate token "-"
      - each digit is separated by spaces
      - separate timesteps by step_sep and channels by channel_sep
    """
    x = canonicalize_ts(np.asarray(x))

    if fmt == "json":
        return json.dumps(x.tolist())

    if fmt == "space":
        arr = x.tolist()
        if x.ndim == 2:
            flat = [v for row in arr for v in row]
        else:
            flat = arr
        return " ".join(str(v) for v in flat)

    if fmt != "digits":
        raise ValueError(f"Unknown fmt: {fmt}")

    scale = 10 ** int(precision)

    def encode_scalar(v: float) -> str:
        if not np.isfinite(v):
            if np.isnan(v):
                return "n a n"
            return "i n f" if v > 0 else "- i n f"

        s = int(np.rint(float(v) * scale))
        if s < 0:
            return "- " + " ".join(list(str(-s)))
        return " ".join(list(str(s)))

    if x.ndim == 1:
        return step_sep.join(encode_scalar(v) for v in x)

    if x.ndim == 2:
        T, C = x.shape
        steps: List[str] = []
        for t in range(T):
            chans = [encode_scalar(x[t, c]) for c in range(C)]
            steps.append(channel_sep.join(chans))
        return step_sep.join(steps)

    raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


def extract_xy(split: Split) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns:
      xs: list of numpy arrays (time series)
      y : (N,) labels as int64 if possible, else float32
    """
    xs: List[np.ndarray] = []
    ys: List[Any] = []

    for row in split:
        xs.append(np.asarray(row.X))
        y = row.y
        if isinstance(y, np.ndarray):
            y = y.item() if y.size == 1 else y
        ys.append(y)

    y_arr = np.asarray(ys)
    if np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    else:
        try:
            as_int = y_arr.astype(np.int64)
            if np.all(as_int.astype(np.float64) == y_arr.astype(np.float64)):
                y_arr = as_int
            else:
                y_arr = y_arr.astype(np.float32)
        except Exception:
            y_arr = y_arr.astype(np.float32)

    return xs, y_arr


# -----------------------------
# Chunk decision
# -----------------------------
def get_token_limit(tokenizer: AutoTokenizer, model: AutoModel) -> int:
    """
    Return a safe maximum sequence length that the model can actually handle.
    For BERT-family encoders this is 512.

    We prefer model.config.max_position_embeddings when available because it
    reflects the actual position embedding table size (hard constraint).
    """
    # Hard constraint from model config (most reliable)
    mpe = getattr(model.config, "max_position_embeddings", None)
    if isinstance(mpe, int) and mpe > 0:
        return int(mpe)

    # Fallback to tokenizer's claim, but guard against sentinel huge values
    ml = getattr(tokenizer, "model_max_length", None)
    if ml is None:
        return 512
    if ml > 100_000:
        return 512
    return int(ml)




# -----------------------------
# Embedding (simple: truncate)
# -----------------------------
@torch.no_grad()
def embed_texts_simple_cls(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    """
    CLS embedding with truncation to max_length.
    """
    model.eval()
    all_embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**batch)
        hidden = out.last_hidden_state  # (B, T, H)
        emb = hidden[:, 0, :]

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embs.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embs, axis=0)


# -----------------------------
# Embedding (chunked: CLS per window, average)
# -----------------------------
@torch.no_grad()
def embed_texts_chunked_cls_batched(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    window_len: int,
    normalize: bool,
) -> np.ndarray:
    """
    Tokenize each text WITHOUT truncation, create non-overlapping windows of size window_len,
    compute CLS per window, average per example.

    window_len is the content length (excluding CLS/SEP). We use stride == window_len.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive")

    model.eval()

    # tokenize all (no special tokens, no truncation)
    all_ids = [tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"] for t in texts]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        pad_id = tokenizer.pad_token_id

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id for CLS chunking.")

    max_len = max(max(len(ids) for ids in all_ids), 1)
    num_chunks = int(math.ceil(max_len / window_len))  # stride == window_len
    H = int(model.config.hidden_size)

    out_embs = np.zeros((len(texts), H), dtype=np.float32)

    def build_batch(batch_ids: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = len(batch_ids)

        padded = np.full((B, max_len), pad_id, dtype=np.int64)
        for bi, ids in enumerate(batch_ids):
            L = min(len(ids), max_len)
            if L > 0:
                padded[bi, :L] = np.asarray(ids[:L], dtype=np.int64)

        chunk_starts = [k * window_len for k in range(num_chunks)]
        L_full = window_len + 2  # CLS + content + SEP

        input_ids = np.full((B, num_chunks, L_full), pad_id, dtype=np.int64)
        attention = np.zeros((B, num_chunks, L_full), dtype=np.int64)
        valid_mask = np.zeros((B, num_chunks), dtype=np.int64)

        for ci, s in enumerate(chunk_starts):
            window = padded[:, s : s + window_len]
            if window.shape[1] < window_len:
                pad_w = window_len - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad_w)), constant_values=pad_id)

            content_attn = (window != pad_id).astype(np.int64)
            valid = (content_attn.sum(axis=1) > 0).astype(np.int64)
            valid_mask[:, ci] = valid

            input_ids[:, ci, 0] = cls_id
            input_ids[:, ci, 1 : 1 + window_len] = window
            input_ids[:, ci, 1 + window_len] = sep_id

            attention[:, ci, 0] = 1
            attention[:, ci, 1 : 1 + window_len] = content_attn
            attention[:, ci, 1 + window_len] = 1

        return input_ids, attention, valid_mask

    for i in range(0, len(texts), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        B = len(batch_ids)

        input_ids, attention_mask, valid_chunk_mask = build_batch(batch_ids)

        flat_ids = torch.tensor(input_ids.reshape(B * num_chunks, -1), device=device)
        flat_attn = torch.tensor(attention_mask.reshape(B * num_chunks, -1), device=device)

        out = model(input_ids=flat_ids, attention_mask=flat_attn)
        hidden = out.last_hidden_state  # (B*num_chunks, L_full, H)
        cls = hidden[:, 0, :].view(B, num_chunks, H)  # (B, num_chunks, H)

        vmask = torch.tensor(valid_chunk_mask, device=device).unsqueeze(-1)  # (B, num_chunks, 1)
        emb = (cls * vmask).sum(dim=1) / vmask.sum(dim=1).clamp(min=1.0)

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        out_embs[i : i + B] = emb.detach().cpu().numpy().astype(np.float32)

    return out_embs


# -----------------------------
# IO
# -----------------------------
def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def save_split_npz(path: str, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y, meta=meta)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    # Load 0-shot splits
    train, test = load_train_test(f"./data/samples/{args.dataset}", n_shots=0)
    train_xs, train_y = extract_xy(train)
    test_xs, test_y = extract_xy(test)

    # Serialize X -> text
    train_texts = [
        ts_to_text(x, args.ts_format, args.precision, args.step_sep, args.channel_sep) for x in train_xs
    ]
    test_texts = [
        ts_to_text(x, args.ts_format, args.precision, args.step_sep, args.channel_sep) for x in test_xs
    ]

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_name)
    mdl = AutoModel.from_pretrained(args.model_name).to(args.device)

    # Single source of truth for token budget:
    # window_len is the content length, excluding CLS/SEP.
    # max_length is the full length including specials -> window_len = max_length - 2
    max_length = 512
    window_len = max_length - 2
    if window_len <= 0:
        raise ValueError(f"Tokenizer max_length too small: {max_length}")

    # Decide whether we need chunking
    mlen = 512
    use_chunked = bool(args.auto_chunk and mlen > max_length)

    # Embed (CLS only)
    if use_chunked:
        train_emb = embed_texts_chunked_cls_batched(
            train_texts,
            tok,
            mdl,
            args.device,
            batch_size=args.batch_size,
            window_len=window_len,
            normalize=args.normalize,
        )
        test_emb = embed_texts_chunked_cls_batched(
            test_texts,
            tok,
            mdl,
            args.device,
            batch_size=args.batch_size,
            window_len=window_len,
            normalize=args.normalize,
        )
    else:
        train_emb = embed_texts_simple_cls(
            train_texts,
            tok,
            mdl,
            args.device,
            batch_size=args.batch_size,
            max_length=max_length,
            normalize=args.normalize,
        )
        test_emb = embed_texts_simple_cls(
            test_texts,
            tok,
            mdl,
            args.device,
            batch_size=args.batch_size,
            max_length=max_length,
            normalize=args.normalize,
        )

    print("Train embeddings:", train_emb.shape)
    print("Test embeddings :", test_emb.shape)

    # Output
    model_tag = sanitize_model_name(args.model_name)
    out_dir = os.path.join(args.out_root, args.dataset, model_tag)

    meta: Dict[str, Any] = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "pool": "cls",
        "batch_size": int(args.batch_size),
        "tokenizer_model_max_length": int(max_length),
        "window_len": int(window_len),  # content len in chunked mode
        "auto_chunk": bool(args.auto_chunk),
        "used_chunked": bool(use_chunked),
        "max_tokens_observed": int(mlen),
        "ts_format": args.ts_format,
        "precision": int(args.precision),
        "step_sep": args.step_sep,
        "channel_sep": args.channel_sep,
        "normalize": bool(args.normalize),
        "device": args.device,
        "train_n": int(len(train_texts)),
        "test_n": int(len(test_texts)),
        "hidden_size": int(train_emb.shape[1]),
    }

    save_split_npz(os.path.join(out_dir, "train.npz"), train_emb, train_y, meta)
    save_split_npz(os.path.join(out_dir, "test.npz"), test_emb, test_y, meta)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
