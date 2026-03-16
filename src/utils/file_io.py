import json
import os
from typing import Any, Iterable

import numpy as np

import csv

def load_tsv(file_path: str) -> list[dict[str, str]]:
    """Load a TSV file with a header row."""
    data = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append(dict(row))
    return data

def append_tsv(file_path: str, row: dict[str, Any]):
    """Append a row to a TSV file. Creates the file with header if it doesn't exist."""
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(output_path: str, data: dict):
    """
    Append a dictionary to the specified output JSONL file.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write('\n')
    

def load_jsonl(file_path: str):
    """Load a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: Iterable[dict[str, Any]], file_path: str):
    """Save an iterable of dicts to a JSON Lines file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(file_path:str) -> dict | list:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def ensure_header(results_path: str, header_cols: list[str]):
    # create parent dir
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = (not os.path.exists(results_path)) or (os.path.getsize(results_path) == 0)
    if write_header:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\t".join(header_cols) + "\n")
            

def append_row(results_path: str, row_vals: list[str]):
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\t".join(map(str, row_vals)) + "\n")


def save_json(results_path: str, data_dict: dict):
    with open(results_path, "w", encoding="utf-8") as fo:
        json.dump(data_dict, fo, indent=2, ensure_ascii=False)


import os
import numpy as np

def save_embeddings(
    train_embed: dict[str, np.ndarray] | np.ndarray | None,
    test_embed: dict[str, np.ndarray] | np.ndarray | None,
    save_path: str,
    file_suffix: str = "_embeddings",
    overwrite: int = 1,   # 0 = safe default
):
    """
    Save embeddings to disk.

    - If train_embed/test_embed is None (or empty dict), that split is NOT saved.
    - If overwrite=0 and the target file exists, raises FileExistsError.
    - Dict -> .npz (per-layer), ndarray -> .npy (single array)
    """
    os.makedirs(save_path, exist_ok=True)

    def _is_empty(x) -> bool:
        return x is None or (isinstance(x, dict) and len(x) == 0)

    def _save_one(split: str, emb):
        if _is_empty(emb):
            return  # nothing to do

        if isinstance(emb, dict):
            out_file = os.path.join(save_path, f"{split}_{file_suffix}.npz")
            if (not overwrite) and os.path.exists(out_file):
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {out_file}. "
                    f"Pass overwrite=1 (or add a CLI flag) if you really mean it."
                )
            np.savez(out_file, **emb)
        else:
            out_file = os.path.join(save_path, f"{split}_{file_suffix}.npy")
            if (not overwrite) and os.path.exists(out_file):
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {out_file}. "
                    f"Pass overwrite=1 (or add a CLI flag) if you really mean it."
                )
            np.save(out_file, emb)

    _save_one("train", train_embed)
    _save_one("test", test_embed)


# def load_embeddings(
#     load_path: str,
#     layer: int | str = -1,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Load embeddings from disk, selecting a specific layer.
    
#     Args:
#         load_path: Directory containing the embedding files.
#         layer: Which layer's embeddings to load.
#                - For .npz files: layer index (int or str). Use -1 for last layer.
#                - For .npy files: ignored (only one layer available).
    
#     Returns:
#         (train_embeddings, test_embeddings) as numpy arrays of shape [N, D].
#     """
#     train_npz = os.path.join(load_path, "train_embeddings.npz")
#     test_npz = os.path.join(load_path, "test_embeddings.npz")
#     train_npy = os.path.join(load_path, "train_embeddings.npy")
#     test_npy = os.path.join(load_path, "test_embeddings.npy")
    
#     # Check for .npz (per-layer) format first
#     if os.path.exists(train_npz) and os.path.exists(test_npz):
#         train_data = np.load(train_npz)
#         test_data = np.load(test_npz)
        
#         # Get available layer keys (should be "0", "1", ..., "L")
#         layer_keys = sorted(train_data.keys(), key=lambda x: int(x))
        
#         # Handle layer=-1 (last layer)
#         if isinstance(layer, int) and layer < 0:
#             layer_key = layer_keys[layer]  # e.g., layer=-1 gets last key
#         else:
#             layer_key = str(layer)
        
#         if layer_key not in train_data:
#             raise ValueError(
#                 f"Layer '{layer_key}' not found. Available layers: {layer_keys}"
#             )
        
#         return train_data[layer_key], test_data[layer_key]
    
#     # Fallback to .npy (single-layer) format
#     elif os.path.exists(train_npy) and os.path.exists(test_npy):
#         return np.load(train_npy), np.load(test_npy)
    
#     else:
#         raise FileNotFoundError(
#             f"No embedding files found in {load_path}. "
#             "Expected train_embeddings.npz/npy and test_embeddings.npz/npy"
#         )

import os
import numpy as np


def _first_existing(path_candidates: list[str]) -> str | None:
    for p in path_candidates:
        if os.path.exists(p):
            return p
    return None


def load_embeddings(
    load_path: str,
    layer: int | str = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from disk, selecting a specific layer.

    Accepts either naming convention:
      - train_embeddings.{npz|npy}, test_embeddings.{npz|npy}
      - train__embeddings.{npz|npy}, test__embeddings.{npz|npy}

    Returns:
        (train_embeddings, test_embeddings) as numpy arrays of shape [N, D].
    """
    # Two supported basenames
    train_bases = ["train_embeddings", "train__embeddings"]
    test_bases = ["test_embeddings", "test__embeddings"]

    # Try NPZ pairs first (must be consistent: both train+test exist)
    for tb, vb in zip(train_bases, test_bases):
        train_npz = os.path.join(load_path, f"{tb}.npz")
        test_npz = os.path.join(load_path, f"{vb}.npz")
        if os.path.exists(train_npz) and os.path.exists(test_npz):
            train_data = np.load(train_npz)
            test_data = np.load(test_npz)

            # Keys like "0","1",... or "layer0","layer1", etc — handle both.
            keys = list(train_data.keys())

            def key_to_int(k: str) -> int:
                # supports "0" or "layer0"
                if k.isdigit():
                    return int(k)
                if k.startswith("layer") and k[5:].isdigit():
                    return int(k[5:])
                raise ValueError(f"Unrecognized layer key: {k}")

            layer_keys_sorted = sorted(keys, key=key_to_int)

            # Choose the layer key
            if isinstance(layer, int) and layer < 0:
                layer_key = layer_keys_sorted[layer]  # -1 => last
            else:
                layer_key = str(layer)
                # also allow passing int 0 meaning key "0" even if stored as "layer0"
                if layer_key not in train_data and f"layer{layer_key}" in train_data:
                    layer_key = f"layer{layer_key}"

            if layer_key not in train_data:
                raise ValueError(
                    f"Layer '{layer}' not found in {train_npz}. "
                    f"Available layers: {layer_keys_sorted}"
                )

            X_train = train_data[layer_key]
            X_test = test_data[layer_key]
            return X_train, X_test

    # Fallback to NPY pairs
    for tb, vb in zip(train_bases, test_bases):
        train_npy = os.path.join(load_path, f"{tb}.npy")
        test_npy = os.path.join(load_path, f"{vb}.npy")
        if os.path.exists(train_npy) and os.path.exists(test_npy):
            return np.load(train_npy), np.load(test_npy)

    # Helpful error
    try:
        found = sorted(os.listdir(load_path))
    except FileNotFoundError:
        found = ["<load_path does not exist>"]

    raise FileNotFoundError(
        f"No embedding files found in {load_path}.\n"
        f"Expected one of:\n"
        f"  train_embeddings.npz + test_embeddings.npz\n"
        f"  train__embeddings.npz + test__embeddings.npz\n"
        f"  train_embeddings.npy + test_embeddings.npy\n"
        f"  train__embeddings.npy + test__embeddings.npy\n"
        f"Found: {found}"
    )
