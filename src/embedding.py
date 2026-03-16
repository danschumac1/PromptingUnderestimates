#!/usr/bin/env python3
"""
blah
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

from utils.file_io import save_embeddings
from utils.setup import setup, standard_args
from utils.constants import build_valid_embedding_strings
from utils.build_prompts import (
    build_classification_system_prompt,
    build_classification_query_prompts,
)

# -------------------------
# Small utilities
# -------------------------
def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _sanitize_layer_key(layer_key: str) -> str:
    # layer keys sometimes include slashes or spaces depending on backend
    return (
        layer_key.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _open_or_create_memmap(
    layer_path: Path,
    n_rows: int,
    d: int,
    dtype: np.dtype,
    resume: bool,
) -> np.memmap:
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and layer_path.exists():
        # open existing
        arr = np.load(layer_path, mmap_mode="r+")
        # sanity checks
        if arr.shape != (n_rows, d):
            raise ValueError(
                f"Existing memmap shape mismatch for {layer_path}: "
                f"found {arr.shape}, expected {(n_rows, d)}"
            )
        if arr.dtype != dtype:
            raise ValueError(
                f"Existing memmap dtype mismatch for {layer_path}: "
                f"found {arr.dtype}, expected {dtype}"
            )
        return arr  # type: ignore[return-value]

    # create new memmap-backed .npy
    return np.lib.format.open_memmap(
        layer_path, mode="w+", dtype=dtype, shape=(n_rows, d)
    )


def _get_partial_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "partial_embeddings"


def _split_paths(out_dir: str | Path, split_name: str) -> Tuple[Path, Path]:
    partial_dir = _get_partial_dir(out_dir) / split_name
    meta_path = partial_dir / "meta.json"
    return partial_dir, meta_path


def additional_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--embedding_types",
        type=str,
        choices=build_valid_embedding_strings(),
        required=True,  # strongly recommended
        help="Comma-separated embedding types, e.g. 'lets,ust' or 'ts,vis,ust'",
    )
    parser.add_argument(
        "--resume",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, resume from partial_embeddings/*/meta.json when present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------
    # 1) Parse args + setup common objects
    # ------------------------------------------------
    arg_parser = standard_args()
    args = additional_args(arg_parser)

    tokens = args.embedding_types.split(",")
    token_set = set(tokens)
    args.include_ts = "ts" in token_set
    args.include_vis = "vis" in token_set
    args.include_LETSCLike = "lets" in token_set
    args.include_user_text = "ust" in token_set

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
        CoT=args.CoT,
    )

    # ------------------------------------------------
    # 2) Classification “system” prompt
    # ------------------------------------------------
    prompter.system_prompt = build_classification_system_prompt(args.dataset, args.CoT)

    # Turn 0/1 CLI flags into bools
    include_ts = bool(args.include_ts)
    include_vis = bool(args.include_vis)
    include_LETSCLike = bool(args.include_LETSCLike)
    include_user_text = bool(args.include_user_text)

    resume = bool(args.resume)

    # ------------------------------------------------
    # 3) Embed train + test (ALL layers, last token)
    #    Incremental write + resume per split
    # ------------------------------------------------
    embeddings: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}

    for split_name, split in (("train", train), ("test", test)):
        n_rows = len(split)
        num_batches = (n_rows + args.batch_size - 1) // args.batch_size

        partial_dir, meta_path = _split_paths(out_dir, split_name)

        # Resume metadata (completed_up_to is exclusive index)
        completed_up_to = 0
        layer_info: Dict[str, Dict[str, Any]] = {}
        if resume and meta_path.exists():
            meta = _load_json(meta_path)

            # basic compatibility checks
            if meta.get("dataset") != args.dataset or meta.get("model") != args.model:
                raise ValueError(
                    f"Found meta for different run in {meta_path}:\n"
                    f"  meta.dataset={meta.get('dataset')} vs args.dataset={args.dataset}\n"
                    f"  meta.model={meta.get('model')} vs args.model={args.model}\n"
                    f"Delete {meta_path.parent} if you want to restart."
                )
            if meta.get("n_rows") != n_rows:
                raise ValueError(
                    f"Split size changed for {split_name}: meta.n_rows={meta.get('n_rows')} "
                    f"vs current n_rows={n_rows}. Delete partial dir to restart."
                )

            completed_up_to = int(meta.get("completed_up_to", 0))
            layer_info = dict(meta.get("layers", {}))
            logger.info(
                f"🔁 Resuming split={split_name} from completed_up_to={completed_up_to}/{n_rows} "
                f"using {meta_path}"
            )

        if completed_up_to >= n_rows:
            logger.info(f"✅ split={split_name} already complete ({n_rows}/{n_rows}); skipping.")
            # load memmaps for final save
            # (if user resumed and wants final .npz without recomputing)
            for lk, info in layer_info.items():
                layer_path = Path(info["path"])
                embeddings[split_name][lk] = np.load(layer_path, mmap_mode="r")  # type: ignore[assignment]
            continue

        # Map from layer_key -> memmap array
        layer_memmaps: Dict[str, np.memmap] = {}

        # If resuming, open existing memmaps so we can keep writing
        if resume and layer_info:
            for lk, info in layer_info.items():
                layer_path = Path(info["path"])
                d = int(info["dim"])
                dtype = np.dtype(info["dtype"])
                layer_memmaps[lk] = _open_or_create_memmap(
                    layer_path=layer_path,
                    n_rows=n_rows,
                    d=d,
                    dtype=dtype,
                    resume=True,
                )

        # Main batching loop, starting at completed_up_to
        for start_idx in tqdm(
            range(completed_up_to, n_rows, args.batch_size),
            desc=(
                f"Embedding {split_name} | {num_batches} batches | {n_rows} rows "
                f"| resume@{completed_up_to}"
            ),
        ):
            end_idx = min(start_idx + args.batch_size, n_rows)
            batch_rows = split[start_idx:end_idx]

            # Build VisPrompt objects for this batch
            query_prompts = build_classification_query_prompts(
                batch_rows=batch_rows,
                dataset=args.dataset,
                model=args.model,
                include_user_text=include_user_text,
                include_ts=include_ts,
                include_LETSCLike=include_LETSCLike,
                include_vis=include_vis,
                CoT=args.CoT,
            )

            # dict[layer_key] = tensor (B, D)
            all_layer_embs = prompter.get_all_layer_embeddings(query_prompts, batch=True)

            # On first encounter of a layer_key, create its memmap file
            for layer_key, emb_tensor in all_layer_embs.items():
                emb_np = emb_tensor.detach().cpu().numpy()
                b, d = emb_np.shape
                if b != (end_idx - start_idx):
                    raise ValueError(
                        f"Batch size mismatch for layer={layer_key}: "
                        f"got {b}, expected {end_idx - start_idx}"
                    )

                if layer_key not in layer_memmaps:
                    safe_lk = _sanitize_layer_key(layer_key)
                    layer_path = partial_dir / f"{safe_lk}.npy"
                    dtype = emb_np.dtype

                    layer_memmaps[layer_key] = _open_or_create_memmap(
                        layer_path=layer_path,
                        n_rows=n_rows,
                        d=d,
                        dtype=dtype,
                        resume=resume,
                    )

                    # record meta for this layer
                    layer_info[layer_key] = {
                        "path": str(layer_path),
                        "dim": int(d),
                        "dtype": str(np.dtype(dtype)),
                    }

                # Write into the correct slice for this batch
                layer_memmaps[layer_key][start_idx:end_idx, :] = emb_np

            # Flush memmaps and update meta after each batch (save-as-you-go)
            for mm in layer_memmaps.values():
                mm.flush()

            completed_up_to = end_idx
            meta = {
                "dataset": args.dataset,
                "model": args.model,
                "embedding_types": args.embedding_types,
                "CoT": int(args.CoT),
                "n_rows": int(n_rows),
                "batch_size": int(args.batch_size),
                "completed_up_to": int(completed_up_to),
                "layers": layer_info,
            }
            _atomic_write_json(meta_path, meta)

        # After finishing split, expose memmaps for final save
        for lk, info in layer_info.items():
            layer_path = Path(info["path"])
            embeddings[split_name][lk] = np.load(layer_path, mmap_mode="r")  # type: ignore[assignment]

        # Log summary
        num_layers = len(embeddings[split_name])
        sample_shape = next(iter(embeddings[split_name].values())).shape if num_layers > 0 else (0, 0)
        logger.info(
            f"✅ Finished embedding split={split_name}, num_layers={num_layers}, sample_shape={sample_shape}. "
            f"Partial cache at {partial_dir}"
        )

    # ------------------------------------------------
    # 4) Save embeddings (per-layer .npz format)
    # ------------------------------------------------
    save_embeddings(
        train_embed=embeddings["train"],  # dict[layer_key, np.ndarray or memmap]
        test_embed=embeddings["test"],    # dict[layer_key, np.ndarray or memmap]
        save_path=out_dir,
    )
    logger.info(
        f"✅ All embeddings saved to {out_dir}/ "
        f"(train_embeddings.npz, test_embeddings.npz with {len(embeddings['train'])} layers)"
    )
