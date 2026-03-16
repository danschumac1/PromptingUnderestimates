import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Required local imports
from utils.prompters import QwenVisionPrompter, LlamaVisionPrompter, MistralVisionPrompter
from utils.prompt_objects import QwenVisPrompt, LlamaVisPrompt, MistralVisPrompt
from utils.file_io import save_embeddings
from utils.forecast_utils import create_univariate_windows, build_forecast_prompt

# --- Configuration Maps ---
PROMPTER_MAP = {"qwen": QwenVisionPrompter, "llama": LlamaVisionPrompter, "mistral": MistralVisionPrompter}
PROMPT_MAP = {"qwen": QwenVisPrompt, "llama": LlamaVisPrompt, "mistral": MistralVisPrompt}

# --- Utilities from Classification Script ---
def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def _sanitize_layer_key(layer_key: str) -> str:
    return layer_key.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")

def _open_or_create_memmap(layer_path: Path, n_rows: int, d: int, dtype: np.dtype, resume: bool) -> np.memmap:
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and layer_path.exists():
        arr = np.load(layer_path, mmap_mode="r+")
        if arr.shape != (n_rows, d):
            raise ValueError(f"Shape mismatch for {layer_path}: {arr.shape} vs {(n_rows, d)}")
        return arr 
    return np.lib.format.open_memmap(layer_path, mode="w+", dtype=dtype, shape=(n_rows, d))

def _split_paths(out_dir: str | Path, split_name: str) -> Tuple[Path, Path]:
    partial_dir = Path(out_dir) / "partial_embeddings" / split_name
    meta_path = partial_dir / "meta.json"
    return partial_dir, meta_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ForecastEmbed")

    out_dir = Path(f"/raid/hdd249/Classification_v2/data/forecasting/embeddings/{args.model}/{args.embedding_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Prep
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    X, y = create_univariate_windows(ot_scaled, args.lookback, args.horizon)
    data_rows = [{"idx": i, "X": X[i], "y": y[i]} for i in range(len(X))]
    
    # Split: 70% Train, 30% Test
    split_idx = int(len(data_rows) * 0.7)
    splits = {
        "train": data_rows[:split_idx],
        "test": data_rows[split_idx:]
    }

    # 2. Setup Prompter
    prompter = PROMPTER_MAP[args.model]() 
    prompter.system_prompt = "You are an expert in energy grid analytics."

    embeddings: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}

    # 3. Embedding Loop
    for split_name, split_data in splits.items():
        n_rows = len(split_data)
        partial_dir, meta_path = _split_paths(out_dir, split_name)
        
        completed_up_to = 0
        layer_info = {}
        resume = bool(args.resume)

        if resume and meta_path.exists():
            meta = _load_json(meta_path)
            completed_up_to = int(meta.get("completed_up_to", 0))
            layer_info = meta.get("layers", {})
            logger.info(f"🔁 Resuming {split_name} at {completed_up_to}/{n_rows}")

        if completed_up_to >= n_rows:
            for lk, info in layer_info.items():
                embeddings[split_name][lk] = np.load(Path(info["path"]), mmap_mode="r")
            continue

        layer_memmaps: Dict[str, np.memmap] = {}
        if resume and layer_info:
            for lk, info in layer_info.items():
                layer_memmaps[lk] = _open_or_create_memmap(Path(info["path"]), n_rows, info["dim"], np.dtype(info["dtype"]), True)

        for start_idx in tqdm(range(completed_up_to, n_rows, args.batch_size), desc=f"Embedding {split_name}"):
            end_idx = min(start_idx + args.batch_size, n_rows)
            batch = split_data[start_idx:end_idx]
            
            query_prompts = [
                build_forecast_prompt(r, scaler, args.lookback, args.horizon, args.embedding_type, split_name, PROMPT_MAP[args.model]) 
                for r in batch
            ]

            all_layer_embs = prompter.get_all_layer_embeddings(query_prompts, batch=True)

            for layer_key, emb_tensor in all_layer_embs.items():
                emb_np = emb_tensor.detach().cpu().numpy()
                b, d = emb_np.shape

                if layer_key not in layer_memmaps:
                    safe_lk = _sanitize_layer_key(layer_key)
                    layer_path = partial_dir / f"{safe_lk}.npy"
                    layer_memmaps[layer_key] = _open_or_create_memmap(layer_path, n_rows, d, emb_np.dtype, resume)
                    layer_info[layer_key] = {"path": str(layer_path), "dim": d, "dtype": str(emb_np.dtype)}

                layer_memmaps[layer_key][start_idx:end_idx, :] = emb_np

            for mm in layer_memmaps.values():
                mm.flush()

            meta = {
                "model": args.model, "embedding_type": args.embedding_type,
                "n_rows": n_rows, "completed_up_to": end_idx, "layers": layer_info
            }
            _atomic_write_json(meta_path, meta)

        for lk, info in layer_info.items():
            embeddings[split_name][lk] = np.load(Path(info["path"]), mmap_mode="r")

    # 4. Final Save
    save_embeddings(train_embed=embeddings["train"], test_embed=embeddings["test"], save_path=str(out_dir))
    logger.info(f"✅ Success. Embeddings saved to {out_dir}")

if __name__ == "__main__":
    main()