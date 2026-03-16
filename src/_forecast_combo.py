import argparse
import json
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Local imports
from utils.prompters import QwenVisionPrompter, LlamaVisionPrompter, MistralVisionPrompter
from utils.prompt_objects import QwenVisPrompt, LlamaVisPrompt, MistralVisPrompt
from utils.file_io import append_jsonl
from utils.forecast_utils import create_univariate_windows, build_forecast_prompt

# --- Configuration Maps ---
PROMPTER_MAP = {"qwen": QwenVisionPrompter, "llama": LlamaVisionPrompter, "mistral": MistralVisionPrompter}
PROMPT_MAP = {"qwen": QwenVisPrompt, "llama": LlamaVisPrompt, "mistral": MistralVisPrompt}


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def _sanitize_layer_key(layer_key: str) -> str:
    return layer_key.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")

def _open_or_create_memmap(path: Path, shape: Tuple, dtype: np.dtype, resume: bool) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists():
        return np.load(path, mmap_mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    # Logging & Paths
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("UnifiedForecast")
    
    # Unified output directory for embeddings and inference logs
    emb_out_dir = Path(f"/raid/hdd249/forecast_embeddings/{args.model}/autoregressive_{args.embedding_type}")
    gen_out_dir = Path(f"data/forecasting/generation/{args.model}")
    emb_out_dir.mkdir(parents=True, exist_ok=True)
    gen_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Prep
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    
    X, y = create_univariate_windows(ot_scaled, args.lookback, args.horizon)
    data_rows = [{"idx": i, "X": X[i], "y": y[i]} for i in range(len(X))]
    
    # 100-sample evaluation slice
    test_start_idx = int(len(data_rows) * 0.7)
    test_data = data_rows[test_start_idx : test_start_idx + 100]

    # 2. Setup Prompter
    prompter = PROMPTER_MAP[args.model]() 
    prompter.system_prompt = (
        "You are an expert in energy grid analytics specializing in transformer Oil Temperature (OT) forecasting.\n\n"
        "### TASK\nForecast the next hour of OT values (a single value).\n"
        "### CONSTRAINTS\n- Your output should strictly follow the format: 'I predict the next hour will be <value>'"
    )

    # 3. Resume & Memmap Initialization
    n_rows = len(test_data)
    n_total_steps = n_rows * args.horizon
    meta_path = emb_out_dir / "meta.json"
    completed_up_to_row = 0
    layer_info = {}
    
    if args.resume and meta_path.exists():
        meta = _load_json(meta_path)
        completed_up_to_row = meta.get("completed_up_to_row", 0)
        layer_info = meta.get("layers", {})

    layer_memmaps: Dict[str, np.memmap] = {}
    target_path = emb_out_dir / "targets_auto.npy"
    targets_mm = _open_or_create_memmap(target_path, (n_total_steps,), np.float32, args.resume)

    # Metric tracking for inference
    all_gt, all_preds = [], []

    # 4. Unified Main Loop
    # Calculate total steps for a smoother tqdm bar
    total_steps = n_rows * args.horizon
    initial_step = completed_up_to_row * args.horizon
    
    pbar = tqdm(total=total_steps, initial=initial_step, desc="Forecast Combo")

    for r_idx in range(completed_up_to_row, n_rows):
        row = test_data[r_idx]
        current_history_scaled = row['X'].flatten().tolist()
        predictions_unscaled = []
        raw_responses = []

        # Autoregressive steps
        for step in range(args.horizon):
            # Update description to show current row/step progress
            pbar.set_description(f"Row {r_idx+1}/{n_rows} | Step {step+1}/{args.horizon}")
            
            prompt_obj = build_forecast_prompt(
                row_idx=int(row['idx']),
                current_X_scaled=current_history_scaled,
                scaler=scaler,
                L=args.lookback,
                H=1, 
                embedding_type=args.embedding_type,
                split_name=f"auto_unified_{args.embedding_type}",
                PromptClass=PROMPT_MAP[args.model]
            )

            # --- THE SINGLE FORWARD PASS ---
            response, all_layer_embs = prompter.get_completions_and_embeddings([prompt_obj], batch=False)
            
            # Save Embedding for current step
            flat_idx = (r_idx * args.horizon) + step
            
            for layer_key, emb_tensor in all_layer_embs.items():
                emb_np = emb_tensor.detach().cpu().numpy().flatten()
                
                if layer_key not in layer_memmaps:
                    safe_lk = _sanitize_layer_key(layer_key)
                    l_path = emb_out_dir / f"{safe_lk}.npy"
                    layer_memmaps[layer_key] = _open_or_create_memmap(l_path, (n_total_steps, emb_np.shape[0]), emb_np.dtype, args.resume)
                    layer_info[layer_key] = {"path": str(l_path), "dim": emb_np.shape[0], "dtype": str(emb_np.dtype)}

                layer_memmaps[layer_key][flat_idx, :] = emb_np

            targets_mm[flat_idx] = row['y'][step].item()

            # Autoregressive logic
            raw_responses.append(response)
            val = extract_prediction(response)
            if val is not None:
                predictions_unscaled.append(val)
                val_scaled = scaler.transform([[val]]).flatten()[0].item()
                current_history_scaled.append(val_scaled)
            else:
                last_scaled = current_history_scaled[-1]
                current_history_scaled.append(last_scaled)
                predictions_unscaled.append(scaler.inverse_transform([[last_scaled]]).flatten()[0].item())
            
            # Move the bar forward by 1 step
            pbar.update(1)

        # 5. Tracking and Saving Inference Results
        y_gt_unscaled = scaler.inverse_transform(row['y'].reshape(-1, 1)).flatten().tolist()
        all_gt.extend(y_gt_unscaled)
        all_preds.extend(predictions_unscaled)

        result_line = {
            "idx": int(row['idx']),
            "ground_truth": [round(x, 3) for x in y_gt_unscaled],
            "forecast": [round(x, 3) for x in predictions_unscaled],
            "raw_responses": raw_responses
        }
        append_jsonl(gen_out_dir / f"autoregressive_{args.embedding_type}.jsonl", result_line)

        # Meta Update & Stats Display
        if (r_idx + 1) % 5 == 0:
            for mm in layer_memmaps.values(): mm.flush()
            targets_mm.flush()
            _atomic_write_json(meta_path, {"completed_up_to_row": r_idx + 1, "layers": layer_info, "horizon": args.horizon})
            
            cur_mae = mean_absolute_error(all_gt, all_preds)
            # Update the stats on the right side of the bar
            pbar.set_postfix({"MAE": f"{cur_mae:.4f}"})

    pbar.close()
    # Final Summary Output
    final_mae = mean_absolute_error(all_gt, all_preds)
    append_jsonl(gen_out_dir / "eval_summary.jsonl", {"model": args.model, "embedding_type": args.embedding_type, "mae": final_mae})
    logger.info(f"✅ Finished. Final MAE: {final_mae:.5f}")

if __name__ == "__main__":
    main()