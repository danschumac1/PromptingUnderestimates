import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from utils.random_prompter import RandomLlamaVisionPrompter, RandomQwenVisionPrompter, RandomQwenVisionPrompter, RandomMistralVisionPrompter
from utils.forecast_utils import create_univariate_windows, build_forecast_prompt
from utils.prompters import QwenVisionPrompter, LlamaVisionPrompter, MistralVisionPrompter
from utils.prompt_objects import QwenVisPrompt, LlamaVisPrompt, MistralVisPrompt

# ---- CONSTANTS ----
PROMPTER_MAP = {
    "qwen": QwenVisionPrompter, 
    "llama": LlamaVisionPrompter, 
    "mistral": MistralVisionPrompter,
    "random_qwen": RandomQwenVisionPrompter,
    "random_llama": RandomLlamaVisionPrompter,
    "random_mistral": RandomMistralVisionPrompter
    
}
PROMPT_MAP = {
    "qwen": QwenVisPrompt, 
    "llama": LlamaVisPrompt, 
    "mistral": MistralVisPrompt,
    "random_qwen": QwenVisPrompt,
    "random_llama": LlamaVisPrompt,
    "random_mistral": MistralVisPrompt
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(PROMPTER_MAP.keys()))
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--lookback", type=int, default=96)
    return parser.parse_args()

def main():
    args = parse_args()

    out_root = Path(f"/raid/hdd249/forecast_embeddings/{args.model}/{args.embedding_type}")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Data Prep
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    
    # We use horizon=6 for targets
    X, y = create_univariate_windows(ot_scaled, args.lookback, 6)
    
    # Define Split Logic
    train_limit = 1000
    test_start = int(len(X) * 0.7)
    test_limit = 100
    
    splits = {
        "train": (X[:train_limit], y[:train_limit]),
        "test": (X[test_start : test_start + test_limit], y[test_start : test_start + test_limit])
    }

    prompter = PROMPTER_MAP[args.model]() 
    prompter.system_prompt = "You are an expert in energy grid analytics."

    for split_name, (X_data, y_data) in splits.items():
        n_rows = len(X_data)
        # Save targets (scaled)
        np.save(out_root / f"targets_{split_name}.npy", y_data)
        
        layer_memmap = None
        
        for i in tqdm(range(n_rows), desc=f"Extracting {split_name} Embeddings"):
            current_X = X_data[i].flatten().tolist()
            
            prompt_obj = build_forecast_prompt(
                row_idx=i, current_X_scaled=current_X,
                scaler=scaler, L=args.lookback, H=1, 
                embedding_type=args.embedding_type,
                split_name=f"{split_name}_extract",
                PromptClass=PROMPT_MAP[args.model]
            )

            # Extract last layer hidden states
            all_layer_embs = prompter.get_all_layer_embeddings([prompt_obj], batch=False)
            last_key = list(all_layer_embs.keys())[-1]
            emb_np = all_layer_embs[last_key].detach().cpu().numpy().flatten()

            if layer_memmap is None:
                shape = (n_rows, emb_np.shape[0])
                layer_memmap = np.lib.format.open_memmap(
                    out_root / f"last_layer_{split_name}.npy", 
                    mode="w+", dtype=emb_np.dtype, shape=shape
                )

            layer_memmap[i, :] = emb_np
            
        layer_memmap.flush()

    print(f"✅ Embeddings and Targets saved to {out_root}")

if __name__ == "__main__":
    main()