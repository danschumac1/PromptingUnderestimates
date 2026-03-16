import argparse
import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# USER DEFINED FUNCTIONS
from utils.random_prompter import RandomQwenVisionPrompter, RandomLlamaVisionPrompter, RandomMistralVisionPrompter
from utils.file_io import append_jsonl
from utils.forecast_utils import create_univariate_windows, build_forecast_prompt
from utils.prompters import QwenVisionPrompter, LlamaVisionPrompter, MistralVisionPrompter
from utils.prompt_objects import QwenVisPrompt, LlamaVisPrompt, MistralVisPrompt

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

def extract_prediction(text):
    if not text: return None
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(PROMPTER_MAP.keys()))
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1) # Added resume flag
    args = parser.parse_args()

    gen_out_dir = Path(f"data/forecasting/generation/{args.model}")
    gen_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = gen_out_dir / f"test_results_{args.embedding_type}.jsonl"

    # 1. Data Prep
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    X, y = create_univariate_windows(ot_scaled, args.lookback, args.horizon)
    
    test_start = int(len(X) * 0.7)
    X_test, y_test = X[test_start : test_start+100], y[test_start : test_start+100]

    # --- RESUME LOGIC ---
    start_idx = 0
    all_gt, all_preds = [], []
    
    if args.resume and out_file.exists():
        with open(out_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                all_gt.extend(data['gt'])
                all_preds.extend(data['pred'])
                start_idx += 1
        print(f"🔁 Resuming from index {start_idx} (loaded {len(all_gt)} historical points)")

    prompter = PROMPTER_MAP[args.model]()
    prompter.system_prompt = "You are an expert in energy grid analytics. Forecast the next hour of OT values."

    # 2. Batched Autoregressive Loop
    n_test = len(X_test)
    # Adjust tqdm to reflect starting position
    pbar = tqdm(total=n_test, initial=start_idx, desc="Batched Inference")

    # Start loop from start_idx
    for i in range(start_idx, n_test, args.batch_size):
        batch_end = min(i + args.batch_size, n_test)
        actual_batch_size = batch_end - i
        
        batch_histories = [X_test[j].flatten().tolist() for j in range(i, batch_end)]
        batch_predictions = [[] for _ in range(actual_batch_size)]
        
        for step in range(args.horizon):
            pbar.set_description(f"Batch {i//args.batch_size + 1} | Step {step+1}/{args.horizon}")
            
            prompts = []
            for b_idx in range(actual_batch_size):
                prompt_obj = build_forecast_prompt(
                    row_idx=i + b_idx, 
                    current_X_scaled=batch_histories[b_idx],
                    scaler=scaler, L=args.lookback, H=1, 
                    embedding_type=args.embedding_type,
                    split_name="test_prompting_batched",
                    PromptClass=PROMPT_MAP[args.model]
                )
                prompts.append(prompt_obj)

            responses = prompter.get_completion(prompts, batch=True)
            
            for b_idx in range(actual_batch_size):
                res = responses[b_idx]
                val = extract_prediction(res)
                
                if val is not None:
                    batch_predictions[b_idx].append(val)
                    val_scaled = scaler.transform([[val]]).flatten()[0].item()
                    batch_histories[b_idx].append(val_scaled)
                else:
                    last_v = batch_histories[b_idx][-1]
                    batch_histories[b_idx].append(last_v)
                    batch_predictions[b_idx].append(scaler.inverse_transform([[last_v]]).flatten()[0].item())

        # 3. Save Results and Update Global Metrics
        for b_idx in range(actual_batch_size):
            global_idx = i + b_idx
            gt_unscaled = scaler.inverse_transform(y_test[global_idx].reshape(-1, 1)).flatten().tolist()
            
            all_gt.extend(gt_unscaled)
            all_preds.extend(batch_predictions[b_idx])
            
            append_jsonl(out_file, {
                "idx": global_idx, 
                "gt": gt_unscaled, 
                "pred": batch_predictions[b_idx]
            })

        running_mae = mean_absolute_error(all_gt, all_preds)
        pbar.set_postfix({"MAE": f"{running_mae:.4f}"})
        pbar.update(actual_batch_size)

    pbar.close()
    print(f"✅ Final Test MAE: {mean_absolute_error(all_gt, all_preds):.4f}")

if __name__ == "__main__":
    main()