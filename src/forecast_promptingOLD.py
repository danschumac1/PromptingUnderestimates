########################
# OLD VERSION OF FORECAST PROMPTING 
########################

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Required local imports
from utils.prompters import QwenVisionPrompter, LlamaVisionPrompter, MistralVisionPrompter
from utils.prompt_objects import QwenVisPrompt, LlamaVisPrompt, MistralVisPrompt
from utils.file_io import append_jsonl
from utils.forecast_utils import create_univariate_windows, build_forecast_prompt, ForecastOutput

# --- Configuration Maps ---
PROMPTER_MAP = {"qwen": QwenVisionPrompter, "llama": LlamaVisionPrompter, "mistral": MistralVisionPrompter}
PROMPT_MAP = {"qwen": QwenVisPrompt, "llama": LlamaVisPrompt, "mistral": MistralVisPrompt}
MAX_RETRIES = 2 # two REtries for a total of three tries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    # Dynamic output pathing
    out_dir = f"/raid/hdd249/Classification_v2/data/forecasting/generation/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.embedding_type}.jsonl")

    # Resume Logic
    completed_idxs = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file, 'r') as f:
            for line in f:
                try: completed_idxs.add(json.loads(line)['idx'])
                except: pass
    elif not args.resume and os.path.exists(out_file):
        os.remove(out_file)

    # 1. Data Prep
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    # truncate the dataset to be 100 rows.
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    X, y = create_univariate_windows(ot_scaled, args.lookback, args.horizon)
    data_rows = [{"idx": i, "X": X[i], "y": y[i]} for i in range(len(X))]
    # truncate
    test_data = data_rows[int(len(data_rows) * 0.7):]
    test_data = [r for r in test_data if r['idx'] not in completed_idxs]
    test_data = test_data[:100] # for quick testing; remove this line for full dataset

    # 2. Setup Prompter
    prompter = PROMPTER_MAP[args.model]() 
    prompter.system_prompt = (
        "You are an expert in energy grid analytics specializing in transformer Oil Temperature (OT) forecasting.\n\n"
        "### TASK\n"
        f"1. Analyze the provided lookback window ({args.lookback} hours) for trends, seasonality, and anomalies.\n"
        f"2. Forecast the next {args.horizon} hours of OT values.\n"
        f"3. Return your response ONLY as a valid JSON object with the keys 'analysis' and 'forecast'.\n\n"
        "### CONSTRAINTS\n"
        "- Do NOT include Python code, markdown blocks (like ```json), or conversational filler.\n"
        "- 'analysis': A concise string explaining the reasoning behind your forecast.\n"
        f"- 'forecast': A JSON list containing EXACTLY {args.horizon} floating-point numbers.\n\n"
        "### OUTPUT FORMAT\n"
        "{\n"
        '  "analysis": "Your trend analysis here...",\n'
        f'  "forecast": [val1, val2, ..., val{args.horizon}]\n'
        "}"
    )

    # 3. Main Loop
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Forecasting"):
        batch = test_data[i : i + args.batch_size]
        query_prompts = [
            build_forecast_prompt(r, scaler, args.lookback, args.horizon, args.embedding_type, "test", PROMPT_MAP[args.model]) 
            for r in batch
        ]

        # Initialize a list to hold the validated objects (None means we need to try/retry)
        validated_outputs = [None] * len(batch)
        # Track raw text just in case we fail all retries
        final_raw_texts = [None] * len(batch)

        for attempt in range(MAX_RETRIES + 1):
            # Determine which indices still need a successful result
            indices_to_fetch = [idx for idx, val in enumerate(validated_outputs) if val is None]
            
            if not indices_to_fetch:
                break
                
            if attempt > 0:
                print(f"⚠️ Retry {attempt}/{MAX_RETRIES} for {len(indices_to_fetch)} items (Schema/Length Validation Failed)")

            # Fetch completions only for missing items
            prompts_to_send = [query_prompts[idx] for idx in indices_to_fetch]
            new_results = prompter.get_completion(prompts_to_send, batch=(len(prompts_to_send) > 1))
            
            # Ensure new_results is a list even if batch=False returned a single string
            if not isinstance(new_results, list):
                new_results = [new_results]

            # Validate the new results
            for local_idx, raw_text in enumerate(new_results):
                global_idx = indices_to_fetch[local_idx]
                final_raw_texts[global_idx] = raw_text

                if not raw_text:
                    continue

                try:
                    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        # This validates the 'analysis' and 'forecast' keys exist via Pydantic
                        val = ForecastOutput(**data)
                        
                        # Manual check for horizon length
                        if len(val.forecast) == args.horizon:
                            validated_outputs[global_idx] = val
                        else:
                            print(f"❌ Length mismatch: Got {len(val.forecast)}, expected {args.horizon}")
                except Exception:
                    pass # Validation failed; validated_outputs[global_idx] remains None for next retry

        # 4. Save results
        for row, val_obj, raw_text in zip(batch, validated_outputs, final_raw_texts):
            y_gt = [round(x, 3) for x in scaler.inverse_transform(row['y'].reshape(-1, 1)).flatten()]
            line = {"idx": int(row['idx']), "ground_truth": y_gt, "raw_output": raw_text}

            if val_obj:
                line.update({
                    "analysis": val_obj.analysis,
                    "forecast": [round(x, 3) for x in val_obj.forecast],
                    "status": "success"
                })
            else:
                line["status"] = "failed"
                line["error"] = "Failed schema validation or horizon length after all retries"
            
            append_jsonl(out_file, line)

if __name__ == "__main__":
    main()