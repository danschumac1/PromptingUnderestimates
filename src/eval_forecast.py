'''
How to run:
   python ./src/eval_forecast.py \
        --model llama  \
        --modality dv \
        --method prompting
'''

import argparse
import os
from utils.file_io import append_tsv, append_jsonl, load_jsonl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecasting models")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--modality", type=str, required=True, help="Data modality")
    parser.add_argument("--method", type=str, required=True, choices=['prompting', 'linear_regression'])
    parser.add_argument("--layer", type=int, help="Layer number")
    return parser.parse_args()

def find_data_path(model, modality, method, layer=None):
    # Includes both the correct spelling and your previous typo just in case!
    base_dirs = [
        # "/raid/hdd249/Classification_v2/data/forecasting/generation",
        # "/raid/hdd249/Classifcation_v2/data/forecasting/generation"
        "./data/forecasting/generation"
    ]
    
    if method == "prompting":
        print("PROMPTONG")
        # data/forecasting/generation/qwen/test_results_d.jsonl
        # Check all possible filename variations you might have used
        filenames = [
            # f"{modality}.jsonl",
            f"test_results_{modality}.jsonl",
            # f"results_{modality}.jsonl",
            # f"_{modality}.jsonl"
        ]
        for b_dir in base_dirs:
            for fname in filenames:
                path = os.path.join(b_dir, model, fname)
                if os.path.exists(path):
                    return path
                    
    elif method == "linear_regression":
        filenames = [f"layer{layer}.jsonl"]
        for b_dir in base_dirs:
            for fname in filenames:
                path = os.path.join(b_dir, model, modality, fname)
                if os.path.exists(path):
                    return path
                    
    return None

def main():
    args = parse_args()
    
    filepath = find_data_path(args.model, args.modality, args.method, args.layer)
    
    if not filepath:
        print(f"FILEPATH: {filepath}")
        print(f"Error: Could not find data for model '{args.model}', modality '{args.modality}', method '{args.method}'.")
        return
        
    gen_data = load_jsonl(filepath)
    print(f"Successfully loaded generation data from: {filepath}")

    valid_rows = []
    for row in gen_data:
        # Dynamically grab the keys whether they are named 'gt' or 'ground_truth'
        gt = row.get('ground_truth', row.get('gt'))
        pred = row.get('forecast', row.get('pred'))
        
        # Only keep rows where predictions and ground truth are valid lists of the same length
        if isinstance(gt, list) and isinstance(pred, list) and len(gt) == len(pred):
            valid_rows.append((gt, pred))

    print(f"Number of entries: {len(gen_data)}")
    print(f"Number of valid entries without errors: {len(valid_rows)}")

    if len(valid_rows) == 0:
        print("Error: No valid rows found to evaluate. Check your JSONL keys.")
        return

    avg_mse = 0
    avg_mae = 0
    for gt, pred in valid_rows:
        mse = sum((g - p) ** 2 for g, p in zip(gt, pred)) / len(gt)
        mae = sum(abs(g - p) for g, p in zip(gt, pred)) / len(gt)
        avg_mse += mse
        avg_mae += mae
        
    avg_mse /= len(valid_rows)
    avg_mae /= len(valid_rows)
    
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    line = {
        "model": args.model,
        "modality": args.modality,
        "method": args.method,
        "layer": args.layer if args.method == "linear_regression" else None, # Fixed logic bug here!
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "num_entries": len(gen_data),
        "num_non_errors": len(valid_rows),
        "num_errors": len(gen_data) - len(valid_rows),
    }

    append_tsv("./data/results/forecast_results.tsv", line)
    print(f"Appended results to ./data/results/forecast_results.jsonl: {line}")

if __name__ == "__main__":
    main()

