# '''
# python ./src/bootstrap_f1.py \
#     --input_file /raid/hdd249/data/sample_generations/llama/ctu/logistic_regression/vis-ust_0-shot_Direct/layer40.jsonl \
#     --output_tsv ./results/bootstrap_f1_results.tsv 
# '''

# import argparse
# import json
# import numpy as np
# import os
# import csv
# from sklearn.metrics import f1_score

# # --- Helper Functions ---

# LAYER_MAP = {
#     "llama": 40, # but if d, then 32
#     "mistral": 40,
#     "random_llama": 40,
#     "mistral_random": 40
# }

# DATASETS = ["ctu", "emg", "tee", "har", "had", "rwc"]

# EMBEDDING_STRS = ["lets-ust_0-shot_Direct", "vis-lets-ust_0-shot_Direct", "vis-ust_0-shot_Direct"]

# def load_data(filepath):
#     """Loads JSONL data. Replaces utils.file_io to make script standalone."""
#     data = []
#     try:
#         with open(filepath, 'r') as f:
#             for line in f:
#                 if line.strip():
#                     data.append(json.loads(line))
#     except FileNotFoundError:
#         print(f"Error: File {filepath} not found.")
#         exit(1)
#     return data

# def determine_model_from_filename(filename):
#     """Determines the model type from the filename."""
#     # Normalize path to just the filename for safer checking
#     base_name = os.path.basename(filename)
    
#     if "random" in filename:
#         if "llama" in filename: return "llama_random"
#         if "qwen" in filename: return "qwen_random"
#         if "mistral" in filename: return "mistral_random"
#     else:
#         if "llama" in filename: return "llama"
#         if "qwen" in filename: return "qwen"
#         if "mistral" in filename: return "mistral"
    
#     # Return 'Unknown' instead of crashing if you want the script to be more robust
#     return "Unknown"

# def determine_dataset_name_from_filename(filename):
#     if "ctu" in filename: return "ctu"
#     if "emg" in filename: return "emg"
#     if "tee" in filename: return "tee"
#     if "har" in filename: return "har"
#     if "had" in filename: return "had"
#     if "rwc" in filename: return "rwc"
#     return "Unknown"

# def determine_modality_from_filename(filename):
#     # FIXED LOGIC: "if 'a' and 'b' in x" is effectively "if 'b' in x".
#     # It must be explicit: "if 'a' in x and 'b' in x"
#     if "lets" in filename and "vis" in filename:
#         return "dv"
#     elif "lets" in filename:
#         return "d"
#     elif "v" in filename: # Assuming 'v' stands for visual/video only if 'lets' isn't present
#         return "v"
#     return "Unknown"

# # --- Bootstrap Logic ---

# def bootstrap_metric(y_true, y_pred, n_bootstraps=10000, ci=95):
#     """Bootstraps the F1 score."""
#     bootstrapped_scores = []
#     n_samples = len(y_true)
    
#     np.random.seed(42)

#     for _ in range(n_bootstraps):
#         # Resample indices with replacement
#         indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        
#         virtual_gt = y_true[indices]
#         virtual_pred = y_pred[indices]
        
#         # zero_division=0 handles samples with no positive predictions
#         score = f1_score(virtual_gt, virtual_pred, average='binary', zero_division=0.0)
#         bootstrapped_scores.append(score)
    
#     alpha = (100 - ci) / 2.0
#     lower_bound = np.percentile(bootstrapped_scores, alpha)
#     upper_bound = np.percentile(bootstrapped_scores, 100 - alpha)
#     mean_score = np.mean(bootstrapped_scores)
    
#     return mean_score, lower_bound, upper_bound

# def save_to_tsv(output_file, data_dict):
#     """Appends results to a TSV file."""
#     file_exists = os.path.isfile(output_file)
    
#     # FIXED: Added 'Model', 'Dataset', 'Modality' to match your data_dict keys
#     fieldnames = [
#         'Model', 'Dataset', 'Modality', 
#         'Metric', 'Mean', 'CI_Lower', 'CI_Upper', 'CI_Level',
#         'Sample_Size', 'Input_File'
#     ]
    
#     with open(output_file, 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        
#         # Write header if file is new
#         if not file_exists:
#             writer.writeheader()
        
#         # If the file exists but has OLD headers, this might append strictly. 
#         # Ideally, delete the old TSV before running if columns changed.
#         writer.writerow(data_dict)
    
#     print(f"Results appended to {output_file}")

# # --- Main ---



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Bootstrap F1 Score from JSONL data.")
#     parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL.")
#     parser.add_argument("--output_tsv", type=str, default="bootstrap_results.tsv", help="Path to output TSV.")
#     parser.add_argument("--n_bootstraps", type=int, default=10000, help="Number of bootstrap iterations.")
#     parser.add_argument("--ci", type=float, default=95.0, help="Confidence Interval level.")

#     args = parser.parse_args()

#     # 1. Parse Metadata
#     model = determine_model_from_filename(args.input_file)
#     dataset = determine_dataset_name_from_filename(args.input_file)
#     modality = determine_modality_from_filename(args.input_file)

#     print(f"Processing: Model={model}, Dataset={dataset}, Modality={modality}")

#     # 2. Load Data
#     data = load_data(args.input_file)
#     y_true = np.array([item['gt'] for item in data])
#     y_pred = np.array([item['pred'] for item in data])
    
#     if len(y_true) == 0:
#         print("Error: No data found in file.")
#         exit(1)

#     # 3. Bootstrap
#     print(f"Bootstrapping F1 ({args.n_bootstraps} iter) on N={len(y_true)}...")
#     mean, lower, upper = bootstrap_metric(y_true, y_pred, args.n_bootstraps, args.ci)

#     # 4. Print
#     print("-" * 30)
#     print(f"F1 Mean: {mean:.4f}")
#     print(f"{args.ci}% CI: [{lower:.4f}, {upper:.4f}]")
#     print("-" * 30)

#     # 5. Save
#     result_row = {
#         'Model': model,
#         'Dataset': dataset,
#         'Modality': modality,
#         'Metric': 'F1',
#         'Mean': f"{mean:.4f}",
#         'CI_Lower': f"{lower:.4f}",
#         'CI_Upper': f"{upper:.4f}",
#         'CI_Level': args.ci,
#         'Sample_Size': len(y_true),
#         'Input_File': args.input_file
#     }
    
#     save_to_tsv(args.output_tsv, result_row)


import argparse
import json
import numpy as np
import os
import csv
from sklearn.metrics import f1_score

# --- Configuration ---

MODELS = [
    # "llama", 
    "mistral", 
    # "random_llama",
    "random_mistral", 
    # "qwen", 
    # "random_qwen"
    ]
DATASETS = [
    "ctu", 
    "emg", 
    "tee", 
    "har", 
    "had", 
    "rwc"
    ]
# Mapping embedding strings to modality shorthand
EMBEDDING_MAP = {
    "lets-ust_0-shot_Direct": "d",
    # "vis-ust_0-shot_Direct": "v",
    # "vis-lets-ust_0-shot_Direct": "dv"
}

def get_layer(model_name, modality):
    """
    Logic: llama/mistral use layer 40, 
    but if modality is 'd' (lets-only), use layer 32.
    """
    if "qwen" in model_name:
        return "64"
    if modality == "d" and model_name in ["llama", "random_llama"]:
        return "32"
    return "40"

# --- Helper Functions ---

def load_data(filepath):
    data = []
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def bootstrap_metric(y_true, y_pred, n_bootstraps=10000, ci=95):
    bootstrapped_scores = []
    n_samples = len(y_true)
    np.random.seed(42)

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        score = f1_score(y_true[indices], y_pred[indices], average='macro', zero_division=0.0)
        bootstrapped_scores.append(score)
    
    alpha = (100 - ci) / 2.0
    return np.mean(bootstrapped_scores), np.percentile(bootstrapped_scores, alpha), np.percentile(bootstrapped_scores, 100 - alpha)

def save_to_tsv(output_file, data_dict):
    file_exists = os.path.isfile(output_file)
    fieldnames = ['Model', 'Dataset', 'Modality', 'Metric', 'Mean', 'CI_Lower', 'CI_Upper', 'CI_Level', 'Sample_Size', 'Input_File']
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Bootstrap F1 Score.")
    parser.add_argument("--base_dir", type=str, default="/raid/hdd249/data/sample_generations", help="Root data dir")
    parser.add_argument("--output_tsv", type=str, default="./results/bootstrap_f1_results.tsv")
    parser.add_argument("--n_bootstraps", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)

    for model in MODELS:
        # Handle directory naming (e.g. folder might be 'llama' even if model is 'llama_random')
        folder_model = model.replace("_random", "")
        
        for dataset in DATASETS:
            for emb_str, modality in EMBEDDING_MAP.items():
                
                layer = get_layer(model, modality)
                
                # Construct Path
                # Pattern: {base}/{model_folder}/{dataset}/logistic_regression/{emb_str}/layer{num}.jsonl
                file_path = os.path.join(
                    args.base_dir, folder_model, dataset, 
                    "logistic_regression", emb_str, f"layer{layer}.jsonl"
                )

                print(f"Checking: {model} | {dataset} | {modality} -> {os.path.basename(file_path)}")
                
                data = load_data(file_path)
                if data is None or len(data) == 0:
                    print(f"  [!] Skipping: File not found or empty.")
                    continue

                y_true = np.array([item['gt'] for item in data])
                y_pred = np.array([item['pred'] for item in data])

                mean, lower, upper = bootstrap_metric(y_true, y_pred, args.n_bootstraps)

                result_row = {
                    'Model': model,
                    'Dataset': dataset,
                    'Modality': modality,
                    'Metric': 'F1',
                    'Mean': f"{mean:.4f}",
                    'CI_Lower': f"{lower:.4f}",
                    'CI_Upper': f"{upper:.4f}",
                    'CI_Level': 95,
                    'Sample_Size': len(y_true),
                    'Input_File': file_path
                }
                
                save_to_tsv(args.output_tsv, result_row)

    print(f"\nDone! All results saved to {args.output_tsv}")