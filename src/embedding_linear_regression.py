'''
How to run:
    python ./src/embedding_linear_regression.py \
     --model mistral \
     --embedding_type v
'''

import os
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Assuming these helpers exist in your environment
from logistic_regression import _layer_sort_key
from utils.forecast_utils import create_univariate_windows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for forecasting")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./data/forecasting/generation")
    return parser.parse_args()

def main():
    args = parse_args()

    ################################################################################################
    # 1. LOAD ORIGINAL DATA
    ################################################################################################
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    
    # Scale target for training
    scaler = StandardScaler()
    ot_scaled = scaler.fit_transform(df[['OT']].values)
    
    _, y = create_univariate_windows(ot_scaled, args.lookback, args.horizon)
    y = y.reshape(y.shape[0], -1) # Shape: (N, horizon)
    
    # Split: 70% Train, 30% Test
    split_idx = int(len(y) * 0.7)
    y_train = y[:split_idx]

    ################################################################################################
    # 2. PATHING & OUTPUT SETUP
    ################################################################################################
    base_path = f"/raid/hdd249/Classification_v2/data/forecasting/embeddings/{args.model}/{args.embedding_type}"
    train_npz_path = os.path.join(base_path, "train__embeddings.npz")
    test_npz_path = os.path.join(base_path, "test__embeddings.npz")

    if not os.path.exists(train_npz_path) or not os.path.exists(test_npz_path):
        raise FileNotFoundError(f"Embedding paths not found: {train_npz_path}")

    # Output root matches logistic_regression structure: {output_dir}/{model}/{embedding_type}/
    out_root = os.path.join(args.output_dir, args.model, args.embedding_type)
    os.makedirs(out_root, exist_ok=True)

    train_data = np.load(train_npz_path)
    test_data = np.load(test_npz_path)
    layer_keys = sorted(list(train_data.files), key=_layer_sort_key)

    ################################################################################################
    # 3. TRAIN/EVAL PER LAYER
    ################################################################################################
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    tscv = TimeSeriesSplit(n_splits=args.cv)

    for layer_key in tqdm(layer_keys, desc="GridSearchCV per layer"):
        train_emb = train_data[layer_key]
        test_emb = test_data[layer_key]

        # Feature Scaling (Crucial for Ridge stability)
        feat_scaler = StandardScaler()
        X_train_scaled = feat_scaler.fit_transform(train_emb)
        X_test_scaled = feat_scaler.transform(test_emb)

        # Multi-output Ridge
        grid_search = GridSearchCV(
            Ridge(),
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Predict & Inverse Scale to get back to original OT units
        preds_scaled = best_model.predict(X_test_scaled)
        preds_final = scaler.inverse_transform(preds_scaled)
        y_test_final = scaler.inverse_transform(y[split_idx:])

        # Save layer-specific file: e.g., layer31.jsonl
        pred_path = os.path.join(out_root, f"layer{layer_key}.jsonl")
        
        with open(pred_path, "w") as f:
            for i in range(len(preds_final)):
                line = {
                    "idx": split_idx + i,
                    "layer": layer_key,
                    "ground_truth": y_test_final[i].tolist(),
                    "forecast": preds_final[i].tolist(),
                    "status": "success"
                }
                f.write(json.dumps(line) + "\n")
        
    print(f"\n✅ Done. Results saved to: {out_root}")

if __name__ == "__main__":
    main()