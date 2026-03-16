import argparse
import os
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.file_io import append_jsonl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "mistral", "random_qwen", "random_llama", "random_mistral"])
    parser.add_argument("--embedding_type", type=str, choices=["d", "v", "dv"], required=True)
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--embedding_type", type=str, required=True, choices=["d", "v", "dv"])
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    # 1. Pathing
    embed_root = f"/raid/hdd249/forecast_embeddings/{args.model}/{args.embedding_type}"
    out_root = f"./data/forecasting/regression_results/{args.model}/{args.embedding_type}"
    os.makedirs(out_root, exist_ok=True)

    # 2. Load Train Data for CV
    X_train = np.load(os.path.join(embed_root, "last_layer_train.npy"))
    y_train = np.load(os.path.join(embed_root, "targets_train.npy")).squeeze()

    # 3. Load Test Data for Inference
    X_test = np.load(os.path.join(embed_root, "last_layer_test.npy"))
    y_test = np.load(os.path.join(embed_root, "targets_test.npy")).squeeze()

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 4. GridSearch on Train Set
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    
    best_ridge = grid_search.best_estimator_
    
    # 5. Inference on Test Set
    y_pred = best_ridge.predict(X_test)

    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # 6. Save results
    summary = {
        "model": args.model,
        "embedding_type": args.embedding_type,
        "best_alpha": grid_search.best_params_["alpha"],
        "mae": round(float(mae), 5),
        "mse": round(float(mse), 5)
    }
    
    append_jsonl(f"{out_root}/eval_summary.jsonl", summary)
    
    # Save raw predictions for later analysis
    np.save(f"{out_root}/test_predictions.npy", y_pred)

    print(f"✅ Done. MAE: {mae:.4f} | MSE: {mse:.4f}")

if __name__ == "__main__":
    main()