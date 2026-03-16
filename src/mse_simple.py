from utils.file_io import load_jsonl


'''
2026-02-18
How to run:
   python ./src/mse_simple.py
'''

def main():
    input_path = "data/forecasting/generation/llama/test_results_v.jsonl"
    data = load_jsonl(input_path)
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    for item in data:
        preds = item["pred"]
        targets = item["gt"]
        for pred, target in zip(preds, targets):
            mse_sum += (pred - target) ** 2
            mae_sum += abs(pred - target)
            count += 1

    mse = mse_sum / count if count > 0 else float("nan")
    mae = mae_sum / count if count > 0 else float("nan")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()