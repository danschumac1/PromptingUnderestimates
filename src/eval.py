'''
How to run:
   see ./bin/eval.sh
'''

import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from utils.file_io import append_row, ensure_header, load_jsonl

def parse_args():
    p = argparse.ArgumentParser(description="Append evaluation metrics to a TSV leaderboard.")
    p.add_argument("--pred_path", type=str, required=True,
                   help="Path to JSONL predictions with fields {pred, gt} per line.")
    p.add_argument("--results_path", type=str, default="./data/raw_results.tsv",
                   help="TSV to append results into (header auto-written).")
    return p.parse_args()


def main():
    args = parse_args()
    results_list = load_jsonl(args.pred_path)
    gts = [line["gt"] for line in results_list]
    preds = [line["pred"] for line in results_list]

    # Metrics
    acc = float(accuracy_score(gts, preds))
    f1m = float(f1_score(gts, preds, average="macro"))

    # Infer fields; allow CLI overrides

    # Results row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "accuracy",
        "macro_f1",
        "pred_path",
        "timestamp"
    ]
    print(f"Saving results to: {args.results_path}")
    ensure_header(args.results_path, header)
    row = [
        f"{acc:.6f}",
        f"{f1m:.6f}",
        args.pred_path,
        timestamp,
    ]
    append_row(args.results_path, row)


    # Console echo
    print(f"File                : {args.pred_path}")
    print(f"Acc                 : {acc:.4f} ({acc:.2%})")
    print(f"Macro F1            : {f1m:.4f} ({f1m:.2%})")


if __name__ == "__main__":
    main()
