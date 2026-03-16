import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score  # Added for F1 calculation

from utils.constants import build_valid_embedding_strings
from utils.setup import setup, standard_args
from utils.build_prompts import (
    build_classification_query_prompts,
    build_classification_system_prompt,
)
from utils.file_io import append_jsonl
from utils.preprocessing import extract_letter_to_idx 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate self-consistency classification.")
    parser.add_argument(
        "--dataset",
        choices=["ctu", "emg", "had", "har", "rwc", "tee"],
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        choices=["llama", "mistral", "qwen", "random_llama", "random_mistral", "random_qwen"],
        type=str,
        required=True,
        help="Model to use for prompting",
    )
    parser.add_argument("--embedding_types", type=str, choices=["d","v","dv"], required=True)
    parser.add_argument("--n_voters", type=int, default=5, help="Number of times to prompt for self-consistency")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    include_vis = "v" in args.embedding_types
    include_lets = "d" in args.embedding_types

    _, _, test, logger, prompter = setup(
        script="zero_shot_sc",
        dataset=args.dataset,
        model=args.model,
        n_shots=0,
        include_ts=False,
        include_vis=include_vis,
        include_LETSCLike=include_lets,
        include_user_text=True,
        sample=1,
        CoT=0
    )
    
    master_out_file = f"./data/generations/{args.model}/{args.dataset}/{args.embedding_types}/self_consistency.jsonl"
    eval_out_file = "./data/results/self_consistency_results.jsonl"


    prompter.system_prompt = build_classification_system_prompt(args.dataset)

    # Metrics Tracking
    all_gt = []
    all_preds = []
    total = 0
    
    # limit test to 100 rows.
    test = test[:100]

    for row in tqdm(test, desc=f"SC Evaluation | {args.n_voters} Voters"):
        idx_scalar = int(np.asarray(row.idx).item())
        y_scalar = int(np.asarray(row.y).item())
        
        individual_attempts = []

        for v_idx in range(args.n_voters):
            query_prompts = build_classification_query_prompts(
                batch_rows=row, 
                dataset=args.dataset,
                model=args.model,
                include_ts=False,
                include_LETSCLike=include_lets,
                include_vis=include_vis,
                CoT=0
            )

            model_output = prompter.get_completion(query_prompts, batch=False)
            letter, pred = extract_letter_to_idx(
                model_output, test.label_maps["letter_to_id"]
            )
            
            individual_attempts.append({
                "round": v_idx,
                "pred": int(pred),
                "letter": letter,
                "raw": model_output
            })

        # --- Aggregate Votes ---
        vote_counts = Counter([attempt["pred"] for attempt in individual_attempts])
        master_pred = vote_counts.most_common(1)[0][0]
        
        # --- Update Metrics ---
        all_gt.append(y_scalar)
        all_preds.append(master_pred)
        total += 1
        
        is_correct = int(master_pred == y_scalar)

        # --- Save Record ---
        final_record = {
            "idx": idx_scalar,
            "gt": y_scalar,
            "master_pred": master_pred,
            "correct": is_correct,
            "vote_tally": dict(vote_counts),
            "individual_outputs": individual_attempts
        }
        append_jsonl(master_out_file, final_record)

        # Periodically log Accuracy & F1
        if total % 10 == 0:
            current_acc = np.mean(np.array(all_gt) == np.array(all_preds))
            current_f1 = f1_score(all_gt, all_preds, average='macro', zero_division=0)
            logger.info(f"Step {total} | Acc: {current_acc:.4f} | F1 (Macro): {current_f1:.4f}")

    # Final Summary
    final_acc = np.mean(np.array(all_gt) == np.array(all_preds))
    final_f1 = f1_score(all_gt, all_preds, average='macro', zero_division=0)
    
    eval_line = {
        "dataset": args.dataset,
        "model": args.model,
        "embedding_types": args.embedding_types,
        "n_voters": args.n_voters,
        "final_acc": final_acc,
        "final_f1_macro": final_f1
    }

    append_jsonl(eval_out_file, eval_line)

    logger.info(f"✅ Finished. Final Acc: {final_acc:.4f} | Final F1: {final_f1:.4f}")
    logger.info(f"Results saved to: {master_out_file}")