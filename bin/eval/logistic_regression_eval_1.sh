#!/bin/bash
# 12345678 ( paste kill id here )
# chmod +x ./bin/eval/logistic_regression_eval_1.sh
# ./bin/eval/logistic_regression_eval_1.sh
# nohup ./bin/eval/logistic_regression_eval_1.sh > ./logs/eval/logistic_regression_eval_1_master.log 2>&1 &
# tail -f ./logs/eval/logistic_regression_eval_1_master.log

set -Eeuo pipefail
shopt -s nullglob

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  # "had"
  # "har"
  # "emg"
  # "ctu"
  # "tee"
  # "rwc" # LATER
  trHARteHAD
)

MODEL_STEMS=(
  "llama"
  # "mistral"
  # "qwen"
  # "random_llama"
  # "random_mistral"
  # "random_qwen"
)

# ONLY ZERO SHOT FOR NOW IS AVAILABLE.
NSHOTS=(
  0
  # 3
  # 5
)

EMBED_TYPES=(
  # "ts-ust"
  "vis-lets-ust"
  # "lets-ust"
  # "vis-ust"
  # "ts-vis-ust"
)

CoT_STRINGS=(
  # "CoT"
  "Direct"
)

ROOT="/raid/hdd249/data/sample_generations"
EVAL_PY="./src/eval.py"

###############################################################################
# MAIN
###############################################################################
fail_count=0
eval_count=0

for dataset in "${DATASETS[@]}"; do
  for model_stem in "${MODEL_STEMS[@]}"; do
    for embed_type in "${EMBED_TYPES[@]}"; do
      for shots in "${NSHOTS[@]}"; do
        for cot_string in "${CoT_STRINGS[@]}"; do

          # Example folder:
          # data/sample_generations/mistral/emg/logistic_regression/lets-ust_CoT/
          run_dir="${ROOT}/${model_stem}/${dataset}/logistic_regression/${embed_type}_${shots}-shot_${cot_string}"

          if [[ ! -d "$run_dir" ]]; then
            echo "[SKIP] missing dir: $run_dir"
            continue
          fi

          # Find all layer jsonls at the TOP LEVEL of run_dir (not best_params/*)
          mapfile -d '' LAYER_FILES < <(
            find "$run_dir" \
              -maxdepth 1 -type f -name 'layer*.jsonl' -print0 \
              | sort -z
          )

          if [[ ${#LAYER_FILES[@]} -eq 0 ]]; then
            echo "[SKIP] no layer*.jsonl files in: $run_dir"
            continue
          fi

          echo "=================================================================="
          echo "[RUN] dataset=$dataset model=$model_stem embed=$embed_type shots=$shots cot=$cot_string"
          echo "[DIR] $run_dir"
          echo "[N]   ${#LAYER_FILES[@]} files"

          for pred_path in "${LAYER_FILES[@]}"; do
            echo "  -> eval: $pred_path"
            if python "$EVAL_PY" --pred_path "$pred_path" --results_path "./data/qwen_random.tsv"; then
              ((eval_count+=1))
            else
              echo "  [FAIL] $pred_path"
              ((fail_count+=1))
            fi
          done
        done
      done
    done
  done
done

echo "=================================================================="
echo "[DONE] eval_count=$eval_count fail_count=$fail_count"

# Make the script exit nonzero if anything failed (useful for monitoring)
if [[ $fail_count -gt 0 ]]; then
  exit 1
fi
