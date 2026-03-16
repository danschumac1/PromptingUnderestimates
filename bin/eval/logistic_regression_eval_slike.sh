#!/bin/bash
# chmod +x ./bin/eval/logistic_regression_eval_slike.sh
# ./bin/eval/logistic_regression_eval_slike.sh
# nohup ./bin/eval/logistic_regression_eval_slike.sh > ./logs/eval/logistic_regression_eval_slike_master.log 2>&1 &
# tail -f ./logs/eval/logistic_regression_eval_slike_master.log

set -Eeuo pipefail
shopt -s nullglob

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  "har"
  "emg"
  "ctu"
  "tee"
  # "rwc"
)

MODEL_STEMS=(
  "llama"
  # "mistral"
  # "qwen"
)

ROOT="/raid/hdd249/Classification_v2/data/sample_generations"
EVAL_PY="./src/eval.py"

###############################################################################
# MAIN
###############################################################################
fail_count=0
eval_count=0

for dataset in "${DATASETS[@]}"; do
  for model_stem in "${MODEL_STEMS[@]}"; do

    run_dir="${ROOT}/${model_stem}/${dataset}/logistic_regression/slike"

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
    echo "[RUN] dataset=$dataset model=$model_stem embed=slike"
    echo "[DIR] $run_dir"
    echo "[N]   ${#LAYER_FILES[@]} files"

    for pred_path in "${LAYER_FILES[@]}"; do
      echo "  -> eval: $pred_path"
      if python "$EVAL_PY" --pred_path "$pred_path"; then
        ((eval_count+=1))
      else
        echo "  [FAIL] $pred_path"
        ((fail_count+=1))
      fi
    done

  done
done

echo "=================================================================="
echo "[DONE] eval_count=$eval_count fail_count=$fail_count"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi
