#!/bin/bash
# chmod +x ./bin/eval/eval_forecast.sh
# ./bin/eval/eval_forecast.sh
# nohup ./bin/eval/eval_forecast.sh > ./logs/eval/forecast_eval_master.log 2>&1 &
# tail -f ./logs/eval/forecast_eval_master.log

set -Eeuo pipefail
shopt -s nullglob

###############################################################################
# CONFIGURATION
###############################################################################
MODELS=(
  "llama"
  "mistral"
  "qwen"
  # "random_llama"
  # "random_mistral"
  # "random_qwen"
)

MODALITIES=(
  "d"
  "v"
  "dv"
)

METHODS=(
  "prompting"
#   "linear_regression"
)

# Define the layers you want to evaluate for linear_regression
LAYERS=(
  10
#   20
#   30
#   40
) 

EVAL_PY="./src/eval_forecast.py"

###############################################################################
# MAIN
###############################################################################
fail_count=0
eval_count=0

echo "Starting forecasting evaluation..."

for model in "${MODELS[@]}"; do
  for modality in "${MODALITIES[@]}"; do
    for method in "${METHODS[@]}"; do

      # PROMPTING METHOD (No layer argument needed)
      if [[ "$method" == "prompting" ]]; then
        echo "=================================================================="
        echo "[RUN] model=$model modality=$modality method=$method"

        if python "$EVAL_PY" --model "$model" --modality "$modality" --method "$method"; then
          ((eval_count+=1))
        else
          echo "  [FAIL] Failed to evaluate: model=$model, modality=$modality, method=$method"
          ((fail_count+=1))
        fi

      # LINEAR REGRESSION METHOD (Requires layer argument)
      elif [[ "$method" == "linear_regression" ]]; then
        for layer in "${LAYERS[@]}"; do
          echo "=================================================================="
          echo "[RUN] model=$model modality=$modality method=$method layer=$layer"

          if python "$EVAL_PY" --model "$model" --modality "$modality" --method "$method" --layer "$layer"; then
            ((eval_count+=1))
          else
            echo "  [FAIL] Failed to evaluate: model=$model, modality=$modality, method=$method, layer=$layer"
            ((fail_count+=1))
          fi
        done
      fi

    done
  done
done

echo "=================================================================="
echo "[DONE] Successful evaluations: $eval_count | Failed evaluations: $fail_count"

# Make the script exit nonzero if anything failed (useful for CI/monitoring)
if [[ $fail_count -gt 0 ]]; then
  exit 1
fi