#!/bin/bash
#                     chmod +x ./bin/generation/forecast_regression.sh
#                     ./bin/generation/forecast_regression.sh
# Run background:     nohup ./bin/generation/forecast_regression.sh > ./logs/forecast_regression.log 2>&1 &
# Monitor:            tail -f ./logs/forecast_regression.log

LOG_FILE="./logs/forecast_regression_analysis.log"
mkdir -p ./logs
source ./bin/_slack_traps.sh 2>/dev/null || echo "Slack traps not found."

MODELS=(
  # "llama"
  # "mistral"
  "qwen"
  # "random_llama"
  # "random_mistral"
  "random_qwen"
  )
EMBEDDING_TYPES=(
  "dv" 
  "d"
  "v"
  )
HORIZON=6
OUTPUT_DIR="./data/forecasting/regression_results"

echo "Starting Potential (Linear Probe) Analysis..." | tee -a "$LOG_FILE"

for model in "${MODELS[@]}"; do
  for etype in "${EMBEDDING_TYPES[@]}"; do
    echo "RUNNING RIDGE: Model: ${model} | Modality: ${etype}" | tee -a "$LOG_FILE"

    python ./src/forecast_regression.py \
      --model "${model}" \
      --embedding_type "${etype}" \
      --horizon "${HORIZON}" \
      2>&1 | tee -a "$LOG_FILE"
  done
done

./bin/_notify_slack.sh "SUCCESS" "✅ Regression Potential Analysis Complete."