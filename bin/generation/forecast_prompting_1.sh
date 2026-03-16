#!/bin/bash
# 1680294
# chmod +x ./bin/generation/forecast_prompting_1.sh
# Usage: ./bin/generation/forecast_prompting_1.sh
# Run background: nohup ./bin/generation/forecast_prompting_1.sh > ./logs/forecast_prompting_1.log 2>&1 &
# Monitor: tail -f ./logs/forecast_prompting_1.log

LOG_FILE="./logs/forecast_prompting_1.log"
mkdir -p ./logs
source ./bin/_slack_traps.sh

MODELS=(
  # "llama"
  # "mistral"
  # "qwen"
  # "random_llama"
  # "random_mistral"
  "random_qwen"
  )
EMBEDDING_TYPES=(
  "dv" 
  "d" 
  "v"
  )
CUDA_DEVICES="1"
LOOKBACK=96
HORIZON=6

for model in "${MODELS[@]}"; do
  for etype in "${EMBEDDING_TYPES[@]}"; do
    echo "----------------------------------------------------------------"
    echo "STARTING TEST PROMPTING: ${model} | modality: ${etype}"
    echo "Autoregressive Horizon: ${HORIZON} | Lookback: ${LOOKBACK}"
    echo "----------------------------------------------------------------"

    LAST_CONTEXT="model=${model} | modality=${etype} | task=test_prompting"
    
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    python ./src/forecast_prompting.py \
      --model "${model}" \
      --embedding_type "${etype}" \
      --lookback "${LOOKBACK}" \
      --horizon "${HORIZON}"

    sleep 2
  done
done

./bin/_notify_slack.sh "SUCCESS" "✅ Test Prompting Complete (100 samples)"