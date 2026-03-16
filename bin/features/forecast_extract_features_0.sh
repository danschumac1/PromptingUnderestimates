#!/bin/bash
# 1577293
# chmod +x ./bin/features/forecast_extract_features_0.sh
# ./bin/features/forecast_extract_features_0.sh
# nohup ./bin/features/forecast_extract_features_0.sh > ./logs/forecast_extract_0.log 2>&1 &
# tail -f ./logs/forecast_extract_0.log

LOG_FILE="./logs/forecast_extract_0.log"
mkdir -p ./logs
source ./bin/_slack_traps.sh 2>/dev/null || echo "Slack traps not found."

###############################################################################
# CONFIGURATION
###############################################################################
MODELS=(
  # "llama"
  "mistral"
  # "qwen"
  # "random_llama"
  # "random_mistral"
  # "random_qwen"
  )
EMBEDDING_TYPES=(
  "dv" 
  "d" 
  "v"
)
CUDA_DEVICES="0"
LOOKBACK=96

###############################################################################
# EXECUTION
###############################################################################
for model in "${MODELS[@]}"; do
  for et in "${EMBEDDING_TYPES[@]}"; do
    echo "=================================================================="
    echo "[FEATURE EXTRACTION] model=$model | type=$et | lookback=$LOOKBACK"
    echo "Processing: 1000 Train samples & 100 Test samples"
    echo "=================================================================="

    # Context for Slack error reporting
    LAST_CONTEXT="model=$model | type=$et | task=feature_extraction"
    LAST_CMD="forecast_embedding.py --model $model --embedding_type $et"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    python ./src/forecast_embedding.py \
      --model "$model" \
      --embedding_type "$et" \
      --lookback "$LOOKBACK" \
      2>&1 | tee -a "$LOG_FILE"

    # Optional: Brief cooldown for GPU memory
    sleep 2
  done
done

###############################################################################
# NOTIFICATION
###############################################################################
./bin/_notify_slack.sh "SUCCESS" \
"✅ Feature Extraction Complete
Model: ${MODELS[*]}
Modalities: ${EMBEDDING_TYPES[*]}
Split: 1000 Train / 100 Test
Log: ${LOG_FILE}"