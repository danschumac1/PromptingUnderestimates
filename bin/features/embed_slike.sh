#!/bin/bash
set -Eeuo pipefail
# 767438 ( paste kill id here )
# chmod +x ./bin/features/embed_slike.sh
# ./bin/features/embed_slike.sh
# nohup ./bin/features/embed_slike.sh > ./logs/embed_slike_master.log 2>&1 &
# tail -f ./logs/embed_slike_master.log

###############################################################################
# PING LOGIC
###############################################################################
LOG_FILE="./logs/embed_slike_gpu2_master.log"
mkdir -p "$(dirname "$LOG_FILE")"
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  # "ctu"
  # "emg"
  # "har"
  # "tee"
  "rwc" # later
)

MODELS=(
  # "llama"
  "mistral"
  "qwen"
)

CUDA_DEVICES="2,3"
BATCH_SIZE=1

# Only include if your standard_args() supports --sample

###############################################################################
# THE PLACE WHERE IT HAPPENS
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    run "SLIKE | dataset=$dataset | model=$model | batch=$BATCH_SIZE" \
      env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
      python ./src/embed_slike.py \
        --dataset "$dataset" \
        --model "$model" \
        --batch_size "$BATCH_SIZE" 
  done
done

###############################################################################
# SUCCESS PING
###############################################################################
./bin/_notify_slack.sh "SUCCESS" \
"✅ Script completed successfully
Script: $(basename "$0")
Host: $(hostname)
CUDA: ${CUDA_DEVICES}
Datasets: ${DATASETS[*]}
Models: ${MODELS[*]}
Batch: ${BATCH_SIZE}
Log: ${LOG_FILE}"
