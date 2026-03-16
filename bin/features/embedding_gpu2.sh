#!/bin/bash
# 2327990 ( paste kill id here )
# chmod +x ./bin/embedding_gpu2.sh
# ./bin/embedding_gpu2.sh
# nohup ./bin/embedding_gpu2.sh > ./logs/embedding_gpu2_master.log 2>&1 &
# tail -f ./logs/embedding_gpu2_master.log

###############################################################################
# PING LOGIC
###############################################################################
LOG_FILE="./logs/embedding_gpu2_master.log"
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
DATASETS=(
    # "ctu"
    # "emg"
    # "har"
    # "tee"
    "rwc" # LATER
)

MODELS=(
    "llama"
    # "mistral"
    # "qwen"
)

# ONLY ZERO SHOT FOR NOW IS AVAILABLE.
NSHOTS=(
    0
    # 3
    # 5
)

# STATIC VARIABLES
# experiments
EMBEDDING_TYPES=(
    # "lets,vis,ust"
    "ts,ust"
    "vis,ust"
    "lets,ust"
    "ts,vis,ust"
)
# config
CUDA_DEVICES="2"
BATCH_SIZE=2
SAMPLE=1
CoT=1

# VIS_METHOD="line" # only change for rwc

###############################################################################
# THE PLACE WHERE IT HAPPENS
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for shots in "${NSHOTS[@]}"; do
      for embedding_types in "${EMBEDDING_TYPES[@]}"; do
        run "dataset=$dataset | model=$model | shots=$shots" \
          env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
          python ./src/embedding.py \
            --dataset "$dataset" \
            --model "$model" \
            --batch_size "$BATCH_SIZE" \
            --n_shots "$shots" \
            --sample "$SAMPLE" \
            --embedding_types "$embedding_types" \
            --CoT "$CoT"
      done
    done
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
Shots: ${NSHOTS[*]}
Flags: user_text=${INCLUDE_USER_TEXT} ts=${INCLUDE_TS} vis=${INCLUDE_VIS} lets=${INCLUDE_LETSCLIKE}
Log: ${LOG_FILE}"
