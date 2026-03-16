#!/bin/bash
# 1708069 ( paste kill id here )
# chmod +x ./bin/embedding_erfan_2.sh
# ./bin/embedding_erfan_2.sh
# nohup ./bin/embedding_erfan_2.sh > ./logs/embedding_erfan_2_master.log 2>&1 &
# tail -f ./logs/embedding_erfan_2_master.log

###############################################################################
# PING LOGIC
###############################################################################
LOG_FILE="./logs/embedding_erfan_2_master.log"
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
    # "rwc" # LATER
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
INCLUDE_USER_TEXT=1
INCLUDE_TS=1
INCLUDE_VIS=0
INCLUDE_LETSCLIKE=0

# config
CUDA_DEVICES="2,3"
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

      run "dataset=$dataset | model=$model | shots=$shots" \
        env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
        python ./src/embedding.py \
          --dataset "$dataset" \
          --model "$model" \
          --batch_size "$BATCH_SIZE" \
          --n_shots "$shots" \
          --sample "$SAMPLE" \
          --include_user_text "$INCLUDE_USER_TEXT" \
          --include_ts "$INCLUDE_TS" \
          --include_vis "$INCLUDE_VIS" \
          --include_LETSCLike "$INCLUDE_LETSCLIKE"\
          --CoT "$CoT"

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
