#!/bin/bash
# 508791 ( paste kill id here )
# chmod +x ./bin/prompting_erfan_1.sh
# ./bin/prompting_erfan_1.sh
# nohup ./bin/prompting_erfan_1.sh > ./logs/prompting_erfan_1_master.log 2>&1 &
# tail -f ./logs/prompting_erfan_1_master.log

LOG_FILE="./logs/prompting_erfan_1_master.log"


# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
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
    "mistral"
    "qwen"
)

NSHOTS=(
    2
    # 3
    # 5
)

# STATIC VARIABLES
# experiments
INCLUDE_TS=1
INCLUDE_VIS=0
INCLUDE_LETSCLIKE=0

# config
CUDA_DEVICES="0,1"
BATCH_SIZE=1
# VIS_METHOD="line"
SAMPLE=1
CoT=0

###############################################################################
# THE PLACE WHERE IT HAPPENS
###############################################################################
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for shots in "${NSHOTS[@]}"; do

            # Context for Slack error reporting
            LAST_CONTEXT="dataset=${dataset} | model=${model} | shots=${shots}"
            LAST_CMD="prompting.py --dataset ${dataset} --model ${model} --n_shots ${shots}"

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
                python ./src/prompting.py \
                    --dataset "${dataset}" \
                    --model "${model}" \
                    --batch_size "${BATCH_SIZE}" \
                    --n_shots "${shots}" \
                    --sample "${SAMPLE}" \
                    --include_ts "${INCLUDE_TS}" \
                    --include_vis "${INCLUDE_VIS}" \
                    --include_LETSCLike "${INCLUDE_LETSCLIKE}" \
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
Flags: ts=${INCLUDE_TS} vis=${INCLUDE_VIS} lets=${INCLUDE_LETSCLIKE}
Log: ${LOG_FILE}"
