#!/bin/bash
# chmod +x ./bin/generation/self_consistancy_gpu2.sh
# ./bin/generation/self_consistancy_gpu2.sh
# nohup ./bin/generation/self_consistancy_gpu2.sh > ./logs/self_consistancy_gpu2.log 2>&1 &
# tail -f ./logs/self_consistancy_gpu2.log

LOG_FILE="./logs/self_consistnacy_gpu2_master.log"

# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
# source ./bin/_slack_traps.sh # Uncomment if using slack notifications

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    # "tee"
    "emg"
    "ctu"
    # "har"
    # "had"
    # "rwc"
)

MODELS=(
    "llama"
    # "qwen"
    # "mistral"
)

# Self-Consistency specific iterables
EMBEDDING_TYPES=(
    "d"
    "v"
    "dv"
)

# Config
CUDA_DEVICES="2"
N_VOTERS=7  

###############################################################################
# EXECUTION LOOP
###############################################################################
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for etype in "${EMBEDDING_TYPES[@]}"; do
            
            echo "Starting: Dataset=${dataset} | Model=${model} | Type=${etype} | Voters=${N_VOTERS}"
            
            # Context for potential error reporting
            LAST_CONTEXT="dataset=${dataset} | model=${model} | embedding_types=${etype} | n_voters=${N_VOTERS}"
            LAST_CMD="self_consistancy.py --dataset ${dataset} --model ${model} --embedding_types ${etype} --n_voters ${N_VOTERS}"

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
                python ./src/self_consistancy.py \
                    --dataset "${dataset}" \
                    --model "${model}" \
                    --embedding_types "${etype}" \
                    --n_voters "${N_VOTERS}"

        done
    done
done

###############################################################################
# SUCCESS PING
###############################################################################
# If you have the slack notify script:
# ./bin/_notify_slack.sh "SUCCESS" \
# "✅ Self-Consistency script completed
# Script: $(basename "$0")
# Host: $(hostname)
# CUDA: ${CUDA_DEVICES}
# Log: ${LOG_FILE}"

echo "✅ All runs completed. Check $LOG_FILE for details."