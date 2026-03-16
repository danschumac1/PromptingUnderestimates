#!/bin/bash
# 2050118 ( paste kill id here )
# chmod +x ./bin/prompting_gpu1.sh
# ./bin/prompting_gpu1.sh
# nohup ./bin/prompting_gpu1.sh > ./logs/prompting_gpu1_master.log 2>&1 &
# tail -f ./logs/prompting_gpu1_master.log

LOG_FILE="./logs/prompting_gpu1_master.log"

# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
DATASETS=(
    # "har"
    # "emg"
    # "ctu"
    # "tee"
    "rwc" # LATER
)

MODELS=(
    "llama"
    # "mistral"
    # "qwen"
)

NSHOTS=(
    0
    # 2
    # 3
    # 5
)

# STATIC VARIABLES
# experiments
EMBEDDING_TYPES=(
    "ts,ust"
    "vis,ust"
    "lets,ust"
    "ts,vis,ust"
    "lets,vis,ust"
)

# config
CUDA_DEVICES="1"
BATCH_SIZE=20
# VIS_METHOD="line"
SAMPLE=1
COT=(
    0
    1
)

###############################################################################
# THE PLACE WHERE IT HAPPENS
###############################################################################
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for embedding_types in "${EMBEDDING_TYPES[@]}"; do
            for shots in "${NSHOTS[@]}"; do
                for cot in "${COT[@]}"; do
                    # Context for Slack error reporting
                    LAST_CONTEXT="dataset=${dataset} | model=${model} | ETs=${embedding_types} | shots=${shots} | --CoT ${cot}"
                    LAST_CMD="prompting.py --dataset ${dataset} --model ${model} --n_shots ${shots} | embedding_types "$embedding_types" | --CoT ${cot}"

                    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
                        python ./src/prompting.py \
                            --dataset "${dataset}" \
                            --model "${model}" \
                            --embedding_types "$embedding_types" \
                            --batch_size "${BATCH_SIZE}" \
                            --n_shots "${shots}" \
                            --sample "${SAMPLE}" \
                            --CoT "$cot"
                done
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
Log: ${LOG_FILE}"
