#!/bin/bash
# 2749113
# chmod +x ./bin/generation/pass_at_k_2.sh
# ./bin/generation/pass_at_k_2.sh
# nohup ./bin/generation/pass_at_k_2.sh > ./logs/pass_at_k_2_master.log 2>&1 &
# tail -f ./logs/pass_at_k_2_master.log

LOG_FILE="./logs/pass_at_k_gpu3_master.log"

# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################

# ITERABLES
DATASETS=(
    # "tee"
    # "emg"
    # "ctu"
    # "har"
    'had'
    # "rwc"   # uncomment when ready
)

MODELS=(
    "qwen"
)

# NOTE:
# --embedding_types must be a valid combo from build_valid_embedding_strings()
# Keep 'ust' unless you explicitly want *no* user text
EMBEDDING_TYPES=(
    # "vis,ust"
    "lets,ust"
    # "lets,vis,ust"
)

###############################################################################
# STATIC ARGS
###############################################################################

N=20          # --n
BATCH_SIZE=5        # attempts per API call (per example)
NORMALIZE=1
COT=0

# batch_size is unused in attempt-batching mode but kept for CLI compatibility
BATCH_SIZE=5

###############################################################################
# RUN
###############################################################################
mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for embedding_types in "${EMBEDDING_TYPES[@]}"; do

      # Context for Slack error reporting
      LAST_CONTEXT="dataset=${dataset} | model=${model} | ETs=${embedding_types} | n=${N} | batch_size=${BATCH_SIZE} | CoT=${COT}"

      LAST_CMD="pass_at_k.py \
        --dataset ${dataset} \
        --model ${model} \
        --embedding_types ${embedding_types} \
        --n ${N} \
        --batch_size ${BATCH_SIZE} \
        --normalize ${NORMALIZE} \
        --CoT ${COT} \
        --resume 1"

      python ./src/pass_at_k.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --embedding_types "${embedding_types}" \
        --n "${N}" \
        --batch_size "${BATCH_SIZE}" \
        --normalize "${NORMALIZE}" \
        --CoT "${COT}" \
        --resume 1
    done
  done
done

###############################################################################
# SUCCESS PING
###############################################################################
./bin/_notify_slack.sh "SUCCESS" \
"✅ pass@k run completed successfully
Script: $(basename "$0")
Host: $(hostname)
CUDA: NA
Datasets: ${DATASETS[*]}
Models: ${MODELS[*]}
EmbeddingTypes: ${EMBEDDING_TYPES[*]}
n: ${N}
batch_size: ${BATCH_SIZE}
CoT: ${COT}
Log: ${LOG_FILE}"
