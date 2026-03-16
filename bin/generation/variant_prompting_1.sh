#!/bin/bash
# 2699976
# chmod +x ./bin/generation/variant_prompting_1.sh
# ./bin/generation/variant_prompting_1.sh
# nohup ./bin/generation/variant_prompting_1.sh > ./logs/variant_prompting_1_master.log 2>&1 &
# tail -f ./logs/variant_prompting_1_master.log

LOG_FILE="./logs/variant_prompting_1_master.log"

# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  # "ctu"
  # "tee"
  # "had"
  # "har"
  # "emg"
  "rwc" # LATER
)

MODELS=(
  "qwen"
)

EMBEDDING_TYPES=(
  "lets,ust"
  # "vis,ust"
  # "vis,lets,ust"
)

# STATIC VARIABLES
BATCH_SIZE=5
COT=0
NORMALIZE=0
MAX_VARIANTS=0

RESUME=0

###############################################################################
# RUN
###############################################################################
mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for embedding_types in "${EMBEDDING_TYPES[@]}"; do

      # Context for Slack error reporting
      LAST_CONTEXT="dataset=${dataset} | model=${model} | ETs=${embedding_types} | --CoT ${COT}"
      LAST_CMD="python ./src/prompt_w_variants.py --dataset ${dataset} --model ${model} --embedding_types ${embedding_types} --batch_size ${BATCH_SIZE} --normalize ${NORMALIZE} --CoT ${COT} --max_variants ${MAX_VARIANTS} --resume ${RESUME}"

      python ./src/prompt_w_variants.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --embedding_types "${embedding_types}" \
        --batch_size "${BATCH_SIZE}" \
        --normalize "${NORMALIZE}" \
        --CoT "${COT}" \
        --max_variants "${MAX_VARIANTS}" \
        --resume "${RESUME}" 
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
CUDA: NA
Datasets: ${DATASETS[*]}
Models: ${MODELS[*]}
EmbeddingTypes: ${EMBEDDING_TYPES[*]}
CoT: ${COT}
Normalize: ${NORMALIZE}
MaxVariants: ${MAX_VARIANTS}
Resume: ${RESUME}
Log: ${LOG_FILE}"
