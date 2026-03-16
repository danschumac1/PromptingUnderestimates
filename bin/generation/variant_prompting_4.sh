#!/bin/bash
# 1470438
# chmod +x ./bin/generation/variant_prompting_4.sh
# ./bin/generation/variant_prompting_4.sh
# nohup ./bin/generation/variant_prompting_4.sh > ./logs/variant_prompting_4_master.log 2>&1 &
# tail -f ./logs/variant_prompting_4_master.log

LOG_FILE="./logs/variant_prompting_4_master.log"

# ------------------------------------------------------------------------------
# Slack traps (shared logic)
# ------------------------------------------------------------------------------
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  "rwc"
)

MODELS=(
  "qwen"
)

EMBEDDING_TYPES=(
  "vis,ust"
  # "lets,ust"
  # "lets,vis,ust"
)

# STATIC VARIABLES
BATCH_SIZE=5
COT=0
NORMALIZE=1
MAX_VARIANTS=0

# ------------------------------------------------------------------------------
# RESUME CONTROLS (match your log for rwc + vis,ust)
# Variant 1 was running and got to 528/901 batches.
# So: skip 0 variants, skip 528 batches within the first resumed variant.
# ------------------------------------------------------------------------------
N_VARIANTS_TO_SKIP=0
N_BATCHES_TO_SKIP=528

###############################################################################
# RUN
###############################################################################
mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for embedding_types in "${EMBEDDING_TYPES[@]}"; do

      # Only apply resume skips for the specific crashed combo: rwc + vis,ust
      if [[ "${dataset}" == "rwc" && "${embedding_types}" == "vis,ust" ]]; then
        VAR_SKIP="${N_VARIANTS_TO_SKIP}"
        BATCH_SKIP="${N_BATCHES_TO_SKIP}"
      else
        VAR_SKIP=0
        BATCH_SKIP=0
      fi

      # Context for Slack error reporting
      LAST_CONTEXT="dataset=${dataset} | model=${model} | ETs=${embedding_types} | --CoT ${COT}"
      LAST_CMD="python ./src/prompt_w_variants.py --dataset ${dataset} --model ${model} --embedding_types ${embedding_types} --batch_size ${BATCH_SIZE} --normalize ${NORMALIZE} --CoT ${COT} --max_variants ${MAX_VARIANTS} --n_variants_to_skip ${VAR_SKIP} --n_batches_to_skip ${BATCH_SKIP}"

      python ./src/prompt_w_variants.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --embedding_types "${embedding_types}" \
        --batch_size "${BATCH_SIZE}" \
        --normalize "${NORMALIZE}" \
        --CoT "${COT}" \
        --max_variants "${MAX_VARIANTS}" \
        --n_variants_to_skip "${VAR_SKIP}" \
        --n_batches_to_skip "${BATCH_SKIP}"

      # If we just used the resume skip for the crashed combo, don't reuse it again
      # in case this script is extended later.
      if [[ "${dataset}" == "rwc" && "${embedding_types}" == "vis,ust" ]]; then
        N_BATCHES_TO_SKIP=0
      fi

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
Resume(rwc vis,ust): n_variants_to_skip=0, n_batches_to_skip=528
Log: ${LOG_FILE}"
