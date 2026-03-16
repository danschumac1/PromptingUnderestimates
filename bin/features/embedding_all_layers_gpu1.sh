#!/bin/bash
# 3149649
# chmod +x ./bin/features/embedding_all_layers_gpu1.sh
# ./bin/features/embedding_all_layers_gpu1.sh
# nohup ./bin/features/embedding_all_layers_gpu1.sh > ./logs/embedding_all_layers_gpu1.log 2>&1 &
# tail -f ./logs/embedding_all_layers_gpu1.log

###############################################################################
# PING LOGIC
###############################################################################
LOG_FILE="./logs/embedding_all_layers.log"
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
DATASETS=(
    "tee"
    # "emg"
    # "ctu"
    # "har"
    # "had"
    # "rwc" 
)

MODELS=(
  # "random_llama"
  "random_mistral"
  # "random_qwen"
  # "llama"
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
# experiments - configure which embedding types to generate
EMBEDDING_TYPES=(
    "lets,vis,ust"
    # "ts,ust"
    # "vis,ust"
    # "lets,ust"
    # "ts,vis,ust"
)


# STATIC VARIABLES
# config
CUDA_DEVICES="1"
BATCH_SIZE=2
SAMPLE=1
# VIS_METHOD="line" # only change for rwc

###############################################################################
# THE PLACE WHERE IT HAPPENS
# NOTE: embedding.py now extracts embeddings from ALL layers (last token)
#       and saves them as .npz files (train_embeddings.npz, test_embeddings.npz)
#       with keys "0", "1", ..., "N" for each layer
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for et in "${EMBEDDING_TYPES[@]}"; do
      for shots in "${NSHOTS[@]}"; do

        echo "=================================================================="
        echo "[RUN] dataset=$dataset | model=$model | shots=$shots"
        echo "      Extracting embeddings from ALL layers (last token position)"
        echo "=================================================================="

        run "dataset=$dataset | model=$model | shots=$shots" \
          env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
          python ./src/embedding.py \
            --dataset "$dataset" \
            --embedding_types "$et" \
            --model "$model" \
            --batch_size "$BATCH_SIZE" \
            --n_shots "$shots" \
            --sample "$SAMPLE"
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
Note: All layer embeddings saved as .npz files
Log: ${LOG_FILE}"
