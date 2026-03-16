#!/bin/bash
# ================================================================
# 2025-11-25
# chmod +x ./bin/get_and_clean_data.sh
# ./bin/data_builders/get_and_clean_data.sh
# ================================================================

set -e  # exit on first error
set -u  # error on undefined variables

BASE_URL="https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets"
RAW_DIR="data/raw_data"

DATASETS=(
    "ctu:CTU"
    "emg:EMG"
    "har:HAR"
    "tee:TEE"
)

for entry in "${DATASETS[@]}"; do
    IFS=":" read -r dataset DATASET_UPPER <<< "$entry"

    echo "=== Processing $DATASET_UPPER ==="

    mkdir -p "${RAW_DIR}/${dataset}"

    wget -O "${RAW_DIR}/${dataset}/${DATASET_UPPER}_TRAIN.ts" \
        "${BASE_URL}/${DATASET_UPPER}/${DATASET_UPPER}_TRAIN.ts"

    wget -O "${RAW_DIR}/${dataset}/${DATASET_UPPER}_TEST.ts" \
        "${BASE_URL}/${DATASET_UPPER}/${DATASET_UPPER}_TEST.ts"

    python ./src/data_management/clean_data.py \
        --dataset "$dataset"
done
