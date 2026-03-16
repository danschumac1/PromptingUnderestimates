#!/bin/bash
# chmod +x ./bin/data_builders/prep_SLLM_data.sh
# ./bin/data_builders/prep_SLLM_data.sh
set -Eeuo pipefail

DATASETS=(
    "ctu"
    "emg"
    # "had"
    "har"
    "tee"
    "rwc"
)


for dataset in "${DATASETS[@]}"; do
  echo -e "---------------------------------------------------------------------------------------------"
  echo "STAGE-1 | $dataset" 
  python ./src/data_management/prep_SLLM_data.py --dataset "$dataset" --overwrite 
  python ./src/data_management/prep_SLLM_data2.py --dataset "$dataset" --overwrite 
  echo -e "---------------------------------------------------------------------------------------------\n\n"
done
