#!/bin/bash
# chmod +x ./bin/data_builders/patch_instruct_time.sh
# ./bin/data_builders/patch_instruct_time.sh
set -Eeuo pipefail

DATASETS=(
    "ctu"
    "emg"
    "had"
    "har"
    "rwc"
    "tee"
)


for dataset in "${DATASETS[@]}"; do
  echo -e "---------------------------------------------------------------------------------------------"
  python ./src/data_management/patch_data_for_instruct_time.py --dataset "$dataset"
  echo -e "---------------------------------------------------------------------------------------------\n\n"
done

