#!/bin/bash
# chmod +x ./bin/data_builders/prep_slike.sh
# ./bin/data_builders/prep_slike.sh
set -Eeuo pipefail

DATASETS=(
    "ctu"
    "emg"
    # "had"
    "har"
    "rwc"
    "tee"
)


for dataset in "${DATASETS[@]}"; do
  echo -e "---------------------------------------------------------------------------------------------"
  python ./src/data_management/slike_summaries.py --dataset "$dataset"
  echo -e "---------------------------------------------------------------------------------------------\n\n"
done
