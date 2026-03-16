#!/bin/bash
# 12345678 ( paste kill id here )
# chmod +x ./bin/eval/random_eval.sh
# ./bin/eval/random_eval.sh
# nohup ./bin/eval/random_eval.sh > ./logs/eval/random_eval_master.log 2>&1 &
# tail -f ./logs/eval/random_eval_master.log

set -Eeuo pipefail
shopt -s nullglob

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
  "ctu"
  # "emg"
  "had"
  "har"
  # "tee"
  "rwc" # LATER
)

MODES=(
    "uniform"
    "prior"
    "majority"
)

ROOT="/raid/hdd249/data/sample_generations/no_model"
EVAL_PY="./src/eval.py"

###############################################################################
# MAIN
###############################################################################
fail_count=0
eval_count=0

for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do

        # Example file:
        # data/sample_generations/no_model/emg/random/uniform.jsonl
        pred_path="${ROOT}/${dataset}/random/${mode}.jsonl"

        python ./src/eval.py \
            --pred_path $pred_path \
            --results_path "./data/random_results.tsv"





  done
done

