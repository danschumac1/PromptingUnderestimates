#!/bin/bash
# chmod +x ./bin/generation/bert_log_reg.sh
# ./bin/generation/bert_log_reg.sh
# nohup ./bin/generation/bert_log_reg.sh > ./logs/bert_log_reg.log &
# tail -f ./logs/bert_log_reg.log
###############################################################################
# set -Eeuo pipefail
# set -o errtrace

###############################################################################
# SLACK TRAPS
###############################################################################
LOG_FILE="./logs/logistic_regression_bert.log"
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    # "ctu"
    # "emg"
    "had"
    # "har"
    # "rwc"
    # "tee"
)

BERT_MODELS=(
    # "bert-base-uncased"
    "bert-large-uncased"
)

NORMALIZE=0
CV_FOLDS=5

# Optional overrides (export before running)
# export C_VALUES="0.01,0.1,1,10"
# export MAX_ITER_VALUES="1000,2000"

mkdir -p ./logs

###############################################################################
# RUN
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for model_name in "${BERT_MODELS[@]}"; do

    echo "=================================================================="
    echo "[RUN] dataset=$dataset model=$model_name"
    echo "=================================================================="

    CMD=(
      python ./src/bert_log_reg.py
        --dataset "$dataset"
        --model_name "$model_name"
        --normalize "$NORMALIZE"
        --cv "$CV_FOLDS"
    )

    if [[ -n "${C_VALUES:-}" ]]; then
      CMD+=(--c_values "$C_VALUES")
    fi

    if [[ -n "${MAX_ITER_VALUES:-}" ]]; then
      CMD+=(--max_iter_values "$MAX_ITER_VALUES")
    fi

    run "GridSearchCV (BERT): dataset=$dataset model=$model_name" "${CMD[@]}"
    echo ""

  done
done

###############################################################################
# SUCCESS PING
###############################################################################
./bin/_notify_slack.sh "SUCCESS" \
"✅ Script completed successfully
Script: $(basename "$0")
Host: $(hostname)
Datasets: ${DATASETS[*]}
Models: ${BERT_MODELS[*]}
Normalize: ${NORMALIZE}
CV: ${CV_FOLDS}
Log: ${LOG_FILE}"
