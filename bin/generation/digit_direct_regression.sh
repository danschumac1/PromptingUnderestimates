#!/bin/bash
# chmod +x ./bin/generation/digit_direct_regression.sh
# ./bin/generation/digit_direct_regression.sh
# nohup ./bin/generation/digit_direct_regression.sh > ./logs/digit_direct.log &

###############################################################################
# SLACK TRAPS & LOGGING
###############################################################################
LOG_FILE="./logs/digit_direct.log"
mkdir -p ./logs
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    "tee"
    "emg"
    "ctu"
    "har"
    "had"
    "rwc" 
)

NORMALIZE=1  # Recommended for Logistic Regression unless data is already scaled
CV_FOLDS=5

###############################################################################
# RUN LOOP
###############################################################################
for dataset in "${DATASETS[@]}"; do

    echo "=================================================================="
    echo "[RUN] Digit Direct Logistic Regression | Dataset: $dataset"
    echo "=================================================================="

    CMD=(
        python ./src/digit_direct_logistic_regression.py
        --dataset "$dataset"
        --normalize "$NORMALIZE"
        --cv "$CV_FOLDS"
    )

    # Optional overrides from environment variables
    if [[ -n "${C_VALUES:-}" ]]; then CMD+=(--c_values "$C_VALUES"); fi
    if [[ -n "${MAX_ITER_VALUES:-}" ]]; then CMD+=(--max_iter_values "$MAX_ITER_VALUES"); fi

    # Using your 'run' wrapper from slack_traps
    run "DigitDirect: $dataset" "${CMD[@]}"
    
    echo -e "Finished $dataset\n"
done

###############################################################################
# SUCCESS PING
###############################################################################
./bin/_notify_slack.sh "SUCCESS" \
"✅ Digit Direct Regression Completed
Script: $(basename "$0")
Datasets: ${DATASETS[*]}
Normalize: ${NORMALIZE}
CV: ${CV_FOLDS}
Log: ${LOG_FILE}"