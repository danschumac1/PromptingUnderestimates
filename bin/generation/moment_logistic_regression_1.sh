#!/bin/bash
# 1961182
# chmod +x ./bin/generation/moment_logistic_regression_1.sh
# ./bin/generation/moment_logistic_regression_1.sh
# nohup ./bin/generation/moment_logistic_regression_1.sh > ./logs/moment_logistic_regression_1.log &
# tail -f ./logs/moment_logistic_regression_1.log
###############################################################################
# set -Eeuo pipefail
# set -o errtrace

###############################################################################
# SLACK TRAPS
###############################################################################
LOG_FILE="./logs/moment_logistic_regression_1.log"
source ./bin/_slack_traps.sh

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    # "ctu"
    # "emg"
    "had"
    # "har"
    # "tee"
    # "rwc" # LATER
)


NORMALIZE=1 # Don't normalize rwc

mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do

        python ./src/moment_logistic_regresion.py \
          --dataset "$dataset" \
          --normalize "$NORMALIZE"

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
Datasets: ${DATASETS[*]}
Log: ${LOG_FILE}"
