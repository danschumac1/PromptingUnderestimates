#!/bin/bash
# 10272
# chmod +x ./bin/generation/logistic_regression_3.sh
# ./bin/generation/logistic_regression_3.sh
# nohup ./bin/generation/logistic_regression_3.sh > ./logs/logistic_regression_3.log &
# tail -f ./logs/logistic_regression_3.log
###############################################################################
# set -Eeuo pipefail
# set -o errtrace

###############################################################################
# SLACK TRAPS
###############################################################################
LOG_FILE="./logs/logistic_regression_3.log"
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

MODEL_STEMS=(
  # "random_llama"
  # "random_mistral"
  "random_qwen"
  # "llama"
  # "mistral"
  # "qwen"

)

EMBEDDING_TYPES=(
    # "ts-ust"
    # "lets-ust"
    # "vis-ust"
    # "ts-vis-ust"
    "vis-lets-ust"  
    # "slike"
)

CoT_STRING=(
  # "CoT"
  "Direct"
)

NORMALIZE=0 # Don't normalize rwc
CV_FOLDS=5

mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do
  for model_stem in "${MODEL_STEMS[@]}"; do
    for emb_type in "${EMBEDDING_TYPES[@]}"; do
      for cot_string in "${CoT_STRING[@]}"; do 

        # EMB_PATH="./data/sample_features/${model_stem}/${dataset}/${emb_type}_0-shot_${cot_string}"
        # NPZ_PATH="${EMB_PATH}/train_embeddings.npz"
        # NPY_PATH="${EMB_PATH}/train_embeddings.npy"

        # if [[ ! -f "$NPZ_PATH" && ! -f "$NPY_PATH" ]]; then
        #   echo "[SKIP] Missing embeddings at: $EMB_PATH"
        #   continue
        # fi

        echo "=================================================================="
        echo "[RUN] dataset=$dataset model=$model_stem emb=$emb_type"
        echo "=================================================================="

        CMD=(
        python ./src/logistic_regression.py
          --dataset "$dataset"
          --model_stem "$model_stem"
          --embedding_types "$emb_type"
          --CoT_string "$cot_string"
          --normalize "$NORMALIZE"
          --cv "$CV_FOLDS"
        )

        if [[ -n "${C_VALUES:-}" ]]; then CMD+=(--c_values "$C_VALUES"); fi
        if [[ -n "${MAX_ITER_VALUES:-}" ]]; then CMD+=(--max_iter_values "$MAX_ITER_VALUES"); fi

        run "GridSearchCV: dataset=$dataset model=$model_stem emb=$emb_type" "${CMD[@]}"
        echo ""
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
Models: ${MODEL_STEMS[*]}
Embeddings: ${EMBEDDING_TYPES[*]}
Normalize: ${NORMALIZE}
CV: ${CV_FOLDS}
Log: ${LOG_FILE}"
