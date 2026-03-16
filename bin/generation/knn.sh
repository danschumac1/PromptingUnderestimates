#!/bin/bash
# ( paste kill id here )
# chmod +x ./bin/knn_gridsearch.sh
# ./bin/knn_gridsearch.sh
# nohup ./bin/knn_gridsearch.sh > ./logs/knn_gridsearch.log 2>&1 &
# tail -f ./logs/knn_gridsearch.log

# set -euo pipefail

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
DATASETS=(
    "ctu"
    # "emg"
    # "har"
    # "tee"
)

MODEL_STEMS=(
    # "llama"
    "mistral"
    # "qwen"
)

EMBEDDING_TYPES=(
    # "ts_ust"
    "lets_ust"
    # "vis_ust"
    # "ts_vis_ust"
    # "vis_lets_ust"
)

# Layers to evaluate (use -1 for last layer, or specific layer indices)
# Typical transformer models have ~32-40 layers, adjust based on your model
LAYERS=(
    -1      # last layer
    # 0     # embedding layer
    # 16    # middle layer (example)
    # 24    # later layer (example)
)

# STATIC VARIABLES
NORMALIZE=1
CV_FOLDS=5
METRIC="cosine"  # options: cosine, euclidean, manhattan

# Grid search parameters (optional - defaults are used if not specified)
# K_VALUES="1,3,5,7,9,11,15,21"

###############################################################################
# THE PLACE WHERE IT HAPPENS
# NOTE: knn.py now uses GridSearchCV internally
#       - Automatically searches over K values
#       - Optimizes for macro F1 score
#       - Saves best parameters to best_params.txt
###############################################################################
mkdir -p ./logs

for dataset in "${DATASETS[@]}"; do
    for model_stem in "${MODEL_STEMS[@]}"; do
        for emb_type in "${EMBEDDING_TYPES[@]}"; do
            for layer in "${LAYERS[@]}"; do

                # Check for embeddings (.npz format for per-layer embeddings)
                EMB_PATH="./data/sample_features/${model_stem}/${dataset}/${emb_type}_0-shot"
                NPZ_PATH="${EMB_PATH}/train_embeddings.npz"
                NPY_PATH="${EMB_PATH}/train_embeddings.npy"  # legacy format

                if [[ ! -f "$NPZ_PATH" && ! -f "$NPY_PATH" ]]; then
                    echo "[SKIP] Missing embeddings at:"
                    echo "  $EMB_PATH"
                    continue
                fi

                echo "=================================================================="
                echo "[RUN] dataset=$dataset model=$model_stem emb=$emb_type layer=$layer"
                echo "      Running GridSearchCV (K sweep with ${CV_FOLDS}-fold CV)"
                echo "=================================================================="

                # Build command with optional custom grid parameters
                CMD="python ./src/knn.py \
                    --dataset \"$dataset\" \
                    --model_stem \"$model_stem\" \
                    --embedding_types \"$emb_type\" \
                    --layer $layer \
                    --normalize $NORMALIZE \
                    --cv $CV_FOLDS \
                    --metric $METRIC"

                # Add custom K values if defined
                if [[ -n "${K_VALUES:-}" ]]; then
                    CMD="$CMD --k_values \"$K_VALUES\""
                fi

                eval $CMD

                echo ""

            done
        done
    done
done

echo "=================================================================="
echo "✅ All KNN runs complete!"
echo "   Results saved to ./data/sample_generations/{model}/{dataset}/knn/{emb_type}/layer{N}/"
echo "   - predictions.jsonl: per-sample predictions"
echo "   - best_params.txt: best hyperparameters and scores"
echo "=================================================================="
