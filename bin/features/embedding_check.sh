#!/bin/bash
# chmod +x ./bin/embedding_check.sh
#  ./bin/embedding_check.sh
# ------------------------------------------------------------
# Embedding existence checklist
# ------------------------------------------------------------

ROOT="./data/sample_features"
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/embedding_checklist_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

MODELS=("llama" "mistral" "qwen")
DATASETS=("ctu" "emg" "har" "tee")
EMBEDDINGS=("ts_ust" "lets_ust" "vis_ust" "ts_vis_ust" "vis_lets_ust")

CHECK_OK="✅"
CHECK_FAIL="❌"

{
    echo -e "model\tdataset\tembedding\tstatus"
    echo "-----------------------------------------------"

    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for emb in "${EMBEDDINGS[@]}"; do
                EMB_DIR="${ROOT}/${model}/${dataset}/${emb}_0-shot"

                TRAIN_OK=false
                TEST_OK=false

                if [[ -d "$EMB_DIR" ]]; then
                    if [[ -f "$EMB_DIR/train_embeddings.npz" || -f "$EMB_DIR/train_embeddings.npy" ]]; then
                        TRAIN_OK=true
                    fi
                    if [[ -f "$EMB_DIR/test_embeddings.npz" || -f "$EMB_DIR/test_embeddings.npy" ]]; then
                        TEST_OK=true
                    fi
                fi

                if [[ "$TRAIN_OK" == true && "$TEST_OK" == true ]]; then
                    STATUS="$CHECK_OK"
                else
                    STATUS="$CHECK_FAIL"
                fi

                echo -e "${model}\t${dataset}\t${emb}\t${STATUS}"
            done
        done
    done
} | tee "$LOG_FILE"

echo
echo "Checklist written to: $LOG_FILE"
