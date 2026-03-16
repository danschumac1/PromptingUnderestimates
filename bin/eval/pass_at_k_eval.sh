#!/bin/bash
# chmod +x bin/eval/pass_at_k_eval.sh
# ./bin/eval/pass_at_k_eval.sh

DATASETS=(
    "tee"
    "emg"
    "ctu"
    "har"
    "had"
    # "rwc" # LATER
)

EMBED_TYPES=(
    "lets-ust"
    "vis-ust"
    # "vis-lets-ust"  
    lets-vis-ust
)

for dataset in "${DATASETS[@]}"; do
    for embed_type in "${EMBED_TYPES[@]}"; do

        pred_path="data/sample_generations/${model_stem}/${dataset}/visual_prompting/${embed_type}_${shots}-shot_${cot_string}.jsonl"
        python ./src/eval_pass_at_k.py \
                --dataset $dataset \
                --embedding_types $embed_type
    done
done
