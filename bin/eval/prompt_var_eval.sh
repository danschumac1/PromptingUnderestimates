#!/bin/bash
# chmod +x bin/eval/prompt_var_eval.sh
# ./bin/eval/prompt_var_eval.sh

DATASETS=(
    "tee"
    "emg"
    "ctu"
    "har"
    "had"
    # "rwc" # LATER
)

EMBED_TYPES=(
    "ts-ust"
    "lets-ust"
    "vis-ust"
    "ts-vis-ust"
    "vis-lets-ust"  
)

for dataset in "${DATASETS[@]}"; do

    pred_path="data/sample_generations/${model_stem}/${dataset}/visual_prompting/${embed_type}_${shots}-shot_${cot_string}.jsonl"
    python ./src/eval_prompt_variants.py \
            --dataset $dataset \
            --model qwen
done
