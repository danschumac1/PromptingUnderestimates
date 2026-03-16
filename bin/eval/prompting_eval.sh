#!/bin/bash
# 12345678 ( paste kill id here )
# chmod +x ./bin/eval/prompting_eval.sh
# ./bin/eval/prompting_eval.sh
# nohup ./bin/eval/prompting_eval.sh > ./logs/eval/prompting_eval_master.log 2>&1 &
# tail -f ./logs/eval/prompting_eval_master.log

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
DATASETS=(
    "har"
    "emg"
    "ctu"
    "tee"
    "rwc" # LATER
)
MODEL_STEMS=(
    "llama"
    "mistral"
    # 'qwen'
)

# ONLY ZERO SHOT FOR NOW IS AVAILABLE.
NSHOTS=(
    # 0
    2
    # 3
    # 5
)

EMBED_TYPES=(
    "ts-ust"
    "lets-ust"
    "vis-ust"
    "ts-vis-ust"
    "vis-lets-ust"  
)

CoT_STRINGS=(
#   "CoT"
  "Direct"
)


###############################################################################
# THE PLACE WHERE IT HAPPENS
###############################################################################
for dataset in "${DATASETS[@]}"; do
    for model_stem in "${MODEL_STEMS[@]}"; do
        for embed_type in "${EMBED_TYPES[@]}"; do
            for shots in "${NSHOTS[@]}"; do
                for cot_string in "${CoT_STRINGS[@]}"; do

                    pred_path="data/sample_generations/${model_stem}/${dataset}/visual_prompting/${embed_type}_${shots}-shot_${cot_string}.jsonl"
                    python ./src/eval.py \
                        --pred_path $pred_path
                done
            done
        done
    done
done
