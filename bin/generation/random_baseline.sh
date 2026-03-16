#!/bin/bash
# chmod +x ./bin/random_baseline.sh
# ./bin/generation/random_baseline.sh

DATASETS=(
    "ctu"
    # "emg"
    "had"
    "har"
    "rwc"
    # "tee"
)

MODES=(
    "uniform"
    "prior"
    "majority"
)

for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        python ./src/random_baseline.py \
            --input_folder "/raid/hdd249/data/samples/$dataset" \
            --mode $mode
    done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"

# not samples
# --input_folder ./Classification/data/datasets/$dataset \