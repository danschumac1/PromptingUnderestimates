#!/bin/bash
# chmod +x ./bin/eval/_mega_eval.sh
set -Eeuo pipefail

ROOT="data/sample_generations"
EVAL_PY="./src/eval.py"
RESULTS="./data/raw_resultsVP.tsv"

# Override like:
#   ./bin/eval/_mega_eval.sh MODELS=llama DATASETS=ctu METHODS=logistic_regression EMBEDS=lets_ust
DATASETS="ctu,emg,har,tee"
MODELS="llama,mistral,qwen" # llama,mistral
METHODS="visual_prompting" # logistic_regression,visual_prompting
EMBEDS="lets_ust,ts_ust,vis_ust,ts_vis_ust,vis_lets_ust"

# For visual_prompting filenames like: lets_0-shot.jsonl
SHOTS="0"

for arg in "$@"; do
  eval "$arg"
done

# mkdir -p "$(dirname "$RESULTS")"
# : > "$RESULTS"

split_csv() { tr ',' ' ' <<< "${1// /}"; }

fail_count=0

for model_stem in $(split_csv "$MODELS"); do
  for dataset in $(split_csv "$DATASETS"); do
    for method in $(split_csv "$METHODS"); do

      if [[ "$method" == "logistic_regression" ]]; then
        for embed_type in $(split_csv "$EMBEDS"); do
          dir="${ROOT}/${model_stem}/${dataset}/logistic_regression/${embed_type}"
          [[ -d "$dir" ]] || continue

          shopt -s nullglob
          layer_files=("$dir"/layer*.jsonl)
          shopt -u nullglob

          if (( ${#layer_files[@]} == 0 )); then
            echo "[SKIP] No layer*.jsonl in: $dir"
            continue
          fi

          for pred_path in "${layer_files[@]}"; do
            [[ "$pred_path" == *"/best_params/"* ]] && continue
            echo "[RUN] $pred_path"
            if ! python "$EVAL_PY" --pred_path "$pred_path" --results_path "$RESULTS"; then
              echo "[FAIL] eval.py failed on: $pred_path"
              ((fail_count+=1))
            fi
          done
        done

      elif [[ "$method" == "visual_prompting" ]]; then
        # visual_prompting files are typically ts/lets/vis/ts_vis/vis_lets (no *_ust)
        for embed_type in $(split_csv "$EMBEDS"); do
          prompt_embed="${embed_type%_ust}"   # lets_ust -> lets, ts_vis_ust -> ts_vis, etc.
          pred_path="${ROOT}/${model_stem}/${dataset}/visual_prompting/${prompt_embed}_${SHOTS}-shot.jsonl"

          [[ -f "$pred_path" ]] || continue

          echo "[RUN] $pred_path"
          if ! python "$EVAL_PY" --pred_path "$pred_path" --results_path "$RESULTS"; then
            echo "[FAIL] eval.py failed on: $pred_path"
            ((fail_count+=1))
          fi
        done

      else
        echo "[WARN] Unknown method: $method (skipping)"
      fi

    done
  done
done

[[ "$fail_count" -eq 0 ]]
