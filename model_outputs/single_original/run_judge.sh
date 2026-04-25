#!/bin/bash

set -e

MODELS=("AudioFlamingo3" "DeSTA" "Qwen")
CATEGORIES=("Animal" "Emotion" "Gender" "Language")

for model in "${MODELS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        # skip desta animal
        if [[ "$model" == "DeSTA" && "$category" == "Animal" ]]; then
            echo "Skipping ${model} ${category}..."
            continue
        fi
        input_file="${model}/${category}.json"
        output_file="${model}/eval_result/${category}.json"
        echo "Judging correctness for ${input_file}..."
        python judge_correctness.py -i "${input_file}" -o "${output_file}"
    done
done