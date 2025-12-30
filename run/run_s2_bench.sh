#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=$1
OUTPUT_FILE=${2:-"results.txt"}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <MODEL_PATH> [OUTPUT_FILE]"
    exit 1
fi

python src/evaluation/eval.py \
    --model "$MODEL_PATH" \
    --evals=AMC23,AIME25,AIME24,MATH500,GPQADiamond \
    --tp=4 \
    --output_file="$OUTPUT_FILE" \
    --temperatures 0.7 \
    --result_dir=./outputs \
    --add_think_tok '<|im_start|>think'
