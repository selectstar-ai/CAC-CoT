#!/usr/bin/env bash
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────
# bash run/run_s1_bench.sh <MODEL_NAME> <MODEL_PATH> [K] [IS_GREEDY]
#
# Examples:
#   bash run/run_s1_bench.sh "Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-7B-Instruct"
#   bash run/run_s1_bench.sh "MyCustomModel" "/path/to/my/model" 5 false
# ──────────────────────────────────────────────────────────────────

MODEL_NAME=${1:-""}
MODEL_PATH=${2:-""}
K=${3:-5}
IS_GREEDY=${4:-false}

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <MODEL_NAME> <MODEL_PATH> [K] [IS_GREEDY]"
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

# Define paths
S1_BENCH_DIR="src/s1_bench"

echo "========================================================"
echo "Starting S1-Bench Pipeline"
echo "Model Name: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "K (Samples): $K"
echo "Greedy: $IS_GREEDY"
echo "========================================================"

# 1. Generate Responses
echo "[1/4] Generating Responses..."
python "${S1_BENCH_DIR}/get_LRM_vllm_response.py" \
    --model "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --k "$K" \
    --greedy "$IS_GREEDY"

# 2. Split Thinking and Answer
echo "[2/4] Splitting Thinking and Answer..."
python "${S1_BENCH_DIR}/split_think_answer.py" \
    --model "$MODEL_NAME" \
    --greedy "$IS_GREEDY"

# 3. Evaluate Results (using GPT-4o)
echo "[3/4] Evaluating Results..."
# Note: This script uses multiprocessing and writes to LRM_acc_eval/
python "${S1_BENCH_DIR}/get_LRM_eval.py" \
    --model "$MODEL_NAME" \
    --greedy "$IS_GREEDY"

# 4. Calculate Scores
echo "[4/4] Calculating Scores..."
# Note: s1bench_len is hardcoded to 1000 in original readme logic typically, 
# but let's check dataset size. Defaulting to 1000 as per common practice or we can fetch it.
# The original script requires --s1bench_len. The dataset seems to have 1000 items.
S1_BENCH_LEN=1000
python "${S1_BENCH_DIR}/get_acc_scores.py" \
    --model "$MODEL_NAME" \
    --greedy "$IS_GREEDY" \
    --s1bench_len "$S1_BENCH_LEN"

echo "========================================================"
echo "S1-Bench Pipeline Completed."
echo "Results should be in ${S1_BENCH_DIR}/LRM_acc_eval/${MODEL_NAME} or similar."
echo "========================================================"
