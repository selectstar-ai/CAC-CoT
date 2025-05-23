#!/usr/bin/env bash
set -euo pipefail

# ── 사용자 정의 가능 변수 ─────────────────────
# SYNTHETIC_MODEL="gemini-2.0-flash"
SYNTHETIC_MODEL="gpt-4o-mini"

DATASET_NAME="datumo/datumo-gemini-short-v2"
MODEL="Qwen/Qwen2.5-7B-Instruct"

PROMPT_YAML="config/synthetic_prompt.yaml"
VERSION="v4"

SYSTEMPROMPT_YAML="config/system_prompt.yaml"
SYSTEM_KEY="qwen"

CONNECTOR_YAML="config/connector.yaml"

OUTPUT_DIR="data/out_sample"
PUSH_TO_HUB=false   # true or false
HF_TOKEN=""         # 허브 푸시할 때만 설정

MIN_LEN=100
MAX_LEN=30000

TEST_MODE=true      # true → 샘플링 실행, false → 전체 실행
SAMPLE_SIZE=10      # TEST_MODE=true일 때만 적용

NUM_WORKERS=4
# ─────────────────────────────────────────────

# ── 실제 실행 ────────────────────────────────
ARGS=(
  --synthetic_model "${SYNTHETIC_MODEL}"
  --dataset_name "${DATASET_NAME}"
  --model "${MODEL}"
  --prompt_yaml "${PROMPT_YAML}"
  --version "${VERSION}"
  --systemprompt_yaml "${SYSTEMPROMPT_YAML}"
  --system_key "${SYSTEM_KEY}"
  --connector_yaml "${CONNECTOR_YAML}"
  --output_dir "${OUTPUT_DIR}"
  --constraint_min_length "${MIN_LEN}"
  --constraint_max_length "${MAX_LEN}"
  --num_workers "${NUM_WORKERS}"
)

# 테스트 모드 설정
if [ "${TEST_MODE}" = true ]; then
  ARGS+=( --test --sample_size "${SAMPLE_SIZE}" )
fi

# HF 허브 푸시 설정
if [ "${PUSH_TO_HUB}" = true ]; then
  ARGS+=( --push_to_hub --token "${HF_TOKEN}" )
fi

python src/synthetic.py "${ARGS[@]}"
