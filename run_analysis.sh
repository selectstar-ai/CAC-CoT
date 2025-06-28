#!/usr/bin/env bash
set -euo pipefail

# ── 사용자 설정 (원하는 값으로 수정) ─────────────────
DATA_PATH="datumo/CAC-CoT"          # HF dataset name or local path
LOAD_TYPE="hf"                       # hf or disk
CONNECTOR_YAML="config/connector.yaml"
GRADING_PROMPT="config/grading_prompt.yaml"
EVAL_MODEL="gpt-4o-mini"
NUM_WORKERS=4

GRADE=true                          # true ▶ grading 실행
# ─────────────────────────────────────────────────────

# ── 실행 ─────────────────────────────────────────────
ARGS=(
  --dataset_dir "${DATA_PATH}"
  --type "${LOAD_TYPE}"
  --connector_yaml "${CONNECTOR_YAML}"
  --evaluate_model "${EVAL_MODEL}"
  --num_workers "${NUM_WORKERS}"
)

if [ "${GRADE}" = true ]; then
  ARGS+=( --grade_accuracy --grading_prompt_yaml "${GRADING_PROMPT}")
fi

python src/analysis.py "${ARGS[@]}"
