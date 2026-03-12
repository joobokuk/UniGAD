#!/usr/bin/env bash
# run_pipeline.sh
# ===============
# 학습 → 4×4 크로스 평가 전체 파이프라인 순서 실행

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_train.sh"
bash "${SCRIPT_DIR}/run_eval.sh"

echo "전체 파이프라인 완료."
