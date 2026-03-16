#!/usr/bin/env bash
# run_eval.sh
# ===========
# UniGAD 4×4 크로스 평가 (4 모델 × 4 데이터셋)
#
# 사용법:
#   ./run_eval.sh
#   ./run_eval.sh --mode zero_shot
#   # custom 데이터셋만 평가:
#   ./run_eval.sh --eval_datasets custom
#   ./run_eval.sh --ckpts custom --eval_datasets custom
#   CUSTOM_ROOT="${DATA_ROOT}/Custom" ./run_eval.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${SCRIPT_DIR}/../Data"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
RESULT="${SCRIPT_DIR}/results_crosseval.json"
DINOV3_REPO="${SCRIPT_DIR}/dinov3"
DINOV3_WTS="${DINOV3_REPO}/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

MODE="both"
EVAL_BATCH=4
FEW_SHOT_KS="1 2 4"

CUSTOM_ROOT="${CUSTOM_ROOT:-${DATA_ROOT}/Custom}"
echo "============================================================"
echo "  UniGAD 4×4 크로스 평가 시작"
echo "  가중치: ${CKPT_DIR}  custom 데이터: ${CUSTOM_ROOT}"
echo "  결과:   ${RESULT}"
echo "============================================================"

python scripts/eval_crosseval.py \
    --mvtec_root     "${DATA_ROOT}/MVTec"   \
    --visa_root      "${DATA_ROOT}/VisA"    \
    --custom_root    "${CUSTOM_ROOT}"      \
    --btad_root      "${DATA_ROOT}/BTAD"    \
    --ckpt_dir       "${CKPT_DIR}"          \
    --result_path    "${RESULT}"            \
    --dinov3_repo    "${DINOV3_REPO}"       \
    --dinov3_weights "${DINOV3_WTS}"        \
    --mode           "${MODE}"              \
    --eval_batch_size "${EVAL_BATCH}"       \
    --few_shot_ks    $FEW_SHOT_KS           \
    --ckpts          mvtec visa custom btad   \
    --eval_datasets  mvtec visa custom btad   \
    "$@"

echo "============================================================"
echo "  평가 완료 – 결과: ${RESULT}"
echo "============================================================"
