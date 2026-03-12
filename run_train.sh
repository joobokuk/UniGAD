#!/usr/bin/env bash
# run_train.sh
# ============
# UniGAD 4개 모델 학습 (MVTec / VisA / JVM / BTAD)
#
# 사용법:
#   ./run_train.sh
#   ./run_train.sh --train_targets mvtec    # 일부만 학습
#   ./run_train.sh --force                  # 강제 재학습

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${SCRIPT_DIR}/../Data"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
DINOV3_REPO="${SCRIPT_DIR}/dinov3"
DINOV3_WTS="${DINOV3_REPO}/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

EPOCHS=50
BATCH_SIZE=256
LR=0.0001
PATIENCE=5

echo "============================================================"
echo "  UniGAD 학습 시작 (MVTec / VisA / JVM / BTAD)"
echo "  체크포인트: ${CKPT_DIR}"
echo "============================================================"

python scripts/train_standard.py \
    --mvtec_root     "${DATA_ROOT}/MVTec"   \
    --visa_root      "${DATA_ROOT}/VisA"    \
    --jvm_root       "${DATA_ROOT}/JVM_mvtec" \
    --btad_root      "${DATA_ROOT}/BTAD"    \
    --ckpt_dir       "${CKPT_DIR}"          \
    --dinov3_repo    "${DINOV3_REPO}"       \
    --dinov3_weights "${DINOV3_WTS}"        \
    --epochs         "${EPOCHS}"            \
    --batch_size     "${BATCH_SIZE}"        \
    --lr             "${LR}"               \
    --patience       "${PATIENCE}"          \
    --train_targets  mvtec visa jvm btad   \
    "$@"

echo "============================================================"
echo "  학습 완료 – 체크포인트: ${CKPT_DIR}"
echo "============================================================"
