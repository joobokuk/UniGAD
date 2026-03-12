"""
unigad/engine/train.py
-----------------------
표준 학습 루프 (train_uniadet) 및 JVM patch-crop 학습 루프 (train_jvm_patch).

개선사항:
  - wrap_multigpu()를 통해 DataParallel 자동 적용 (Multi-GPU 지원)
  - best_ckpt_path 하나만 저장 (중복 파일 없음)
  - 학습 진입 시 DataLoader 배치 수 검증 (ZeroDivisionError 방지)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unigad.losses              import FocalLoss, DiceLoss
from unigad.models.multigpu     import wrap_multigpu, inner
from unigad.models.uniadet      import UniADet
from unigad.transforms          import PATCH_SIZE_DINOV3


# ─────────────────────────────────────────────────────────────────────
# 표준 학습 (MVTec / VisA / BTAD / JVM)
# ─────────────────────────────────────────────────────────────────────
def train_uniadet(
    model:          UniADet,
    dataloader:     DataLoader,
    device:         torch.device,
    epochs:         int   = 10,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    patience:       int   = 5,
    ckpt_path:      str | None = None,
    patch_size:     int   = PATCH_SIZE_DINOV3,
    img_size:       int   = 448,
) -> None:
    """
    Decoupled Classifiers만 학습. Backbone은 항상 Frozen.

    Args:
        ckpt_path : best epoch 가중치 저장 경로 (단일 파일)
    """
    wrapped = wrap_multigpu(model)
    wrapped.to(device)

    classifiers = inner(wrapped).classifiers
    classifiers.train()
    inner(wrapped).backbone.eval()

    optimizer = torch.optim.AdamW(
        [p for p in classifiers.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce    = nn.CrossEntropyLoss()
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice  = DiceLoss()
    N_SIDE          = img_size // patch_size

    if len(dataloader) == 0:
        raise RuntimeError(
            "DataLoader 배치 수가 0입니다. "
            "dataset 크기 및 batch_size/drop_last 설정을 확인하세요."
        )

    print(f"\n{'='*60}")
    print(f"  train_uniadet | epochs={epochs}, lr={lr}, patience={patience}")
    print(f"{'='*60}")

    best_loss  = float("inf")
    no_improve = 0
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        inner(wrapped).backbone.eval()
        classifiers.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch+1:02d}/{epochs}")

        for imgs, labels, masks in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)
            B      = imgs.size(0)

            optimizer.zero_grad()
            cls_logit, seg_logit, _ = wrapped(imgs)

            loss_cls = criterion_ce(cls_logit, labels.long())

            masks_p = F.interpolate(
                masks.unsqueeze(1), size=(N_SIDE, N_SIDE), mode="nearest",
            ).view(B, -1)
            anomaly_logit = seg_logit[..., 1] - seg_logit[..., 0]
            loss_seg = criterion_focal(anomaly_logit, masks_p) + \
                       criterion_dice(anomaly_logit, masks_p)

            loss = loss_cls + loss_seg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifiers.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
                seg=f"{loss_seg.item():.4f}",
            )

        scheduler.step()
        avg = epoch_loss / len(dataloader)

        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in classifiers.state_dict().items()}
            tag = "  ★ best"
        else:
            no_improve += 1
            tag = f"  (no improve {no_improve}/{patience})"

        print(f"[Train] Epoch {epoch+1:02d}/{epochs} – avg_loss: {avg:.4f}{tag}")

        if no_improve >= patience:
            print(f"[EarlyStopping] {patience}회 연속 미개선 → 조기 종료")
            break

    if best_state is not None:
        classifiers.load_state_dict(best_state)
        print(f"[Train] Best 가중치 복원 완료 (best_loss={best_loss:.4f})")
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            print(f"[Train] 저장: {ckpt_path}")

    # 학습된 classifiers를 원본 model에도 반영
    model.classifiers.load_state_dict(classifiers.state_dict())
    print("[Train] 완료.\n")


# ─────────────────────────────────────────────────────────────────────
# JVM Patch-Crop 학습
# ─────────────────────────────────────────────────────────────────────
def train_jvm_patch(
    model:          nn.Module,
    dataloader:     DataLoader,
    device:         torch.device,
    epochs:         int   = 30,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    patience:       int   = 7,
    ckpt_path:      str   = "checkpoints/ckpt_jvm_patch.pth",
) -> None:
    """
    JVM Patch-Crop UniADet 학습 (MultiGPU 래퍼 전달 시 그대로 사용).
    """
    from unigad.utils.patch import FINAL_SIZE, N_PATCHES
    classifiers = inner(model).classifiers
    N_SIDE      = FINAL_SIZE // PATCH_SIZE_DINOV3

    optimizer       = torch.optim.AdamW(
        [p for p in classifiers.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce    = nn.CrossEntropyLoss()
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice  = DiceLoss()

    if len(dataloader) == 0:
        raise RuntimeError(
            f"DataLoader 배치 수가 0입니다 (샘플={len(dataloader.dataset)}, "
            f"배치={dataloader.batch_size})."
        )

    print(f"\n{'='*60}")
    print(f"  train_jvm_patch | epochs={epochs}, lr={lr}, patience={patience}")
    print(f"  GPUs={torch.cuda.device_count()}")
    print(f"{'='*60}")

    best_loss  = float("inf")
    no_improve = 0
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        model.train()
        inner(model).backbone.eval()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch+1:02d}/{epochs}")

        for imgs, labels, masks in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)
            B      = imgs.size(0)

            optimizer.zero_grad()
            cls_logit, seg_logit, _ = model(imgs)

            loss_cls = criterion_ce(cls_logit, labels.long())
            masks_p  = F.interpolate(
                masks.unsqueeze(1), size=(N_SIDE, N_SIDE), mode="nearest",
            ).view(B, -1)
            anomaly_logit = seg_logit[..., 1] - seg_logit[..., 0]
            loss_focal    = criterion_focal(anomaly_logit, masks_p)
            loss_dice     = criterion_dice(anomaly_logit,  masks_p)
            loss          = loss_cls + loss_focal + loss_dice

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifiers.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
                focal=f"{loss_focal.item():.4f}",
                dice=f"{loss_dice.item():.4f}",
            )

        scheduler.step()
        avg = epoch_loss / len(dataloader)

        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in classifiers.state_dict().items()}
            tag = "  ★ best"
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1
            tag = f"  (no improve {no_improve}/{patience})"

        print(f"[Train] Epoch {epoch+1:02d}/{epochs} – avg_loss: {avg:.4f}{tag}")

        if no_improve >= patience:
            print(f"[EarlyStopping] {patience}회 연속 미개선 → 조기 종료")
            break

    if best_state is not None:
        classifiers.load_state_dict(best_state)
        print(f"[Train] Best 가중치 복원 완료 (best_loss={best_loss:.4f})")

    print(f"[Train] 완료. 저장: {ckpt_path}")
