"""
unigad/models/multigpu.py
--------------------------
DataParallel 호환 래퍼 및 유틸리티.

MultiGPUUniADet: patch_feats를 list 대신 단일 텐서로 반환하여
DataParallel의 gather 단계에서 문제가 없도록 한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unigad.models.uniadet import UniADet


class MultiGPUUniADet(nn.Module):
    """
    DataParallel 호환 UniADet 래퍼.

    forward()가 반환하는 patch_feats(list)는
    DataParallel의 gather가 처리하지 못하므로
    [B, L, N, C] 단일 텐서로 스택하여 반환한다.
    """

    def __init__(self, base_model: UniADet):
        super().__init__()
        self.backbone    = base_model.backbone
        self.classifiers = base_model.classifiers

    def forward(self, x: torch.Tensor):
        feats                          = self.backbone(x)
        cls_list, seg_list, patch_list = self.classifiers.compute_scores(feats)

        cls_logit  = torch.stack(cls_list,   dim=1).mean(1)   # [B, 2]
        seg_logit  = torch.stack(seg_list,   dim=1).mean(1)   # [B, N, 2]
        patch_feat = torch.stack(patch_list, dim=1)            # [B, L, N, C]
        return cls_logit, seg_logit, patch_feat


def wrap_multigpu(base_model: UniADet) -> nn.Module:
    """
    GPU 수에 따라 DataParallel을 자동 적용.

    항상 MultiGPUUniADet으로 래핑하고,
    GPU가 2개 이상이면 DataParallel도 적용한다.
    """
    model = MultiGPUUniADet(base_model)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[MultiGPU] DataParallel – {n_gpus}개 GPU 병렬 사용")
        model = nn.DataParallel(model)
    return model


def inner(model: nn.Module) -> MultiGPUUniADet:
    """DataParallel 래퍼 안의 실제 모델 반환."""
    return model.module if isinstance(model, nn.DataParallel) else model


def patch_feat_to_list(pf: torch.Tensor) -> list[torch.Tensor]:
    """[B, L, N, C] → L × [B, N, C]"""
    return [pf[:, i] for i in range(pf.size(1))]
