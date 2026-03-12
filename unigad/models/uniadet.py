"""
unigad/models/uniadet.py
-------------------------
UniADet 통합 모델 (backbone + classifiers 조립).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unigad.models.backbone    import HooklessBackbone
from unigad.models.classifiers import DecoupledClassifiers
from unigad.transforms         import EXTRACT_LAYERS


class UniADet(nn.Module):
    """
    UniADet 통합 모델.

    구성:
      - HooklessBackbone     : DINOv3-L/16 또는 DINOv2-L/14 (Frozen, DataParallel-safe)
      - DecoupledClassifiers : 학습 가능한 분류기 (~0.02M params)
    """

    def __init__(
        self,
        layers: list[int]          = EXTRACT_LAYERS,
        backbone: str              = "dinov3",
        dinov3_repo: str | None    = None,
        dinov3_weights: str | None = None,
        patch_size: int            = 16,
    ):
        super().__init__()
        self.backbone = HooklessBackbone(
            layers=layers,
            backbone=backbone,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
            patch_size=patch_size,
        )
        self.classifiers = DecoupledClassifiers(
            embed_dim=self.backbone.embed_dim,
            n_layers=len(layers),
        )
        n_train = sum(p.numel() for p in self.classifiers.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[UniADet] 백본: {backbone.upper()} | "
              f"학습 파라미터: {n_train:,} / 전체: {n_total:,}")

    def forward(self, x: torch.Tensor):
        """
        Returns:
            cls_logit  : [B, 2]
            seg_logit  : [B, N, 2]
            patch_feats: list of [B, N, C]
        """
        feats = self.backbone(x)
        return self.classifiers(feats)
