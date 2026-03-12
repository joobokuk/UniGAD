"""
unigad/models/classifiers.py
-----------------------------
DecoupledClassifiers: 학습 대상 파라미터 (~0.01M).

각 추출 레이어마다 독립적인 W_cls / W_seg 방향 벡터를 유지하고,
코사인 유사도 + 온도 계수(logit_scale)로 이상 점수를 계산한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledClassifiers(nn.Module):
    """
    레이어별 이미지/픽셀 이상 분류기.

    파라미터:
        W_cls[i]     : [2, C]  – CLS token 기반 이미지 이상 점수
        W_seg[i]     : [2, C]  – Patch token 기반 픽셀 이상 점수
        logit_scale  : scalar  – 온도 계수 τ의 역수 (CLIP 방식, 학습 가능)

    파라미터 수 (embed_dim=1024, n_layers=5):
        W_cls + W_seg : 2 × 5 × 2 × 1024 = 20,480
        logit_scale   : 1
        합계           : ~0.02M
    """

    def __init__(self, embed_dim: int = 1024, n_layers: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers  = n_layers

        self.W_cls = nn.ParameterList([
            nn.Parameter(torch.empty(2, embed_dim)) for _ in range(n_layers)
        ])
        self.W_seg = nn.ParameterList([
            nn.Parameter(torch.empty(2, embed_dim)) for _ in range(n_layers)
        ])
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self._init_weights()

    def _init_weights(self):
        for p in list(self.W_cls) + list(self.W_seg):
            nn.init.normal_(p, std=0.02)

    def compute_scores(
        self,
        backbone_features: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list, list, list]:
        """
        Args:
            backbone_features: list of (cls_tok [B,C], patch_tok [B,N,C]) per layer

        Returns:
            cls_logits_all : list of [B, 2]    (레이어별 이미지 로짓)
            seg_logits_all : list of [B, N, 2] (레이어별 패치 로짓)
            patch_feats_all: list of [B, N, C] (L2 정규화 패치 특징)
        """
        scale           = self.logit_scale.clamp(max=100)
        cls_logits_all  = []
        seg_logits_all  = []
        patch_feats_all = []

        for i, (cls_feat, patch_feat) in enumerate(backbone_features):
            cls_n   = F.normalize(cls_feat,       p=2, dim=-1)
            patch_n = F.normalize(patch_feat,     p=2, dim=-1)
            w_cls_n = F.normalize(self.W_cls[i],  p=2, dim=-1)
            w_seg_n = F.normalize(self.W_seg[i],  p=2, dim=-1)

            cos_cls = (cls_n @ w_cls_n.t()) * scale          # [B, 2]
            cos_seg = torch.einsum("bnc,kc->bnk", patch_n, w_seg_n) * scale  # [B,N,2]

            cls_logits_all.append(cos_cls)
            seg_logits_all.append(cos_seg)
            patch_feats_all.append(patch_n)

        return cls_logits_all, seg_logits_all, patch_feats_all

    def forward(
        self,
        backbone_features: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        레이어별 점수를 평균 집계하여 최종 예측 반환.

        Returns:
            cls_logit  : [B, 2]
            seg_logit  : [B, N, 2]
            patch_feats: list of [B, N, C]
        """
        cls_list, seg_list, patch_list = self.compute_scores(backbone_features)
        cls_logit = torch.stack(cls_list, dim=1).mean(dim=1)
        seg_logit = torch.stack(seg_list, dim=1).mean(dim=1)
        return cls_logit, seg_logit, patch_list
