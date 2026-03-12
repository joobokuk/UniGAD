"""
unigad/models/backbone.py
--------------------------
DataParallel-safe 백본 (HooklessBackbone).

get_intermediate_layers() 기반으로 특징을 추출하여
forward hook의 race condition 문제를 제거한다.
모든 학습/추론 경로에서 동일하게 사용한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unigad.transforms import EXTRACT_LAYERS


class HooklessBackbone(nn.Module):
    """
    DINOv3-Large/16 또는 DINOv2-Large/14 백본 (완전 Frozen).

    get_intermediate_layers() 를 사용하여 특징을 추출하므로
    DataParallel 환경에서도 thread-safe 하게 동작한다.

    Args:
        layers        : 추출할 Transformer 블록 인덱스 목록 (0-indexed)
        backbone      : 'dinov3' (기본값) 또는 'dinov2'
        dinov3_repo   : DINOv3 로컬 클론 경로
        dinov3_weights: DINOv3 체크포인트 경로
        patch_size    : 입력 해상도 기준 패치 크기 (DINOv3=16, DINOv2=14)
    """

    def __init__(
        self,
        layers: list[int]    = EXTRACT_LAYERS,
        backbone: str        = "dinov3",
        dinov3_repo: str | None    = None,
        dinov3_weights: str | None = None,
        patch_size: int      = 16,
    ):
        super().__init__()
        self.layers     = layers
        self.patch_size = patch_size

        if backbone == "dinov3":
            if dinov3_repo is None or dinov3_weights is None:
                raise ValueError(
                    "backbone='dinov3' 사용 시 dinov3_repo와 dinov3_weights가 필요합니다."
                )
            print(f"[HooklessBackbone] DINOv3-Large/16 로딩...")
            print(f"  repo   : {dinov3_repo}")
            print(f"  weights: {dinov3_weights}")
            self.model = torch.hub.load(
                dinov3_repo, "dinov3_vitl16",
                source="local", weights=dinov3_weights,
            )
        else:
            print("[HooklessBackbone] DINOv2-Large/14 로딩...")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True,
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.embed_dim = self.model.embed_dim  # 1024 (ViT-L 공통)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            list of (cls_token [B,C], patch_tokens [B,N,C]) per layer
        """
        H, W      = x.shape[2], x.shape[3]
        n_patches = (H // self.patch_size) * (W // self.patch_size)

        try:
            outputs = self.model.get_intermediate_layers(
                x, n=self.layers, return_class_token=True, norm=False,
            )
        except TypeError:
            outputs = self.model.get_intermediate_layers(
                x, n=self.layers, return_class_token=True,
            )

        results = []
        for patch_and_reg, cls_tok in outputs:
            patch_tok = patch_and_reg[:, -n_patches:, :]
            results.append((cls_tok, patch_tok))
        return results
