"""
unigad/engine/memory_bank.py
-----------------------------
메모리 뱅크 구성 및 Few-shot 점수 계산.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unigad.models.multigpu import inner, patch_feat_to_list
from unigad.utils.patch import crop_patch, N_PATCHES, FINAL_SIZE as PATCH_FINAL
from torchvision import transforms
from PIL import Image

_EVAL_TF = transforms.Compose([
    transforms.Resize((PATCH_FINAL, PATCH_FINAL)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────
# 표준 메모리 뱅크 (전체 이미지 단위)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_memory_bank(
    model:          nn.Module,
    support_loader: DataLoader,
    device:         torch.device,
) -> list[torch.Tensor]:
    """
    정상 이미지에서 패치 특징을 추출하여 메모리 뱅크 구성.

    Returns:
        list of [M, C] per layer (L2 정규화 완료)
    """
    _inner = inner(model)
    _inner.eval()
    _inner.backbone.eval()

    bank_per_layer: list[list[torch.Tensor]] = []
    print("[MemBank] 정상 이미지 패치 특징 추출 중...")

    for imgs, _, _ in tqdm(support_loader, desc="[MemBank]"):
        imgs = imgs.to(device)
        cls_logit, seg_logit, pf = _inner(imgs)
        patch_feats = patch_feat_to_list(pf)

        if not bank_per_layer:
            bank_per_layer = [[] for _ in patch_feats]
        for i, feat in enumerate(patch_feats):
            bank_per_layer[i].append(feat.reshape(-1, feat.size(-1)).cpu())

    memory_bank = [torch.cat(feats, dim=0) for feats in bank_per_layer]
    print(f"[MemBank] 완료 – 패치 수/레이어: {memory_bank[0].shape[0]}")
    return memory_bank


# ─────────────────────────────────────────────────────────────────────
# 위치별 메모리 뱅크 (patch-crop 전용)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_memory_banks_per_pos(
    model:       nn.Module,
    support_dir: str,
    device:      torch.device,
    n_shot:      int = 4,
) -> Optional[list[list[torch.Tensor]]]:
    """
    support_dir/<cat>/train/good/ 내 이미지로 위치별(P0~P3) 메모리 뱅크 구성.

    Returns:
        memory_banks[pidx][layer_idx] = [M, C] or None
    """
    _inner = inner(model)
    _inner.eval()
    _inner.backbone.eval()

    n_layers = len(_inner.backbone.layers)
    banks: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(n_layers)] for _ in range(N_PATCHES)
    ]
    total_ok = 0

    for cat_dir in sorted(Path(support_dir).iterdir()):
        if not cat_dir.is_dir():
            continue
        good_dir = cat_dir / "train" / "good"
        if not good_dir.exists():
            continue

        img_paths = sorted(
            list(good_dir.glob("*.png")) +
            list(good_dir.glob("*.jpg")) +
            list(good_dir.glob("*.JPG"))
        )[:n_shot]

        if not img_paths:
            continue

        total_ok += len(img_paths)
        for img_path in tqdm(img_paths, desc=f"  [MemBank] {cat_dir.name}", leave=False):
            img = Image.open(img_path).convert("RGB")
            for pidx in range(N_PATCHES):
                patch   = crop_patch(img, pidx)
                patch_t = _EVAL_TF(patch).unsqueeze(0).to(device)
                _, _, pf_t = _inner(patch_t)
                for lidx, pf in enumerate(patch_feat_to_list(pf_t)):
                    banks[pidx][lidx].append(pf.squeeze(0).cpu())

    if total_ok == 0:
        print(f"  [MemBank] 지지 이미지 없음: {support_dir}")
        return None

    memory_banks = [
        [torch.cat(banks[pidx][lidx], dim=0) for lidx in range(n_layers)]
        for pidx in range(N_PATCHES)
    ]
    print(f"  [MemBank] 완료 – 위치당 벡터: {memory_banks[0][0].shape[0]}")
    return memory_banks


# ─────────────────────────────────────────────────────────────────────
# Few-shot 점수 계산 (NN 코사인 거리)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_fewshot_score(
    patch_feats_norm: list[torch.Tensor],
    memory_bank:      list[torch.Tensor],
    device:           torch.device,
) -> torch.Tensor:
    """
    쿼리 패치 vs 메모리 뱅크 정상 패치의 NN 코사인 거리.

    score_patch = 1 - max_over_m { cosine_sim(q, m) }

    Returns:
        [B, N] – 레이어 평균 이상 점수 (높을수록 이상)
    """
    layer_scores = []
    for q_feat, m_feat in zip(patch_feats_norm, memory_bank):
        q   = q_feat.to(device)
        m   = m_feat.to(device)
        sim = torch.einsum("bnc,mc->bnm", q, m)
        max_sim, _ = sim.max(dim=-1)
        layer_scores.append(1.0 - max_sim)
    return torch.stack(layer_scores, dim=1).mean(dim=1)
