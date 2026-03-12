"""
unigad/engine/evaluate.py
--------------------------
표준 평가 루프 (evaluate_uniadet) 및 JVM patch 평가 루프 (eval_jvm_patch).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from unigad.models.multigpu import inner, patch_feat_to_list
from unigad.transforms      import PATCH_SIZE_DINOV3
from unigad.utils.patch     import (
    crop_patch, stitch_heatmaps, seg_score_to_heatmap,
    save_jet_heatmap, N_PATCHES, ORIG_SIZE, FINAL_SIZE,
)
from unigad.engine.memory_bank import compute_fewshot_score

from torchvision import transforms

_EVAL_TF = transforms.Compose([
    transforms.Resize((FINAL_SIZE, FINAL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────
# 표준 평가 (전체 이미지 단위)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_uniadet(
    model:       nn.Module,
    dataloader:  DataLoader,
    device:      torch.device,
    memory_bank: Optional[list[torch.Tensor]] = None,
    category:    str = "unknown",
    patch_size:  int = PATCH_SIZE_DINOV3,
) -> dict:
    """
    Zero-shot 또는 Few-shot 평가.

    Returns:
        dict(category, mode, img_auroc, img_aupr, pix_auroc, pix_aupr)
    """
    _inner = inner(model)
    _inner.eval()
    _inner.backbone.eval()

    all_img_scores   = []
    all_img_labels   = []
    all_pixel_scores = []
    all_pixel_masks  = []
    mode_str = "Few-shot" if memory_bank is not None else "Zero-shot"

    print(f"\n[Eval] {mode_str} – category: {category}")

    for imgs, labels, masks in tqdm(dataloader, desc=f"[Eval-{category}]"):
        imgs = imgs.to(device)
        B, _, H, W = imgs.shape
        H_p = H // patch_size
        W_p = W // patch_size

        cls_logit, seg_logit, pf = _inner(imgs)
        patch_feats = patch_feat_to_list(pf)

        img_score_zs = F.softmax(cls_logit, dim=-1)[:, 1]
        seg_score_zs = F.softmax(seg_logit, dim=-1)[..., 1]

        if memory_bank is not None:
            seg_score_fs = compute_fewshot_score(patch_feats, memory_bank, device)
            final_seg    = seg_score_zs + seg_score_fs
            final_img    = img_score_zs + final_seg.max(dim=1).values
        else:
            final_seg = seg_score_zs
            final_img = 0.5 * img_score_zs + 0.5 * final_seg.max(dim=1).values

        seg_map = final_seg.view(B, 1, H_p, W_p)
        seg_map = F.interpolate(seg_map, size=(H, W), mode="bilinear", align_corners=False)

        all_img_scores.append(final_img.cpu().numpy())
        all_img_labels.append(labels.numpy())
        all_pixel_scores.append(seg_map.squeeze(1).cpu().numpy())
        all_pixel_masks.append(masks.numpy())

    arr_img_s = np.concatenate(all_img_scores).flatten()
    arr_img_l = np.concatenate(all_img_labels).flatten()
    arr_pix_s = np.concatenate(all_pixel_scores).flatten()
    arr_pix_m = np.concatenate(all_pixel_masks).flatten()

    img_auroc = roc_auc_score(arr_img_l, arr_img_s)
    img_aupr  = average_precision_score(arr_img_l, arr_img_s)
    if arr_pix_m.sum() > 0:
        pix_auroc = roc_auc_score(arr_pix_m, arr_pix_s)
        pix_aupr  = average_precision_score(arr_pix_m, arr_pix_s)
    else:
        pix_auroc = pix_aupr = float("nan")

    print(f"  [{category}] ImgAUROC={img_auroc:.4f} ImgAUPR={img_aupr:.4f} "
          f"PixAUROC={pix_auroc:.4f} PixAUPR={pix_aupr:.4f}")

    return {
        "category":  category,
        "mode":      mode_str,
        "img_auroc": img_auroc,
        "img_aupr":  img_aupr,
        "pix_auroc": pix_auroc,
        "pix_aupr":  pix_aupr,
    }


# ─────────────────────────────────────────────────────────────────────
# JVM Patch-Crop 평가
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_jvm_patch(
    model:            nn.Module,
    jvm_root:         str,
    device:           torch.device,
    memory_banks:     Optional[list[list[torch.Tensor]]] = None,
    mode_name:        str = "zero_shot",
    save_heatmap_dir: Optional[str] = None,
) -> dict:
    """
    JVM_mvtec 테스트셋 전체 평가 (patch-crop 기반).

    이미지 점수: 4 패치 max 집계
    픽셀 히트맵: stitch → 1024×1024 vs GT 마스크
    """
    _inner = inner(model)
    _inner.eval()
    _inner.backbone.eval()

    all_img_scores: list[float]      = []
    all_img_labels: list[int]        = []
    all_pix_scores: list[np.ndarray] = []
    all_pix_masks:  list[np.ndarray] = []

    for cat_dir in sorted(Path(jvm_root).iterdir()):
        if not cat_dir.is_dir():
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue

        for cls_dir in sorted(test_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            img_label = 0 if cls_dir.name == "good" else 1

            for img_path in tqdm(sorted(cls_dir.glob("*.png")),
                                 desc=f"  [{mode_name}] {cat_dir.name}/{cls_dir.name}"):
                img = Image.open(img_path).convert("RGB")

                if img_label == 1:
                    gt_path = (cat_dir / "ground_truth" / cls_dir.name
                               / (img_path.stem + "_mask.png"))
                    gt_mask = (np.array(Image.open(gt_path).convert("L")) > 0
                               if gt_path.exists()
                               else np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=bool))
                else:
                    gt_mask = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=bool)

                patch_img_scores: list[float]      = []
                patch_hmaps:      list[np.ndarray] = []

                for pidx in range(N_PATCHES):
                    patch   = crop_patch(img, pidx)
                    patch_t = _EVAL_TF(patch).unsqueeze(0).to(device)

                    cls_logit, seg_logit, pf_t = _inner(patch_t)
                    pf_list     = patch_feat_to_list(pf_t)
                    img_prob_zs = F.softmax(cls_logit, dim=-1)[0, 1].item()
                    seg_prob_zs = F.softmax(seg_logit, dim=-1)[0, :, 1]

                    if memory_banks is not None:
                        seg_score_fs = compute_fewshot_score(
                            pf_list, memory_banks[pidx], device
                        )[0]
                        final_seg = seg_prob_zs + seg_score_fs
                        img_score = img_prob_zs + final_seg.max().item()
                    else:
                        final_seg = seg_prob_zs
                        img_score = 0.5 * img_prob_zs + 0.5 * final_seg.max().item()

                    patch_img_scores.append(img_score)
                    patch_hmaps.append(seg_score_to_heatmap(final_seg))

                final_img = max(patch_img_scores)
                full_hmap = stitch_heatmaps(patch_hmaps)

                all_img_scores.append(final_img)
                all_img_labels.append(img_label)
                all_pix_scores.append(full_hmap.flatten())
                all_pix_masks.append(gt_mask.flatten().astype(np.float32))

                if save_heatmap_dir:
                    hdir = Path(save_heatmap_dir) / mode_name / cat_dir.name / cls_dir.name
                    save_jet_heatmap(full_hmap, hdir / f"{img_path.stem}_heatmap.png")

    arr_img_s = np.array(all_img_scores)
    arr_img_l = np.array(all_img_labels)
    arr_pix_s = np.concatenate(all_pix_scores)
    arr_pix_m = np.concatenate(all_pix_masks)

    img_auroc = roc_auc_score(arr_img_l, arr_img_s)
    img_aupr  = average_precision_score(arr_img_l, arr_img_s)
    if arr_pix_m.sum() > 0:
        pix_auroc = roc_auc_score(arr_pix_m, arr_pix_s)
        pix_aupr  = average_precision_score(arr_pix_m, arr_pix_s)
    else:
        pix_auroc = pix_aupr = float("nan")

    print(f"  [{mode_name}] ImgAUROC={img_auroc:.4f} ImgAUPR={img_aupr:.4f} "
          f"PixAUROC={pix_auroc:.4f} PixAUPR={pix_aupr:.4f}")

    return {
        "mode":      mode_name,
        "img_auroc": img_auroc,
        "img_aupr":  img_aupr,
        "pix_auroc": pix_auroc,
        "pix_aupr":  pix_aupr,
    }
