"""
unigad/datasets/jvm_patch.py
-----------------------------
JVM_mvtec 전용 Patch-Crop 학습 데이터셋.

패치 레이블 부여 규칙:
  - 정상 이미지의 4패치 → label=0
  - 이상 이미지의 각 패치:
      마스크 크롭이 완전 검정(all-zero)이면 label=0
      1픽셀이라도 이상 있으면 label=1
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from unigad.utils.patch import PATCH_BBOXES, N_PATCHES, FINAL_SIZE, crop_patch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TF = transforms.Compose([
    transforms.Resize((FINAL_SIZE, FINAL_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_MASK_TF = transforms.Compose([
    transforms.Resize((FINAL_SIZE, FINAL_SIZE), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])


class JVMPatchTrainDataset(Dataset):
    """
    Args:
        root     : JVM_mvtec 루트 경로
        use_test : True 이면 test 이미지도 학습에 포함 (이상 감독 학습)
    """

    def __init__(self, root: str, use_test: bool = False):
        self.root     = Path(root)
        self.use_test = use_test
        self.samples: list[tuple] = []  # (img_path, pidx, label, mask_path|None)
        self._build()

    def _collect_raw(self) -> list[tuple]:
        raw: list[tuple] = []
        for cat_dir in sorted(self.root.iterdir()):
            if not cat_dir.is_dir():
                continue

            good_dir = cat_dir / "train" / "good"
            if good_dir.exists():
                for p in sorted(good_dir.glob("*.png")):
                    raw.append((str(p), 0, None))

            if self.use_test:
                test_dir = cat_dir / "test"
                if test_dir.exists():
                    for cls_dir in sorted(test_dir.iterdir()):
                        if not cls_dir.is_dir():
                            continue
                        is_good = cls_dir.name == "good"
                        for p in sorted(cls_dir.glob("*.png")):
                            if is_good:
                                raw.append((str(p), 0, None))
                            else:
                                gt = cat_dir / "ground_truth" / cls_dir.name / (p.stem + "_mask.png")
                                raw.append((str(p), 1, str(gt) if gt.exists() else None))
        return raw

    def _build(self):
        raw = self._collect_raw()
        for img_path, img_label, mask_path in raw:
            if img_label == 0:
                for pidx in range(N_PATCHES):
                    self.samples.append((img_path, pidx, 0, None))
            else:
                mask_full: Optional[Image.Image] = None
                if mask_path and os.path.exists(mask_path):
                    mask_full = Image.open(mask_path).convert("L")

                for pidx in range(N_PATCHES):
                    if mask_full is not None:
                        pm          = crop_patch(mask_full, pidx)
                        patch_label = 1 if (np.array(pm) > 0).any() else 0
                    else:
                        patch_label = 0
                    self.samples.append((img_path, pidx, patch_label, mask_path))

        n_anom = sum(1 for s in self.samples if s[2] == 1)
        n_norm = len(self.samples) - n_anom
        print(f"[JVMPatchDataset] 전체: {len(self.samples)}, 정상: {n_norm}, 이상: {n_anom}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pidx, label, mask_path = self.samples[idx]

        img   = Image.open(img_path).convert("RGB")
        patch = crop_patch(img, pidx)
        patch = _TRAIN_TF(patch)

        if mask_path and os.path.exists(mask_path):
            mask_full  = Image.open(mask_path).convert("L")
            patch_mask = _MASK_TF(crop_patch(mask_full, pidx)).squeeze(0)
            patch_mask = (patch_mask > 0.5).float()
        else:
            patch_mask = torch.zeros(FINAL_SIZE, FINAL_SIZE)

        return patch, torch.tensor(label, dtype=torch.long), patch_mask


def make_weighted_sampler(dataset: JVMPatchTrainDataset) -> WeightedRandomSampler:
    """anomaly:normal = 1:1 비율 WeightedRandomSampler."""
    labels = [s[2] for s in dataset.samples]
    n_anom = max(sum(labels), 1)
    n_norm = max(len(labels) - n_anom, 1)
    weights = [1.0 / n_anom if l == 1 else 1.0 / n_norm for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
