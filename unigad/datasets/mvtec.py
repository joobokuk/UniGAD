"""
unigad/datasets/mvtec.py
-------------------------
MVTec AD 데이터셋 및 MVTec 포맷을 따르는 데이터셋(JVM_mvtec 등) 로더.

디렉토리 구조:
    <root>/
      <category>/
        train/
          good/  *.png
        test/
          good/          *.png   (정상, label=0)
          <defect_type>/ *.png   (이상, label=1)
        ground_truth/
          <defect_type>/ *_mask.png
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from unigad.transforms import make_eval_transform, IMG_SIZE_DINOV3


class MVTecADDataset(Dataset):
    """
    Args:
        root     : 루트 경로 (MVTec 또는 JVM_mvtec)
        category : 평가할 단일 카테고리
        split    : 'train' 또는 'test'
        transform: 이미지 변환 (기본: eval transform)
        img_size : 기본 마스크 크기 (transform 없을 때 사용)
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "test",
        transform=None,
        img_size: int = IMG_SIZE_DINOV3,
    ):
        from unigad.transforms import make_mask_transform
        self.transform      = transform or make_eval_transform(img_size)
        self.mask_transform = make_mask_transform(img_size)
        self.img_size       = img_size
        self.samples: list[tuple[str, int, str | None]] = []

        split_dir = Path(root) / category / split
        if not split_dir.exists():
            raise RuntimeError(f"MVTec 경로를 찾을 수 없습니다: {split_dir}")

        for subdir in sorted(split_dir.iterdir()):
            if not subdir.is_dir():
                continue
            label = 0 if subdir.name == "good" else 1
            for p in sorted(subdir.glob("*.png")):
                if label == 0:
                    self.samples.append((str(p), 0, None))
                else:
                    gt = Path(root) / category / "ground_truth" / subdir.name / (p.stem + "_mask.png")
                    self.samples.append((str(p), 1, str(gt) if gt.exists() else None))

        print(f"[MVTec-{category}/{split}] samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))

        if mask_path and os.path.exists(mask_path):
            mask = self.mask_transform(Image.open(mask_path).convert("L")).squeeze(0)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(self.img_size, self.img_size)

        return img, torch.tensor(label, dtype=torch.long), mask
