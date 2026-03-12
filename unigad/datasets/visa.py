"""
unigad/datasets/visa.py
------------------------
VisA 데이터셋 로더.

디렉토리 구조:
    <root>/
      <category>/
        Data/
          Images/
            Normal/  *.JPG
            Anomaly/ *.JPG
          Masks/
            Anomaly/ *.png
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from unigad.transforms import IMG_SIZE_DINOV3


class VisADataset(Dataset):
    """
    Args:
        root       : VisA 루트 경로
        categories : 학습할 카테고리 목록 (None이면 전체)
        transform  : 이미지 변환
        img_size   : 기본 마스크 크기
    """

    def __init__(
        self,
        root: str,
        categories=None,
        transform=None,
        img_size: int = IMG_SIZE_DINOV3,
    ):
        from unigad.transforms import make_eval_transform, make_mask_transform
        self.transform      = transform or make_eval_transform(img_size)
        self.mask_transform = make_mask_transform(img_size)
        self.img_size       = img_size
        self.samples: list[tuple[str, int, str | None]] = []

        root = Path(root)
        cats = sorted(os.listdir(root)) if categories is None else categories

        for cat in cats:
            normal_dir  = root / cat / "Data" / "Images" / "Normal"
            anomaly_dir = root / cat / "Data" / "Images" / "Anomaly"
            mask_dir    = root / cat / "Data" / "Masks"  / "Anomaly"

            if normal_dir.exists():
                for p in sorted(normal_dir.glob("*")):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.samples.append((str(p), 0, None))

            if anomaly_dir.exists():
                for p in sorted(anomaly_dir.glob("*")):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        mask_p = mask_dir / (p.stem + ".png")
                        self.samples.append((str(p), 1, str(mask_p) if mask_p.exists() else None))

        if not self.samples:
            raise RuntimeError(f"VisA 데이터를 찾을 수 없습니다: {root}")
        print(f"[VisA] samples: {len(self.samples)}")

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
