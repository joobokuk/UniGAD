"""
unigad/datasets/btad.py
------------------------
BTAD (BeanTech Anomaly Detection) 데이터셋 로더.

MVTec과 다른 점:
  - 정상 폴더: good → ok
  - 이상 폴더: <defect_type> → ko (단일 폴더)
  - 마스크 파일명: <stem>_mask.png → <stem>.png (접미사 없음)
  - 이미지 확장자: .png → .bmp

디렉토리 구조:
    <root>/
      <category>/   (예: 01, 02, 03)
        train/
          ok/  *.bmp
        test/
          ok/  *.bmp   (label=0)
          ko/  *.bmp   (label=1)
        ground_truth/
          ko/  *.png
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from unigad.transforms import IMG_SIZE_DINOV3

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


class BTADDataset(Dataset):
    """
    Args:
        root     : BTAD 루트 경로
        category : 평가할 단일 카테고리 (예: "01")
        split    : 'train' 또는 'test'
        transform: 이미지 변환
        img_size : 기본 마스크 크기
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "test",
        transform=None,
        img_size: int = IMG_SIZE_DINOV3,
    ):
        from unigad.transforms import make_eval_transform, make_mask_transform
        self.transform      = transform or make_eval_transform(img_size)
        self.mask_transform = make_mask_transform(img_size)
        self.img_size       = img_size
        self.samples: list[tuple[str, int, str | None]] = []

        root      = Path(root)
        split_dir = root / category / split
        if not split_dir.exists():
            raise RuntimeError(f"BTAD 경로를 찾을 수 없습니다: {split_dir}")

        if split == "train":
            for p in sorted((split_dir / "ok").iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), 0, None))
        else:
            ok_dir = split_dir / "ok"
            ko_dir = split_dir / "ko"
            gt_dir = root / category / "ground_truth" / "ko"

            if ok_dir.exists():
                for p in sorted(ok_dir.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        self.samples.append((str(p), 0, None))

            if ko_dir.exists():
                for p in sorted(ko_dir.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        mask_p = gt_dir / (p.stem + ".png")
                        self.samples.append((str(p), 1, str(mask_p) if mask_p.exists() else None))

        print(f"[BTAD-{category}/{split}] samples: {len(self.samples)}")

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
