"""
transform_masking.py
===================

목적:
  VisA 데이터셋 루트를 입력받아,
  각 클래스(카테고리)별 Data/Masks/Anomaly 경로에 있는 마스크 이미지를
  이진화(0 → 0, 0이 아닌 값 → 255)하여 outputs 아래에 저장.

VisA 구조:
  <visa_root>/
    <category>/
      Data/
        Masks/
          Anomaly/   *.png  ← 여기만 대상

출력:
  outputs/binarized_masks/<category>/Data/Masks/Anomaly/ (기본)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# VisA 내 마스크 경로 (카테고리 기준 상대 경로)
MASKS_ANOMALY_SUBDIR = Path("Data") / "Masks" / "Anomaly"


def discover_visa_categories(visa_root: Path) -> list[str]:
    """VisA 루트에서 Data/ 가 있는 카테고리 폴더 목록 반환."""
    cats: list[str] = []
    if not visa_root.exists():
        return cats
    for d in sorted(visa_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "Data").exists():
            cats.append(d.name)
    return cats


def collect_masks_per_category(visa_root: Path, categories: list[str]) -> list[tuple[str, Path]]:
    """
    (category, image_path) 쌍 목록 반환.
    각 카테고리의 <visa_root>/<cat>/Data/Masks/Anomaly/* 이미지만 수집.
    """
    pairs: list[tuple[str, Path]] = []
    for cat in categories:
        mask_dir = visa_root / cat / MASKS_ANOMALY_SUBDIR
        if not mask_dir.exists():
            continue
        for p in sorted(mask_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                pairs.append((cat, p))
    return pairs


def binarize_mask(arr: np.ndarray) -> np.ndarray:
    """
    픽셀 값이 0이면 0, 0이 아니면 255로 이진화.
    - arr: [H,W] 또는 [H,W,C] (grayscale / RGB / RGBA 등)
    - 반환: [H,W] uint8, 0 또는 255
    """
    if arr.ndim == 3:
        nonzero = (arr != 0).any(axis=-1)
    else:
        nonzero = arr != 0
    return np.where(nonzero, 255, 0).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisA 각 클래스별 Masks/Anomaly 이미지를 이진화하여 저장"
    )
    parser.add_argument(
        "--visa_root",
        type=str,
        required=True,
        help="VisA 데이터셋 루트 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/binarized_masks",
        help="이진 마스크 저장 루트 (기본: outputs/binarized_masks)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="처리할 카테고리 목록 (미지정 시 visa_root에서 자동 탐색)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visa_root = Path(args.visa_root)
    output_root = Path(args.output_dir)

    if not visa_root.exists():
        raise FileNotFoundError(f"VisA 루트가 없습니다: {visa_root}")
    if not visa_root.is_dir():
        raise NotADirectoryError(f"VisA 루트가 폴더가 아닙니다: {visa_root}")

    categories = args.categories or discover_visa_categories(visa_root)
    if not categories:
        print(f"[결과] VisA 카테고리를 찾지 못했습니다: {visa_root}")
        return

    pairs = collect_masks_per_category(visa_root, categories)
    if not pairs:
        print(
            f"[결과] 각 카테고리의 {MASKS_ANOMALY_SUBDIR} 아래 이미지가 없습니다. "
            f"카테고리: {categories}"
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    errors = []

    for cat, p in tqdm(pairs, desc="이진화"):
        try:
            with Image.open(p) as im:
                arr = np.asarray(im)
            bin_arr = binarize_mask(arr)
            out_im = Image.fromarray(bin_arr, mode="L")

            # 출력: output_root / <category> / Data / Masks / Anomaly / <filename>
            out_path = output_root / cat / MASKS_ANOMALY_SUBDIR / p.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_im.save(out_path)
            n_ok += 1
        except Exception as e:
            errors.append((str(p), str(e)))

    print(f"\n저장 완료: {n_ok}개 → {output_root}")
    if errors:
        print(f"실패: {len(errors)}개")
        for path, msg in errors[:10]:
            print(f"  - {path}: {msg}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개")


if __name__ == "__main__":
    main()

# python transform_masking.py --visa_root /home/user/jupyter/bk/paperworks/UniADet/VisA
# python transform_masking.py --visa_root /home/user/jupyter/bk/paperworks/UniADet/VisA --output_dir outputs/my_binary
# python transform_masking.py --visa_root /home/user/jupyter/bk/paperworks/UniADet/VisA --categories candle capsules