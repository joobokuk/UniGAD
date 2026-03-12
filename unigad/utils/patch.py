"""
unigad/utils/patch.py
----------------------
JVM patch-crop 관련 상수 및 유틸리티 함수.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ORIG_SIZE  = 1024
CROP_SIZE  = 576
STRIDE     = 448
FINAL_SIZE = 448
N_PATCHES  = 4

PATCH_BBOXES: list[tuple[int, int, int, int]] = [
    (0,      CROP_SIZE, 0,      CROP_SIZE),
    (0,      CROP_SIZE, STRIDE, ORIG_SIZE),
    (STRIDE, ORIG_SIZE, 0,      CROP_SIZE),
    (STRIDE, ORIG_SIZE, STRIDE, ORIG_SIZE),
]


def crop_patch(img: Image.Image, pidx: int) -> Image.Image:
    """PIL 이미지에서 pidx번 패치(576×576) 크롭."""
    rs, re, cs, ce = PATCH_BBOXES[pidx]
    return img.crop((cs, rs, ce, re))


def stitch_heatmaps(patch_hmaps: list[np.ndarray]) -> np.ndarray:
    """
    4개 패치 히트맵(각 576×576 numpy) → 1024×1024 stitch.
    중첩 128px 영역은 관여한 패치들의 평균값 사용.
    """
    acc   = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.float32)
    count = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.float32)
    for pidx, hmap in enumerate(patch_hmaps):
        rs, re, cs, ce = PATCH_BBOXES[pidx]
        h_t, w_t = re - rs, ce - cs
        if hmap.shape != (h_t, w_t):
            t    = torch.from_numpy(hmap).float().unsqueeze(0).unsqueeze(0)
            hmap = F.interpolate(t, size=(h_t, w_t),
                                 mode="bilinear", align_corners=False).squeeze().numpy()
        acc[rs:re, cs:ce]   += hmap
        count[rs:re, cs:ce] += 1.0
    return acc / np.maximum(count, 1e-8)


def seg_score_to_heatmap(seg_score: torch.Tensor) -> np.ndarray:
    """[N] 패치 이상 점수 → [CROP_SIZE, CROP_SIZE] numpy 히트맵."""
    n_side = int(seg_score.numel() ** 0.5)
    t = seg_score.view(1, 1, n_side, n_side)
    t = F.interpolate(t, size=(CROP_SIZE, CROP_SIZE),
                      mode="bilinear", align_corners=False)
    return t.squeeze().cpu().numpy()


def save_jet_heatmap(score_map: np.ndarray, out_path) -> None:
    """numpy [H, W] float → JET 컬러 PNG 저장."""
    from pathlib import Path
    vmin, vmax = float(score_map.min()), float(score_map.max())
    norm = (score_map - vmin) / (vmax - vmin + 1e-8)
    v    = norm
    rgb  = np.zeros((*norm.shape, 3), dtype=np.float32)

    m0 = v < 0.25
    m1 = (v >= 0.25) & (v < 0.5)
    m2 = (v >= 0.5)  & (v < 0.75)
    m3 = v >= 0.75

    rgb[m0, 0] = 0.0;              rgb[m0, 1] = 4.0*v[m0];              rgb[m0, 2] = 1.0
    rgb[m1, 0] = 0.0;              rgb[m1, 1] = 1.0;                     rgb[m1, 2] = 1.0 - 4.0*(v[m1]-0.25)
    rgb[m2, 0] = 4.0*(v[m2]-0.5); rgb[m2, 1] = 1.0;                     rgb[m2, 2] = 0.0
    rgb[m3, 0] = 1.0;              rgb[m3, 1] = 1.0 - 4.0*(v[m3]-0.75); rgb[m3, 2] = 0.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), "RGB").save(out_path)
