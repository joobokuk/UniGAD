#!/usr/bin/env python3
"""
scripts/generate_patch_heatmap.py
===================================
Patch-Crop 방식으로 Custom(MVTec 포맷) 데이터셋의 이상 히트맵을 생성한다.

원본 1024×1024 이미지를 576×576 패치 4장(128px 중첩)으로 나누어
각 패치를 독립 추론 후 1024×1024 히트맵으로 stitch 저장.

추론 모드:
  zero_shot      : W_cls / W_seg 점수만 사용
  few_shot       : 위치별 메모리 뱅크 기반 NN 점수만 사용
  both (기본)    : zero_shot + few_shot 합산

few-shot support 이미지:
  기본값 : --custom_root 내 각 카테고리 train/good/
  golden : --golden_root 지정 시 해당 경로 사용

출력 구조:
  <output_root>/
    <mode>/                    (zero_shot | few_shot | both)
      <category>/
        <defect_type>/
          <stem>_heatmap.png

사용 예시:
  # zero-shot 히트맵
  python scripts/generate_patch_heatmap.py \\
      --custom_root /path/to/Custom \\
      --ckpt_path   checkpoints/ckpt_custom_patch.pth \\
      --mode        zero_shot

  # golden template few-shot 히트맵
  python scripts/generate_patch_heatmap.py \\
      --custom_root /path/to/Custom \\
      --golden_root /path/to/CustomGolden \\
      --ckpt_path   checkpoints/ckpt_custom_patch.pth \\
      --mode        both

  # 특정 카테고리만
  python scripts/generate_patch_heatmap.py \\
      --custom_root /path/to/Custom \\
      --ckpt_path   checkpoints/ckpt_custom_patch.pth \\
      --categories  medication_pouch_left__bright
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from unigad.models.uniadet      import UniADet
from unigad.models.multigpu     import wrap_multigpu, inner
from unigad.engine.memory_bank  import build_memory_banks_per_pos
from unigad.transforms          import EXTRACT_LAYERS, PATCH_SIZE_DINOV3
from unigad.utils.checkpoint    import load_ckpt
from unigad.utils.patch         import (
    crop_patch, stitch_heatmaps, seg_score_to_heatmap,
    save_jet_heatmap, N_PATCHES, FINAL_SIZE,
)
from unigad.engine.memory_bank  import compute_fewshot_score
from unigad.models.multigpu     import patch_feat_to_list

BASE      = Path(__file__).parent.parent
DATA_ROOT = BASE.parent / "Data"

_EVAL_TF = transforms.Compose([
    transforms.Resize((FINAL_SIZE, FINAL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 데이터 경로
    p.add_argument("--custom_root",    required=True,
                   help="Custom 데이터셋 루트 (MVTec 포맷)")
    p.add_argument("--golden_root",    default=None,
                   help="few-shot support용 Golden Template 루트 (미지정 시 custom_root 사용)")
    p.add_argument("--categories",     nargs="+", default=None,
                   help="처리할 카테고리 목록 (미지정 시 자동 탐색)")
    p.add_argument("--target",         default="anomaly",
                   choices=["anomaly", "normal", "all"],
                   help="히트맵을 생성할 이미지 대상")

    # 체크포인트
    p.add_argument("--ckpt_path",      required=True,
                   help="학습된 가중치 경로 (.pth)")

    # 추론 모드
    p.add_argument("--mode",           default="both",
                   choices=["zero_shot", "few_shot", "both"],
                   help="추론 모드")
    p.add_argument("--n_shot",         type=int, default=4,
                   help="few-shot 위치별 메모리 뱅크에 사용할 support 이미지 수")

    # 출력
    p.add_argument("--output_root",    default="outputs/patch_heatmaps",
                   help="히트맵 저장 루트 디렉토리")

    # 백본
    p.add_argument("--backbone",       default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--dinov3_repo",    default=str(BASE / "dinov3"))
    p.add_argument("--dinov3_weights", default=str(
        BASE / "dinov3" / "pretrained"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--layers",         nargs="+", type=int, default=EXTRACT_LAYERS)

    return p.parse_args()


def discover_categories(root: Path) -> list[str]:
    """test/ 폴더가 있는 하위 디렉토리를 카테고리로 간주."""
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / "test").exists())


@torch.no_grad()
def process_category(
    model:        torch.nn.Module,
    cat_dir:      Path,
    args,
    device:       torch.device,
    memory_banks: list | None,
):
    """단일 카테고리의 히트맵 생성."""
    _inner = inner(model)
    _inner.eval()
    _inner.backbone.eval()

    test_dir = cat_dir / "test"
    if not test_dir.exists():
        print(f"  [Skip] test/ 없음: {cat_dir}")
        return

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        is_anomaly = cls_dir.name != "good"

        if args.target == "anomaly" and not is_anomaly:
            continue
        if args.target == "normal" and is_anomaly:
            continue

        img_paths = sorted(
            list(cls_dir.glob("*.png")) +
            list(cls_dir.glob("*.jpg")) +
            list(cls_dir.glob("*.JPG"))
        )
        if not img_paths:
            continue

        out_dir = Path(args.output_root) / args.mode / cat_dir.name / cls_dir.name

        for img_path in tqdm(img_paths,
                             desc=f"  [{cat_dir.name}/{cls_dir.name}]"):
            img = Image.open(img_path).convert("RGB")

            patch_img_scores: list[float]      = []
            patch_hmaps:      list[np.ndarray] = []

            for pidx in range(N_PATCHES):
                patch   = crop_patch(img, pidx)
                patch_t = _EVAL_TF(patch).unsqueeze(0).to(device)

                cls_logit, seg_logit, pf_t = _inner(patch_t)
                pf_list     = patch_feat_to_list(pf_t)
                img_prob_zs = F.softmax(cls_logit, dim=-1)[0, 1].item()
                seg_prob_zs = F.softmax(seg_logit, dim=-1)[0, :, 1]

                if memory_banks is not None and args.mode != "zero_shot":
                    seg_score_fs = compute_fewshot_score(
                        pf_list, memory_banks[pidx], device
                    )[0]
                    if args.mode == "few_shot":
                        final_seg = seg_score_fs
                        img_score = final_seg.max().item()
                    else:  # both
                        final_seg = seg_prob_zs + seg_score_fs
                        img_score = img_prob_zs + final_seg.max().item()
                else:
                    final_seg = seg_prob_zs
                    img_score = 0.5 * img_prob_zs + 0.5 * final_seg.max().item()

                patch_img_scores.append(img_score)
                patch_hmaps.append(seg_score_to_heatmap(final_seg))

            full_hmap = stitch_heatmaps(patch_hmaps)
            save_jet_heatmap(full_hmap, out_dir / f"{img_path.stem}_heatmap.png")

    print(f"  저장 완료: {Path(args.output_root) / args.mode / cat_dir.name}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device} | GPU: {torch.cuda.device_count()}")

    custom_root = Path(args.custom_root)
    categories  = args.categories or discover_categories(custom_root)
    if not categories:
        raise RuntimeError(f"카테고리를 찾을 수 없습니다: {custom_root}")
    print(f"카테고리: {categories}")
    print(f"모드: {args.mode} | 대상: {args.target} | n_shot: {args.n_shot}")

    # ── 모델 로드 ────────────────────────────────────────────────
    print("\n[Init] 모델 로딩...")
    model = UniADet(
        layers=args.layers, backbone=args.backbone,
        dinov3_repo=args.dinov3_repo, dinov3_weights=args.dinov3_weights,
        patch_size=PATCH_SIZE_DINOV3,
    )
    model = wrap_multigpu(model)
    load_ckpt(model, args.ckpt_path)
    model.to(device)

    # ── 위치별 메모리 뱅크 구성 (few_shot / both) ────────────────
    memory_banks = None
    if args.mode in ("few_shot", "both"):
        support_root = args.golden_root or args.custom_root
        src_label    = "Golden" if args.golden_root else "custom_root train/good"
        print(f"\n[MemBank] Support 소스: {src_label} ({support_root})")
        memory_banks = build_memory_banks_per_pos(
            model, support_root, device, args.n_shot
        )
        if memory_banks is None:
            print("[MemBank] 메모리 뱅크 구성 실패 → zero_shot 모드로 대체")
            args.mode = "zero_shot"

    # ── 카테고리별 히트맵 생성 ────────────────────────────────────
    for cat_name in categories:
        cat_dir = custom_root / cat_name
        if not cat_dir.is_dir():
            print(f"[Skip] 경로 없음: {cat_dir}")
            continue
        print(f"\n{'─'*55}")
        print(f"  카테고리: {cat_name}")
        print(f"{'─'*55}")
        process_category(model, cat_dir, args, device, memory_banks)

    print(f"\n히트맵 생성 완료 → {Path(args.output_root) / args.mode}")


if __name__ == "__main__":
    main()
