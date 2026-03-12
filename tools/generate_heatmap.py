"""
generate_heatmap.py
====================

목적:
  UniADet (uniadet_paper_aligned) 학습 가중치를 로드하여
  MVTec / VisA / BTAD 어떤 포맷의 데이터셋에도 이상 히트맵을 생성한다.

포맷 자동 감지 규칙:
  - <root>/<cat>/Data/               → visa
  - <root>/<cat>/test/ko/            → btad
  - <root>/<cat>/test/               → mvtec
  --dataset_format 인자로 명시적으로 지정할 수도 있다.

few-shot 지지 이미지:
  기본적으로 각 데이터셋의 정상(train) 이미지를 사용한다.
  --support_root 를 지정하면 해당 경로의 train/good/ 이미지를 메모리 뱅크로 사용한다.
  (예: JVM Golden Template)

출력:
  --output_root/<category>/<original_stem>_heatmap.png  (JET 컬러 히트맵)

사용 예시:
  # MVTec 포맷 (JVM_mvtec)
  python generate_heatmap.py \\
      --dataset_root  JVM_mvtec \\
      --load_path     checkpoints_260309/ckpt_trained_on_jvm.pth

  # MVTec 포맷 + Golden Template을 few-shot support로
  python generate_heatmap.py \\
      --dataset_root  JVM_mvtec \\
      --support_root  JVM_goldentemplate \\
      --load_path     checkpoints_260309/ckpt_trained_on_jvm.pth \\
      --mode          both

  # VisA
  python generate_heatmap.py \\
      --dataset_root  VisA \\
      --load_path     checkpoints_260309/ckpt_trained_on_visa.pth

  # BTAD
  python generate_heatmap.py \\
      --dataset_root  BTAD \\
      --load_path     checkpoints_260309/ckpt_trained_on_btad.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image

import uniadet_paper_aligned as paper


# ──────────────────────────────────────────────────────────────────────
# 포맷 자동 감지
# ──────────────────────────────────────────────────────────────────────
def detect_format(root: Path) -> str:
    """
    루트 하위 첫 번째 카테고리 폴더의 구조를 보고 포맷을 추론한다.
    반환값: "visa" | "btad" | "mvtec"
    """
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "Data").exists():
            return "visa"
        if (d / "test" / "ko").exists():
            return "btad"
        if (d / "test").exists():
            return "mvtec"
    return "mvtec"


# ──────────────────────────────────────────────────────────────────────
# 카테고리 탐색
# ──────────────────────────────────────────────────────────────────────
def discover_categories(root: Path, fmt: str) -> list[str]:
    cats = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if fmt == "visa" and (d / "Data").exists():
            cats.append(d.name)
        elif fmt == "btad" and (d / "test" / "ko").exists():
            cats.append(d.name)
        elif fmt == "mvtec" and (d / "test").exists():
            cats.append(d.name)
    return cats


# ──────────────────────────────────────────────────────────────────────
# 데이터셋 로더 팩토리
# ──────────────────────────────────────────────────────────────────────
def load_dataset(root: str, cat: str, fmt: str, split: str = "test"):
    """split: 'test'(평가용) 또는 'train'(support용)"""
    if fmt == "visa":
        return paper.VisADataset(root=root, categories=[cat],
                                 transform=paper.eval_transform)
    elif fmt == "btad":
        return paper.BTADDataset(root=root, category=cat,
                                 split=split, transform=paper.eval_transform)
    else:  # mvtec
        return paper.MVTecADDataset(root=root, category=cat,
                                    split=split, transform=paper.eval_transform)


# ──────────────────────────────────────────────────────────────────────
# JET 히트맵 저장
# ──────────────────────────────────────────────────────────────────────
def save_heatmap(score_map: torch.Tensor, img_path: str, out_dir: Path):
    """
    score_map : [H, W] CPU 텐서
    out_path  : <out_dir>/<original_stem>_heatmap.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(img_path)
    out_path = out_dir / f"{p.stem}_heatmap.png"

    arr = score_map.numpy().astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    norm = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr)

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    v = norm

    m0 = v < 0.25
    m1 = (v >= 0.25) & (v < 0.5)
    m2 = (v >= 0.5)  & (v < 0.75)
    m3 = v >= 0.75

    rgb[m0, 0] = 0.0;               rgb[m0, 1] = 4.0*v[m0];            rgb[m0, 2] = 1.0
    rgb[m1, 0] = 0.0;               rgb[m1, 1] = 1.0;                   rgb[m1, 2] = 1.0 - 4.0*(v[m1]-0.25)
    rgb[m2, 0] = 4.0*(v[m2]-0.5);  rgb[m2, 1] = 1.0;                   rgb[m2, 2] = 0.0
    rgb[m3, 0] = 1.0;               rgb[m3, 1] = 1.0 - 4.0*(v[m3]-0.75); rgb[m3, 2] = 0.0

    Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), "RGB").save(out_path)


# ──────────────────────────────────────────────────────────────────────
# 메모리 뱅크 구성
# ──────────────────────────────────────────────────────────────────────
def build_support_bank(model, support_root: str, cat: str, fmt: str,
                       n_shot: int, device) -> list | None:
    """
    support_root 에서 정상 이미지를 n_shot 장 로드하여 메모리 뱅크를 구성한다.
    """
    try:
        sup_ds = load_dataset(support_root, cat, fmt, split="train")
    except RuntimeError as e:
        print(f"    [few-shot skip] support 데이터 로드 실패: {e}")
        return None

    normal_idx = [i for i, s in enumerate(sup_ds.samples) if s[1] == 0]
    if not normal_idx:
        print(f"    [few-shot skip] 정상 이미지 없음: {cat}")
        return None

    selected = normal_idx[:n_shot]
    if len(selected) < n_shot:
        print(f"    [few-shot] 요청 {n_shot}-shot이지만 {len(selected)}장만 사용합니다.")

    print(f"    [few-shot] 메모리 뱅크 구성: {len(selected)}장 사용")
    sup_dl = DataLoader(Subset(sup_ds, selected), batch_size=len(selected), shuffle=False)
    return paper.build_memory_bank(model, sup_dl, device)


# ──────────────────────────────────────────────────────────────────────
# 단일 카테고리 히트맵 생성
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_for_category(model, cat: str, args, device):
    fmt          = args.dataset_format
    support_fmt  = args.support_format or fmt

    # ── 평가 데이터셋 로드 ────────────────────────────────────
    try:
        full_ds = load_dataset(args.dataset_root, cat, fmt, split="test")
    except RuntimeError as e:
        print(f"  [Skip] {e}")
        return

    # 처리 대상 선택
    target = args.target
    if target == "anomaly":
        indices = [i for i, s in enumerate(full_ds.samples) if s[1] == 1]
    elif target == "normal":
        indices = [i for i, s in enumerate(full_ds.samples) if s[1] == 0]
    else:  # all
        indices = list(range(len(full_ds)))

    if not indices:
        print(f"  [Skip] target='{target}' 이미지가 없습니다: {cat}")
        return

    print(f"  대상 이미지: {len(indices)}장 (target={target})")
    ds = Subset(full_ds, indices)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)

    # ── 메모리 뱅크 구성 (few-shot / both) ────────────────────
    memory_bank = None
    if args.mode in ("few_shot", "both"):
        support_root = args.support_root or args.dataset_root
        memory_bank  = build_support_bank(model, support_root, cat, support_fmt,
                                          args.n_shot, device)
        if memory_bank is None and args.mode == "few_shot":
            print(f"  [Skip] few_shot 모드인데 메모리 뱅크 구성 실패: {cat}")
            return

    # ── 추론 ──────────────────────────────────────────────────
    out_dir   = Path(args.output_root) / cat
    base_idx  = 0

    for imgs, _, _ in dl:
        imgs = imgs.to(device)
        B    = imgs.size(0)
        H, W = imgs.shape[-2], imgs.shape[-1]
        H_p  = H // paper.PATCH_SIZE
        W_p  = W // paper.PATCH_SIZE

        cls_logit, seg_logit, patch_feats = model(imgs)

        # Zero-shot 세그멘테이션 점수
        anomaly_logit = seg_logit[..., 1] - seg_logit[..., 0]  # [B, N]
        seg_score_zs  = torch.sigmoid(anomaly_logit)
        final_score   = seg_score_zs

        # Few-shot 점수
        if memory_bank is not None:
            seg_score_fs = paper.compute_fewshot_score(patch_feats, memory_bank, device)
            if args.mode == "few_shot":
                final_score = seg_score_fs
            else:  # both
                final_score = seg_score_zs + seg_score_fs

        # 패치 해상도 → 원본 해상도 업샘플
        seg_map = final_score.view(B, 1, H_p, W_p)
        seg_map = F.interpolate(seg_map, size=(H, W), mode="bilinear", align_corners=False)

        for b in range(B):
            global_idx = ds.indices[base_idx + b]
            img_path   = full_ds.samples[global_idx][0]
            save_heatmap(seg_map[b, 0].cpu(), img_path, out_dir)

        base_idx += B

    print(f"  저장 완료: {out_dir}")


# ──────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    BASE = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate anomaly heatmaps for MVTec / VisA / BTAD datasets"
    )

    # 데이터셋
    parser.add_argument("--dataset_root",   type=str, required=True,
                        help="평가 데이터셋 루트 경로")
    parser.add_argument("--dataset_format", type=str, default=None,
                        choices=["mvtec", "visa", "btad"],
                        help="데이터셋 포맷 (미지정 시 자동 감지)")
    parser.add_argument("--categories",     nargs="+", default=None,
                        help="처리할 카테고리 목록 (미지정 시 자동 탐색)")
    parser.add_argument("--target",         type=str, default="anomaly",
                        choices=["anomaly", "normal", "all"],
                        help="히트맵을 생성할 이미지 대상 (기본: anomaly)")

    # 체크포인트
    parser.add_argument("--load_path",      type=str, required=True,
                        help="학습된 분류기 가중치 경로 (.pth)")

    # 출력
    parser.add_argument("--output_root",    type=str, default="outputs/heatmaps",
                        help="히트맵 저장 루트 디렉토리")

    # few-shot support
    parser.add_argument("--mode",           type=str, default="both",
                        choices=["zero_shot", "few_shot", "both"],
                        help="추론 모드 (기본: both)")
    parser.add_argument("--n_shot",         type=int, default=1,
                        help="few-shot 시 사용할 support 이미지 수 (기본: 1)")
    parser.add_argument("--support_root",   type=str, default=None,
                        help="few-shot 메모리 뱅크용 별도 경로 (미지정 시 dataset_root 사용). "
                             "예: JVM_goldentemplate")
    parser.add_argument("--support_format", type=str, default=None,
                        choices=["mvtec", "visa", "btad"],
                        help="support_root 의 포맷 (미지정 시 dataset_format 과 동일)")

    # 백본
    parser.add_argument("--backbone",       type=str, default="dinov3",
                        choices=["dinov3", "dinov2"])
    parser.add_argument("--dinov3_repo",    type=str,
                        default=str(BASE / "dinov3"))
    parser.add_argument("--dinov3_weights", type=str,
                        default=str(BASE / "dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    parser.add_argument("--layers",         nargs="+", type=int,
                        default=paper.EXTRACT_LAYERS)

    # 기타
    parser.add_argument("--batch_size",     type=int, default=4)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 백본 설정 ──────────────────────────────────────────────
    if args.backbone == "dinov3":
        paper.IMG_SIZE   = paper.IMG_SIZE_DINOV3
        paper.PATCH_SIZE = paper.PATCH_SIZE_DINOV3
    else:
        paper.IMG_SIZE   = paper.IMG_SIZE_DINOV2
        paper.PATCH_SIZE = paper.PATCH_SIZE_DINOV2
    paper.eval_transform = paper.make_eval_transform(paper.IMG_SIZE)
    paper.mask_transform = paper.make_mask_transform(paper.IMG_SIZE)

    print(f"\n사용 장치: {device}")
    print(f"백본: {args.backbone.upper()} | IMG_SIZE={paper.IMG_SIZE}, PATCH_SIZE={paper.PATCH_SIZE}")

    # ── 포맷 결정 ──────────────────────────────────────────────
    dataset_root = Path(args.dataset_root)
    if args.dataset_format:
        fmt = args.dataset_format
    else:
        fmt = detect_format(dataset_root)
        print(f"포맷 자동 감지: {fmt}")
    args.dataset_format = fmt

    # ── 카테고리 결정 ──────────────────────────────────────────
    categories = args.categories or discover_categories(dataset_root, fmt)
    if not categories:
        raise RuntimeError(f"처리할 카테고리를 찾을 수 없습니다: {dataset_root}")
    print(f"카테고리: {categories}")

    # ── 모델 로드 ──────────────────────────────────────────────
    model = paper.UniADet(
        layers=args.layers,
        backbone=args.backbone,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
    )
    ckpt = torch.load(args.load_path, map_location="cpu")
    model.classifiers.load_state_dict(ckpt)
    model.to(device).eval()
    model.backbone.eval()
    print(f"[Checkpoint] 로드: {args.load_path}")

    print(f"\n모드: {args.mode}  |  대상: {args.target}  |  출력: {args.output_root}")
    if args.support_root:
        print(f"few-shot support: {args.support_root} (format={args.support_format or fmt})")

    # ── 카테고리별 히트맵 생성 ────────────────────────────────
    for cat in categories:
        print(f"\n{'─'*50}")
        print(f"  카테고리: {cat}")
        print(f"{'─'*50}")
        generate_for_category(model, cat, args, device)

    print(f"\n히트맵 생성 완료 → {args.output_root}")


if __name__ == "__main__":
    main()
