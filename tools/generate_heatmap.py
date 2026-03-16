"""
generate_heatmap.py
====================

목적:
  UniGAD 학습 가중치를 로드하여
  MVTec / VisA / BTAD 등 어떤 포맷의 데이터셋에도 이상 히트맵을 생성한다.
  (unigad 패키지 사용, uniadet_paper_aligned 미사용)

포맷 자동 감지 규칙:
  - <root>/<cat>/Data/               → visa
  - <root>/<cat>/test/ko/            → btad
  - <root>/<cat>/test/               → mvtec
  --dataset_format 인자로 명시적으로 지정할 수도 있다.

few-shot 지지 이미지:
  기본적으로 각 데이터셋의 정상(train) 이미지를 사용한다.
  --support_root 를 지정하면 해당 경로의 train/good/ 이미지를 메모리 뱅크로 사용한다.
  (예: Golden Template 생성 결과 폴더)

출력:
  --output_root/<category>/<original_stem>_heatmap.png  (JET 컬러 히트맵)

히트맵 뷰 모드 (--heatmap_mode):
  full_image : 이미지 전체를 한 번에 리사이즈 후 추론(기본). 빠름.
  patch_tiled: 이미지를 타일 단위로 잘라 각각 추론 후 병합. 원본 해상도에 가깝게 확인 가능.

사용 예시:
  # MVTec 형식 custom 데이터셋
  python generate_heatmap.py \\
      --dataset_root  /path/to/Custom \\
      --load_path     checkpoints/ckpt_trained_on_Custom.pth

  # Golden Template을 few-shot support로
  python generate_heatmap.py \\
      --dataset_root  /path/to/Custom \\
      --support_root  /path/to/Custom_goldentemplate \\
      --load_path     checkpoints/ckpt_trained_on_Custom.pth \\
      --mode          both

  # VisA / BTAD
  python generate_heatmap.py --dataset_root /path/to/VisA --load_path checkpoints/ckpt_trained_on_visa.pth
  python generate_heatmap.py --dataset_root /path/to/BTAD --load_path checkpoints/ckpt_trained_on_btad.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# UniGAD 루트를 path에 넣어 unigad 패키지 사용 (tools/ 에서 실행해도 동작)
_SCRIPT_DIR = Path(__file__).resolve().parent
_UNIGAD_ROOT = _SCRIPT_DIR.parent
if str(_UNIGAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNIGAD_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image

from unigad.models.uniadet import UniADet
from unigad.models.multigpu import wrap_multigpu, inner, patch_feat_to_list
from unigad.datasets.mvtec import MVTecADDataset
from unigad.datasets.visa import VisADataset
from unigad.datasets.btad import BTADDataset
from unigad.engine.memory_bank import build_memory_bank, compute_fewshot_score
from unigad.transforms import (
    EXTRACT_LAYERS,
    IMG_SIZE_DINOV3,
    IMG_SIZE_DINOV2,
    PATCH_SIZE_DINOV3,
    PATCH_SIZE_DINOV2,
    make_eval_transform,
)
from unigad.utils.checkpoint import load_ckpt


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
def load_dataset(root: str, cat: str, fmt: str, split: str, eval_transform):
    """split: 'test'(평가용) 또는 'train'(support용). eval_transform은 main에서 img_size 기준으로 생성."""
    if fmt == "visa":
        return VisADataset(root=root, categories=[cat], transform=eval_transform)
    elif fmt == "btad":
        return BTADDataset(root=root, category=cat, split=split, transform=eval_transform)
    else:  # mvtec
        return MVTecADDataset(root=root, category=cat, split=split, transform=eval_transform)


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
                       n_shot: int, device, eval_transform) -> list | None:
    """
    support_root 에서 정상 이미지를 n_shot 장 로드하여 메모리 뱅크를 구성한다.
    """
    try:
        sup_ds = load_dataset(support_root, cat, fmt, split="train", eval_transform=eval_transform)
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
    return build_memory_bank(model, sup_dl, device)


# ──────────────────────────────────────────────────────────────────────
# 패치(타일) 단위 추론 → 전체 해상도 히트맵 병합
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _heatmap_one_image_patch_tiled(model, img_path: str, memory_bank, args, device) -> torch.Tensor:
    """
    이미지를 IMG_SIZE 타일로 잘라 각 타일 추론 후 하나의 히트맵으로 병합.
    반환: [H_orig, W_orig] CPU 텐서
    """
    img = Image.open(img_path).convert("RGB")
    w_orig, h_orig = img.size
    sz = args.img_size
    transform = args.eval_transform

    # 타일 그리드 (경계 넘어가면 패딩)
    canvas_sum = np.zeros((h_orig, w_orig), dtype=np.float64)
    canvas_cnt = np.zeros((h_orig, w_orig), dtype=np.float64)

    for y0 in range(0, h_orig, sz):
        for x0 in range(0, w_orig, sz):
            x1, y1 = min(x0 + sz, w_orig), min(y0 + sz, h_orig)
            crop = img.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            if cw < sz or ch < sz:
                pad_img = Image.new("RGB", (sz, sz), (0, 0, 0))
                pad_img.paste(crop, (0, 0))
                crop = pad_img
            tile = transform(crop).unsqueeze(0).to(device)
            _, seg_logit, patch_feat = model(tile)
            seg_score_zs = F.softmax(seg_logit, dim=-1)[..., 1]
            final_score = seg_score_zs
            if memory_bank is not None:
                patch_feats_list = patch_feat_to_list(patch_feat)
                seg_score_fs = compute_fewshot_score(patch_feats_list, memory_bank, device)
                if args.mode == "few_shot":
                    final_score = seg_score_fs
                else:
                    final_score = seg_score_zs + seg_score_fs
            H_p = tile.shape[-2] // args.patch_size
            W_p = tile.shape[-1] // args.patch_size
            seg_map = final_score.view(1, 1, H_p, W_p)
            seg_map = F.interpolate(seg_map, size=(sz, sz), mode="bilinear", align_corners=False)
            arr = seg_map[0, 0].cpu().numpy()
            dy = min(sz, h_orig - y0)
            dx = min(sz, w_orig - x0)
            canvas_sum[y0 : y0 + dy, x0 : x0 + dx] += arr[:dy, :dx]
            canvas_cnt[y0 : y0 + dy, x0 : x0 + dx] += 1.0

    out = np.divide(canvas_sum, canvas_cnt, where=canvas_cnt > 0)
    return torch.from_numpy(out.astype(np.float32))


# ──────────────────────────────────────────────────────────────────────
# 단일 카테고리 히트맵 생성
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_for_category(model, cat: str, args, device):
    fmt          = args.dataset_format
    support_fmt  = args.support_format or fmt

    # ── 평가 데이터셋 로드 ────────────────────────────────────
    try:
        full_ds = load_dataset(args.dataset_root, cat, fmt, split="test", eval_transform=args.eval_transform)
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

    print(f"  대상 이미지: {len(indices)}장 (target={target})  heatmap_mode={args.heatmap_mode}")
    ds = Subset(full_ds, indices)
    out_dir = Path(args.output_root) / cat

    # ── 메모리 뱅크 구성 (few-shot / both) ────────────────────
    memory_bank = None
    if args.mode in ("few_shot", "both"):
        support_root = args.support_root or args.dataset_root
        memory_bank  = build_support_bank(model, support_root, cat, support_fmt,
                                          args.n_shot, device, args.eval_transform)
        if memory_bank is None and args.mode == "few_shot":
            print(f"  [Skip] few_shot 모드인데 메모리 뱅크 구성 실패: {cat}")
            return

    # ── 추론: patch_tiled ─────────────────────────────────────
    if args.heatmap_mode == "patch_tiled":
        for i in indices:
            img_path = full_ds.samples[i][0]
            seg_map = _heatmap_one_image_patch_tiled(model, img_path, memory_bank, args, device)
            save_heatmap(seg_map, img_path, out_dir)
        print(f"  저장 완료: {out_dir}")
        return

    # ── 추론: full_image (기본) ────────────────────────────────
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)
    base_idx = 0
    for imgs, _, _ in dl:
        imgs = imgs.to(device)
        B    = imgs.size(0)
        H, W = imgs.shape[-2], imgs.shape[-1]
        H_p  = H // args.patch_size
        W_p  = W // args.patch_size

        cls_logit, seg_logit, patch_feat = model(imgs)
        patch_feats_list = patch_feat_to_list(patch_feat)

        # Zero-shot 세그멘테이션 점수 (UniGAD과 동일: softmax)
        seg_score_zs = F.softmax(seg_logit, dim=-1)[..., 1]
        final_score  = seg_score_zs

        # Few-shot 점수
        if memory_bank is not None:
            seg_score_fs = compute_fewshot_score(patch_feats_list, memory_bank, device)
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
    base = _UNIGAD_ROOT

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
                        default=str(base / "dinov3"))
    parser.add_argument("--dinov3_weights", type=str,
                        default=str(base / "dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    parser.add_argument("--layers",         nargs="+", type=int,
                        default=EXTRACT_LAYERS)

    # 히트맵 뷰 모드
    parser.add_argument("--heatmap_mode",  type=str, default="full_image",
                        choices=["full_image", "patch_tiled"],
                        help="full_image: 이미지 전체를 한 번에 리사이즈 후 추론(기본). "
                             "patch_tiled: 이미지를 타일 단위로 잘라 각각 추론 후 합쳐서 전체 해상도 히트맵 생성")

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
        args.img_size   = IMG_SIZE_DINOV3
        args.patch_size = PATCH_SIZE_DINOV3
    else:
        args.img_size   = IMG_SIZE_DINOV2
        args.patch_size = PATCH_SIZE_DINOV2
    args.eval_transform = make_eval_transform(args.img_size)

    print(f"\n사용 장치: {device}")
    print(f"백본: {args.backbone.upper()} | IMG_SIZE={args.img_size}, PATCH_SIZE={args.patch_size}")

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
    model = UniADet(
        layers=args.layers,
        backbone=args.backbone,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        patch_size=args.patch_size,
    )
    model = wrap_multigpu(model)
    load_ckpt(model, args.load_path)
    model.to(device).eval()
    inner(model).backbone.eval()
    print(f"[Checkpoint] 로드: {args.load_path}")

    print(f"\n모드: {args.mode}  |  heatmap_mode: {args.heatmap_mode}  |  대상: {args.target}  |  출력: {args.output_root}")
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

## 전체 이미지 한 번에 (기본, 빠름)
# python generate_heatmap.py --dataset_root /home/user/jupyter/bk/paperworks/Data/PXK --load_path checkpoints/ckpt_trained_on_PXK.pth --output_root outputs

# 타일 단위로 잘라서 원본 해상도에 가깝게
# python generate_heatmap.py --dataset_root PXK --load_path checkpoints/ckpt_trained_on_PXK.pth --output_root out --heatmap_mode patch_tiled