#!/usr/bin/env python3
"""
scripts/eval_custom_patch_crosseval.py
========================================
MVTec / VisA / BTAD 학습 가중치를 각각 로드하여
Custom 이미지를 Patch-Crop 방식으로 추론하고 지표를 비교한다.

사용 예시:
    python scripts/eval_custom_patch_crosseval.py
    python scripts/eval_custom_patch_crosseval.py --models mvtec visa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from unigad.models.uniadet      import UniADet
from unigad.models.multigpu     import wrap_multigpu
from unigad.engine.evaluate     import eval_custom_patch
from unigad.engine.memory_bank  import build_memory_banks_per_pos
from unigad.transforms          import EXTRACT_LAYERS, PATCH_SIZE_DINOV3
from unigad.utils.checkpoint    import load_ckpt
from unigad.utils.metrics       import print_cross_summary

BASE      = Path(__file__).parent.parent
DATA_ROOT = BASE.parent / "Data"

CKPT_FILES = {
    "mvtec": "ckpt_trained_on_mvtec.pth",
    "visa":  "ckpt_trained_on_visa.pth",
    "btad":  "ckpt_trained_on_btad.pth",
    "custom_patch": "ckpt_custom_patch.pth"
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--custom_root",       default=str(DATA_ROOT / "JVM_mvtec"))
    p.add_argument("--support_root",   default=None)
    p.add_argument("--golden_root",    default=str(DATA_ROOT / "JVM_goldentemplate"))
    p.add_argument("--ckpt_dir",       default=str(BASE / "checkpoints"))
    p.add_argument("--result_path",    default=str(BASE / "results_custom_patch_crosseval.json"))
    p.add_argument("--models",         nargs="+",
                   default=list(CKPT_FILES.keys()),
                   choices=list(CKPT_FILES.keys()))
    p.add_argument("--backbone",        default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--dinov3_repo",     default=str(BASE / "dinov3"))
    p.add_argument("--dinov3_weights",  default=str(
        BASE / "dinov3" / "pretrained"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--layers",          nargs="+", type=int, default=EXTRACT_LAYERS)
    p.add_argument("--n_shot",          type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device} | GPU: {torch.cuda.device_count()}")

    support = args.support_root or args.custom_root

    # ── DINOv3 백본은 모든 모델에서 공유 → 1회만 로딩 ─────────────────
    print("\n[Init] DINOv3 백본 로딩")
    model = UniADet(
        layers=args.layers, backbone=args.backbone,
        dinov3_repo=args.dinov3_repo, dinov3_weights=args.dinov3_weights,
        patch_size=PATCH_SIZE_DINOV3,
    )
    model = wrap_multigpu(model)
    model.to(device)

    all_results: dict[str, dict] = {}

    for model_name in args.models:
        ckpt_path = Path(args.ckpt_dir) / CKPT_FILES[model_name]
        if not ckpt_path.exists():
            print(f"[Skip] 체크포인트 없음: {ckpt_path}"); continue

        print(f"\n{'='*60}")
        print(f"  모델: {model_name.upper()} | {ckpt_path.name}")
        print(f"{'='*60}")

        # classifiers 가중치만 교체 (백본은 그대로)
        load_ckpt(model, str(ckpt_path))
        model_results = {}

        # [A] Zero-shot
        model_results["zero_shot"] = eval_custom_patch(
            model, args.custom_root, device, mode_name="zero_shot",
        )

        # [B] Standard Few-shot
        std_banks = build_memory_banks_per_pos(model, support, device, args.n_shot)
        if std_banks:
            model_results["few_shot_standard"] = eval_custom_patch(
                model, args.custom_root, device,
                memory_banks=std_banks, mode_name="few_shot_standard",
            )

        # [C] Golden Few-shot
        if args.golden_root and Path(args.golden_root).exists():
            gold_banks = build_memory_banks_per_pos(
                model, args.golden_root, device, args.n_shot,
            )
            if gold_banks:
                model_results["few_shot_golden"] = eval_custom_patch(
                    model, args.custom_root, device,
                    memory_banks=gold_banks, mode_name="few_shot_golden",
                )

        all_results[model_name] = model_results

    # 요약 출력 (cross_summary 형식에 맞게 래핑)
    wrapped = {k: {"custom": {mk: [mv] for mk, mv in v.items()}}
               for k, v in all_results.items()}
    print_cross_summary(wrapped)

    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.result_path}")


if __name__ == "__main__":
    main()
