#!/usr/bin/env python3
"""
scripts/train_eval_custom_patch.py
====================================
Custom 데이터셋 전용 Patch-Crop 학습 + 추론 (Multi-GPU 지원).

사용 예시:
    python scripts/train_eval_custom_patch.py \\
        --custom_root /path/to/Custom \\
        --golden_root /path/to/CustomGolden \\
        --epochs 50 --batch_size 384 --patience 5

추론만:
    python scripts/train_eval_custom_patch.py --skip_train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from unigad.models.uniadet      import UniADet
from unigad.models.multigpu     import wrap_multigpu, inner
from unigad.datasets.custom_patch import CustomPatchTrainDataset, make_weighted_sampler
from unigad.engine.train        import train_custom_patch
from unigad.engine.evaluate     import eval_custom_patch
from unigad.engine.memory_bank  import build_memory_banks_per_pos
from unigad.transforms          import EXTRACT_LAYERS, PATCH_SIZE_DINOV3
from unigad.utils.checkpoint    import load_ckpt, should_skip
from torch.utils.data           import DataLoader

BASE      = Path(__file__).parent.parent
DATA_ROOT = BASE.parent / "Data"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--custom_root",    default=str(DATA_ROOT / "Custom"))
    p.add_argument("--support_root",   default=None)
    p.add_argument("--golden_root",    default=str(DATA_ROOT / "JVM_goldentemplate"))
    p.add_argument("--ckpt_path",      default=str(BASE / "checkpoints" / "ckpt_custom_patch.pth"))
    p.add_argument("--result_path",    default=str(BASE / "results_custom_patch.json"))
    p.add_argument("--heatmap_dir",    default=None)
    p.add_argument("--backbone",        default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--dinov3_repo",     default=str(BASE / "dinov3"))
    p.add_argument("--dinov3_weights",  default=str(
        BASE / "dinov3" / "pretrained"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--layers",          nargs="+", type=int, default=EXTRACT_LAYERS)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=384)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int,   default=7)
    p.add_argument("--use_test",        action="store_true")
    p.add_argument("--num_workers",     type=int,   default=12)
    p.add_argument("--n_shot",          type=int,   default=4)
    p.add_argument("--force",           action="store_true")
    p.add_argument("--skip_train",      action="store_true")
    p.add_argument("--skip_eval",       action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device} | GPU: {torch.cuda.device_count()}")

    base_model = UniADet(
        layers=args.layers, backbone=args.backbone,
        dinov3_repo=args.dinov3_repo, dinov3_weights=args.dinov3_weights,
        patch_size=PATCH_SIZE_DINOV3,
    )
    model = wrap_multigpu(base_model)
    model.to(device)

    # ── 학습 ──────────────────────────────────────────────────────
    if not args.skip_train:
        if Path(args.ckpt_path).exists() and not args.force:
            print(f"[Skip Train] 체크포인트 존재 → 로드: {args.ckpt_path}")
            load_ckpt(model, args.ckpt_path)
        else:
            train_ds = CustomPatchTrainDataset(args.custom_root, use_test=args.use_test)
            sampler  = make_weighted_sampler(train_ds)
            n        = len(train_ds)
            eff_bs   = min(args.batch_size, n)
            train_dl = DataLoader(
                train_ds, batch_size=eff_bs, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True,
                drop_last=(n >= args.batch_size),
            )
            train_custom_patch(
                model, train_dl, device,
                epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                ckpt_path=args.ckpt_path,
            )

    # ── 추론 ──────────────────────────────────────────────────────
    if not args.skip_eval:
        if args.skip_train:
            load_ckpt(model, args.ckpt_path)

        results = {}
        support = args.support_root or args.custom_root

        # [A] Zero-shot
        results["zero_shot"] = eval_custom_patch(
            model, args.custom_root, device,
            mode_name="zero_shot", save_heatmap_dir=args.heatmap_dir,
        )

        # [B] Standard Few-shot
        std_banks = build_memory_banks_per_pos(model, support, device, args.n_shot)
        if std_banks:
            results["few_shot_standard"] = eval_custom_patch(
                model, args.custom_root, device,
                memory_banks=std_banks, mode_name="few_shot_standard",
                save_heatmap_dir=args.heatmap_dir,
            )

        # [C] Golden Few-shot
        if args.golden_root and Path(args.golden_root).exists():
            gold_banks = build_memory_banks_per_pos(model, args.golden_root, device, args.n_shot)
            if gold_banks:
                results["few_shot_golden"] = eval_custom_patch(
                    model, args.custom_root, device,
                    memory_banks=gold_banks, mode_name="few_shot_golden",
                    save_heatmap_dir=args.heatmap_dir,
                )

        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {args.result_path}")


if __name__ == "__main__":
    main()
