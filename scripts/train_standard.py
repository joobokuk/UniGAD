#!/usr/bin/env python3
"""
scripts/train_standard.py
==========================
MVTec / VisA / JVM / BTAD 데이터로 4개 모델을 각각 학습하여
체크포인트를 저장한다. 이미 체크포인트가 존재하면 학습을 건너뛴다.

사용 예시:
    python scripts/train_standard.py \\
        --ckpt_dir checkpoints \\
        --epochs 50 --batch_size 256 --patience 5 \\
        --train_targets mvtec visa jvm btad

특정 데이터셋만 재학습:
    python scripts/train_standard.py --train_targets mvtec --force
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from unigad.models.uniadet      import UniADet
from unigad.models.multigpu     import wrap_multigpu
from unigad.datasets.mvtec      import MVTecADDataset
from unigad.datasets.visa       import VisADataset
from unigad.datasets.btad       import BTADDataset
from unigad.engine.train        import train_uniadet
from unigad.transforms          import (
    EXTRACT_LAYERS, MVTEC_CATEGORIES,
    IMG_SIZE_DINOV3, IMG_SIZE_DINOV2,
    PATCH_SIZE_DINOV3, PATCH_SIZE_DINOV2,
    make_train_transform,
)
from unigad.utils.checkpoint    import should_skip, save_ckpt
from unigad.utils.dataloader    import make_dataloader

BASE      = Path(__file__).parent.parent
DATA_ROOT = BASE.parent / "Data"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mvtec_root", default=str(DATA_ROOT / "MVTec"))
    p.add_argument("--visa_root",  default=str(DATA_ROOT / "VisA"))
    p.add_argument("--jvm_root",   default=str(DATA_ROOT / "JVM_mvtec"))
    p.add_argument("--btad_root",  default=str(DATA_ROOT / "BTAD"))
    p.add_argument("--ckpt_dir",   default=str(BASE / "checkpoints"))
    p.add_argument("--backbone",   default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--dinov3_repo",     default=str(BASE.parent / "UniADet" / "dinov3"))
    p.add_argument("--dinov3_weights",  default=str(
        BASE.parent / "UniADet" / "dinov3" / "pretrained"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--layers",      nargs="+", type=int, default=EXTRACT_LAYERS)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--train_targets", nargs="+",
                   choices=["mvtec", "visa", "jvm", "btad"],
                   default=["mvtec", "visa", "jvm", "btad"])
    p.add_argument("--mvtec_categories", nargs="+", default=MVTEC_CATEGORIES)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_model(args, img_size, patch_size) -> UniADet:
    return UniADet(
        layers=args.layers,
        backbone=args.backbone,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        patch_size=patch_size,
    )


def train_one(tag, args, device, dataset, img_size, patch_size):
    ckpt_path = str(Path(args.ckpt_dir) / f"ckpt_trained_on_{tag}.pth")
    if should_skip(ckpt_path, args.force):
        return

    tf  = make_train_transform(img_size)
    dl  = make_dataloader(dataset(tf), args.batch_size, num_workers=args.num_workers)
    model = build_model(args, img_size, patch_size).to(device)
    train_uniadet(
        model, dl, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        ckpt_path=ckpt_path, img_size=img_size, patch_size=patch_size,
    )


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size   = IMG_SIZE_DINOV3   if args.backbone == "dinov3" else IMG_SIZE_DINOV2
    patch_size = PATCH_SIZE_DINOV3 if args.backbone == "dinov3" else PATCH_SIZE_DINOV2
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"장치: {device} | backbone: {args.backbone.upper()} | img_size: {img_size}")

    tf = make_train_transform(img_size)

    if "mvtec" in args.train_targets:
        ckpt = str(Path(args.ckpt_dir) / "ckpt_trained_on_mvtec.pth")
        if not should_skip(ckpt, args.force):
            datasets = []
            for cat in args.mvtec_categories:
                try:
                    datasets.append(MVTecADDataset(args.mvtec_root, cat, "train",
                                                   transform=tf, img_size=img_size))
                except RuntimeError as e:
                    print(f"  [Skip] {cat}: {e}")
            if datasets:
                dl = make_dataloader(ConcatDataset(datasets), args.batch_size,
                                     num_workers=args.num_workers)
                model = build_model(args, img_size, patch_size).to(device)
                train_uniadet(model, dl, device,
                              epochs=args.epochs, lr=args.lr, patience=args.patience,
                              ckpt_path=ckpt, img_size=img_size, patch_size=patch_size)

    if "visa" in args.train_targets:
        ckpt = str(Path(args.ckpt_dir) / "ckpt_trained_on_visa.pth")
        if not should_skip(ckpt, args.force):
            ds    = VisADataset(args.visa_root, transform=tf, img_size=img_size)
            dl    = make_dataloader(ds, args.batch_size, num_workers=args.num_workers)
            model = build_model(args, img_size, patch_size).to(device)
            train_uniadet(model, dl, device,
                          epochs=args.epochs, lr=args.lr, patience=args.patience,
                          ckpt_path=ckpt, img_size=img_size, patch_size=patch_size)

    if "jvm" in args.train_targets:
        ckpt = str(Path(args.ckpt_dir) / "ckpt_trained_on_jvm.pth")
        if not should_skip(ckpt, args.force):
            cats = [d.name for d in sorted(Path(args.jvm_root).iterdir())
                    if d.is_dir() and (d / "train").exists()]
            datasets = []
            for cat in cats:
                try:
                    datasets.append(MVTecADDataset(args.jvm_root, cat, "train",
                                                   transform=tf, img_size=img_size))
                except RuntimeError as e:
                    print(f"  [Skip] {cat}: {e}")
            if datasets:
                dl = make_dataloader(ConcatDataset(datasets), args.batch_size,
                                     num_workers=args.num_workers)
                model = build_model(args, img_size, patch_size).to(device)
                train_uniadet(model, dl, device,
                              epochs=args.epochs, lr=args.lr, patience=args.patience,
                              ckpt_path=ckpt, img_size=img_size, patch_size=patch_size)

    if "btad" in args.train_targets:
        ckpt = str(Path(args.ckpt_dir) / "ckpt_trained_on_btad.pth")
        if not should_skip(ckpt, args.force):
            cats = [d.name for d in sorted(Path(args.btad_root).iterdir())
                    if d.is_dir() and (d / "train" / "ok").exists()]
            datasets = []
            for cat in cats:
                try:
                    datasets.append(BTADDataset(args.btad_root, cat, "train",
                                                transform=tf, img_size=img_size))
                except RuntimeError as e:
                    print(f"  [Skip] {cat}: {e}")
            if datasets:
                dl = make_dataloader(ConcatDataset(datasets), args.batch_size,
                                     num_workers=args.num_workers)
                model = build_model(args, img_size, patch_size).to(device)
                train_uniadet(model, dl, device,
                              epochs=args.epochs, lr=args.lr, patience=args.patience,
                              ckpt_path=ckpt, img_size=img_size, patch_size=patch_size)

    print("\n모든 학습 완료.")


if __name__ == "__main__":
    main()
