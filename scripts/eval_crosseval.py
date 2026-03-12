#!/usr/bin/env python3
"""
scripts/eval_crosseval.py
==========================
학습된 4개 가중치(MVTec/VisA/JVM/BTAD)를 로드하여
4개 데이터셋 전체에 대해 크로스 평가를 수행한다.

사용 예시:
    python scripts/eval_crosseval.py
    python scripts/eval_crosseval.py --ckpts mvtec --eval_datasets jvm
    python scripts/eval_crosseval.py --mode zero_shot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from unigad.models.uniadet      import UniADet
from unigad.models.multigpu     import wrap_multigpu
from unigad.datasets.mvtec      import MVTecADDataset
from unigad.datasets.visa       import VisADataset
from unigad.datasets.btad       import BTADDataset
from unigad.engine.evaluate     import evaluate_uniadet
from unigad.engine.memory_bank  import build_memory_bank
from unigad.transforms          import (
    EXTRACT_LAYERS, MVTEC_CATEGORIES,
    IMG_SIZE_DINOV3, IMG_SIZE_DINOV2,
    PATCH_SIZE_DINOV3, PATCH_SIZE_DINOV2,
    make_eval_transform,
)
from unigad.utils.checkpoint    import load_ckpt
from unigad.utils.metrics       import print_summary_table, print_cross_summary

BASE      = Path(__file__).parent.parent
DATA_ROOT = BASE.parent / "Data"

CKPT_FILES = {
    "mvtec": "ckpt_trained_on_mvtec.pth",
    "visa":  "ckpt_trained_on_visa.pth",
    "jvm":   "ckpt_trained_on_jvm.pth",
    "btad":  "ckpt_trained_on_btad.pth",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mvtec_root",  default=str(DATA_ROOT / "MVTec"))
    p.add_argument("--visa_root",   default=str(DATA_ROOT / "VisA"))
    p.add_argument("--jvm_root",    default=str(DATA_ROOT / "JVM_mvtec"))
    p.add_argument("--btad_root",   default=str(DATA_ROOT / "BTAD"))
    p.add_argument("--ckpt_dir",    default=str(BASE / "checkpoints"))
    p.add_argument("--result_path", default=str(BASE / "results_crosseval.json"))
    p.add_argument("--backbone",    default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--dinov3_repo",     default=str(BASE / "dinov3"))
    p.add_argument("--dinov3_weights",  default=str(
        BASE / "dinov3" / "pretrained"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    p.add_argument("--layers",          nargs="+", type=int, default=EXTRACT_LAYERS)
    p.add_argument("--mode",            default="both",
                   choices=["zero_shot", "few_shot", "both"])
    p.add_argument("--few_shot_ks",     nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--mvtec_categories", nargs="+", default=MVTEC_CATEGORIES)
    p.add_argument("--visa_categories",  nargs="+", default=None)
    p.add_argument("--jvm_categories",   nargs="+", default=None)
    p.add_argument("--btad_categories",  nargs="+", default=None)
    p.add_argument("--ckpts",         nargs="+",
                   choices=list(CKPT_FILES.keys()),
                   default=list(CKPT_FILES.keys()))
    p.add_argument("--eval_datasets", nargs="+",
                   choices=["mvtec", "visa", "jvm", "btad"],
                   default=["mvtec", "visa", "jvm", "btad"])
    return p.parse_args()


def key_for_k(k: int) -> str:
    return "zero_shot" if k == 0 else f"few_shot_{k}"


def build_kshot_bank(model, support_ds, normal_idx, k, device):
    selected = normal_idx[:k]
    if not selected:
        return None
    dl = DataLoader(Subset(support_ds, selected), batch_size=len(selected), shuffle=False)
    return build_memory_bank(model, dl, device)


def eval_mvtec_style(model, root, cats, device, args, tag):
    buckets = {"zero_shot": []}
    for k in args.few_shot_ks:
        buckets[key_for_k(k)] = []
    tf = make_eval_transform(IMG_SIZE_DINOV3 if args.backbone == "dinov3" else IMG_SIZE_DINOV2)

    for cat in cats:
        try:
            test_ds = MVTecADDataset(root, cat, "test", transform=tf)
        except RuntimeError as e:
            print(f"  [Skip] {e}")
            continue
        test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

        if args.mode in ("zero_shot", "both"):
            r = evaluate_uniadet(model, test_dl, device, category=cat)
            r["dataset"] = tag; r["k_shot"] = 0
            buckets["zero_shot"].append(r)

        if args.mode in ("few_shot", "both"):
            try:
                support_ds = MVTecADDataset(root, cat, "train", transform=tf)
            except RuntimeError:
                continue
            normal_idx = [i for i, s in enumerate(support_ds.samples) if s[1] == 0]
            for k in args.few_shot_ks:
                bank = build_kshot_bank(model, support_ds, normal_idx, k, device)
                if bank is None:
                    continue
                r = evaluate_uniadet(model, test_dl, device, memory_bank=bank, category=cat)
                r["dataset"] = tag; r["k_shot"] = k
                buckets[key_for_k(k)].append(r)
    return buckets


def eval_visa(model, root, cats, device, args):
    buckets = {"zero_shot": []}
    for k in args.few_shot_ks:
        buckets[key_for_k(k)] = []
    tf = make_eval_transform(IMG_SIZE_DINOV3 if args.backbone == "dinov3" else IMG_SIZE_DINOV2)

    for cat in cats:
        try:
            test_ds = VisADataset(root, [cat], transform=tf)
        except RuntimeError as e:
            print(f"  [Skip] {e}"); continue
        test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

        if args.mode in ("zero_shot", "both"):
            r = evaluate_uniadet(model, test_dl, device, category=cat)
            r["dataset"] = "visa"; r["k_shot"] = 0
            buckets["zero_shot"].append(r)

        if args.mode in ("few_shot", "both"):
            normal_idx = [i for i, s in enumerate(test_ds.samples) if s[1] == 0]
            for k in args.few_shot_ks:
                bank = build_kshot_bank(model, test_ds, normal_idx, k, device)
                if bank is None:
                    continue
                r = evaluate_uniadet(model, test_dl, device, memory_bank=bank, category=cat)
                r["dataset"] = "visa"; r["k_shot"] = k
                buckets[key_for_k(k)].append(r)
    return buckets


def eval_btad(model, root, cats, device, args):
    buckets = {"zero_shot": []}
    for k in args.few_shot_ks:
        buckets[key_for_k(k)] = []
    tf = make_eval_transform(IMG_SIZE_DINOV3 if args.backbone == "dinov3" else IMG_SIZE_DINOV2)

    for cat in cats:
        try:
            test_ds = BTADDataset(root, cat, "test", transform=tf)
        except RuntimeError as e:
            print(f"  [Skip] {e}"); continue
        test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

        if args.mode in ("zero_shot", "both"):
            r = evaluate_uniadet(model, test_dl, device, category=cat)
            r["dataset"] = "btad"; r["k_shot"] = 0
            buckets["zero_shot"].append(r)

        if args.mode in ("few_shot", "both"):
            try:
                support_ds = BTADDataset(root, cat, "train", transform=tf)
            except RuntimeError:
                continue
            normal_idx = [i for i, s in enumerate(support_ds.samples) if s[1] == 0]
            for k in args.few_shot_ks:
                bank = build_kshot_bank(model, support_ds, normal_idx, k, device)
                if bank is None:
                    continue
                r = evaluate_uniadet(model, test_dl, device, memory_bank=bank, category=cat)
                r["dataset"] = "btad"; r["k_shot"] = k
                buckets[key_for_k(k)].append(r)
    return buckets


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size   = IMG_SIZE_DINOV3   if args.backbone == "dinov3" else IMG_SIZE_DINOV2
    patch_size = PATCH_SIZE_DINOV3 if args.backbone == "dinov3" else PATCH_SIZE_DINOV2
    print(f"장치: {device} | backbone: {args.backbone.upper()} | few_shot_ks: {args.few_shot_ks}")

    visa_cats = args.visa_categories or [
        d.name for d in sorted(Path(args.visa_root).iterdir())
        if d.is_dir() and (d / "Data").exists()
    ]
    jvm_cats  = args.jvm_categories  or [
        d.name for d in sorted(Path(args.jvm_root).iterdir())
        if d.is_dir() and (d / "test").exists()
    ]
    btad_cats = args.btad_categories or [
        d.name for d in sorted(Path(args.btad_root).iterdir())
        if d.is_dir() and (d / "test" / "ko").exists()
    ]

    all_results = {}

    for ckpt_tag in args.ckpts:
        ckpt_path = Path(args.ckpt_dir) / CKPT_FILES[ckpt_tag]
        if not ckpt_path.exists():
            print(f"[Skip] 체크포인트 없음: {ckpt_path}"); continue

        model = UniADet(layers=args.layers, backbone=args.backbone,
                        dinov3_repo=args.dinov3_repo, dinov3_weights=args.dinov3_weights,
                        patch_size=patch_size)
        model = wrap_multigpu(model)
        load_ckpt(model, str(ckpt_path))
        model.to(device)

        all_results[ckpt_tag] = {}

        for ds_tag in args.eval_datasets:
            print(f"\n{'─'*60}")
            print(f"  [{ckpt_tag}] → [{ds_tag}]")
            print(f"{'─'*60}")

            if ds_tag == "mvtec":
                buckets = eval_mvtec_style(model, args.mvtec_root,
                                           args.mvtec_categories, device, args, "mvtec")
            elif ds_tag == "visa":
                buckets = eval_visa(model, args.visa_root, visa_cats, device, args)
            elif ds_tag == "jvm":
                buckets = eval_mvtec_style(model, args.jvm_root,
                                           jvm_cats, device, args, "jvm")
            elif ds_tag == "btad":
                buckets = eval_btad(model, args.btad_root, btad_cats, device, args)
            else:
                continue

            all_results[ckpt_tag][ds_tag] = buckets
            for mode_key, results in buckets.items():
                if results:
                    print_summary_table(results)

        del model
        torch.cuda.empty_cache()

    print_cross_summary(all_results)

    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.result_path}")


if __name__ == "__main__":
    main()
