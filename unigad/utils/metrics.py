"""
unigad/utils/metrics.py
------------------------
평가 결과 출력 유틸리티.
"""
from __future__ import annotations

import numpy as np


def print_summary_table(results: list[dict]) -> None:
    """카테고리별 평가 결과를 정리된 표로 출력."""
    header = (f"{'Category':<14} {'Mode':<18} "
              f"{'ImgAUROC':>9} {'ImgAUPR':>8} {'PixAUROC':>9} {'PixAUPR':>8}")
    print(f"\n{'='*len(header)}")
    print("  UniGAD 평가 요약")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"  {r.get('category',''):<12} {r.get('mode',''):<18} "
              f"{r['img_auroc']:>9.4f} {r['img_aupr']:>8.4f} "
              f"{r['pix_auroc']:>9.4f} {r['pix_aupr']:>8.4f}")
    if results:
        img_a = np.mean([r["img_auroc"] for r in results])
        img_p = np.mean([r["img_aupr"]  for r in results])
        pix_a = float(np.nanmean([r["pix_auroc"] for r in results]))
        pix_p = float(np.nanmean([r["pix_aupr"]  for r in results]))
        print("-" * len(header))
        print(f"  {'MEAN':<12} {'':<18} "
              f"{img_a:>9.4f} {img_p:>8.4f} {pix_a:>9.4f} {pix_p:>8.4f}")
    print(f"{'='*len(header)}\n")


def print_cross_summary(all_results: dict) -> None:
    """
    크로스 평가 요약 출력.
    all_results[ckpt_tag][ds_tag][mode_key] = [result, ...]
    """
    header = (f"{'Ckpt':<10} {'Dataset':<10} {'Mode':<20} "
              f"{'ImgAUROC':>9} {'ImgAUPR':>8} {'PixAUROC':>9} {'PixAUPR':>8}")
    sep = "─" * len(header)

    print(f"\n{'='*len(header)}")
    print("  UniGAD 크로스 평가 요약 (Zero-shot + k-shot)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for ckpt_tag, datasets in all_results.items():
        for ds_tag, modes in datasets.items():
            for mode_key, results in modes.items():
                if not results:
                    continue
                img_a = np.mean([r["img_auroc"] for r in results])
                img_p = np.mean([r["img_aupr"]  for r in results])
                pix_a = float(np.nanmean([r["pix_auroc"] for r in results]))
                pix_p = float(np.nanmean([r["pix_aupr"]  for r in results]))
                if mode_key == "zero_shot":
                    mode_str = "Zero-shot"
                else:
                    k        = mode_key.split("_")[-1]
                    mode_str = f"Few-shot ({k}-shot)"
                print(f"  {ckpt_tag:<8} {ds_tag:<10} {mode_str:<20} "
                      f"{img_a:>9.4f} {img_p:>8.4f} {pix_a:>9.4f} {pix_p:>8.4f}")
        print(sep)
    print(f"{'='*len(header)}")
