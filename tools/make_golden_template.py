"""
make_golden_template.py
========================

목적:
  JVM_mvtec 폴더의 각 클래스별 good 이미지(train/good + test/good)에서
  랜덤하게 10장을 선택하여 픽셀 평균 이미지(Golden Template)를 생성한다.

  - 시행 횟수: 4회 (각 시행은 독립적이며 이전 시행과 겹칠 수 있음)
  - 출력 구조: MVTec 포맷(train/good/)으로 저장하여
    few-shot 추론 시 support 데이터셋으로 바로 사용 가능

사용법:
  python make_golden_template.py
  python make_golden_template.py --jvm_root JVM_mvtec --output_root JVM_goldentemplate
  python make_golden_template.py --n_select 10 --n_trials 4 --seed 42

출력 구조:
  JVM_goldentemplate/
    <category>/
      train/
        good/
          golden_01.png   ← 1회차 10장 평균
          golden_02.png   ← 2회차 10장 평균
          golden_03.png
          golden_04.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_good_images(class_dir: Path) -> list[Path]:
    """train/good 와 test/good 아래 이미지 파일을 모두 수집한다."""
    images: list[Path] = []
    for split in ("train", "test"):
        good_dir = class_dir / split / "good"
        if good_dir.exists():
            for p in sorted(good_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    images.append(p)
    return images


def average_images(paths: list[Path]) -> Image.Image:
    """주어진 이미지 경로 목록을 픽셀 단위로 평균하여 PIL Image로 반환한다."""
    arrays = []
    base_size = None
    for p in paths:
        img = Image.open(p).convert("RGB")
        if base_size is None:
            base_size = img.size
        elif img.size != base_size:
            img = img.resize(base_size, Image.LANCZOS)
        arrays.append(np.array(img, dtype=np.float32))

    avg = np.mean(arrays, axis=0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg, mode="RGB")


def discover_classes(jvm_root: Path) -> list[str]:
    return [
        d.name for d in sorted(jvm_root.iterdir())
        if d.is_dir() and (d / "train" / "good").exists()
    ]


def main():
    parser = argparse.ArgumentParser(description="JVM Golden Template Generator")
    parser.add_argument("--jvm_root",    type=str, default="JVM_mvtec",
                        help="JVM_mvtec 루트 경로")
    parser.add_argument("--output_root", type=str, default="JVM_goldentemplate",
                        help="출력 폴더 (기본: JVM_goldentemplate)")
    parser.add_argument("--n_select",    type=int, default=10,
                        help="시행당 랜덤 선택 이미지 수 (기본: 10)")
    parser.add_argument("--n_trials",    type=int, default=4,
                        help="시행 횟수 = 생성할 Golden Template 수 (기본: 4)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="재현성을 위한 랜덤 시드 (미지정 시 매번 다른 결과)")
    args = parser.parse_args()

    jvm_root    = Path(args.jvm_root)
    output_root = Path(args.output_root)

    if not jvm_root.exists():
        raise FileNotFoundError(f"JVM 루트를 찾을 수 없습니다: {jvm_root}")

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[Seed] {args.seed}")

    classes = discover_classes(jvm_root)
    if not classes:
        raise RuntimeError(f"클래스를 찾을 수 없습니다: {jvm_root}")

    print(f"발견된 클래스: {classes}")
    print(f"시행 횟수: {args.n_trials}, 시행당 선택 수: {args.n_select}\n")

    for cat in classes:
        class_dir = jvm_root / cat
        images    = collect_good_images(class_dir)

        print(f"[{cat}] 수집된 good 이미지: {len(images)}장")

        if len(images) == 0:
            print(f"  [Skip] good 이미지 없음.")
            continue

        if len(images) < args.n_select:
            print(f"  [주의] 이미지 수({len(images)})가 n_select({args.n_select})보다 적습니다. "
                  f"복원 추출(with replacement)로 전환합니다.")
            use_replacement = True
        else:
            use_replacement = False

        # 출력 폴더: MVTec 포맷 (train/good/)
        out_dir = output_root / cat / "train" / "good"
        out_dir.mkdir(parents=True, exist_ok=True)

        for trial in range(1, args.n_trials + 1):
            if use_replacement:
                selected = random.choices(images, k=args.n_select)
            else:
                # 시행 간 중복 허용, 시행 내 중복 없음
                selected = random.sample(images, k=args.n_select)

            avg_img  = average_images(selected)
            out_path = out_dir / f"golden_{trial:02d}.png"
            avg_img.save(out_path)

            selected_names = [p.name for p in selected]
            print(f"  Trial {trial}: {selected_names}  → {out_path.name}")

        print()

    print(f"Golden Template 생성 완료: {output_root}")
    print()
    print("few-shot 추론 시 활용 방법:")
    print(f"  --jvm_root {output_root}  (기존 JVM_mvtec 대신 이 경로 사용)")


if __name__ == "__main__":
    main()
