"""
unigad/utils/checkpoint.py
---------------------------
체크포인트 저장/로드 유틸리티.

best epoch 가중치 하나만 저장 (중복 파일 없음).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from unigad.models.multigpu import inner


def save_ckpt(model: nn.Module, path: str) -> None:
    """classifiers state_dict를 파일로 저장."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(inner(model).classifiers.state_dict(), path)
    print(f"[Checkpoint] 저장: {path}")


def load_ckpt(model: nn.Module, path: str, device: str = "cpu") -> None:
    """체크포인트를 로드하여 classifiers에 복원."""
    ckpt = torch.load(path, map_location=device)
    inner(model).classifiers.load_state_dict(ckpt)
    print(f"[Checkpoint] 로드: {path}")


def should_skip(path: str, force: bool = False) -> bool:
    """체크포인트가 이미 존재하고 --force 가 없으면 True 반환."""
    if not force and Path(path).exists():
        print(f"  [Skip] 체크포인트 존재: {path}")
        print("  (재학습하려면 --force 옵션을 사용하세요.)")
        return True
    return False
