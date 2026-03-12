"""
unigad/utils/dataloader.py
---------------------------
DataLoader 안전 생성 헬퍼.

batch_size가 샘플 수를 초과하면 자동으로 줄이고,
drop_last가 빈 DataLoader를 만들지 않도록 조정한다.
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    n         = len(dataset)
    eff_bs    = min(batch_size, n)
    drop_last = n >= batch_size
    if eff_bs != batch_size:
        print(f"  [DataLoader] 샘플 수({n}) < batch_size({batch_size}); "
              f"batch_size={eff_bs}, drop_last=False 로 조정")
    return DataLoader(
        dataset,
        batch_size=eff_bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
