"""
unigad/losses.py
----------------
세그멘테이션 및 분류 손실 함수.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    L_focal = -alpha * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t   = torch.where(targets == 1, probs, 1 - probs)
        loss  = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss.
    L_dice = 1 - (2|P∩G| + ε) / (|P| + |G| + ε)
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        probs_f  = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter    = (probs_f * targets_f).sum(1)
        dice     = (2.0 * inter + self.smooth) / (probs_f.sum(1) + targets_f.sum(1) + self.smooth)
        return (1 - dice).mean()
