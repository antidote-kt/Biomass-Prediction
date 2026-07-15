import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedHuberLoss(nn.Module):
    """按目标列加权的 Huber loss，权重顺序与 cfg.targets 一致。"""

    def __init__(self, weights, delta: float = 1.0, normalize_weights: bool = True):
        super().__init__()
        weight = torch.as_tensor(weights, dtype=torch.float32)
        if weight.ndim != 1:
            raise ValueError("target loss 权重必须是一维序列")
        if torch.any(weight < 0):
            raise ValueError("target loss 权重不能为负数")
        if float(weight.sum()) <= 0:
            raise ValueError("target loss 权重之和必须大于 0")
        if normalize_weights:
            weight = weight / weight.sum()
        self.register_buffer("weights", weight)
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"预测和目标形状不一致: pred={pred.shape}, target={target.shape}")
        if pred.shape[-1] != self.weights.numel():
            raise ValueError(
                f"预测目标数 {pred.shape[-1]} 与 loss 权重数 {self.weights.numel()} 不一致"
            )
        loss = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        return (loss * self.weights.view(1, -1)).sum(dim=1).mean()


def rmse_per_target(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐目标计算 RMSE，主要用于训练过程监控。"""
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def weighted_r2_global(pred: np.ndarray, true: np.ndarray, weights: np.ndarray) -> float:
    """按比赛口径计算加权全局 R^2。"""
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)

    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    w = np.broadcast_to(w, true.shape).reshape(-1)

    y = true.reshape(-1)
    yhat = pred.reshape(-1)

    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("权重之和必须大于 0")

    ybar = np.sum(w * y) / wsum
    rss = np.sum(w * (y - yhat) ** 2)
    tss = np.sum(w * (y - ybar) ** 2)
    if tss <= 0:
        return 0.0
    return 1.0 - (rss / tss)
