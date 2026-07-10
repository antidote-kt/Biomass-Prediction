import numpy as np


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
