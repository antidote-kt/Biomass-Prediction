import numpy as np


def rmse_per_target(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-target RMSE for monitoring."""
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def weighted_r2_global(pred: np.ndarray, true: np.ndarray, weights: np.ndarray) -> float:
    """Competition-style weighted global R^2."""
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)

    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    w = np.broadcast_to(w, true.shape).reshape(-1)

    y = true.reshape(-1)
    yhat = pred.reshape(-1)

    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("weights sum must be positive")

    ybar = np.sum(w * y) / wsum
    rss = np.sum(w * (y - yhat) ** 2)
    tss = np.sum(w * (y - ybar) ** 2)
    if tss <= 0:
        return 0.0
    return 1.0 - (rss / tss)
