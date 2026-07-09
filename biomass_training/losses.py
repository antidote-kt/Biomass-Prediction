import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """SmoothL1 loss with per-target weights."""

    def __init__(self, weights, beta: float = 1.0):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("loss weights must be a 1D sequence")
        if torch.any(weights < 0):
            raise ValueError("loss weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("loss weights sum must be positive")
        self.beta = beta
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
        if pred.size(-1) != self.weights.numel():
            raise ValueError(
                f"expected {self.weights.numel()} targets, got {pred.size(-1)}"
            )

        per_target_loss = nn.functional.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
            reduction="none",
        )
        weighted_loss = per_target_loss * self.weights.view(1, -1)
        return weighted_loss.sum(dim=1).mean()
