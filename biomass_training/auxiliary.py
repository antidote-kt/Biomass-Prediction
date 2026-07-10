from typing import Dict

import torch
import torch.nn as nn

from .config import CFG


def compute_auxiliary_loss(
    outputs: Dict[str, torch.Tensor],
    aux_targets: Dict[str, torch.Tensor],
    cfg: CFG,
) -> torch.Tensor:
    """根据训练集元数据计算加权辅助任务损失。"""
    total = None

    if "height" in outputs and "height" in aux_targets:
        loss = nn.functional.smooth_l1_loss(outputs["height"], aux_targets["height"])
        total = _add_loss(total, cfg.aux_height_weight * loss)

    if "ndvi" in outputs and "ndvi" in aux_targets:
        loss = nn.functional.smooth_l1_loss(outputs["ndvi"], aux_targets["ndvi"])
        total = _add_loss(total, cfg.aux_ndvi_weight * loss)

    if "species" in outputs and "species" in aux_targets:
        loss = nn.functional.cross_entropy(outputs["species"], aux_targets["species"])
        total = _add_loss(total, cfg.aux_species_weight * loss)

    if "state" in outputs and "state" in aux_targets:
        loss = nn.functional.cross_entropy(outputs["state"], aux_targets["state"])
        total = _add_loss(total, cfg.aux_state_weight * loss)

    if "month" in outputs and "month" in aux_targets:
        loss = nn.functional.cross_entropy(outputs["month"], aux_targets["month"])
        total = _add_loss(total, cfg.aux_month_weight * loss)

    if total is None:
        device = outputs["biomass"].device
        return torch.tensor(0.0, device=device)
    return total


def _add_loss(current: torch.Tensor | None, new_loss: torch.Tensor) -> torch.Tensor:
    if current is None:
        return new_loss
    return current + new_loss
