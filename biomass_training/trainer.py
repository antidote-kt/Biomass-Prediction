from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .auxiliary import compute_auxiliary_loss
from .config import CFG
from .distributed import DistributedContext, gather_objects
from .metrics import rmse_per_target, weighted_r2_global


def move_model_input_to_device(x, device: torch.device):
    """把单视角或双视角输入搬到目标设备。"""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        imgs1, imgs2 = x
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        return (imgs1, imgs2)
    return x.to(device, non_blocking=True)


def load_model_weights(model: nn.Module, weight_path: Path, strict: bool = True) -> None:
    """加载保存的 state_dict，并兼容多卡训练产生的 module. 前缀。"""
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到续训权重: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    print(f"已加载续训权重: {weight_path} (strict={strict})")


def resolve_resume_weight_path(cfg: CFG, fold: Optional[int] = None) -> Optional[Path]:
    """解析指定 fold 对应的续训权重路径。"""
    if not cfg.resume_training:
        return None

    if cfg.resume_model_paths:
        idx = 0 if fold is None else fold
        if idx >= len(cfg.resume_model_paths):
            raise ValueError(
                f"resume_model_paths 只有 {len(cfg.resume_model_paths)} 个路径，"
                f"但当前请求的是第 {idx} 折"
        )
        return Path(cfg.resume_model_paths[idx])

    raise ValueError("resume_training=True，但 resume_model_paths 为空")


def train_one_epoch(
    model: nn.Module,
    cfg: CFG,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: Optional[torch.amp.GradScaler],
    amp: bool,
    device: torch.device,
    dist_ctx: DistributedContext,
) -> float:
    """训练一个 epoch，并返回平均 loss。"""
    model.train()
    running = 0.0
    n = 0

    for batch in tqdm(loader, desc="训练", leave=False, disable=not dist_ctx.is_main_process):
        x, y, aux_targets = batch
        model_input = move_model_input_to_device(x, device)
        y = y.to(device, non_blocking=True)
        aux_targets = {
            name: value.to(device, non_blocking=True) for name, value in aux_targets.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = model(model_input)
            biomass_pred = outputs["biomass"]
            loss = loss_fn(biomass_pred, y)
            if cfg.use_auxiliary_tasks:
                # 辅助任务 loss 只在训练阶段参与优化。
                loss = loss + compute_auxiliary_loss(outputs, aux_targets, cfg)

        if scaler is not None and amp:
            # CUDA 混合精度训练：缩放梯度以减少 underflow 风险。
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        bs = y.size(0)
        running += loss.item() * bs
        n += bs

    return running / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    cfg: CFG,
    loader: DataLoader,
    loss_fn: nn.Module,
    amp: bool,
    device: torch.device,
    dist_ctx: DistributedContext,
):
    """执行验证，并返回 loss、逐目标 RMSE 和加权 R^2。"""
    model.eval()
    running = 0.0
    n = 0
    preds, trues = [], []

    for batch in tqdm(loader, desc="验证", leave=False, disable=not dist_ctx.is_main_process):
        x, y, aux_targets = batch
        model_input = move_model_input_to_device(x, device)
        y = y.to(device, non_blocking=True)
        aux_targets = {
            name: value.to(device, non_blocking=True) for name, value in aux_targets.items()
        }

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = model(model_input)
            biomass_pred = outputs["biomass"]
            loss = loss_fn(biomass_pred, y)
            if cfg.use_auxiliary_tasks:
                loss = loss + compute_auxiliary_loss(outputs, aux_targets, cfg)

        bs = y.size(0)
        running += loss.item() * bs
        n += bs

        preds.append(biomass_pred.float().cpu().numpy())
        trues.append(y.float().cpu().numpy())

    local_preds = np.vstack(preds) if preds else np.empty((0, len(cfg.targets)), dtype=np.float32)
    local_trues = np.vstack(trues) if trues else np.empty((0, len(cfg.targets)), dtype=np.float32)
    gathered_preds = gather_objects(local_preds, dist_ctx)
    gathered_trues = gather_objects(local_trues, dist_ctx)
    # DDP 验证时收集所有进程的预测，按完整验证集计算指标。
    preds = np.vstack(gathered_preds)
    trues = np.vstack(gathered_trues)
    rmse = rmse_per_target(preds, trues)
    score_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
    r2 = weighted_r2_global(preds, trues, score_weights)
    return running / max(n, 1), rmse, r2
