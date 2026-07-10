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


def load_model_weights(model: nn.Module, weight_path: Path, strict: bool = True) -> None:
    """Load a saved model state_dict, handling DataParallel prefixes."""
    if not weight_path.exists():
        raise FileNotFoundError(f"resume weight not found: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded resume weights: {weight_path} (strict={strict})")


def resolve_resume_weight_path(cfg: CFG, fold: Optional[int] = None) -> Optional[Path]:
    """Resolve the checkpoint path for full-data training or a specific fold."""
    if not cfg.resume_training:
        return None

    if cfg.resume_model_paths:
        idx = 0 if fold is None else fold
        if idx >= len(cfg.resume_model_paths):
            raise ValueError(
                f"resume_model_paths has {len(cfg.resume_model_paths)} paths, "
                f"but fold {idx} was requested"
            )
        return Path(cfg.resume_model_paths[idx])

    if cfg.resume_model_dir is not None:
        if fold is None:
            return cfg.resume_model_dir / "biomass_all_train.pth"
        return cfg.resume_model_dir / f"biomass_fold{fold}.pth"

    raise ValueError(
        "resume_training=True, but neither resume_model_paths nor resume_model_dir is set"
    )


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
    """Train one epoch and return mean loss."""
    model.train()
    running = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False, disable=not dist_ctx.is_main_process):
        x, y, aux_targets = batch
        if isinstance(x, tuple) and len(x) == 2:
            imgs1, imgs2 = x
            imgs1 = imgs1.to(device, non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            model_input = (imgs1, imgs2)
        else:
            model_input = x.to(device, non_blocking=True)
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
                loss = loss + compute_auxiliary_loss(outputs, aux_targets, cfg)

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    """Run validation and return loss, RMSE and weighted R^2."""
    model.eval()
    running = 0.0
    n = 0
    preds, trues = [], []

    for batch in tqdm(loader, desc="valid", leave=False, disable=not dist_ctx.is_main_process):
        x, y, aux_targets = batch
        if isinstance(x, tuple) and len(x) == 2:
            imgs1, imgs2 = x
            imgs1 = imgs1.to(device, non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            model_input = (imgs1, imgs2)
        else:
            model_input = x.to(device, non_blocking=True)
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
    preds = np.vstack(gathered_preds)
    trues = np.vstack(gathered_trues)
    rmse = rmse_per_target(preds, trues)
    score_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
    r2 = weighted_r2_global(preds, trues, score_weights)
    return running / max(n, 1), rmse, r2
