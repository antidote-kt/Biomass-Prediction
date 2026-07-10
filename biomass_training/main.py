import gc

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import CFG
from .data import (
    PastureImageTrainDataset,
    build_transforms,
    collate_fn_train,
    make_train_wide,
    prepare_auxiliary_metadata,
)
from .distributed import DistributedContext, barrier, cleanup_distributed, reduce_mean, setup_distributed
from .losses import WeightedSmoothL1Loss
from .model import BiomassModel
from .trainer import load_model_weights, resolve_resume_weight_path, train_one_epoch, validate
from .utils import seed_everything


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model for saving or backbone access."""
    return model.module if hasattr(model, "module") else model


def build_model(cfg: CFG, device: torch.device, aux_dims=None) -> nn.Module:
    """Construct the configured model."""
    return BiomassModel(
        model_name=cfg.backbone,
        pretrained=True,
        dual_view=cfg.dual_view,
        use_mamba=cfg.use_mamba,
        use_cross_attention=cfg.use_cross_attention,
        use_self_attention=cfg.use_self_attention,
        num_heads=cfg.num_heads,
        num_mamba_layers=cfg.num_mamba_layers,
        aux_dims=aux_dims,
    ).to(device)


def build_loader(
    dataset,
    cfg: CFG,
    shuffle: bool,
    drop_last: bool,
    dist_ctx: DistributedContext,
) -> DataLoader:
    """Construct a dataloader, using DistributedSampler when DDP is enabled."""
    sampler = None
    if dist_ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn_train,
    )


def prepare_training(
    model: nn.Module,
    cfg: CFG,
    device: torch.device,
    dist_ctx: DistributedContext,
):
    """Prepare wrappers and optimizers for training."""
    base_model = unwrap_model(model)
    if cfg.freeze_backbone:
        for p in base_model.backbone.parameters():
            p.requires_grad = False

    if dist_ctx.enabled:
        ddp_kwargs = {
            "device_ids": [dist_ctx.local_rank],
            "output_device": dist_ctx.local_rank,
            "find_unused_parameters": cfg.find_unused_parameters,
        }
        model = DDP(model, **ddp_kwargs)
    elif cfg.use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    return model, optimizer, scheduler, scaler, amp_enabled


def cleanup_training(*objects) -> None:
    """Release large training objects between folds."""
    del objects
    torch.cuda.empty_cache()
    gc.collect()


def make_dataset(cfg: CFG, df, image_root, transform):
    """Construct a train/validation dataset with optional auxiliary targets."""
    return PastureImageTrainDataset(
        df,
        image_root,
        transform=transform,
        dual_view=cfg.dual_view,
        log_targets=cfg.log_targets,
        use_auxiliary_tasks=cfg.use_auxiliary_tasks,
    )


def log(msg: str, dist_ctx: DistributedContext) -> None:
    """Print only on the main process."""
    if dist_ctx.is_main_process:
        print(msg)


def run_full_data_training(
    cfg: CFG,
    train_wide,
    train_tfms,
    loss_fn,
    device: torch.device,
    aux_dims,
    dist_ctx: DistributedContext,
) -> None:
    """Train on the full dataset without validation."""
    log(f"\n{'=' * 60}", dist_ctx)
    log("Using full training data mode without validation split", dist_ctx)
    log(f"{'=' * 60}", dist_ctx)

    tr_df = train_wide.reset_index(drop=True)
    tr_ds = make_dataset(cfg, tr_df, cfg.data_dir, train_tfms)
    tr_loader = build_loader(tr_ds, cfg, shuffle=True, drop_last=True, dist_ctx=dist_ctx)

    model = build_model(cfg, device, aux_dims=aux_dims)
    resume_path = resolve_resume_weight_path(cfg, fold=None)
    if resume_path is not None:
        load_model_weights(model, resume_path, strict=cfg.resume_strict)

    model, optimizer, scheduler, scaler, amp_enabled = prepare_training(model, cfg, device, dist_ctx)
    best_path = cfg.save_dir / "biomass_all_train.pth"

    log(f"\n{'=' * 60}", dist_ctx)
    log(f"Full training set | train={len(tr_ds)} | save={best_path}", dist_ctx)
    log(f"{'=' * 60}", dist_ctx)

    for epoch in range(cfg.epochs):
        if isinstance(tr_loader.sampler, DistributedSampler):
            tr_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            cfg,
            tr_loader,
            optimizer,
            loss_fn,
            scaler,
            amp=amp_enabled,
            device=device,
            dist_ctx=dist_ctx,
        )
        train_loss = reduce_mean(train_loss, device, dist_ctx)
        scheduler.step()
        log(f"epoch {epoch + 1}/{cfg.epochs} | train_loss {train_loss:.5f}", dist_ctx)

    if dist_ctx.is_main_process:
        model_to_save = unwrap_model(model)
        torch.save(model_to_save.state_dict(), best_path)
        print(f"Training completed, model saved to {best_path}")

    barrier(dist_ctx)
    cleanup_training(model, tr_loader, tr_ds)


def run_kfold_training(
    cfg: CFG,
    train_wide,
    train_tfms,
    val_tfms,
    loss_fn,
    device: torch.device,
    aux_dims,
    dist_ctx: DistributedContext,
) -> None:
    """Run K-fold training."""
    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_wide)):
        if cfg.one_fold and fold != 0:
            continue

        tr_df = train_wide.iloc[tr_idx].reset_index(drop=True)
        va_df = train_wide.iloc[va_idx].reset_index(drop=True)

        tr_ds = make_dataset(cfg, tr_df, cfg.data_dir, train_tfms)
        va_ds = make_dataset(cfg, va_df, cfg.data_dir, val_tfms)
        tr_loader = build_loader(tr_ds, cfg, shuffle=True, drop_last=True, dist_ctx=dist_ctx)
        va_loader = build_loader(va_ds, cfg, shuffle=False, drop_last=False, dist_ctx=dist_ctx)

        model = build_model(cfg, device, aux_dims=aux_dims)
        resume_path = resolve_resume_weight_path(cfg, fold=fold)
        if resume_path is not None:
            load_model_weights(model, resume_path, strict=cfg.resume_strict)

        model, optimizer, scheduler, scaler, amp_enabled = prepare_training(model, cfg, device, dist_ctx)
        best_score = -float("inf")
        best_path = cfg.save_dir / f"biomass_fold{fold}.pth"

        log(f"\n{'=' * 60}", dist_ctx)
        log(f"Fold {fold} | train={len(tr_ds)} val={len(va_ds)} | save={best_path}", dist_ctx)
        log(f"{'=' * 60}", dist_ctx)

        for epoch in range(cfg.epochs):
            if isinstance(tr_loader.sampler, DistributedSampler):
                tr_loader.sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model,
                cfg,
                tr_loader,
                optimizer,
                loss_fn,
                scaler,
                amp=amp_enabled,
                device=device,
                dist_ctx=dist_ctx,
            )
            val_loss, val_rmse, val_r2 = validate(
                model,
                cfg,
                va_loader,
                loss_fn,
                amp=amp_enabled,
                device=device,
                dist_ctx=dist_ctx,
            )

            train_loss = reduce_mean(train_loss, device, dist_ctx)
            val_loss = reduce_mean(val_loss, device, dist_ctx)
            scheduler.step()

            score = float(val_r2)
            log(
                f"epoch {epoch + 1}/{cfg.epochs} | "
                f"train_loss {train_loss:.5f} | val_loss {val_loss:.5f} | "
                f"weighted_r2 {val_r2:.6f} | rmse {val_rmse}",
                dist_ctx,
            )

            if score > best_score and dist_ctx.is_main_process:
                best_score = score
                model_to_save = unwrap_model(model)
                torch.save(model_to_save.state_dict(), best_path)

        log(f"Fold {fold} best weighted_r2: {best_score:.6f}", dist_ctx)
        barrier(dist_ctx)
        cleanup_training(model, tr_loader, va_loader, tr_ds, va_ds)


def main():
    cfg = CFG()
    dist_ctx = setup_distributed(cfg.use_distributed, backend=cfg.distributed_backend)
    seed_everything(cfg.seed + dist_ctx.rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", dist_ctx.local_rank if dist_ctx.enabled else 0)
    else:
        device = torch.device("cpu")
    log(f"Current Device: {device}", dist_ctx)
    if dist_ctx.enabled:
        log(f"Distributed training enabled | world_size={dist_ctx.world_size}", dist_ctx)

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_wide = make_train_wide(cfg.data_dir / cfg.train_csv)
        train_wide, aux_dims = prepare_auxiliary_metadata(train_wide)
        log(f"Train wide shape: {train_wide.shape}", dist_ctx)
        if dist_ctx.is_main_process:
            if "display" in globals():
                display(train_wide.head())
            else:
                print(train_wide.head())

        train_tfms, val_tfms = build_transforms(cfg.img_size)
        loss_fn = WeightedSmoothL1Loss(cfg.loss_weights, beta=cfg.loss_beta).to(device)
        log(f"Loss: WeightedSmoothL1Loss(beta={cfg.loss_beta}, weights={cfg.loss_weights})", dist_ctx)
        if cfg.log_targets:
            log("Biomass targets are trained in log(1+y) space.", dist_ctx)
        if cfg.use_auxiliary_tasks:
            log(f"Auxiliary tasks enabled: {sorted(aux_dims.keys())}", dist_ctx)

        if cfg.use_all_train_data:
            run_full_data_training(cfg, train_wide, train_tfms, loss_fn, device, aux_dims, dist_ctx)
        else:
            run_kfold_training(cfg, train_wide, train_tfms, val_tfms, loss_fn, device, aux_dims, dist_ctx)
    finally:
        cleanup_distributed(dist_ctx)
