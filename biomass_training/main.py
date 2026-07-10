import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
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
from .model import BiomassModel
from .trainer import load_model_weights, resolve_resume_weight_path, train_one_epoch, validate
from .utils import seed_everything


def unwrap_model(model: nn.Module) -> nn.Module:
    """返回被 DDP 包裹前的原始模型，便于保存权重或访问 backbone。"""
    return model.module if hasattr(model, "module") else model


def build_model(cfg: CFG, device: torch.device, aux_dims=None) -> nn.Module:
    """根据配置构建模型。"""
    return BiomassModel(
        model_name=cfg.backbone,
        pretrained=True,
        dual_view=cfg.dual_view,
        use_mamba=cfg.use_mamba,
        use_cross_attention=cfg.use_cross_attention,
        use_self_attention=cfg.use_self_attention,
        num_heads=cfg.num_heads,
        num_mamba_layers=cfg.num_mamba_layers,
        log_targets=cfg.log_targets,
        aux_dims=aux_dims,
    ).to(device)


def build_loader(
    dataset,
    cfg: CFG,
    shuffle: bool,
    drop_last: bool,
    dist_ctx: DistributedContext,
) -> DataLoader:
    """构建 DataLoader；开启 DDP 时自动使用 DistributedSampler。"""
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
    """准备训练包装器、优化器、学习率调度器和混合精度组件。"""
    base_model = unwrap_model(model)
    if cfg.freeze_backbone:
        # 冻结 backbone 时只训练融合层、回归头和辅助头。
        for p in base_model.backbone.parameters():
            p.requires_grad = False

    if dist_ctx.enabled:
        # torchrun/DDP 场景：每个进程绑定一张本地 GPU。
        ddp_kwargs = {
            "device_ids": [dist_ctx.local_rank],
            "output_device": dist_ctx.local_rank,
            "find_unused_parameters": cfg.find_unused_parameters,
        }
        model = DDP(model, **ddp_kwargs)

    base_model = unwrap_model(model)
    backbone_params = [p for p in base_model.backbone.parameters() if p.requires_grad]
    head_params = [
        p for name, p in base_model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    ]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs - warmup_epochs, 1),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs, 1),
        )
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    # CPU 下不创建 GradScaler，避免无意义的 CUDA 组件。
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    return model, optimizer, scheduler, scaler, amp_enabled


def cleanup_training(*objects) -> None:
    """在不同 fold 之间释放大对象和显存缓存。"""
    del objects
    torch.cuda.empty_cache()
    gc.collect()


def make_dataset(cfg: CFG, df, image_root, transform):
    """创建训练或验证数据集，可按配置带上辅助目标。"""
    return PastureImageTrainDataset(
        df,
        image_root,
        transform=transform,
        dual_view=cfg.dual_view,
        log_targets=cfg.log_targets,
        use_auxiliary_tasks=cfg.use_auxiliary_tasks,
    )


def log(msg: str, dist_ctx: DistributedContext) -> None:
    """只在主进程打印日志，避免 DDP 多进程重复刷屏。"""
    if dist_ctx.is_main_process:
        print(msg)


def make_dry_total_quintiles(train_wide, bins: int) -> np.ndarray:
    """按 Dry_Total_g 生成等频分层标签，用于 StratifiedGroupKFold。"""
    dry_total = train_wide["Dry_Total_g"].to_numpy(dtype=np.float64)
    order = np.argsort(dry_total, kind="mergesort")
    labels = np.empty(len(dry_total), dtype=np.int64)
    labels[order] = np.minimum(np.arange(len(dry_total)) * bins // len(dry_total), bins - 1)
    return labels


def make_image_id_groups(train_wide) -> np.ndarray:
    """从 image_path 提取图片 ID，作为分组依据避免同图泄漏。"""
    return (
        train_wide["image_path"]
        .astype(str)
        .map(lambda path: path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        .to_numpy()
    )


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
    """执行按 Dry_Total 分层、按图片 ID 分组的 K 折交叉验证训练。"""
    stratify_labels = make_dry_total_quintiles(train_wide, cfg.stratify_bins)
    groups = make_image_id_groups(train_wide)
    kf = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_wide, stratify_labels, groups)):
        # 每个 fold 都重新切分数据集、构建模型和优化器，互不共享训练状态。
        tr_df = train_wide.iloc[tr_idx].reset_index(drop=True)
        va_df = train_wide.iloc[va_idx].reset_index(drop=True)

        tr_ds = make_dataset(cfg, tr_df, cfg.data_dir, train_tfms)
        va_ds = make_dataset(cfg, va_df, cfg.data_dir, val_tfms)
        tr_loader = build_loader(tr_ds, cfg, shuffle=True, drop_last=True, dist_ctx=dist_ctx)
        va_loader = build_loader(va_ds, cfg, shuffle=False, drop_last=False, dist_ctx=dist_ctx)

        model = build_model(cfg, device, aux_dims=aux_dims)
        resume_path = resolve_resume_weight_path(cfg, fold=fold)
        if resume_path is not None:
            # K 折训练时按 fold 加载对应的初始或续训权重。
            load_model_weights(model, resume_path, strict=cfg.resume_strict)

        model, optimizer, scheduler, scaler, amp_enabled = prepare_training(model, cfg, device, dist_ctx)
        best_score = -float("inf")
        best_path = cfg.save_dir / f"biomass_fold{fold}.pth"

        log(f"\n{'=' * 60}", dist_ctx)
        log(f"第 {fold} 折 | 训练图片数={len(tr_ds)} 验证图片数={len(va_ds)} | 保存路径={best_path}", dist_ctx)
        log(
            f"Dry_Total 五分位分布 | "
            f"训练={np.bincount(stratify_labels[tr_idx], minlength=cfg.stratify_bins).tolist()} "
            f"验证={np.bincount(stratify_labels[va_idx], minlength=cfg.stratify_bins).tolist()}",
            dist_ctx,
        )
        log(f"{'=' * 60}", dist_ctx)

        for epoch in range(cfg.epochs):
            if isinstance(tr_loader.sampler, DistributedSampler):
                # DDP 中每个 epoch 重设 sampler，保证各进程 shuffle 一致且不重复。
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
                f"轮次 {epoch + 1}/{cfg.epochs} | "
                f"训练损失 {train_loss:.5f} | 验证损失 {val_loss:.5f} | "
                f"加权 R2 {val_r2:.6f} | RMSE {val_rmse}",
                dist_ctx,
            )

            if score > best_score and dist_ctx.is_main_process:
                best_score = score
                model_to_save = unwrap_model(model)
                # 只保存验证集 weighted R2 最好的权重。
                torch.save(model_to_save.state_dict(), best_path)

        log(f"第 {fold} 折最佳加权 R2: {best_score:.6f}", dist_ctx)
        barrier(dist_ctx)
        cleanup_training(model, tr_loader, va_loader, tr_ds, va_ds)


def main():
    cfg = CFG()
    dist_ctx = setup_distributed(cfg.use_distributed, backend=cfg.distributed_backend)
    seed_everything(cfg.seed + dist_ctx.rank)

    # DDP 时每个进程使用自己的 local_rank；单进程时默认使用 cuda:0。
    if torch.cuda.is_available():
        device = torch.device("cuda", dist_ctx.local_rank if dist_ctx.enabled else 0)
    else:
        device = torch.device("cpu")
    log(f"当前设备: {device}", dist_ctx)
    if dist_ctx.enabled:
        log(f"已启用分布式训练 | 进程数={dist_ctx.world_size}", dist_ctx)

    if dist_ctx.is_main_process:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
    barrier(dist_ctx)

    try:
        train_wide = make_train_wide(cfg.data_dir / cfg.train_csv)
        train_wide, aux_dims = prepare_auxiliary_metadata(train_wide)
        log(f"训练宽表形状: {train_wide.shape}", dist_ctx)
        if dist_ctx.is_main_process:
            if "display" in globals():
                display(train_wide.head())
            else:
                print(train_wide.head())

        train_tfms, val_tfms = build_transforms(cfg.img_size)
        loss_fn = nn.HuberLoss(delta=cfg.loss_beta).to(device)
        log(f"损失函数: HuberLoss(delta={cfg.loss_beta})，作用于 log(1+y) 目标", dist_ctx)
        log(
            f"优化器: AdamW(backbone 学习率={cfg.backbone_lr}, head 学习率={cfg.head_lr}, "
            f"权重衰减={cfg.weight_decay})",
            dist_ctx,
        )
        log(f"学习率调度: 预热 {cfg.warmup_epochs} 轮 + 余弦退火", dist_ctx)
        log(f"梯度裁剪: {cfg.grad_clip}", dist_ctx)
        if cfg.log_targets:
            log("生物量目标将在 log(1+y) 空间训练。", dist_ctx)
        if cfg.use_auxiliary_tasks:
            log(f"已启用辅助任务: {sorted(aux_dims.keys())}", dist_ctx)

        run_kfold_training(cfg, train_wide, train_tfms, val_tfms, loss_fn, device, aux_dims, dist_ctx)
    finally:
        cleanup_distributed(dist_ctx)
