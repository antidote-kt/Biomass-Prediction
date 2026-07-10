from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class CFG:
    """训练流程的集中配置。"""

    # 基础训练参数
    seed: int = 42
    folds: int = 5 #
    img_size: int = 512
    batch_size: int = 4 #
    num_workers: int = 0
    epochs: int = 20 #
    backbone_lr: float = 1e-5
    head_lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5 #
    grad_clip: float = 1.0
    stratify_bins: int = 5

    # 损失函数与目标值处理
    loss_beta: float = 5.0
    # y = log(1 + y)变换
    log_targets: bool = True
    amp: bool = True
    freeze_backbone: bool = False

    # 断点或外部权重加载配置
    resume_training: bool = False #
    resume_strict: bool = False
    resume_model_paths: Tuple[str, ...] = () #
    backbone: str = "vit_huge_plus_patch16_dinov3.lvd1689m"

    # 数据与输出路径
    data_dir: Path = Path("../csiro-biomass") #
    train_csv: str = "train.csv"
    save_dir: Path = Path("./weights") #
    targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    # 模型结构开关
    dual_view: bool = True
    use_mamba: bool = True
    use_cross_attention: bool = True
    use_self_attention: bool = False
    num_heads: int = 8
    num_mamba_layers: int = 2
    use_distributed: bool = True
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = False

    # 训练时使用元数据做辅助监督，推理时不会输出这些辅助头
    use_auxiliary_tasks: bool = True
    aux_ndvi_weight: float = 0.10
    aux_height_weight: float = 0.10
    aux_species_weight: float = 0.05
    aux_state_weight: float = 0.05
    aux_month_weight: float = 0.05
