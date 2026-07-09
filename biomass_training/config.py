from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class CFG:
    """Centralized training configuration."""

    seed: int = 42
    folds: int = 3
    img_size: int = 512
    batch_size: int = 2
    num_workers: int = 0
    one_fold: bool = False
    use_all_train_data: bool = False
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-2
    loss_beta: float = 1.0
    log_targets: bool = True
    loss_weights: Tuple[float, float, float, float, float] = (
        0.1,
        0.1,
        0.1,
        0.2,
        0.5,
    )
    amp: bool = True
    freeze_backbone: bool = False
    resume_training: bool = True
    resume_strict: bool = True
    resume_model_paths: Tuple[str, ...] = (
        "/kaggle/input/csiro-dual-view-attention-weight-model/weights/biomass_fold0.pth",
        "/kaggle/input/csiro-dual-view-attention-weight-model/weights/biomass_fold1.pth",
        "/kaggle/input/csiro-dual-view-attention-weight-model/weights/biomass_fold2.pth",
    )
    resume_model_dir: Optional[Path] = None
    backbone: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    data_dir: Path = Path("/kaggle/input/competitions/csiro-biomass")
    train_csv: str = "train.csv"
    save_dir: Path = Path("/kaggle/working/weights")
    targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    dual_view: bool = True
    use_mamba: bool = True
    use_cross_attention: bool = True
    use_self_attention: bool = False
    num_heads: int = 8
    num_mamba_layers: int = 2
    use_data_parallel: bool = False
    use_auxiliary_tasks: bool = True
    aux_ndvi_weight: float = 0.10
    aux_height_weight: float = 0.10
    aux_species_weight: float = 0.05
    aux_state_weight: float = 0.05
    aux_month_weight: float = 0.05
