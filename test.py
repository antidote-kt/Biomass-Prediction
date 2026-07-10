import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

@dataclass(frozen=True)
class InferenceCFG:
    """单文件推理配置。"""

    seed: int = 42
    data_dir: Path = Path("../csiro-biomass")
    model_dir: Path = Path("./weights")
    output_csv: Path = Path("./submission.csv")
    model_paths: Sequence[str] = ()
    img_size: int = 512
    batch_size: int = 4
    num_workers: int = 0
    backbone: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    log_targets: bool = True
    dual_view: bool = True
    use_mamba: bool = True
    use_cross_attention: bool = True
    use_self_attention: bool = False
    num_heads: int = 8
    num_mamba_layers: int = 2
    targets: Sequence[str] = (
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    )

def seed_everything(seed: int) -> None:
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inverse_log_targets(x):
    """将 log(1+y) 空间的预测值还原到原始生物量空间。"""
    if isinstance(x, torch.Tensor):
        return torch.expm1(x)
    return np.expm1(x)


class LocalMambaBlock(nn.Module):
    """轻量级序列建模模块，用深度卷积模拟局部 token 交互。"""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class SelfAttentionBlock(nn.Module):
    """单视角建模使用的自注意力模块。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = shortcut + attn_out

        shortcut = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = shortcut + x
        return x


class CrossAttentionBlock(nn.Module):
    """双视角特征融合使用的交叉注意力模块。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_l = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fused = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        x_l_norm = self.norm_l(x_l)
        x_r_norm = self.norm_r(x_r)

        attn_l2r, _ = self.cross_attn(x_l_norm, x_r_norm, x_r_norm)
        attn_r2l, _ = self.cross_attn(x_r_norm, x_l_norm, x_l_norm)

        x_l = x_l + attn_l2r
        x_r = x_r + attn_r2l

        x_fused = torch.cat([x_l, x_r], dim=1)
        x_fused = x_fused + self.ffn(self.norm_fused(x_fused))
        return x_fused


class BiomassModel(nn.Module):
    """可配置的生物量回归模型。"""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        backbone_path: Optional[Path] = None,
        dual_view: bool = True,
        use_mamba: bool = True,
        use_cross_attention: bool = False,
        use_self_attention: bool = False,
        num_heads: int = 8,
        num_mamba_layers: int = 2,
        log_targets: bool = True,
        aux_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        self.dual_view = dual_view
        self.use_mamba = use_mamba
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        self.log_targets = log_targets
        self.aux_dims = aux_dims or {}

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        nf = self.backbone.num_features

        if self.dual_view:
            # 双视角模式下，左右图像先分别过同一个 backbone，再进行特征融合。
            if self.use_cross_attention:
                self.cross_attn = CrossAttentionBlock(nf, num_heads=num_heads, dropout=0.1)
            if self.use_mamba:
                self.mamba_fusion = nn.Sequential(*[
                    LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
                    for _ in range(num_mamba_layers)
                ])
        else:
            if self.use_self_attention:
                self.self_attn = SelfAttentionBlock(nf, num_heads=num_heads, dropout=0.1)
            if self.use_mamba:
                self.mamba_fusion = nn.Sequential(*[
                    LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
                    for _ in range(num_mamba_layers)
                ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        def make_head() -> nn.Sequential:
            """创建单个回归头，每个目标独立预测。"""
            return nn.Sequential(
                nn.Linear(nf, nf // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(nf // 2, 1),
                nn.Softplus(),
            )

        self.head_green = make_head()
        self.head_dead = make_head()
        self.head_clover = make_head()
        self.aux_heads = nn.ModuleDict()

        # 辅助头只在提供 aux_dims 时创建，推理脚本默认不加载这些头。
        if "height" in self.aux_dims:
            self.aux_heads["height"] = nn.Linear(nf, 1)
        if "ndvi" in self.aux_dims:
            self.aux_heads["ndvi"] = nn.Linear(nf, 1)
        if "species" in self.aux_dims:
            self.aux_heads["species"] = nn.Linear(nf, self.aux_dims["species"])
        if "state" in self.aux_dims:
            self.aux_heads["state"] = nn.Linear(nf, self.aux_dims["state"])
        if "month" in self.aux_dims:
            self.aux_heads["month"] = nn.Linear(nf, self.aux_dims["month"])

    def forward(self, x) -> torch.Tensor:
        if self.dual_view:
            if not isinstance(x, (tuple, list)) or len(x) != 2:
                raise ValueError("dual_view=True 时，输入必须是 (left, right) 形式的二元序列")
            left, right = x
            x_l = self.backbone(left)
            x_r = self.backbone(right)

            x_fused = None
            if self.use_cross_attention:
                # 让左右视角互相查询信息，融合互补区域。
                x_fused = self.cross_attn(x_l, x_r)

            if self.use_mamba:
                if x_fused is None:
                    x_fused = torch.cat([x_l, x_r], dim=1)
                # 在拼接后的 token 序列上继续做局部序列建模。
                x_fused = self.mamba_fusion(x_fused)

            if x_fused is None:
                x_fused = torch.cat([x_l, x_r], dim=1)

            x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        else:
            if isinstance(x, (tuple, list)):
                x = x[0]
            x_feat = self.backbone(x)
            if self.use_self_attention:
                x_feat = self.self_attn(x_feat)
            if self.use_mamba:
                x_feat = self.mamba_fusion(x_feat)
            x_pool = self.pool(x_feat.transpose(1, 2)).flatten(1)

        green = self.head_green(x_pool)
        dead = self.head_dead(x_pool)
        clover = self.head_clover(x_pool)
        if self.log_targets:
            # 三个头预测 log(1+y) 空间的 Green/Dead/Clover；
            # GDM 和 Total 先还原到原始尺度相加，再映射回 log 空间对齐训练目标。
            green_raw = torch.expm1(green)
            dead_raw = torch.expm1(dead)
            clover_raw = torch.expm1(clover)
            gdm = torch.log1p(green_raw + clover_raw)
            total = torch.log1p(green_raw + clover_raw + dead_raw)
        else:
            gdm = green + clover
            total = gdm + dead

        outputs = {
            "biomass": torch.cat([green, dead, clover, gdm, total], dim=1)
        }
        for name, head in self.aux_heads.items():
            aux_output = head(x_pool)
            outputs[name] = aux_output.squeeze(-1) if aux_output.shape[-1] == 1 else aux_output
        return outputs





class PastureImageTestDataset(Dataset):
    """生物量预测推理数据集。"""

    def __init__(self, df: pd.DataFrame, image_root: Path, img_size: int, dual_view: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.dual_view = dual_view
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.dual_view:
            # 推理阶段沿用训练时的双视角裁剪方式。
            left = img.crop((0, 0, h, h))
            right = img.crop((w - h, 0, w, h))
            img1 = self.transform(image=np.array(left))["image"]
            img2 = self.transform(image=np.array(right))["image"]
            return (img1, img2), row["image_path"]

        img1 = self.transform(image=np.array(img))["image"]
        return img1, row["image_path"]


def collate_fn_test(batch):
    """拼接单视角或双视角的测试 batch。"""
    first_x, _ = batch[0]
    image_paths = [image_path for _, image_path in batch]

    if isinstance(first_x, (tuple, list)):
        imgs1 = torch.stack([x[0] for x, _ in batch])
        imgs2 = torch.stack([x[1] for x, _ in batch])
        return (imgs1, imgs2), image_paths

    imgs = torch.stack([x for x, _ in batch])
    return imgs, image_paths


def resolve_model_paths(cfg: InferenceCFG) -> list[Path]:
    """解析需要参与集成推理的模型权重路径。"""
    if cfg.model_paths:
        return [Path(p) for p in cfg.model_paths]
    return sorted(cfg.model_dir.glob("*.pth"))


def load_model_for_inference(cfg: InferenceCFG, weight_path: Path, device: torch.device) -> BiomassModel:
    """加载生物量模型；推理时忽略训练专用的辅助头。"""
    model = BiomassModel(
        model_name=cfg.backbone,
        pretrained=False,
        dual_view=cfg.dual_view,
        use_mamba=cfg.use_mamba,
        use_cross_attention=cfg.use_cross_attention,
        use_self_attention=cfg.use_self_attention,
        num_heads=cfg.num_heads,
        num_mamba_layers=cfg.num_mamba_layers,
        log_targets=cfg.log_targets,
        aux_dims={},
    ).to(device)

    state_dict = torch.load(weight_path, map_location="cpu")
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_with_model(model: BiomassModel, loader: DataLoader, device: torch.device, log_targets: bool) -> np.ndarray:
    """执行推理，只返回五个生物量目标的预测。"""
    preds_all = []
    amp_enabled = device.type == "cuda"

    for x, _ in tqdm(loader, desc="推理", leave=False):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            imgs1 = x[0].to(device, non_blocking=True)
            imgs2 = x[1].to(device, non_blocking=True)
            model_input = (imgs1, imgs2)
        else:
            model_input = x.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(model_input)

        biomass_pred = outputs["biomass"] if isinstance(outputs, dict) else outputs
        biomass_pred = biomass_pred.float().cpu()
        if log_targets:
            # 训练时若使用 log1p 目标，推理输出需要还原到原始尺度。
            biomass_pred = inverse_log_targets(biomass_pred)
        biomass_pred = torch.clamp(biomass_pred, min=0.0)
        preds_all.append(biomass_pred.numpy())

    return np.vstack(preds_all)


def build_submission(test_df: pd.DataFrame, test_wide: pd.DataFrame, predictions: np.ndarray, targets: Sequence[str]) -> pd.DataFrame:
    """将每张图片的预测结果转回官方提交要求的长表格式。"""
    target_to_idx = {target: i for i, target in enumerate(targets)}
    image_to_idx = {image_path: i for i, image_path in enumerate(test_wide["image_path"].tolist())}

    submission_rows = []
    for _, row in test_df.iterrows():
        img_i = image_to_idx[row["image_path"]]
        tgt_i = target_to_idx[row["target_name"]]
        prediction = float(predictions[img_i, tgt_i])
        submission_rows.append({"sample_id": row["sample_id"], "target": prediction})

    return pd.DataFrame(submission_rows)


def main():
    cfg = InferenceCFG()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    test_df = pd.read_csv(cfg.data_dir / "test.csv")
    test_wide = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
    print("测试集行数:", len(test_df))
    print("测试图片数:", len(test_wide))

    dataset = PastureImageTestDataset(
        test_wide,
        image_root=cfg.data_dir,
        img_size=cfg.img_size,
        dual_view=cfg.dual_view,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_test,
    )

    model_paths = resolve_model_paths(cfg)
    if len(model_paths) == 0:
        raise ValueError("未找到模型权重。请设置 InferenceCFG.model_paths，或把权重文件放到 model_dir。")

    print("找到以下权重:")
    for path in model_paths:
        print("-", path)

    all_preds = []
    for weight_path in model_paths:
        model = load_model_for_inference(cfg, weight_path, device)
        preds = predict_with_model(model, loader, device, log_targets=cfg.log_targets)
        all_preds.append(preds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 多个 fold 权重做简单平均，得到最终集成预测。
    ensemble_predictions = np.mean(np.stack(all_preds, axis=0), axis=0)
    print("集成预测形状:", ensemble_predictions.shape)

    submission_df = build_submission(test_df, test_wide, ensemble_predictions, cfg.targets)
    submission_df.to_csv(cfg.output_csv, index=False)
    print(f"提交文件已保存到 {cfg.output_csv}")
    print(submission_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
