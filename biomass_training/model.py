from pathlib import Path
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn


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
            if not isinstance(x, tuple) or len(x) != 2:
                raise ValueError("Input must be a tuple of (left, right) when dual_view=True")
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
            if isinstance(x, tuple):
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
