from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from biomass_training.model import BiomassModel


# 五个回归目标的顺序必须和训练时模型输出顺序保持一致。
# Dry_Total_g 位于最后一维的 index=4，是这次主要解释的目标。
TARGETS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")

# 与训练/验证 transform 保持一致的 ImageNet 归一化参数。
# 解释热力图时，预处理必须和训练时一致，否则梯度对应的输入分布会偏掉。
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class HeatmapCFG:
    """热力图脚本配置。

    日常使用时只需要改这里，然后直接运行：
    python heatmap.py

    如果要对比无辅助模型、辅助任务模型、最终模型，可以在 weights 里写多个权重：
    weights = (
        "no_aux=models/no_aux_fold0.pth",
        "aux=models/aux_fold0.pth",
        "final=models/biomass_fold0.pth",
    )
    等号左边会作为输出文件名中的模型标签。
    """

    # 输入图片和模型权重。
    image: Path = Path("../csiro-biomass/train/ID193102215.jpg")
    weights: Tuple[str, ...] = (
        "bi=models/biomass_fold0.pth",
    )

    # 输出目录。
    output_dir: Path = Path("./heatmaps")
    target: str = "Dry_Total_g" # 表示你要看模型预测哪个目标时关注了哪里。现在是看总干重 Dry_Total_g

    # 图像预处理和可视化参数。
    img_size: int = 512
    alpha: float = 0.45
    # 是热力图叠加到原图上的透明度。
    #0.45表示45% 热力图颜色 + 55% 原图。
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型结构参数；必须和训练该权重时的配置保持一致。
    backbone: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    dual_view: bool = True
    use_mamba: bool = True
    use_cross_attention: bool = True
    use_self_attention: bool = False
    num_heads: int = 8
    num_mamba_layers: int = 2
    log_targets: bool = True
    enable_5_head: bool = True


def parse_weight_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
    """解析多个权重输入。

    支持两种形式：
    1. name=/path/to/model.pth：输出文件里使用 name，适合对比 no_aux/aux/final。
    2. /path/to/model.pth：自动使用权重文件名作为输出前缀。
    """
    parsed = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
            parsed.append((name.strip(), Path(path)))
        else:
            path = Path(spec)
            parsed.append((path.stem, path))
    return parsed


def build_transform(img_size: int):
    """构建热力图推理用 transform。

    这里故意只使用 Resize + Normalize，不做任何随机增强。
    生成解释图时必须是确定性的，同一张图片才能在不同模型之间公平比较。
    """
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


def load_image_inputs(
    image_path: Path,
    img_size: int,
    dual_view: bool,
    device: torch.device,
) -> Tuple[Image.Image, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
    """读取图片并转成模型输入。

    返回两部分：
    - original：未归一化的原始 RGB 图，用来最后叠加彩色热力图。
    - model_input：送进模型的 tensor。双视角模型返回 (left, right)，单视角返回一个 tensor。

    项目里的 dual_view 训练方式会把一张横向牧场图裁成左右两个正方形视角：
    左视角取 [0:h]，右视角取 [w-h:w]。这里必须复用同样裁剪逻辑。
    """
    original = Image.open(image_path).convert("RGB")
    transform = build_transform(img_size)

    if dual_view:
        w, h = original.size
        # 原始图通常是宽图，左右各取一个 h*h 正方形，对应训练集 Dataset 的逻辑。
        left = original.crop((0, 0, h, h))
        right = original.crop((w - h, 0, w, h))
        # Albumentations 输出 C*H*W，再补 batch 维度，变成 1*C*H*W。
        left_tensor = transform(image=np.array(left))["image"].unsqueeze(0).to(device)
        right_tensor = transform(image=np.array(right))["image"].unsqueeze(0).to(device)
        return original, (left_tensor, right_tensor)

    tensor = transform(image=np.array(original))["image"].unsqueeze(0).to(device)
    return original, tensor


def load_model(cfg: HeatmapCFG, weight_path: Path, device: torch.device) -> BiomassModel:
    """按配置重建模型结构并加载权重。

    注意：热力图只解释五个 biomass 主目标，辅助任务头不参与推理，所以 aux_dims={}。
    如果权重来自 DDP 训练，state_dict 的 key 可能带有 module. 前缀，这里会自动去掉。
    """
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
        enable_5_head=cfg.enable_5_head,
    ).to(device)

    state_dict = torch.load(weight_path, map_location="cpu")
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # strict=False 是为了兼容“训练时带辅助头、解释时不创建辅助头”的场景。
    # unexpected 多半是辅助头参数，missing 则提示当前结构里有参数没有从权重加载到。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[{weight_path.name}] 忽略未使用权重: {unexpected[:8]}")
    if missing:
        print(f"[{weight_path.name}] 未加载参数: {missing[:8]}")
    model.eval()
    return model


def target_score(model: BiomassModel, model_input, target_idx: int) -> torch.Tensor:
    """取指定目标的标量分数，作为反向传播的解释目标。

    对 batch 求 sum 是常见写法；这里 batch=1，等价于解释这一张图的该目标预测值。
    如果训练使用 log_targets=True，解释的是 log1p 空间的 Dry_Total_g 输出。
    """
    outputs = model(model_input)
    biomass = outputs["biomass"] if isinstance(outputs, dict) else outputs
    return biomass[:, target_idx].sum()


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    """把任意热力图归一化到 [0, 1]，便于保存和叠加。

    不同模型的原始梯度量级不可直接比较，归一化后主要比较“关注区域分布”。
    """
    heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap -= heatmap.min()
    denom = heatmap.max()
    if denom > 1e-8:
        heatmap /= denom
    return heatmap


def token_to_grid(token_map: torch.Tensor) -> np.ndarray:
    """把 ViT token 重要性向量恢复成二维网格。

    ViT/DINO 类模型通常输出形状为 B*N*C，其中 N 是 token 数。
    有些模型会在 patch token 前面放 cls/register token，所以这里尝试跳过 0~15 个前缀 token，
    找到剩余 token 数能组成正方形的位置，再 reshape 成 side*side。
    """
    values = token_map.detach().float().cpu().numpy()
    n_tokens = values.shape[0]
    for prefix in range(0, min(16, n_tokens)):
        grid_tokens = n_tokens - prefix
        side = int(round(grid_tokens ** 0.5))
        if side * side == grid_tokens:
            return normalize_map(values[prefix:].reshape(side, side))
    return normalize_map(values.reshape(1, n_tokens))


def select_gradcam_module(model: BiomassModel):
    """选择最接近预测头、且仍保留空间/token 结构的 Grad-CAM 目标层。"""
    if model.dual_view:
        if model.use_mamba and hasattr(model, "mamba_fusion"):
            return "mamba_fusion", model.mamba_fusion, True
        if model.use_cross_attention and hasattr(model, "cross_attn"):
            return "cross_attn", model.cross_attn, True
        return "backbone", model.backbone, False

    if model.use_mamba and hasattr(model, "mamba_fusion"):
        return "mamba_fusion", model.mamba_fusion, False
    if model.use_self_attention and hasattr(model, "self_attn"):
        return "self_attn", model.self_attn, False
    return "backbone", model.backbone, False


def tensor_from_hook_output(output) -> torch.Tensor:
    """从 hook 输出中取出用于 Grad-CAM 的 tensor。"""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Grad-CAM 目标层输出不是 tensor: {type(output).__name__}")


def activation_to_map(activation: torch.Tensor) -> np.ndarray:
    """把一次 hook 捕获到的激活和梯度转成热力图。"""
    if activation.grad is None:
        raise RuntimeError("Grad-CAM 目标层没有梯度，请确认该层参与了目标输出计算")

    if activation.ndim == 4:
        # CNN 风格输出：B*C*H*W。
        grads = activation.grad[0]
        acts = activation.detach()[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        return normalize_map(cam.cpu().numpy())

    if activation.ndim == 3:
        # ViT/token 风格输出：B*N*C。
        grads = activation.grad[0]
        acts = activation.detach()[0]
        cam = torch.relu((grads * acts).sum(dim=1))
        return token_to_grid(cam)

    raise ValueError(f"不支持的 Grad-CAM 目标层输出维度: {tuple(activation.shape)}")


def fused_activation_to_view_maps(activation: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """把双视角融合后的 B*N*C token 热力图拆成 left/right 两张图。"""
    if activation.grad is None:
        raise RuntimeError("Grad-CAM 目标层没有梯度，请确认该层参与了目标输出计算")
    if activation.ndim != 3:
        raise ValueError(
            "双视角融合层 Grad-CAM 目前需要 B*N*C token 输出，"
            f"当前维度为: {tuple(activation.shape)}"
        )

    grads = activation.grad[0]
    acts = activation.detach()[0]
    cam = torch.relu((grads * acts).sum(dim=1))
    n_tokens = cam.shape[0]
    if n_tokens % 2 != 0:
        raise ValueError(f"融合 token 数不能平均拆成左右视角: {n_tokens}")
    mid = n_tokens // 2
    return token_to_grid(cam[:mid]), token_to_grid(cam[mid:])


def gradcam_maps(
    model: BiomassModel,
    model_input,
    target_idx: int,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """生成 Grad-CAM 风格热力图。

    传统 Grad-CAM 适合 CNN 的 B*C*H*W 特征：
    - 对目标输出反传；
    - 用梯度在空间维度上的均值作为每个通道权重；
    - 对加权后的激活求和并 ReLU，得到关注区域。

    对 ViT/token 模型，这里使用 token 级近似：
    - 目标层输出 B*N*C；
    - 对每个 token 计算 grad * activation；
    - 再把 token 重要性恢复成二维 patch 网格。

    默认会自适应选择最接近预测头的 token/feature 层：
    mamba_fusion -> cross_attn/self_attn -> backbone。
    这样在启用 cross attention + LocalMamba 时，解释的是融合后、真正送去 pooling/head 的特征。
    如果没有这些模块，则自动回退到 backbone。

    这个方法更接近“模型预测前最后一层 token 关注区域”，但会依赖目标层输出形状。
    """
    activations: List[torch.Tensor] = []
    target_name, target_module, fused_dual_view = select_gradcam_module(model)

    def save_activation(_module, _inputs, output):
        # retain_grad 让非叶子张量在 backward 后也保留梯度。
        tensor = tensor_from_hook_output(output)
        tensor.retain_grad()
        activations.append(tensor)

    print(f"Grad-CAM 目标层: {target_name}")
    handle = target_module.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        score = target_score(model, model_input, target_idx)
        score.backward()
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError(f"Grad-CAM 目标层没有捕获到激活: {target_name}")

    if isinstance(model_input, tuple):
        if fused_dual_view:
            return fused_activation_to_view_maps(activations[-1])

        # backbone 在 dual_view 下会被调用两次，因此 activations 依次保存 left/right。
        maps = [activation_to_map(activation) for activation in activations]
        if len(maps) < 2:
            raise RuntimeError(f"dual_view=True 时没有捕获到左右两个视角的 {target_name} 激活")
        return maps[0], maps[1]

    return activation_to_map(activations[-1])


def compose_original_size_map(
    original: Image.Image,
    heatmap,
    dual_view: bool,
) -> np.ndarray:
    """把模型输入尺寸上的热力图映射回原始图片尺寸。

    单视角：直接 resize 回原图宽高。
    双视角：左/右两个 h*h 热力图分别贴回原图左侧和右侧；
    如果左右视角在中间有重叠区域，使用 counts 做平均，避免重叠区域被加倍。
    """
    w, h = original.size
    if dual_view:
        left_map, right_map = heatmap
        canvas = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        left_resized = cv2.resize(left_map, (h, h), interpolation=cv2.INTER_CUBIC)
        right_resized = cv2.resize(right_map, (h, h), interpolation=cv2.INTER_CUBIC)
        canvas[:, :h] += left_resized
        counts[:, :h] += 1
        canvas[:, w - h:w] += right_resized
        counts[:, w - h:w] += 1
        counts[counts == 0] = 1
        return normalize_map(canvas / counts)

    return normalize_map(cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC))


def save_heatmap_outputs(original: Image.Image, heatmap: np.ndarray, output_prefix: Path, alpha: float) -> None:
    """保存两种输出文件。

    - .overlay.jpg：彩色热力图按 alpha 叠加到原图，适合展示模型关注区域。
    - .heatmap.png：单通道灰度重要性图，适合后续定量分析或重新配色。
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(original)
    heat_uint8 = np.uint8(255 * normalize_map(heatmap))
    # OpenCV 的 applyColorMap 输出是 BGR，需要转成 RGB 再和 PIL/原图一致。
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip((1 - alpha) * rgb + alpha * color, 0, 255))

    Image.fromarray(overlay).save(output_prefix.with_suffix(".overlay.jpg"), quality=95)
    Image.fromarray(heat_uint8).save(output_prefix.with_suffix(".heatmap.png"))


def validate_cfg(cfg: HeatmapCFG) -> None:
    """检查配置是否可用，尽早给出清楚的错误信息。"""
    if cfg.target not in TARGETS:
        raise ValueError(f"target 必须是 {TARGETS} 之一，当前为: {cfg.target}")
    if not cfg.image.exists():
        raise FileNotFoundError(f"找不到输入图片: {cfg.image}")
    if len(cfg.weights) == 0:
        raise ValueError("weights 不能为空，请至少配置一个模型权重")
    for _name, weight_path in parse_weight_specs(cfg.weights):
        if not weight_path.exists():
            raise FileNotFoundError(f"找不到模型权重: {weight_path}")


def main() -> None:
    """脚本入口：同一张图可同时对多个模型权重生成热力图。"""
    cfg = HeatmapCFG()
    validate_cfg(cfg)
    device = torch.device(cfg.device)
    target_idx = TARGETS.index(cfg.target)
    weight_specs = parse_weight_specs(cfg.weights)

    # 图片只需要预处理一次；多个权重复用同一个输入，保证对比严格来自模型差异。
    original, model_input = load_image_inputs(cfg.image, cfg.img_size, cfg.dual_view, device)
    for name, weight_path in weight_specs:
        print(f"生成热力图: {name} <- {weight_path}")
        model = load_model(cfg, weight_path, device)
        heatmap = gradcam_maps(model, model_input, target_idx)

        # 将模型输入空间的热力图拼回原图空间，再保存叠加图和灰度图。
        full_map = compose_original_size_map(original, heatmap, cfg.dual_view)
        output_prefix = cfg.output_dir / f"{cfg.image.stem}.{name}.{cfg.target}.gradcam"
        save_heatmap_outputs(original, full_map, output_prefix, cfg.alpha)
        print(f"已保存: {output_prefix.with_suffix('.overlay.jpg')}")
        print(f"已保存: {output_prefix.with_suffix('.heatmap.png')}")


if __name__ == "__main__":
    main()
