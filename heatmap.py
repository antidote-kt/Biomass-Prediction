import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from biomass_training.model import BiomassModel


TARGETS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_weight_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
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
    original = Image.open(image_path).convert("RGB")
    transform = build_transform(img_size)

    if dual_view:
        w, h = original.size
        left = original.crop((0, 0, h, h))
        right = original.crop((w - h, 0, w, h))
        left_tensor = transform(image=np.array(left))["image"].unsqueeze(0).to(device)
        right_tensor = transform(image=np.array(right))["image"].unsqueeze(0).to(device)
        return original, (left_tensor, right_tensor)

    tensor = transform(image=np.array(original))["image"].unsqueeze(0).to(device)
    return original, tensor


def load_model(args, weight_path: Path, device: torch.device) -> BiomassModel:
    model = BiomassModel(
        model_name=args.backbone,
        pretrained=False,
        dual_view=args.dual_view,
        use_mamba=args.use_mamba,
        use_cross_attention=args.use_cross_attention,
        use_self_attention=args.use_self_attention,
        num_heads=args.num_heads,
        num_mamba_layers=args.num_mamba_layers,
        log_targets=args.log_targets,
        aux_dims={},
        enable_5_head=args.enable_5_head,
    ).to(device)

    state_dict = torch.load(weight_path, map_location="cpu")
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[{weight_path.name}] 忽略未使用权重: {unexpected[:8]}")
    if missing:
        print(f"[{weight_path.name}] 未加载参数: {missing[:8]}")
    model.eval()
    return model


def target_score(model: BiomassModel, model_input, target_idx: int) -> torch.Tensor:
    outputs = model(model_input)
    biomass = outputs["biomass"] if isinstance(outputs, dict) else outputs
    return biomass[:, target_idx].sum()


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap -= heatmap.min()
    denom = heatmap.max()
    if denom > 1e-8:
        heatmap /= denom
    return heatmap


def tensor_saliency(tensor: torch.Tensor) -> np.ndarray:
    grad = tensor.grad.detach()
    saliency = grad.abs().mean(dim=1)[0].float().cpu().numpy()
    return normalize_map(saliency)


def input_importance_map(
    model: BiomassModel,
    model_input,
    target_idx: int,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    if isinstance(model_input, tuple):
        left, right = model_input
        left.requires_grad_(True)
        right.requires_grad_(True)
        left.grad = None
        right.grad = None
        score = target_score(model, (left, right), target_idx)
        score.backward()
        return tensor_saliency(left), tensor_saliency(right)

    model_input.requires_grad_(True)
    model_input.grad = None
    score = target_score(model, model_input, target_idx)
    score.backward()
    return tensor_saliency(model_input)


def token_to_grid(token_map: torch.Tensor) -> np.ndarray:
    values = token_map.detach().float().cpu().numpy()
    n_tokens = values.shape[0]
    for prefix in range(0, min(16, n_tokens)):
        grid_tokens = n_tokens - prefix
        side = int(round(grid_tokens ** 0.5))
        if side * side == grid_tokens:
            return normalize_map(values[prefix:].reshape(side, side))
    return normalize_map(values.reshape(1, n_tokens))


def gradcam_maps(
    model: BiomassModel,
    model_input,
    target_idx: int,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    activations: List[torch.Tensor] = []

    def save_activation(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = model.backbone.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        score = target_score(model, model_input, target_idx)
        score.backward()
    finally:
        handle.remove()

    maps = []
    for activation in activations:
        if activation.ndim == 4:
            grads = activation.grad[0]
            acts = activation.detach()[0]
            weights = grads.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=0))
            maps.append(normalize_map(cam.cpu().numpy()))
        elif activation.ndim == 3:
            grads = activation.grad[0]
            acts = activation.detach()[0]
            cam = torch.relu((grads * acts).sum(dim=1))
            maps.append(token_to_grid(cam))
        else:
            raise ValueError(f"不支持的 backbone 输出维度: {tuple(activation.shape)}")

    if isinstance(model_input, tuple):
        if len(maps) < 2:
            raise RuntimeError("dual_view=True 时没有捕获到左右两个视角的 backbone 激活")
        return maps[0], maps[1]
    return maps[0]


def compose_original_size_map(
    original: Image.Image,
    heatmap,
    dual_view: bool,
) -> np.ndarray:
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
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(original)
    heat_uint8 = np.uint8(255 * normalize_map(heatmap))
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip((1 - alpha) * rgb + alpha * color, 0, 255))

    Image.fromarray(overlay).save(output_prefix.with_suffix(".overlay.jpg"), quality=95)
    Image.fromarray(heat_uint8).save(output_prefix.with_suffix(".heatmap.png"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 Dry_Total_g 等目标生成 token importance / Grad-CAM 风格热力图。")
    parser.add_argument("--image", type=Path, required=True, help="输入牧场图片路径")
    parser.add_argument("--weights", nargs="+", required=True, help="权重路径；可写成 name=path 以控制输出文件名")
    parser.add_argument("--output-dir", type=Path, default=Path("./heatmaps"), help="热力图输出目录")
    parser.add_argument("--method", choices=("input", "gradcam"), default="input", help="input 更稳健；gradcam 使用 backbone token/feature 梯度")
    parser.add_argument("--target", choices=TARGETS, default="Dry_Total_g", help="要解释的回归目标")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", default="vit_huge_plus_patch16_dinov3.lvd1689m")
    parser.add_argument("--dual-view", type=parse_bool, default=True)
    parser.add_argument("--use-mamba", type=parse_bool, default=True)
    parser.add_argument("--use-cross-attention", type=parse_bool, default=True)
    parser.add_argument("--use-self-attention", type=parse_bool, default=False)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-mamba-layers", type=int, default=2)
    parser.add_argument("--log-targets", type=parse_bool, default=True)
    parser.add_argument("--enable-5-head", type=parse_bool, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    target_idx = TARGETS.index(args.target)
    weight_specs = parse_weight_specs(args.weights)

    original, model_input = load_image_inputs(args.image, args.img_size, args.dual_view, device)
    for name, weight_path in weight_specs:
        print(f"生成热力图: {name} <- {weight_path}")
        model = load_model(args, weight_path, device)
        if args.method == "gradcam":
            heatmap = gradcam_maps(model, model_input, target_idx)
        else:
            heatmap = input_importance_map(model, model_input, target_idx)

        full_map = compose_original_size_map(original, heatmap, args.dual_view)
        output_prefix = args.output_dir / f"{args.image.stem}.{name}.{args.target}.{args.method}"
        save_heatmap_outputs(original, full_map, output_prefix, args.alpha)
        print(f"已保存: {output_prefix.with_suffix('.overlay.jpg')}")
        print(f"已保存: {output_prefix.with_suffix('.heatmap.png')}")


if __name__ == "__main__":
    main()
