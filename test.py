import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from biomass_training.config import CFG
from biomass_training.model import BiomassModel
from biomass_training.utils import inverse_log_targets, seed_everything


@dataclass(frozen=True)
class TestCFG:
    """推理配置。"""

    seed: int = 42
    data_dir: Path = Path("./csiro-biomass")
    model_dir: Path = Path("./weights")
    output_csv: Path = Path("./submission.csv")
    model_paths: Sequence[str] = ()
    model_glob: str = "biomass_fold*.pth"


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

    if isinstance(first_x, tuple):
        imgs1 = torch.stack([x[0] for x, _ in batch])
        imgs2 = torch.stack([x[1] for x, _ in batch])
        return (imgs1, imgs2), image_paths

    imgs = torch.stack([x for x, _ in batch])
    return imgs, image_paths


def resolve_model_paths(test_cfg: TestCFG) -> list[Path]:
    """解析需要参与集成推理的模型权重路径。"""
    if test_cfg.model_paths:
        return [Path(p) for p in test_cfg.model_paths]
    return sorted(test_cfg.model_dir.glob(test_cfg.model_glob))


def load_model_for_inference(train_cfg: CFG, weight_path: Path, device: torch.device) -> BiomassModel:
    """加载生物量模型；推理时忽略训练专用的辅助头。"""
    model = BiomassModel(
        model_name=train_cfg.backbone,
        pretrained=False,
        dual_view=train_cfg.dual_view,
        use_mamba=train_cfg.use_mamba,
        use_cross_attention=train_cfg.use_cross_attention,
        use_self_attention=train_cfg.use_self_attention,
        num_heads=train_cfg.num_heads,
        num_mamba_layers=train_cfg.num_mamba_layers,
        log_targets=train_cfg.log_targets,
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

    for x, _ in tqdm(loader, desc="Inference", leave=False):
        if isinstance(x, tuple) and len(x) == 2:
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
    test_cfg = TestCFG()
    train_cfg = CFG()
    seed_everything(test_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Device:", device)

    test_df = pd.read_csv(test_cfg.data_dir / "test.csv")
    test_wide = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
    print("Test rows:", len(test_df))
    print("Test images:", len(test_wide))

    dataset = PastureImageTestDataset(
        test_wide,
        image_root=test_cfg.data_dir,
        img_size=train_cfg.img_size,
        dual_view=train_cfg.dual_view,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_test,
    )

    model_paths = resolve_model_paths(test_cfg)
    if len(model_paths) == 0:
        raise ValueError("No model checkpoints found. Set TestCFG.model_paths or provide files in model_dir.")

    print("Found weights:")
    for path in model_paths:
        print("-", path)

    all_preds = []
    for weight_path in model_paths:
        model = load_model_for_inference(train_cfg, weight_path, device)
        preds = predict_with_model(model, loader, device, log_targets=train_cfg.log_targets)
        all_preds.append(preds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 多个 fold 权重做简单平均，得到最终集成预测。
    ensemble_predictions = np.mean(np.stack(all_preds, axis=0), axis=0)
    print("Ensemble predictions shape:", ensemble_predictions.shape)

    submission_df = build_submission(test_df, test_wide, ensemble_predictions, train_cfg.targets)
    submission_df.to_csv(test_cfg.output_csv, index=False)
    print(f"Saved submission to {test_cfg.output_csv}")
    print(submission_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
