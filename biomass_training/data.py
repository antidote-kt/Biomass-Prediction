from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def make_train_wide(train_csv: Path) -> pd.DataFrame:
    """将官方长表格式的 train.csv 转成每张图片一行的宽表。"""
    df = pd.read_csv(train_csv)
    required_cols = {"image_path", "target_name", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    # 官方 CSV 中同一张图片有多行目标值，这里 pivot 成多列目标。
    wide = (
        df.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )

    # 保留可用的元数据字段，后续可作为辅助任务监督信号。
    metadata_cols = [
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    available_metadata = [col for col in metadata_cols if col in df.columns]
    if available_metadata:
        metadata = (
            df.groupby("image_path", as_index=False)[available_metadata]
            .first()
            .copy()
        )
        wide = wide.merge(metadata, on="image_path", how="left")

    for col in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]:
        if col not in wide.columns:
            raise ValueError(f"train.csv missing target after pivot: {col}")

    if "GDM_g" not in wide.columns:
        wide["GDM_g"] = wide["Dry_Green_g"] + wide["Dry_Clover_g"]
    if "Dry_Total_g" not in wide.columns:
        wide["Dry_Total_g"] = wide["GDM_g"] + wide["Dry_Dead_g"]

    return wide


def prepare_auxiliary_metadata(train_wide: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """编码训练时辅助监督会用到的元数据目标。"""
    train_wide = train_wide.copy()
    aux_dims: Dict[str, int] = {}

    if "Sampling_Date" in train_wide.columns:
        sampling_date = pd.to_datetime(train_wide["Sampling_Date"], errors="coerce")
        month = sampling_date.dt.month.fillna(0).astype(int)
        train_wide["Month"] = month
        aux_dims["month"] = int(month.max())

    if "Species" in train_wide.columns:
        species = train_wide["Species"].fillna("Unknown").astype(str)
        species_codes, species_uniques = pd.factorize(species, sort=True)
        train_wide["Species_Code"] = species_codes.astype(np.int64)
        aux_dims["species"] = len(species_uniques)

    if "State" in train_wide.columns:
        state = train_wide["State"].fillna("Unknown").astype(str)
        state_codes, state_uniques = pd.factorize(state, sort=True)
        train_wide["State_Code"] = state_codes.astype(np.int64)
        aux_dims["state"] = len(state_uniques)

    if "Pre_GSHH_NDVI" in train_wide.columns:
        train_wide["Pre_GSHH_NDVI"] = train_wide["Pre_GSHH_NDVI"].astype(np.float32)
        aux_dims["ndvi"] = 1

    if "Height_Ave_cm" in train_wide.columns:
        train_wide["Height_Ave_cm"] = train_wide["Height_Ave_cm"].astype(np.float32)
        aux_dims["height"] = 1

    return train_wide, aux_dims


def build_transforms(img_size: int):
    """构建训练集和验证集的数据增强。"""
    train_tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.75,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])

    val_tfms = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])

    return train_tfms, val_tfms


class PastureImageTrainDataset(Dataset):
    """牧草生物量图片的训练数据集。"""

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        transform=None,
        dual_view: bool = True,
        log_targets: bool = True,
        use_auxiliary_tasks: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.dual_view = dual_view
        self.log_targets = log_targets
        self.use_auxiliary_tasks = use_auxiliary_tasks

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.dual_view:
            # 原图左右两侧是两个视角，分别裁成正方形后送入模型。
            left = img.crop((0, 0, h, h))
            right = img.crop((w - h, 0, w, h))
            if self.transform is not None:
                img1 = self.transform(image=np.array(left))["image"]
                img2 = self.transform(image=np.array(right))["image"]
            else:
                img1, img2 = left, right
        else:
            if self.transform is not None:
                img1 = self.transform(image=np.array(img))["image"]
            else:
                img1 = img

        y = torch.tensor(
            [
                row["Dry_Green_g"],
                row["Dry_Dead_g"],
                row["Dry_Clover_g"],
                row["GDM_g"],
                row["Dry_Total_g"],
            ],
            dtype=torch.float32,
        )
        if self.log_targets:
            # 对目标值做 log1p，缓解生物量数值跨度较大的问题。
            y = torch.log1p(y)

        aux_targets = {}
        if self.use_auxiliary_tasks:
            # 这些辅助目标只在训练中使用，帮助 backbone 学到更稳定的表征。
            if "Height_Ave_cm" in row.index and pd.notna(row["Height_Ave_cm"]):
                aux_targets["height"] = torch.tensor(row["Height_Ave_cm"], dtype=torch.float32)
            if "Pre_GSHH_NDVI" in row.index and pd.notna(row["Pre_GSHH_NDVI"]):
                aux_targets["ndvi"] = torch.tensor(row["Pre_GSHH_NDVI"], dtype=torch.float32)
            if "Species_Code" in row.index and pd.notna(row["Species_Code"]):
                aux_targets["species"] = torch.tensor(int(row["Species_Code"]), dtype=torch.long)
            if "State_Code" in row.index and pd.notna(row["State_Code"]):
                aux_targets["state"] = torch.tensor(int(row["State_Code"]), dtype=torch.long)
            if "Month" in row.index and pd.notna(row["Month"]) and int(row["Month"]) > 0:
                aux_targets["month"] = torch.tensor(int(row["Month"]) - 1, dtype=torch.long)

        if self.dual_view:
            return (img1, img2), y, aux_targets
        return img1, y, aux_targets


def collate_fn_train(batch):
    """拼接单视角或双视角 batch。"""
    first_sample = batch[0]
    if isinstance(first_sample[0], tuple):
        imgs1, imgs2, ys = [], [], []
        aux_batch = {}
        for (img1, img2), y, aux_targets in batch:
            imgs1.append(img1)
            imgs2.append(img2)
            ys.append(y)
            for name, value in aux_targets.items():
                aux_batch.setdefault(name, []).append(value)
        collated_aux = {name: torch.stack(values) for name, values in aux_batch.items()}
        return (torch.stack(imgs1), torch.stack(imgs2)), torch.stack(ys), collated_aux

    imgs, ys = [], []
    aux_batch = {}
    for img, y, aux_targets in batch:
        imgs.append(img)
        ys.append(y)
        for name, value in aux_targets.items():
            aux_batch.setdefault(name, []).append(value)
    collated_aux = {name: torch.stack(values) for name, values in aux_batch.items()}
    return torch.stack(imgs), torch.stack(ys), collated_aux
