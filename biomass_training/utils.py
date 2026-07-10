import os
import random

import numpy as np
import torch


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
