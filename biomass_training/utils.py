import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inverse_log_targets(x):
    """Map log(1+y) predictions back to original biomass space."""
    if isinstance(x, torch.Tensor):
        return torch.expm1(x)
    return np.expm1(x)
