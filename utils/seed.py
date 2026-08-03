"""Reproducible seeding for all random number generators."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for full reproducibility.

    Also enables cudnn.benchmark when CUDA is available, which picks the
    fastest convolution/attention kernel for a fixed input resolution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True: finds the fastest algorithm for a fixed input size.
    # Safe here because resolution is fixed within a training run.
    torch.backends.cudnn.benchmark = True
