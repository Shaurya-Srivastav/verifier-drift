"""Seed every random number generator the pipeline may touch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - seeds third-party code that uses the global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass
