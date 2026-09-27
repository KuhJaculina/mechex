"""Single seeding entry point.

Every stochastic step in the release draws from the generator seeded here, and the seed
travels in each checkpoint so a resumed run continues the same stream.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 20260927


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and torch, and return the seed that was applied.

    Call this before any tensor is allocated; seeding after a model exists leaves the
    initial weights in the process-random stream.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def numpy_generator(seed: int | None = None) -> np.random.Generator:
    """Independent numpy generator for a data-producing routine."""
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def torch_generator(seed: int | None = None) -> torch.Generator:
    """Independent torch generator for a data-producing routine."""
    generator = torch.Generator()
    generator.manual_seed(DEFAULT_SEED if seed is None else seed)
    return generator


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the compute device, honouring an explicit request."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int) -> None:
    """Seed a dataloader worker from the parent's torch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
