"""Learning-rate schedule.

Ref: Sec. 3.8, AdamW with cosine decay, warmup fraction and a floor exposed in config.
"""

from __future__ import annotations

import math

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from mechex.settings import OptimiserConfig


def warmup_steps(total_steps: int, warmup_fraction: float) -> int:
    """Number of linear warmup steps implied by a fraction of the run."""
    if total_steps <= 0:
        return 0
    return int(max(0.0, min(1.0, warmup_fraction)) * total_steps)


def cosine_factor(
    step: int,
    total_steps: int,
    warmup: int,
    min_lr_factor: float,
) -> float:
    """Multiplier applied to the base learning rate at ``step``."""
    if total_steps <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_factor + (1.0 - min_lr_factor) * cosine)


def build_scheduler(optimizer: Optimizer, total_steps: int, config: OptimiserConfig) -> LambdaLR:
    """Cosine schedule with linear warmup and a floor."""
    warmup = warmup_steps(total_steps, config.warmup_fraction)

    def schedule(step: int) -> float:
        return cosine_factor(step, total_steps, warmup, config.min_lr_factor)

    return LambdaLR(optimizer, lr_lambda=schedule)


def current_lr(optimizer: Optimizer) -> float:
    """Learning rate of the first parameter group."""
    groups = optimizer.param_groups
    return float(groups[0]["lr"]) if groups else float("nan")


def steps_per_epoch(n_samples: int, batch_size: int, grad_accum: int = 1) -> int:
    """Optimiser steps in one epoch."""
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError("batch size and accumulation must be positive")
    return max(1, math.ceil(n_samples / (batch_size * grad_accum)))


def total_steps(
    n_samples: int,
    config: OptimiserConfig,
    epochs: int | None = None,
) -> int:
    """Total optimiser steps for a run."""
    horizon = config.epochs if epochs is None else epochs
    return steps_per_epoch(n_samples, config.batch_size, config.grad_accum) * max(1, horizon)


def describe_reference_schedule(config: OptimiserConfig, n_samples: int) -> dict[str, object]:
    """The pre-specified schedule, recorded in the report so a reader can audit it."""
    return {
        "optimiser": config.name,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "scheduler": "cosine_with_warmup",
        "warmup_fraction": config.warmup_fraction,
        "warmup_steps": warmup_steps(total_steps(n_samples, config), config.warmup_fraction),
        "total_steps": total_steps(n_samples, config),
        "min_lr_factor": config.min_lr_factor,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "epochs": config.epochs,
        "early_stopping_patience": config.patience,
        "precision": config.precision,
        "world_size": config.world_size,
        "effective_batch_size": config.batch_size * config.grad_accum * config.world_size,
    }


def apply_warmup_only(optimizer: Optimizer, warmup: int) -> Optimizer:
    """Trivial schedule used by the smoke test when a single epoch is enough."""
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optimizer


def module_groups(module: nn.Module) -> list[str]:
    """Names of the submodules carrying trainable parameters."""
    return sorted(
        {
            name.split(".")[0]
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
    )
