"""Optimiser construction and early stopping.

Ref: Sec. 3.8, AdamW with cosine decay and early stopping on the validation partition
with a patience of twelve epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer

from mechex.settings import OptimiserConfig

PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def build_optimizer(module: nn.Module, config: OptimiserConfig) -> Optimizer:
    """AdamW over the parameters that carry a gradient.

    Frozen encoders are excluded rather than given a zero learning rate, so their state
    never appears in the optimiser and a resumed run cannot drift them.
    """
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("the module has no trainable parameters")
    if config.name != "adamw":
        raise ValueError(f"unsupported optimiser: {config.name}")
    return torch.optim.AdamW(
        trainable,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Autocast dtype for a precision setting, or ``None`` for full precision."""
    if precision == "fp32":
        return None
    if precision not in PRECISION_DTYPES:
        raise ValueError(f"unknown precision: {precision}")
    return PRECISION_DTYPES[precision]


def clip_gradients(module: nn.Module, max_norm: float) -> float:
    """Gradient-norm clipping; returns the pre-clip norm."""
    parameters = [parameter for parameter in module.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(norm)


def grad_norm(module: nn.Module) -> float:
    """Current gradient norm, monitored to catch a stalled step."""
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return float(total**0.5)


@dataclass
class EarlyStopping:
    """Patience-based stopping on a monitored quantity."""

    patience: int = 12
    min_delta: float = 1e-5
    mode: str = "min"
    best: float = float("inf")
    best_epoch: int = 0
    counter: int = 0
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if self.mode == "max":
            self.best = float("-inf")

    def update(self, value: float, epoch: int) -> bool:
        """Record a value and report whether training should stop."""
        self.history.append(float(value))
        improved = False
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
        else:
            self.counter += 1
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def as_dict(self) -> dict[str, object]:
        return {
            "best": float(self.best),
            "best_epoch": int(self.best_epoch),
            "epochs_without_improvement": int(self.counter),
            "patience": int(self.patience),
            "monitored": len(self.history),
        }
