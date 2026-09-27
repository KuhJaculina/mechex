"""Margin cost on the conservation residual.

Ref: Sec. 3.6, the conservation head introduces a differential cost once the residual
exceeds a threshold.
"""

from __future__ import annotations

import torch


def margin_penalty(residual: torch.Tensor, tolerance: float, margin: float) -> torch.Tensor:
    """Hinge cost that activates only after the residual passes ``tolerance``.

    A smooth margin rather than an indicator keeps the penalty differentiable at the
    boundary, so a record crossing the tolerance is nudged back instead of switched.
    """
    if margin <= 0:
        raise ValueError("the margin must be positive")
    excess = residual - tolerance
    return torch.nn.functional.softplus(excess / margin) * margin


def conservation_penalty(
    residual: torch.Tensor,
    tolerance: float,
    margin: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Weighted mean margin cost over a batch."""
    return weight * margin_penalty(residual, tolerance, margin).mean()


def penalty_scale(residual: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Share of the batch already past the tolerance, monitored during training."""
    return (residual > tolerance).to(residual.dtype).mean()
