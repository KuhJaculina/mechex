"""Differentiable concordance objective.

Ref: Sec. 3.8, a concordance-oriented regression loss on the exposure index.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def concordance_ccc(
    predicted: torch.Tensor, reference: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Lin's concordance correlation of a batch, in differentiable form."""
    x = predicted.reshape(-1)
    y = reference.reshape(-1)
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = ((x - mean_x) ** 2).mean()
    var_y = ((y - mean_y) ** 2).mean()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    denominator = var_x + var_y + (mean_x - mean_y) ** 2 + eps
    return 2.0 * covariance / denominator


def concordance_loss(
    predicted: torch.Tensor, reference: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """One minus concordance, the minimisation form of the objective."""
    return 1.0 - concordance_ccc(predicted, reference, eps)


def log_concordance_loss(
    log_predicted: torch.Tensor,
    log_reference: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Concordance loss computed in log space.

    Exposure spans orders of magnitude across molecules, so the concordance objective is
    evaluated on the log index where a fold error is a symmetric distance.
    """
    return concordance_loss(log_predicted, log_reference, eps)


def squared_log_error(log_predicted: torch.Tensor, log_reference: torch.Tensor) -> torch.Tensor:
    """Mean squared log error, reported alongside the concordance term."""
    return (log_predicted.reshape(-1) - log_reference.reshape(-1)).pow(2).mean()


def pearson_penalty(
    predicted: torch.Tensor, reference: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Scatter-only penalty, useful when an offset is already corrected elsewhere."""
    x = predicted.reshape(-1)
    y = reference.reshape(-1)
    centred_x = x - x.mean()
    centred_y = y - y.mean()
    denominator = torch.sqrt(centred_x.pow(2).sum() * centred_y.pow(2).sum()).clamp_min(eps)
    return 1.0 - (centred_x * centred_y).sum() / denominator
