"""Small tensor utilities shared by the branches and the objectives."""

from __future__ import annotations

import torch
from torch import nn

EPS = 1e-8


def call_module(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Call a tensor-returning submodule and pin the result type.

    ``nn.Module.__call__`` is annotated as returning ``Any``, which would erase the
    tensor type at every call site; binding it here keeps the type visible.
    """
    output = module(*inputs)
    assert isinstance(output, torch.Tensor)
    return output


def safe_divide(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Divide with the denominator clamped away from zero.

    Exposure ratios are undefined when systemic availability collapses; clamping keeps
    the gradient finite instead of producing an inf that poisons the whole step.
    """
    return numerator / denominator.clamp_min(eps)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Mean over the entries of ``values`` selected by a boolean ``mask``."""
    weights = mask.to(values.dtype)
    total = (values * weights).sum()
    return total / weights.sum().clamp_min(eps)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``."""
    return torch.where(
        x < -0.6931471805599453, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
    )


def standardise(values: torch.Tensor, dim: int = 0, eps: float = EPS) -> torch.Tensor:
    """Zero-mean unit-variance standardisation along ``dim``."""
    centred = values - values.mean(dim=dim, keepdim=True)
    scale = values.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
    return centred / scale


def unit_scale(values: torch.Tensor, dim: int = 0, eps: float = EPS) -> torch.Tensor:
    """Rescale a non-negative quantity so that its maximum along ``dim`` is one."""
    return values / values.amax(dim=dim, keepdim=True).clamp_min(eps)


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with a clamped variance.

    The evidential head can emit a near-zero variance for a confident case; the clamp
    bounds the per-sample loss so a single outlier cannot dominate the batch.
    """
    clamped = variance.clamp_min(eps)
    return 0.5 * (torch.log(clamped) + (target - mean).pow(2) / clamped)


def one_hot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode an integer index tensor."""
    return torch.nn.functional.one_hot(index.long(), num_classes=num_classes).to(torch.float32)


def segment_sum(values: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sum rows of ``values`` grouped by an integer ``index``.

    Implemented with ``scatter_add`` on a flat buffer rather than a python loop so the
    molecular encoder can pool atom embeddings without leaving the graph.
    """
    out = torch.zeros((num_segments, *values.shape[1:]), dtype=values.dtype, device=values.device)
    flat_index = index.long()
    for _ in range(values.dim() - 1):
        flat_index = flat_index.unsqueeze(-1)
    expanded = flat_index.expand_as(values)
    return out.scatter_add_(0, expanded, values)


def segment_mean(values: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Mean of rows of ``values`` grouped by an integer ``index``."""
    counts = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    counts = counts.scatter_add_(0, index.long(), torch.ones_like(index, dtype=values.dtype))
    shape = (-1,) + (1,) * (values.dim() - 1)
    return segment_sum(values, index, num_segments) / counts.view(shape).clamp_min(1.0)
