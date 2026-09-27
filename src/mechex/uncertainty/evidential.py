"""Evidential uncertainty head.

Ref: Sec. 3.6, the uncertainty head combines an evidential output with a post-hoc
isotonic recalibration to produce a calibrated predictive standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mechex.settings import UncertaintyConfig
from mechex.utils.tensors import gaussian_nll

MIN_ALPHA = 1.0 + 1e-3


@dataclass
class EvidentialOutput:
    """Normal-inverse-gamma parameters and the derived predictive moments."""

    mean: torch.Tensor
    nu: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    variance: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.variance.clamp_min(1e-12))


def nig_parameters(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the four raw outputs into ``mean``, ``nu``, ``alpha`` and ``beta``.

    ``alpha`` is offset above one because the predictive variance ``beta/(alpha-1)`` is
    undefined at the boundary.
    """
    mean = raw[..., 0]
    nu = torch.nn.functional.softplus(raw[..., 1]).clamp_min(1e-6)
    alpha = torch.nn.functional.softplus(raw[..., 2]).clamp_min(1e-6) + MIN_ALPHA
    beta = torch.nn.functional.softplus(raw[..., 3]).clamp_min(1e-6)
    return mean, nu, alpha, beta


def evidential_variance(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Predictive variance of the normal-inverse-gamma posterior."""
    aleatoric = beta / (alpha - 1.0).clamp_min(1e-6)
    epistemic = aleatoric / nu.clamp_min(1e-6)
    return aleatoric + epistemic


def nig_nll(
    target: torch.Tensor, output: EvidentialOutput, coefficient: float = 0.0
) -> torch.Tensor:
    """Normal-inverse-gamma negative log-likelihood with an optional evidence penalty."""
    residual2 = (target - output.mean).pow(2)
    two_beta = 2.0 * output.beta.clamp_min(1e-6)
    log_term = torch.log(torch.pi / output.nu.clamp_min(1e-6))
    term_a = output.alpha.clamp_min(1e-6) * torch.log(two_beta * (1.0 + output.nu))
    term_b = (output.alpha + 0.5) * torch.log(residual2 * output.nu + two_beta)
    gamma = torch.lgamma(output.alpha + 0.5) - torch.lgamma(output.alpha.clamp_min(1e-6))
    loss = 0.5 * log_term - term_a + term_b + gamma
    if coefficient:
        loss = loss + coefficient * residual2.abs()
    return loss.mean()


class EvidentialHead(nn.Module):
    """Maps a shared representation to normal-inverse-gamma parameters."""

    def __init__(self, width: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, hidden: torch.Tensor) -> EvidentialOutput:
        raw = self.trunk(hidden)
        mean, nu, alpha, beta = nig_parameters(raw)
        variance = evidential_variance(nu, alpha, beta)
        return EvidentialOutput(mean=mean, nu=nu, alpha=alpha, beta=beta, variance=variance)

    def loss(self, target: torch.Tensor, output: EvidentialOutput) -> torch.Tensor:
        """Training objective for this head."""
        return nig_nll(target, output)


class PointHead(nn.Module):
    """Regression head without a learned uncertainty; the ablation's no-head variant.

    The predictive deviation comes from a constant residual scale fitted on the
    calibration set, so the head still emits a deviation for the calibration-error
    comparison while carrying no evidence model.
    """

    residual_scale: torch.Tensor

    def __init__(self, width: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("residual_scale", torch.tensor([1.0]), persistent=True)
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_residual_scale(self, scale: float) -> None:
        self.residual_scale = torch.tensor([float(scale)])

    def forward(self, hidden: torch.Tensor) -> EvidentialOutput:
        mean = self.trunk(hidden).squeeze(-1)
        variance = self.residual_scale.expand_as(mean).pow(2)
        ones = torch.ones_like(mean)
        return EvidentialOutput(
            mean=mean, nu=ones, alpha=ones + MIN_ALPHA, beta=variance, variance=variance
        )

    def loss(self, target: torch.Tensor, output: EvidentialOutput) -> torch.Tensor:
        """Squared error, since this variant carries no evidence model."""
        return gaussian_nll(target, output.mean, output.variance.clamp_min(1e-8)).mean()


def uncertainty_config_for(mode: str) -> UncertaintyConfig:
    """Config preset matching one of the reported uncertainty variants."""
    return UncertaintyConfig(mode=mode)
