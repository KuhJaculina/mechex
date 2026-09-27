"""Adequacy and evidential terms of the composite objective.

Ref: Sec. 3.8, a binary cross-entropy term on the adequate/inadequate label alongside a
regression loss on the exposure index.
"""

from __future__ import annotations

import torch

from mechex.decision.threshold import adequacy_boundary
from mechex.uncertainty.evidential import EvidentialOutput, nig_nll


def adequacy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on the adequacy label."""
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.reshape(-1).to(logits.dtype)
    )


def adequacy_labels(
    exposure: torch.Tensor, tau: float, adequacy_relative: float = 1.0
) -> torch.Tensor:
    """Hard adequacy labels at the fixed relative boundary."""
    boundary = adequacy_boundary(tau, adequacy_relative)
    return (exposure >= boundary).to(torch.float32)


def evidential_loss(
    target: torch.Tensor, output: EvidentialOutput, coefficient: float = 0.0
) -> torch.Tensor:
    """Normal-inverse-gamma loss for the uncertainty head."""
    return nig_nll(target, output, coefficient=coefficient)


def brier_term(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Brier score as a differentiable calibration monitor."""
    return (probabilities.reshape(-1) - labels.reshape(-1).to(probabilities.dtype)).pow(2).mean()


def calibration_gap(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> torch.Tensor:
    """Differentiable surrogate for the expected calibration error.

    The exact expected calibration error needs a bin assignment, which is not
    differentiable; this surrogate soft-assigns cases to bins through a softmax over
    bin distances so a training run can report a calibration signal too.
    """
    p = probabilities.reshape(-1)
    y = labels.reshape(-1).to(p.dtype)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=p.device, dtype=p.dtype)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = torch.softmax(-((p.unsqueeze(-1) - centres.unsqueeze(0)) / 0.05).pow(2), dim=-1)
    mass = weights.mean(dim=0)
    confidence = (weights * p.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-6)
    accuracy = (weights * y.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-6)
    return (mass * (confidence - accuracy).abs()).sum()
