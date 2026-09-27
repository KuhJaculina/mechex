"""Conservation head: the residual of Eq. (1) and its margin penalty.

Ref: Sec. 3.6, conservation head; the head applies a differential cost to the residual
once it exceeds a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mechex.conservation.law import implied_delivered_fraction, systemic_availability
from mechex.conservation.penalty import conservation_penalty
from mechex.settings import ConservationConfig
from mechex.utils.tensors import safe_divide


@dataclass
class ConservationRecord:
    """Everything the conservation head emits for one batch."""

    availability_reference: torch.Tensor
    availability_reconstruction: torch.Tensor
    residual: torch.Tensor
    implied_fraction: torch.Tensor

    def mean_residual(self) -> torch.Tensor:
        return self.residual.mean()


def relative_residual(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative residual ``|A_ref - A_hat| / A_ref``.

    The manuscript's Eq. (1) writes the denominator as the model product while the
    sentence that follows calls it the concentration-derived quantity; the release takes
    the denominator as the concentration-derived reference, which is what makes the
    residual a relative error rather than a ratio against the fit itself.
    """
    return safe_divide((reference - reconstruction).abs(), reference.abs(), eps=eps)


def margin_penalty(residual: torch.Tensor, tolerance: float, margin: float) -> torch.Tensor:
    """Hinge cost that activates only after the residual passes ``tolerance``.

    A smooth margin rather than an indicator keeps the penalty differentiable at the
    boundary.
    """
    from mechex.conservation.penalty import margin_penalty as _margin_penalty

    return _margin_penalty(residual, tolerance, margin)


class ConservationHead(nn.Module):
    """Applies the conservation law and reports its residual and penalty."""

    def __init__(self, config: ConservationConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConservationConfig()

    def forward(
        self,
        partition: torch.Tensor,
        delivered_fraction: torch.Tensor,
        interaction: torch.Tensor,
        availability_reference: torch.Tensor,
    ) -> ConservationRecord:
        reconstruction = systemic_availability(partition, delivered_fraction, interaction)
        residual = relative_residual(availability_reference, reconstruction)
        implied = implied_delivered_fraction(
            availability_reference.detach(), partition, interaction
        )
        return ConservationRecord(
            availability_reference=availability_reference,
            availability_reconstruction=reconstruction,
            residual=residual,
            implied_fraction=implied,
        )

    def penalty(self, residual: torch.Tensor) -> torch.Tensor:
        """Weighted margin penalty for the composite objective."""
        return conservation_penalty(
            residual,
            tolerance=self.config.residual_tolerance,
            margin=self.config.penalty_margin,
            weight=self.config.penalty_weight,
        )
