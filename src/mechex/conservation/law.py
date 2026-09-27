"""The exposure-conservation relation.

Ref: Sec. 3.6, Eq. (1): systemic availability is the product of the partition term, the
delivered fraction and an interaction factor.
"""

from __future__ import annotations

import torch


def systemic_availability(
    partition: torch.Tensor,
    delivered_fraction: torch.Tensor,
    interaction: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct systemic availability as ``kappa * f_perf * (1 + epsilon)``.

    ``kappa`` is bounded below by the partition head's softplus, ``f_perf`` lies in the
    open unit interval by construction of the vascular head, and ``(1 + epsilon)`` is
    kept strictly positive by the coupling offset, so the product cannot change sign.
    """
    return partition * delivered_fraction * (1.0 + interaction)


def interaction_factor(interaction: torch.Tensor) -> torch.Tensor:
    """The ``(1 + epsilon)`` factor, exposed for the positivity check."""
    return 1.0 + interaction


def implied_delivered_fraction(
    availability: torch.Tensor,
    partition: torch.Tensor,
    interaction: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Delivered fraction implied by an availability, inverted from Eq. (1).

    Used by the inconsistency definition, which admits a prediction only when this
    quantity lands inside the physically admissible interval.
    """
    denominator = (partition * (1.0 + interaction)).clamp_min(eps)
    return availability / denominator
