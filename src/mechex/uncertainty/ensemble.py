"""Deep-ensemble alternative to the evidential head.

Ref: Sec. 4, the evidential head was kept over deep ensembles under an identical budget.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.uncertainty.evidential import MIN_ALPHA, EvidentialOutput


class DeepEnsembleHeads(nn.Module):
    """Several independent regression heads over one shared representation.

    The predictive deviation is the spread of the member means, which captures
    disagreement between members rather than a per-sample evidence estimate.
    """

    def __init__(self, width: int, n_members: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        if n_members < 2:
            raise ValueError("an ensemble needs at least two members")
        self.n_members = n_members
        self.members = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_members)
        )

    def forward(self, hidden: torch.Tensor) -> EvidentialOutput:
        stacked = torch.stack([member(hidden).squeeze(-1) for member in self.members], dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False).clamp_min(1e-8)
        shape = mean.shape
        return EvidentialOutput(
            mean=mean,
            nu=torch.ones(shape, dtype=mean.dtype, device=mean.device),
            alpha=torch.full(shape, 1.0 + MIN_ALPHA, dtype=mean.dtype, device=mean.device),
            beta=variance,
            variance=variance,
        )

    def loss(self, target: torch.Tensor, output: EvidentialOutput) -> torch.Tensor:
        """Mean squared error of the ensemble mean."""
        return (output.mean - target).pow(2).mean()
