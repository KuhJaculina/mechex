"""Adapted projection head turning a molecular embedding into the partition term.

Ref: Sec. 3.6, modified projection head; Sec. 4 retention attribution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mechex.basis.chemistry import DESCRIPTOR_NAMES, descriptor_matrix
from mechex.basis.graph import MoleculeGraph
from mechex.basis.transformer import MolecularBasisEncoder, collate_graphs
from mechex.settings import MolecularBasisConfig

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


class PartitionHead(nn.Module):
    """Maps the frozen embedding plus the descriptor read-out to ``kappa(m)``.

    The log-partition is the sum of an interpretable descriptor term and a learned
    embedding term, so the retention attribution the manuscript reports is a fitted
    quantity of this head rather than a post-hoc correlate.
    """

    descriptor_weights: torch.Tensor
    descriptor_bias: torch.Tensor
    descriptor_scale: torch.Tensor

    def __init__(self, latent_dim: int, n_descriptors: int = N_DESCRIPTORS) -> None:
        super().__init__()
        self.register_buffer("descriptor_weights", torch.zeros(n_descriptors), persistent=True)
        self.register_buffer("descriptor_bias", torch.zeros(1), persistent=True)
        self.register_buffer("descriptor_scale", torch.ones(n_descriptors), persistent=True)
        self.embedding = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        self.embedding_scale = nn.Parameter(torch.tensor(0.1))

    def set_descriptor_scaling(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        """Install the descriptor standardisation measured on the development split."""
        self.descriptor_scale = scale.detach().clone()
        self.descriptor_bias = bias.detach().clone()

    def forward(self, embedding: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        standardised = (descriptors - self.descriptor_bias) / self.descriptor_scale.clamp_min(1e-6)
        interpretable = standardised @ self.descriptor_weights
        learned = self.embedding_scale * self.embedding(embedding).squeeze(-1)
        log_partition = interpretable + learned
        return torch.nn.functional.softplus(log_partition) + 1e-4

    def attribution(self) -> dict[str, float]:
        """Signed contribution of each descriptor to the log-partition.

        A negative entry lowers the partition term, which is the direction the
        manuscript reports for efflux-substrate propensity.
        """
        weights = self.descriptor_weights.detach().cpu().numpy()
        return {name: float(value) for name, value in zip(DESCRIPTOR_NAMES, weights, strict=True)}

    def ranked_attribution(self) -> list[tuple[str, float]]:
        """Descriptors ordered from most to least negative."""
        return sorted(self.attribution().items(), key=lambda item: item[1])


class MolecularBranch(nn.Module):
    """Structure branch: frozen encoder plus the adapted partition head."""

    def __init__(self, config: MolecularBasisConfig | None = None) -> None:
        super().__init__()
        self.config = config or MolecularBasisConfig()
        self.encoder = MolecularBasisEncoder(self.config)
        self.head = PartitionHead(self.config.latent_dim)

    def forward(
        self,
        batch_atoms: torch.Tensor,
        batch_coords: torch.Tensor,
        batch_mask: torch.Tensor,
        batch_bond: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from mechex.basis.transformer import MolecularBatch

        batch = MolecularBatch(
            atoms=batch_atoms,
            coords=batch_coords,
            mask=batch_mask,
            bond_index=batch_bond,
        )
        embedding = self.encoder(batch)
        partition = self.head(embedding, descriptors)
        return partition, embedding

    def forward_graphs(
        self,
        graphs: list[MoleculeGraph],
        descriptors: torch.Tensor | None = None,
        efflux_affinities: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience path for a python-side list of graphs."""
        batch = collate_graphs(graphs)
        if descriptors is None:
            matrix = descriptor_matrix(graphs, efflux_affinities)
            descriptors = torch.as_tensor(matrix, dtype=torch.float32, device=batch.atoms.device)
        return self.forward(batch.atoms, batch.coords, batch.mask, batch.bond_index, descriptors)

    def calibrate_descriptor_scaling(self, matrix: np.ndarray) -> None:
        """Fit the descriptor standardisation used inside the head."""
        values = torch.as_tensor(matrix, dtype=torch.float32)
        scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        bias = values.mean(dim=0)
        self.head.set_descriptor_scaling(scale, bias)
