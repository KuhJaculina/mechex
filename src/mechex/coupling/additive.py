"""Additive and alternative couplings used by the architectural-substitution ablation.

Ref: Sec. 3.6 and Sec. 4, cross-attention against concatenation, bilinear pooling and
gated fusion.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.coupling.bidirectional import CouplingOutput
from mechex.settings import CouplingConfig
from mechex.utils.tensors import call_module


class _BaseCoupling(nn.Module):
    """Shared interaction head over a merged representation."""

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__()
        self.config = config
        self.interaction = nn.Linear(config.width, 1)
        self.offset = config.interaction_offset

    def _epsilon(self, merged: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.interaction(merged).squeeze(-1)) + self.offset

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        molecule_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> CouplingOutput:
        merged = self.merge(molecule_embedding, image_embedding)
        return CouplingOutput(merged, self._epsilon(merged))


class ConcatenationCoupling(_BaseCoupling):
    """Concatenates the two embeddings and mixes them with a shared trunk."""

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__(config)
        self.trunk = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        return call_module(self.trunk, torch.cat([molecule_embedding, image_embedding], dim=-1))


class BilinearPoolingCoupling(_BaseCoupling):
    """Bilinear pooling of the pair, which represents the product explicitly."""

    def __init__(self, config: CouplingConfig, rank: int = 16) -> None:
        super().__init__(config)
        self.molecule_factor = nn.Linear(config.width, rank, bias=False)
        self.image_factor = nn.Linear(config.width, rank, bias=False)
        self.expand = (
            nn.Linear(rank, config.width, bias=False) if rank != config.width else nn.Identity()
        )

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.molecule_factor(molecule_embedding) * self.image_factor(image_embedding)
        return call_module(self.expand, pooled)


class GatedFusionCoupling(_BaseCoupling):
    """Learns a per-dimension gate between the two embeddings."""

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__(config)
        self.gate = nn.Linear(2 * config.width, config.width)

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.sigmoid(self.gate(torch.cat([molecule_embedding, image_embedding], dim=-1)))
        return weight * molecule_embedding + (1.0 - weight) * image_embedding


class StackedBranchCoupling(_BaseCoupling):
    """Stacks independently trained branch contributions with no cross-branch attention.

    Each branch contributes an additive scalar to the interaction term, so the pair can
    only be combined additively rather than multiplicatively.
    """

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__(config)
        self.molecule_scalar = nn.Linear(config.width, 1)
        self.image_scalar = nn.Linear(config.width, 1)
        self.merge_projection = nn.Linear(2 * config.width, config.width)

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        return call_module(
            self.merge_projection, torch.cat([molecule_embedding, image_embedding], dim=-1)
        )

    def forward(
        self,
        molecule_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> CouplingOutput:
        merged = self.merge(molecule_embedding, image_embedding)
        additive = self.molecule_scalar(molecule_embedding) + self.image_scalar(image_embedding)
        # The additive variant still has to respect the physical admissibility the
        # conservation law asserts, so its interaction passes through the same positive map.
        epsilon = torch.nn.functional.softplus(additive.squeeze(-1)) + self.offset
        return CouplingOutput(merged, epsilon)


class RandomInitCoupling(_BaseCoupling):
    """Identical to the chosen coupling but fed branch embeddings without adaptation.

    Used by the ablation that replaces both pretrained branches with randomly initialised
    encoders; the coupling itself is unchanged.
    """

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__(config)
        self.trunk = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )

    def merge(
        self, molecule_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        return call_module(self.trunk, torch.cat([molecule_embedding, image_embedding], dim=-1))


COUPLING_MODES = (
    "cross_attention",
    "concatenation",
    "bilinear_pooling",
    "gated_fusion",
    "single_branch_stack",
    "random_init",
)


def build_coupling(config: CouplingConfig) -> nn.Module:
    """Construct the coupling named by ``config.mode``."""
    from mechex.coupling.bidirectional import CrossAttentionCoupling

    registry: dict[str, type[nn.Module]] = {
        "cross_attention": CrossAttentionCoupling,
        "concatenation": ConcatenationCoupling,
        "bilinear_pooling": BilinearPoolingCoupling,
        "gated_fusion": GatedFusionCoupling,
        "single_branch_stack": StackedBranchCoupling,
        "random_init": RandomInitCoupling,
    }
    if config.mode not in registry:
        raise ValueError(f"unknown coupling mode: {config.mode}")
    return registry[config.mode](config)
