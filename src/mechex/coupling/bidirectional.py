"""Bidirectional cross-attention coupling of the two branch embeddings.

Ref: Sec. 3.6, exposure-coupling module; the interaction term of Eq. (1).
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.settings import CouplingConfig


class BidirectionalCrossAttention(nn.Module):
    """Attends each branch embedding to the other in both directions.

    The two embeddings describe factors that act multiplicatively, so the coupling emits
    an interaction embedding from the pair rather than concatenating the two streams.
    """

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__()
        self.config = config
        self.molecule_to_image = nn.MultiheadAttention(
            config.width, config.heads, dropout=config.dropout, batch_first=True
        )
        self.image_to_molecule = nn.MultiheadAttention(
            config.width, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm_molecule = nn.LayerNorm(config.width)
        self.norm_image = nn.LayerNorm(config.width)
        self.merge = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, config.width),
        )
        self.interaction = nn.Linear(config.width, 1)
        self.offset = config.interaction_offset

    def forward(
        self,
        molecule_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        molecule_tokens = molecule_embedding.unsqueeze(1)
        image_tokens = image_embedding.unsqueeze(1)
        molecule_attended, _ = self.molecule_to_image(
            molecule_tokens, image_tokens, image_tokens, need_weights=False
        )
        image_attended, _ = self.image_to_molecule(
            image_tokens, molecule_tokens, molecule_tokens, need_weights=False
        )
        molecule_out = self.norm_molecule(molecule_tokens + molecule_attended).squeeze(1)
        image_out = self.norm_image(image_tokens + image_attended).squeeze(1)
        merged = self.merge(torch.cat([molecule_out, image_out], dim=-1))
        epsilon = torch.nn.functional.softplus(self.interaction(merged).squeeze(-1)) + self.offset
        return merged, epsilon


class CouplingOutput:
    """Coupling result carrying both the interaction embedding and the interaction term."""

    def __init__(self, embedding: torch.Tensor, epsilon: torch.Tensor) -> None:
        self.embedding = embedding
        self.epsilon = epsilon

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embedding, self.epsilon


class CrossAttentionCoupling(nn.Module):
    """Adapter exposing :class:`BidirectionalCrossAttention` behind the coupling API."""

    def __init__(self, config: CouplingConfig) -> None:
        super().__init__()
        self.core = BidirectionalCrossAttention(config)

    def forward(
        self,
        molecule_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> CouplingOutput:
        embedding, epsilon = self.core(molecule_embedding, image_embedding)
        return CouplingOutput(embedding, epsilon)
