"""Adapted projection head turning a volumetric embedding into the delivered fraction.

Ref: Sec. 3.6, modified projection head on the vascular line.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.settings import VascularEncoderConfig
from mechex.vascular.encoder import VolumetricCTEncoder

MINIMUM_DELIVERED_FRACTION = 0.05
MAXIMUM_DELIVERED_FRACTION = 0.95


class DeliveredFractionHead(nn.Module):
    """Projects the vascular embedding onto the admissible delivered-fraction range."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1),
        )
        self.low = MINIMUM_DELIVERED_FRACTION
        self.high = MAXIMUM_DELIVERED_FRACTION

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.trunk(embedding).squeeze(-1))
        return self.low + (self.high - self.low) * raw


class VascularBranch(nn.Module):
    """Delivery branch: frozen volumetric encoder plus the adapted head."""

    def __init__(self, config: VascularEncoderConfig | None = None, in_channels: int = 2) -> None:
        super().__init__()
        self.config = config or VascularEncoderConfig()
        self.encoder = VolumetricCTEncoder(self.config, in_channels=in_channels)
        self.head = DeliveredFractionHead(self.config.latent_dim)

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(volume)
        fraction = self.head(embedding)
        return fraction, embedding
