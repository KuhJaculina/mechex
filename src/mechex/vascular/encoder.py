"""Frozen volumetric CT foundation encoder for the delivery branch.

Ref: Sec. 3.6, vascular line; a volumetric CT base encoder held frozen with an adapted
projection head.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.settings import VascularEncoderConfig
from mechex.utils.tensors import call_module


class ConvStem(nn.Module):
    """Strided 3D convolutional stem producing token embeddings."""

    def __init__(
        self, in_channels: int, channels: tuple[int, ...], kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        padding = kernel_size // 2
        for index, width in enumerate(channels):
            stride = 2 if index > 0 else 1
            layers.append(
                nn.Conv3d(
                    current,
                    width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.GroupNorm(num_groups=min(8, width), num_channels=width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            current = width
        self.blocks = nn.Sequential(*layers)
        self.out_channels = current

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return call_module(self.blocks, volume)


class TokenAttention(nn.Module):
    """Transformer block over flattened volumetric tokens."""

    def __init__(self, width: int, heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalised = self.norm_attention(tokens)
        attended, _ = self.attention(normalised, normalised, normalised, need_weights=False)
        attended_dropped: torch.Tensor = self.dropout(attended)
        tokens = tokens + attended_dropped
        fed: torch.Tensor = call_module(self.ffn, self.norm_ffn(tokens))
        fed = self.dropout(fed)
        return tokens + fed


class VolumetricCTEncoder(nn.Module):
    """Frozen encoder mapping a two-phase CTA to a delivery-capacity embedding."""

    def __init__(self, config: VascularEncoderConfig, in_channels: int = 2) -> None:
        super().__init__()
        self.config = config
        self.stem = ConvStem(in_channels, config.stem_channels, config.kernel_size, config.dropout)
        self.token_projection = nn.Linear(self.stem.out_channels, config.latent_dim)
        self.blocks = nn.ModuleList(
            TokenAttention(config.latent_dim, 4, config.dropout)
            for _ in range(max(1, config.depth // 3))
        )
        self.norm = nn.LayerNorm(config.latent_dim)
        if config.frozen:
            self.freeze()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> VolumetricCTEncoder:
        """A frozen encoder stays in eval mode whatever the outer module is doing."""
        super().train(mode and not self.config.frozen)
        return self

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = self.stem(volume)
        batch, channels, depth, height, width = features.shape
        tokens = features.view(batch, channels, depth * height * width).transpose(1, 2)
        tokens = call_module(self.token_projection, tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = call_module(self.norm, tokens)
        return tokens.mean(dim=1)

    def embed_volumes(self, volume: torch.Tensor) -> torch.Tensor:
        """Embed an already batched volume tensor."""
        return self.forward(volume)


def block_tokens(volume: torch.Tensor, config: VascularEncoderConfig) -> int:
    """Number of tokens the encoder produces for a configured grid, for budget checks."""
    depth, height, width = config.volume_shape
    total = depth * height * width
    for _ in range(1, len(config.stem_channels)):
        total = max(1, total // 8)
    return total
