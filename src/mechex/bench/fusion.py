"""Fusion baselines.

Ref: Table 1, early-concatenation fusion, the unconstrained cross-attention fusion, and
the contrastive-alignment fusion.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.utils.tensors import call_module


class BranchEncoder(nn.Module):
    """Small encoder lifting a feature block to the fusion width."""

    def __init__(self, input_dim: int, width: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return call_module(self.trunk, features)


class EarlyConcatenationFusion(nn.Module):
    """Concatenates the two blocks before any mixing.

    Ref: Table 1, B10. The two blocks meet only as stacked coordinates, so the fusion can
    express an additive combination and nothing else.
    """

    def __init__(self, molecular_dim: int, imaging_dim: int, width: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(molecular_dim + imaging_dim, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, molecular: torch.Tensor, imaging: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([molecular, imaging], dim=-1)
        return call_module(self.trunk, merged).squeeze(-1)


class UnconstrainedCrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with no conservation constraint and no uncertainty head.

    Ref: Table 1, B11. This is the strongest unconstrained comparator: it has the coupling
    architecture but none of the physical structure, which is what isolates the effect of
    the constraint.
    """

    def __init__(
        self, molecular_dim: int, imaging_dim: int, width: int = 32, heads: int = 4
    ) -> None:
        super().__init__()
        self.molecular_encoder = BranchEncoder(molecular_dim, width)
        self.imaging_encoder = BranchEncoder(imaging_dim, width)
        self.molecular_to_image = nn.MultiheadAttention(width, heads, batch_first=True)
        self.image_to_molecule = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(2 * width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, molecular: torch.Tensor, imaging: torch.Tensor) -> torch.Tensor:
        molecule_tokens = self.molecular_encoder(molecular).unsqueeze(1)
        image_tokens = self.imaging_encoder(imaging).unsqueeze(1)
        molecule_attended, _ = self.molecular_to_image(
            molecule_tokens, image_tokens, image_tokens, need_weights=False
        )
        image_attended, _ = self.image_to_molecule(
            image_tokens, molecule_tokens, molecule_tokens, need_weights=False
        )
        merged = self.norm(
            torch.cat([molecule_attended.squeeze(1), image_attended.squeeze(1)], dim=-1)
        )
        return call_module(self.head, merged).squeeze(-1)


class ContrastiveAlignmentFusion(nn.Module):
    """Aligns the two branch embeddings in a shared space, then regresses.

    Ref: Table 1, B12. The alignment term pulls paired embeddings together and pushes
    unpaired ones apart, which is a representation-level alternative to coupling.
    """

    def __init__(self, molecular_dim: int, imaging_dim: int, width: int = 32) -> None:
        super().__init__()
        self.molecular_encoder = BranchEncoder(molecular_dim, width)
        self.imaging_encoder = BranchEncoder(imaging_dim, width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def encode(
        self, molecular: torch.Tensor, imaging: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.molecular_encoder(molecular), self.imaging_encoder(imaging)

    def alignment_loss(self, molecular: torch.Tensor, imaging: torch.Tensor) -> torch.Tensor:
        """InfoNCE-style contrastive loss over the paired embeddings."""
        molecule_embedding, image_embedding = self.encode(molecular, imaging)
        molecule_normalised = torch.nn.functional.normalize(molecule_embedding, dim=-1)
        image_normalised = torch.nn.functional.normalize(image_embedding, dim=-1)
        logits = molecule_normalised @ image_normalised.t() / self.temperature.clamp_min(1e-3)
        targets = torch.arange(logits.shape[0], device=logits.device)
        forward = torch.nn.functional.cross_entropy(logits, targets)
        backward = torch.nn.functional.cross_entropy(logits.t(), targets)
        return 0.5 * (forward + backward)

    def forward(self, molecular: torch.Tensor, imaging: torch.Tensor) -> torch.Tensor:
        molecule_embedding, image_embedding = self.encode(molecular, imaging)
        merged = torch.cat([molecule_embedding, image_embedding], dim=-1)
        return call_module(self.head, merged).squeeze(-1)


FUSION_BUILDERS = {
    "early_concatenation": EarlyConcatenationFusion,
    "cross_attention_unconstrained": UnconstrainedCrossAttentionFusion,
    "contrastive_alignment": ContrastiveAlignmentFusion,
}


def build_fusion(name: str, molecular_dim: int, imaging_dim: int) -> nn.Module:
    """Construct one of the fusion baselines by name."""
    if name not in FUSION_BUILDERS:
        raise ValueError(f"unknown fusion baseline: {name}")
    built: nn.Module = FUSION_BUILDERS[name](molecular_dim, imaging_dim)
    return built


def train_fusion(
    model: nn.Module,
    molecular: torch.Tensor,
    imaging: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 40,
    learning_rate: float = 5e-3,
) -> list[float]:
    """Fit a fusion baseline on a fixed split under the shared tuning budget."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        prediction: torch.Tensor = model(molecular, imaging)
        loss: torch.Tensor = torch.nn.functional.mse_loss(prediction, target)
        if isinstance(model, ContrastiveAlignmentFusion):
            loss = loss + model.alignment_loss(molecular, imaging)
        torch.autograd.backward(loss)
        optimiser.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def predict_fusion(
    model: nn.Module, molecular: torch.Tensor, imaging: torch.Tensor
) -> torch.Tensor:
    """Evaluate a fusion baseline."""
    model.eval()
    result: torch.Tensor = model(molecular, imaging)
    return result
