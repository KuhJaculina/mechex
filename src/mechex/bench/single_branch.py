"""Single-branch baselines.

Ref: Table 1, B1 molecular foundation transformer (molecular only) and B6 volumetric CTA
encoder (imaging only), which fix how much of the result either branch carries alone.
"""

from __future__ import annotations

import torch
from torch import nn

from mechex.utils.tensors import call_module


class SingleBranchRegressor(nn.Module):
    """One feature block lifted to a scalar exposure through a small MLP."""

    def __init__(self, input_dim: int, width: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        layers.append(nn.Linear(current, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return call_module(self.trunk, features).squeeze(-1)


def train_single_branch(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 60,
    learning_rate: float = 5e-3,
) -> list[float]:
    """Fit a single-branch baseline under the shared tuning budget."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        prediction: torch.Tensor = model(features)
        loss: torch.Tensor = torch.nn.functional.mse_loss(prediction, target)
        torch.autograd.backward(loss)
        optimiser.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def predict_single_branch(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Evaluate a single-branch baseline."""
    model.eval()
    result: torch.Tensor = model(features)
    return result


class FrozenHeadModel(nn.Module):
    """Encodes with a frozen trunk and adapts only a projection head.

    Ref: Sec. 3.6 and Sec. 4, frozen encoders with modified heads were preferred to full
    fine-tuning under an identical budget; the ablation needs both forms.
    """

    def __init__(self, input_dim: int, width: int = 32, frozen: bool = True) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, width))
        self.head = nn.Linear(width, 1)
        self.frozen = frozen
        if frozen:
            for parameter in self.trunk.parameters():
                parameter.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if self.frozen else torch.enable_grad():
            embedding = call_module(self.trunk, features)
        return call_module(self.head, embedding).squeeze(-1)
