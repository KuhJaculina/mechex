"""Calibrated uncertainty head with a post-hoc isotonic recalibration.

Ref: Sec. 3.6, uncertainty head; the three reported variants are the evidential head,
the deep ensemble and the head without recalibration.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mechex.settings import UncertaintyConfig
from mechex.uncertainty.ensemble import DeepEnsembleHeads
from mechex.uncertainty.evidential import EvidentialHead, EvidentialOutput, PointHead
from mechex.uncertainty.isotonic import IsotonicRecalibrator

UNCERTAINTY_MODES = ("evidential", "deep_ensemble", "none")


class UncertaintyHead(nn.Module):
    """Emits a mean and a calibrated predictive deviation for the exposure index."""

    def __init__(self, config: UncertaintyConfig | None = None, width: int = 128) -> None:
        super().__init__()
        self.config = config or UncertaintyConfig()
        self.core: nn.Module
        if self.config.mode == "evidential":
            self.core = EvidentialHead(width, self.config.hidden_dim)
        elif self.config.mode == "deep_ensemble":
            self.core = DeepEnsembleHeads(width, hidden_dim=self.config.hidden_dim)
        elif self.config.mode == "none":
            self.core = PointHead(width, self.config.hidden_dim)
        else:
            raise ValueError(f"unknown uncertainty mode: {self.config.mode}")
        self.recalibrator = IsotonicRecalibrator(self.config.isotonic_bins)
        self.recalibration_enabled = bool(self.config.recalibrate and self.config.mode != "none")

    def raw(self, hidden: torch.Tensor) -> EvidentialOutput:
        """Uncalibrated head output."""
        output = self.core(hidden)
        assert isinstance(output, EvidentialOutput)
        return output

    def forward(self, hidden: torch.Tensor) -> EvidentialOutput:
        output = self.raw(hidden)
        if not self.recalibration_enabled or not self.recalibrator.is_fitted():
            return output
        recalibrated = self.recalibrator.predict(output.std.detach().cpu().numpy())
        variance = torch.as_tensor(recalibrated, dtype=output.variance.dtype).pow(2)
        return EvidentialOutput(
            mean=output.mean,
            nu=output.nu,
            alpha=output.alpha,
            beta=output.beta,
            variance=variance,
        )

    def loss(self, target: torch.Tensor, output: EvidentialOutput) -> torch.Tensor:
        """Delegates to the active core's objective."""
        core = self.core
        if isinstance(core, DeepEnsembleHeads):
            return core.loss(target, output)
        if isinstance(core, PointHead):
            return core.loss(target, output)
        assert isinstance(core, EvidentialHead)
        return core.loss(target, output)

    def recalibrate(self, hidden: torch.Tensor, target: torch.Tensor) -> None:
        """Fit the isotonic map on a held-out calibration set."""
        if not self.recalibration_enabled:
            return
        with torch.no_grad():
            output = self.raw(hidden)
            predicted = output.std.detach().cpu().numpy()
            error = (target - output.mean).abs().detach().cpu().numpy()
        self.recalibrator.fit(predicted, error)
        if isinstance(self.core, PointHead):
            residual = float(np.mean(error))
            self.core.set_residual_scale(max(residual, 1e-6))


def build_uncertainty_head(mode: str, width: int = 128, n_bins: int = 32) -> UncertaintyHead:
    """Construct one of the reported uncertainty variants."""
    return UncertaintyHead(UncertaintyConfig(mode=mode, isotonic_bins=n_bins), width=width)
