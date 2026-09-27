"""Adequacy boundary and the exposure index.

Ref: Sec. 3.3, exposure index as the concentration-time integral scaled by the delivery
term; Sec. 3.5, the threshold is the 30% minimum clinically important difference fixed on
the development split.
"""

from __future__ import annotations

import numpy as np
import torch

ADEQUATE = 1
INADEQUATE = 0
ABSTAIN = -1


def concentration_time_integral(
    times: np.ndarray,
    concentrations: np.ndarray,
) -> float:
    """Trapezoidal integral of a plasma concentration-time record."""
    t = np.asarray(times, dtype=np.float64)
    c = np.asarray(concentrations, dtype=np.float64)
    if t.shape != c.shape:
        raise ValueError("times and concentrations must share a shape")
    if t.size < 2:
        return float(c.sum())
    order = np.argsort(t)
    return float(np.trapezoid(c[order], t[order]))


def exposure_index(integral: float, delivery: float, reference_integral: float = 1.0) -> float:
    """Local exposure index ``Eloc``.

    Ref: Sec. 3.3 and Sec. 3.5. The concentration-time integral is expressed relative to
    the molecule's reference integral, so the index is dimensionless and comparable
    across molecules, then scaled by the dimensionless delivery term.
    """
    if reference_integral <= 0:
        raise ValueError("reference integral must be positive")
    return float(integral) / float(reference_integral) * float(delivery)


def adequacy_boundary(tau: float, adequacy_relative: float = 1.0) -> float:
    """Relative-exposure boundary separating adequate from inadequate.

    ``tau`` is the minimum clinically important difference as a fractional change; the
    cut is placed at ``adequacy_relative * (1 + tau)``.
    """
    return float(adequacy_relative) * (1.0 + float(tau))


def is_adequate(exposure: np.ndarray, tau: float, adequacy_relative: float = 1.0) -> np.ndarray:
    """Label records against the fixed relative boundary."""
    cut = adequacy_boundary(tau, adequacy_relative)
    return (np.asarray(exposure, dtype=np.float64) >= cut).astype(np.int64)


def adequacy_from_baseline(
    exposure: np.ndarray,
    baseline: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Label records by their change from a per-molecule baseline.

    Used where the reference standard is expressed as a fractional change from the
    systemic exposure rather than as an absolute index.
    """
    values = np.asarray(exposure, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    ratio = values / np.clip(base, 1e-12, None)
    return (ratio >= 1.0 + float(tau)).astype(np.int64)


class ThresholdDecision:
    """Applies the adequacy boundary to a batch of exposure estimates."""

    def __init__(self, tau: float, adequacy_relative: float = 1.0) -> None:
        self.tau = float(tau)
        self.adequacy_relative = float(adequacy_relative)
        self.boundary = adequacy_boundary(tau, adequacy_relative)

    def forward_tensor(self, exposure: torch.Tensor) -> torch.Tensor:
        return (exposure >= self.boundary).long()

    def classify(self, exposure: np.ndarray) -> np.ndarray:
        return (np.asarray(exposure, dtype=np.float64) >= self.boundary).astype(np.int64)

    def decision_curve_point(
        self, exposure: np.ndarray, label: np.ndarray
    ) -> tuple[float, float, float]:
        """False-positive rate, false-negative rate and net benefit at the boundary."""
        predicted = self.classify(exposure)
        truth = np.asarray(label, dtype=np.int64)
        positives = truth.sum()
        negatives = truth.size - positives
        false_positive = float(np.sum((predicted == ADEQUATE) & (truth == INADEQUATE)))
        false_negative = float(np.sum((predicted == INADEQUATE) & (truth == ADEQUATE)))
        true_positive = float(np.sum((predicted == ADEQUATE) & (truth == ADEQUATE)))
        fpr = false_positive / max(negatives, 1)
        fnr = false_negative / max(positives, 1)
        prevalence = positives / max(truth.size, 1)
        net = true_positive / max(truth.size, 1) - false_positive / max(truth.size, 1) * (
            prevalence / max(1.0 - prevalence, 1e-6)
        )
        return fpr, fnr, float(net)
