"""Input perturbation and its response matrix.

Ref: Sec. 3.10, robustness under distribution shift was assessed by the shift conditions
and by input perturbation, with the perturbation-response matrix reported in the
supplementary material.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    """One named perturbation with its magnitude."""

    name: str
    magnitude: float
    channel: str = "both"

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "magnitude": self.magnitude, "channel": self.channel}


DEFAULT_PERTURBATIONS = (
    Perturbation("intensity_gain", 0.05),
    Perturbation("intensity_gain", 0.15),
    Perturbation("contrast_attenuation", 0.10),
    Perturbation("contrast_attenuation", 0.25),
    Perturbation("phase_delay_shift", 1.0, channel="delayed"),
    Perturbation("phase_delay_shift", 2.0, channel="delayed"),
    Perturbation("arterial_dropout", 0.10, channel="arterial"),
)


def apply_gain(volume: np.ndarray, magnitude: float, channel: str = "both") -> np.ndarray:
    """Scale the intensity of one channel."""
    perturbed = np.asarray(volume, dtype=np.float64).copy()
    if perturbed.ndim < 3:
        raise ValueError("expected a channel-first volume")
    if channel == "both":
        return perturbed * (1.0 + magnitude)
    index = 0 if channel == "arterial" else 1
    perturbed[index] = perturbed[index] * (1.0 + magnitude)
    return perturbed


def apply_attenuation(volume: np.ndarray, magnitude: float, channel: str = "both") -> np.ndarray:
    """Pull contrast toward the background of the scan."""
    perturbed = np.asarray(volume, dtype=np.float64).copy()
    if channel == "both":
        return perturbed * (1.0 - magnitude)
    index = 0 if channel == "arterial" else 1
    perturbed[index] = perturbed[index] * (1.0 - magnitude)
    return perturbed


def apply_phase_shift(volume: np.ndarray, magnitude: float) -> np.ndarray:
    """Reduce the delayed-phase enhancement as a timing shift would.

    A timing shift lowers the delayed phase toward the arterial one, which is the direction
    that changes the perfused-fraction estimate.
    """
    perturbed = np.asarray(volume, dtype=np.float64).copy()
    shift = float(np.clip(magnitude / 4.0, 0.0, 0.9))
    perturbed[1] = perturbed[1] * (1.0 - shift) + perturbed[0] * shift
    return perturbed


def apply_dropout(volume: np.ndarray, magnitude: float, channel: str = "arterial") -> np.ndarray:
    """Replace a fraction of a channel with the volume mean."""
    perturbed = np.asarray(volume, dtype=np.float64).copy()
    index = 0 if channel == "arterial" else 1
    flat = perturbed[index].reshape(-1)
    count = round(float(np.clip(magnitude, 0.0, 1.0)) * flat.size)
    if count > 0:
        drop = np.random.default_rng(0).permutation(flat.size)[:count]
        flat[drop] = float(flat.mean())
    return perturbed


def apply_perturbation(volume: np.ndarray, perturbation: Perturbation) -> np.ndarray:
    """Dispatch one perturbation by name."""
    if perturbation.name == "intensity_gain":
        return apply_gain(volume, perturbation.magnitude, perturbation.channel)
    if perturbation.name == "contrast_attenuation":
        return apply_attenuation(volume, perturbation.magnitude, perturbation.channel)
    if perturbation.name == "phase_delay_shift":
        return apply_phase_shift(volume, perturbation.magnitude)
    if perturbation.name == "arterial_dropout":
        return apply_dropout(volume, perturbation.magnitude, perturbation.channel)
    raise ValueError(f"unknown perturbation: {perturbation.name}")


def response_matrix(
    predictor: Callable[[np.ndarray], float],
    volume: np.ndarray,
    perturbations: tuple[Perturbation, ...] = DEFAULT_PERTURBATIONS,
) -> list[dict[str, float | str]]:
    """Response of a scalar predictor to each perturbation.

    ``predictor`` receives a channel-first volume and returns a scalar, so the matrix can
    be built for the delivered fraction and for the exposure index alike.
    """
    baseline = float(predictor(np.asarray(volume, dtype=np.float64)))
    rows: list[dict[str, float | str]] = []
    for perturbation in perturbations:
        perturbed = apply_perturbation(volume, perturbation)
        value = float(predictor(perturbed))
        rows.append(
            {
                "name": perturbation.name,
                "magnitude": perturbation.magnitude,
                "channel": perturbation.channel,
                "baseline": baseline,
                "perturbed": value,
                "relative_change": (value - baseline) / max(abs(baseline), 1e-9),
            }
        )
    return rows


def worst_relative_change(matrix: list[dict[str, float | str]]) -> float:
    """Largest absolute relative change in a response matrix."""
    if not matrix:
        return 0.0
    return float(max(abs(float(row["relative_change"])) for row in matrix))
