"""Perfused-fraction estimation from the CTA phases.

Ref: Sec. 3.3, the perfused-fraction estimate converted into a dimensionless delivery
term; Sec. 3.5, Tier-3 reference construction.
"""

from __future__ import annotations

import numpy as np


def arterial_input(arterial: np.ndarray, delayed: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Fraction of the arterial inflow that has reached the muscle bed.

    The two phases bound the delivery: a bed that is fully perfused reaches its delayed
    plateau, while an occluded bed keeps the arterial-to-delayed ratio close to one.
    """
    a = np.asarray(arterial, dtype=np.float64)
    d = np.asarray(delayed, dtype=np.float64)
    numerator = np.maximum(a - d, 0.0)
    denominator = np.maximum(a, eps)
    return np.clip(numerator / denominator, 0.0, 1.0)


def perfused_fraction(arterial: np.ndarray, delayed: np.ndarray) -> float:
    """Region-averaged perfused fraction for one imaging episode."""
    field = arterial_input(arterial, delayed)
    foreground = field[field > 0]
    if foreground.size == 0:
        return 0.0
    return float(foreground.mean())


def perfused_fraction_map(arterial: np.ndarray, delayed: np.ndarray) -> np.ndarray:
    """Voxelwise perfused fraction, used by the calibration of the reference tier."""
    return arterial_input(arterial, delayed)


def delivery_term(
    arterial: np.ndarray,
    delayed: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
) -> float:
    """Dimensionless delivery term built from the perfused fraction.

    The admissible interval keeps the term strictly inside the open unit interval so the
    conservation ratio never inverts the sign of the exposure.
    """
    fraction = perfused_fraction(arterial, delayed)
    return float(np.clip(fraction, minimum, maximum))


def is_monotone(fractions: np.ndarray, exposures: np.ndarray) -> bool:
    """Check that exposure is non-decreasing in perfused fraction.

    The manuscript lists a monotonicity violation among the three ways a prediction can
    be physically inconsistent.
    """
    x = np.asarray(fractions, dtype=np.float64)
    y = np.asarray(exposures, dtype=np.float64)
    if x.size < 2:
        return True
    order = np.argsort(x)
    diffs = np.diff(y[order])
    return bool(np.all(diffs >= -1e-9))


def phase_fixture(
    size: tuple[int, int, int] = (16, 16, 12),
    occlusion: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a schema-compatible arterial and delayed phase pair.

    The occlusion parameter sets how much of the arterial inflow fails to reach the
    delayed plateau, which is the property the perfusion estimator has to recover.
    """
    generator = np.random.default_rng(seed)
    depth, height, width = size
    coords = np.stack(
        np.meshgrid(
            np.linspace(0, 1, depth),
            np.linspace(0, 1, height),
            np.linspace(0, 1, width),
            indexing="ij",
        ),
        axis=-1,
    )
    centre = coords - 0.5
    radius = np.sqrt((centre**2).sum(axis=-1))
    vessel = np.exp(-((radius - 0.18) ** 2) / (2 * 0.06**2))
    noise = generator.normal(0.0, 0.02, size=size)
    arterial = np.clip(400.0 * vessel + noise * 40.0, 0.0, 900.0)
    delivered = np.clip(1.0 - occlusion, 0.05, 1.0)
    delayed = np.clip(
        arterial * (1.0 - 0.55 * delivered) + generator.normal(0.0, 6.0, size=size), 0.0, 900.0
    )
    return arterial.astype(np.float64), delayed.astype(np.float64)
