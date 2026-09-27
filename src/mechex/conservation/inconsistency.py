"""Physical-inconsistency rate.

Ref: Sec. 2.3, the fraction of predictions whose conservation residual exceeds the
pre-specified tolerance, whose implied delivered fraction falls outside the physically
admissible range, or that violate monotonicity of exposure with respect to perfused
fraction.
"""

from __future__ import annotations

import numpy as np
import torch

from mechex.vascular.projection import MAXIMUM_DELIVERED_FRACTION, MINIMUM_DELIVERED_FRACTION


def residue_exceedance(residual: np.ndarray, tolerance: float) -> np.ndarray:
    """Boolean flag for residuals above the pre-specified tolerance."""
    return np.asarray(residual, dtype=np.float64) > float(tolerance)


def inadmissible_fraction(
    implied: np.ndarray,
    low: float = MINIMUM_DELIVERED_FRACTION,
    high: float = MAXIMUM_DELIVERED_FRACTION,
) -> np.ndarray:
    """Boolean flag for an implied delivered fraction outside the admissible interval."""
    values = np.asarray(implied, dtype=np.float64)
    return (values < low) | (values > high)


def monotonicity_violation(
    fractions: np.ndarray, exposures: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Per-record flag for a local decrease of exposure in perfused fraction.

    Records are ordered by perfused fraction; any negative step in exposure flags both
    members of the offending pair.
    """
    x = np.asarray(fractions, dtype=np.float64)
    y = np.asarray(exposures, dtype=np.float64)
    flags = np.zeros(x.shape[0], dtype=bool)
    if x.shape[0] < 2:
        return flags
    order = np.argsort(x)
    ordered_y = y[order]
    steps = np.diff(ordered_y)
    bad = np.where(steps < -tolerance)[0]
    for index in bad:
        flags[order[index]] = True
        flags[order[index + 1]] = True
    return flags


def inconsistency_flags(
    residual: np.ndarray,
    implied: np.ndarray,
    fractions: np.ndarray,
    exposures: np.ndarray,
    tolerance: float,
) -> dict[str, np.ndarray]:
    """The three component flags plus their union."""
    residual_flag = residue_exceedance(residual, tolerance)
    fraction_flag = inadmissible_fraction(implied)
    monotone_flag = monotonicity_violation(fractions, exposures)
    return {
        "residual": residual_flag,
        "implied_fraction": fraction_flag,
        "monotonicity": monotone_flag,
        "any": residual_flag | fraction_flag | monotone_flag,
    }


def inconsistency_rate(flags: dict[str, np.ndarray]) -> float:
    """Percentage of records flagged by at least one component."""
    union = flags["any"]
    if union.size == 0:
        return 0.0
    return float(100.0 * union.mean())


def residual_from_tensors(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
) -> np.ndarray:
    """Detach the residual of a batch as a numpy array."""
    from mechex.conservation.residual import relative_residual

    return relative_residual(reference, reconstruction).detach().cpu().numpy()
