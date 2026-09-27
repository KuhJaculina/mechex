"""Fold error between predicted and reference exposure.

Ref: Sec. 2.2, the geometric fold-error reported for tissue-level quantities.
"""

from __future__ import annotations

import numpy as np

POSITIVE_FLOOR = 1e-6


def geometric_fold_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    floor: float = POSITIVE_FLOOR,
) -> float:
    """Geometric fold error ``exp(mean |ln(pred/ref)|)``.

    A geometric mean is the right summary for a ratio: a two-fold over-prediction and a
    two-fold under-prediction are the same error and cancel symmetrically.
    """
    p = np.asarray(predicted, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    if p.size != r.size:
        raise ValueError("predicted and reference must have the same length")
    if p.size == 0:
        return float("nan")
    ratio = np.clip(p, floor, None) / np.clip(r, floor, None)
    return float(np.exp(np.mean(np.abs(np.log(ratio)))))


def fold_error_distribution(
    predicted: np.ndarray,
    reference: np.ndarray,
    floor: float = POSITIVE_FLOOR,
) -> dict[str, float]:
    """Fold error alongside its dispersion, which the mean alone hides."""
    p = np.asarray(predicted, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    log_ratio = np.abs(np.log(np.clip(p, floor, None) / np.clip(r, floor, None)))
    return {
        "geometric_fold_error": float(np.exp(log_ratio.mean())),
        "median_fold_error": float(np.exp(np.median(log_ratio))),
        "within_two_fold": float(np.mean(log_ratio <= np.log(2.0))),
        "within_three_fold": float(np.mean(log_ratio <= np.log(3.0))),
    }


def zone_position(fold_error: float, low: float = 1.5, high: float = 3.0) -> str:
    """Where a fold error sits relative to a declared admissible zone."""
    if fold_error < low:
        return "below"
    if fold_error > high:
        return "above"
    return "inside"


def fold_error_report(predicted: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Fold-error summary with its position in the declared zone attached."""
    distribution = fold_error_distribution(predicted, reference)
    value = distribution["geometric_fold_error"]
    return {
        **distribution,
        "zone_position": zone_position(value),
        "zone_low": 1.5,
        "zone_high": 3.0,
    }
