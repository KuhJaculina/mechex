"""Decision-curve analysis.

Ref: Sec. 3.9, net benefit from decision-curve analysis; Sec. 3.10, each operating point
reported with its false-positive rate and net-benefit curve.
"""

from __future__ import annotations

import numpy as np


def net_benefit(
    probability: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """Net benefit of treating everyone whose probability exceeds ``threshold``.

    The treat-all and treat-none references are returned by :func:`decision_curve` so the
    model's curve can be judged against them rather than in isolation.
    """
    p = np.asarray(probability, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.int64).ravel()
    if p.size == 0:
        return 0.0
    predicted = p >= threshold
    true_positive = float(np.sum(predicted & (truth == 1)))
    false_positive = float(np.sum(predicted & (truth == 0)))
    odds = threshold / max(1.0 - threshold, 1e-12)
    return float(true_positive / truth.size - false_positive / truth.size * odds)


def treat_all_net_benefit(labels: np.ndarray, threshold: float) -> float:
    """Net benefit of treating every case."""
    truth = np.asarray(labels, dtype=np.int64).ravel()
    prevalence = float(truth.mean()) if truth.size else 0.0
    return float(prevalence - (1.0 - prevalence) * threshold / max(1.0 - threshold, 1e-12))


def decision_curve(
    probability: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Model, treat-all and treat-none net benefit across a threshold grid."""
    grid = (
        np.linspace(0.01, 0.99, 99)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64).ravel()
    )
    rows: list[dict[str, float]] = []
    for threshold in grid:
        rows.append(
            {
                "threshold": float(threshold),
                "model": net_benefit(probability, labels, float(threshold)),
                "treat_all": treat_all_net_benefit(labels, float(threshold)),
                "treat_none": 0.0,
            }
        )
    return rows


def maximum_net_benefit_advantage(
    probability: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> float:
    """Largest gap between the model curve and the better of the two references."""
    curve = decision_curve(probability, labels, thresholds)
    if not curve:
        return 0.0
    best = max(row["model"] - max(row["treat_all"], row["treat_none"]) for row in curve)
    return float(best)


def stable_around(
    probability: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    relative_window: float = 0.30,
    points: int = 13,
) -> dict[str, float]:
    """Net benefit near a fixed threshold.

    Ref: Sec. 4, the net benefit stays stable within a window around the fixed threshold,
    which bounds how much the decision depends on that exact value.
    """
    low = max(0.0, threshold * (1.0 - relative_window))
    high = min(1.0, threshold * (1.0 + relative_window))
    grid = np.linspace(low, high, points)
    values = np.array([net_benefit(probability, labels, float(cut)) for cut in grid])
    return {
        "lower_threshold": float(low),
        "upper_threshold": float(high),
        "minimum_net_benefit": float(values.min()),
        "maximum_net_benefit": float(values.max()),
        "net_benefit_range": float(values.max() - values.min()),
    }
