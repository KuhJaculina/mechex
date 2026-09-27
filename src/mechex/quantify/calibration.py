"""Calibration of the adequacy probability.

Ref: Sec. 3.9, expected calibration error over fifteen equal-width bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU = 1e-12
LOGIT_CLIP = 30.0


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Sigmoid evaluated on clipped logits, so a large score does not overflow."""
    value: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)))
    return value


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected calibration error over equal-width probability bins.

    Each bin contributes its mass times the absolute gap between the mean predicted
    probability and the observed rate, so the value is the bin-mass-weighted mean gap.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.float64).ravel()
    if probabilities.size != truth.size:
        raise ValueError("probabilities and labels must have the same length")
    if probabilities.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        low = edges[index]
        high = edges[index + 1]
        if index == n_bins - 1:
            selected = (probabilities >= low) & (probabilities <= high)
        else:
            selected = (probabilities >= low) & (probabilities < high)
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(truth[selected].mean())
        total += float(selected.mean()) * abs(confidence - accuracy)
    return float(total)


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    """Per-bin confidence, accuracy and count, the curve behind the ECE."""
    described = describe_bins(
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        n_bins,
    )
    limits = limits_for(n_bins)
    return [
        {
            "bin_lower": float(edge),
            "bin_upper": float(limits[index + 1]),
            "confidence": float(described["confidence"][index]),
            "accuracy": float(described["accuracy"][index]),
            "count": float(described["count"][index]),
        }
        for index, edge in enumerate(limits[:-1])
    ]


def describe_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> dict[str, np.ndarray]:
    """Bin-wise confidence, accuracy and count arrays."""
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    confidence = np.zeros(n_bins, dtype=np.float64)
    accuracy = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.float64)
    for position in range(n_bins):
        selected = index == position
        count[position] = float(selected.sum())
        if count[position] > 0:
            confidence[position] = float(probabilities[selected].mean())
            accuracy[position] = float(labels[selected].mean())
    return {"confidence": confidence, "accuracy": accuracy, "count": count}


def limits_for(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of the adequacy probability."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


@dataclass
class PlattCalibrator:
    """Logistic recalibration of a continuous score onto an adequacy probability.

    The exposure index is not a probability, so a calibration map is needed before an
    expected calibration error is defined for it.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        """Fit by Newton steps on the two-parameter logistic likelihood."""
        x = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        centre = float(x.mean())
        scale = float(x.std()) if float(x.std()) > TAU else 1.0
        z = (x - centre) / scale
        slope = 1.0
        intercept = 0.0
        for _ in range(100):
            logits = intercept + slope * z
            probability = _stable_sigmoid(logits)
            residual = probability - y
            weights = np.clip(probability * (1.0 - probability), 1e-6, None)
            gradient = np.array([residual.sum(), (residual * z).sum()])
            hessian = np.array(
                [
                    [weights.sum(), (weights * z).sum()],
                    [(weights * z).sum(), (weights * z * z).sum()],
                ]
            )
            step = np.linalg.solve(hessian + 1e-8 * np.eye(2), gradient)
            intercept -= float(step[0])
            slope -= float(step[1])
            if float(np.abs(step).max()) < 1e-9:
                break
        self.slope = float(slope / scale)
        self.intercept = float(intercept - slope * centre / scale)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted logistic map."""
        x = np.asarray(scores, dtype=np.float64)
        return _stable_sigmoid(self.intercept + self.slope * x)


def exposure_probabilities(
    scores: np.ndarray,
    boundary: float,
    scale: float | None = None,
) -> np.ndarray:
    """Map an exposure index to an adequacy probability around a fixed boundary.

    Used when no calibration set is available; the scale defaults to the spread of the
    scores themselves.
    """
    x = np.asarray(scores, dtype=np.float64)
    width = float(scale) if scale is not None else max(float(x.std()), 1e-6)
    return _stable_sigmoid((x - boundary) / width)
