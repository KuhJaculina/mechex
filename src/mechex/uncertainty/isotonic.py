"""Post-hoc isotonic recalibration of the predictive deviation.

Ref: Sec. 3.6, the uncertainty head's calibrated standard deviation is produced by
combining its evidential output with an isotonic recalibration developed on a separate
calibration set.
"""

from __future__ import annotations

from typing import cast

import numpy as np

# A Gaussian's mean absolute deviation is sqrt(2/pi) sigma; the calibrator is fitted on
# absolute residuals and converted back to the standard-deviation scale with this factor.
MEAN_ABS_TO_SIGMA = float(np.sqrt(np.pi / 2.0))


def pool_adjacent_violators(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Isotonic (non-decreasing) least-squares fit by pool-adjacent-violators.

    Blocks are held as value, weight and length triples rather than as an array indexed by
    the leading element: a merged block's pooled mean belongs to the whole run, and reading
    it back from the leading index alone would leave the trailing members of the run
    carrying their unmerged values.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    block_values: list[float] = []
    block_weights: list[float] = []
    block_lengths: list[int] = []
    for index in range(x.size):
        value = float(x[index])
        weight = float(w[index])
        length = 1
        while block_values and block_values[-1] > value + 1e-12:
            previous_value = block_values.pop()
            previous_weight = block_weights.pop()
            previous_length = block_lengths.pop()
            total_weight = previous_weight + weight
            value = (previous_value * previous_weight + value * weight) / total_weight
            weight = total_weight
            length = previous_length + length
        block_values.append(value)
        block_weights.append(weight)
        block_lengths.append(length)
    fitted = np.empty_like(x)
    cursor = 0
    for value, length in zip(block_values, block_lengths, strict=True):
        fitted[cursor : cursor + length] = value
        cursor += length
    return fitted


class IsotonicRecalibrator:
    """Maps a raw predictive deviation to a recalibrated one.

    The fit is monotone by construction, so a case the model already considers uncertain
    can never be assigned a smaller deviation than a case it considers confident.
    """

    def __init__(self, n_bins: int = 32) -> None:
        self.n_bins = int(n_bins)
        self.edges: np.ndarray | None = None
        self.fitted: np.ndarray | None = None
        self.global_scale = 1.0

    def fit(self, predicted_std: np.ndarray, absolute_error: np.ndarray) -> IsotonicRecalibrator:
        """Fit the calibration map from predicted deviations to observed absolute errors."""
        predicted = np.asarray(predicted_std, dtype=np.float64).ravel()
        observed = np.asarray(absolute_error, dtype=np.float64).ravel()
        if predicted.size != observed.size:
            raise ValueError("predicted and observed arrays must have the same length")
        if predicted.size == 0:
            self.edges = np.array([0.0, 1.0])
            self.fitted = np.array([0.0, 0.0])
            return self
        self.global_scale = float(observed.mean() / max(predicted.mean(), 1e-12))
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(predicted, quantiles)
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * 1e-12)
        bin_index = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, self.n_bins - 1)
        means = np.zeros(self.n_bins, dtype=np.float64)
        counts = np.zeros(self.n_bins, dtype=np.float64)
        for index in range(self.n_bins):
            selected = bin_index == index
            counts[index] = float(selected.sum())
            if counts[index] > 0:
                means[index] = float(observed[selected].mean())
        populated = counts > 0
        if populated.any():
            filled = np.interp(
                np.arange(self.n_bins),
                np.flatnonzero(populated),
                means[populated],
            )
        else:
            filled = np.full(self.n_bins, float(observed.mean()))
        self.edges = edges
        self.fitted = pool_adjacent_violators(filled, counts.clip(min=1.0))
        return self

    def predict(self, predicted_std: np.ndarray) -> np.ndarray:
        """Apply the fitted map, returning a recalibrated standard deviation."""
        if self.edges is None or self.fitted is None:
            raise RuntimeError("the recalibrator must be fitted before use")
        predicted = np.asarray(predicted_std, dtype=np.float64).ravel()
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        interpolated: np.ndarray = np.interp(predicted, centres, self.fitted)
        return interpolated / MEAN_ABS_TO_SIGMA

    def is_fitted(self) -> bool:
        return self.edges is not None and self.fitted is not None

    def as_dict(self) -> dict[str, object]:
        """Serialisable state, so a checkpoint can restore the calibration."""
        if self.edges is None or self.fitted is None:
            return {"fitted": False}
        return {
            "is_fitted": True,
            "n_bins": self.n_bins,
            "edges": self.edges.tolist(),
            "fitted_values": self.fitted.tolist(),
            "global_scale": self.global_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IsotonicRecalibrator:
        """Restore a calibrator written by :meth:`as_dict`."""
        n_bins = int(cast(int, payload.get("n_bins", 32)))
        calibrator = cls(n_bins=n_bins)
        if payload.get("is_fitted"):
            calibrator.edges = np.asarray(payload["edges"], dtype=np.float64)
            calibrator.fitted = np.asarray(payload["fitted_values"], dtype=np.float64)
            calibrator.global_scale = float(cast(float, payload.get("global_scale", 1.0)))
        return calibrator
