"""Silent-failure monitoring.

Ref: Sec. 3.10, silent-failure monitoring is a drift detector over the input distribution
and the residual distribution, with an alerting rule calibrated so that the alert rate on
stationary data stays below one per thousand cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ALERT_RATE_CEILING = 1e-3


@dataclass
class RunningMoments:
    """Streaming mean and variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def as_dict(self) -> dict[str, float]:
        return {"count": float(self.count), "mean": self.mean, "std": self.std}


@dataclass
class StandardisedDriftDetector:
    """Flags a case when its residual exceeds a calibrated multiple of the reference spread."""

    threshold: float = float("inf")
    inputs: RunningMoments = field(default_factory=RunningMoments)
    residuals: RunningMoments = field(default_factory=RunningMoments)
    alerts: int = 0
    seen: int = 0

    def observe(self, residual: float, summary: float) -> bool:
        """Record one case and report whether it alerts."""
        self.seen += 1
        alerted = self._is_alert(residual, summary)
        if alerted:
            self.alerts += 1
        self.inputs.update(float(summary))
        self.residuals.update(float(residual))
        return alerted

    def _is_alert(self, residual: float, summary: float) -> bool:
        if not np.isfinite(self.threshold) or self.residuals.count < 30:
            return False
        scale = max(self.residuals.std, 1e-9)
        return abs(float(residual)) > self.threshold * scale

    @property
    def alert_rate(self) -> float:
        if self.seen == 0:
            return 0.0
        return self.alerts / self.seen

    def as_dict(self) -> dict[str, object]:
        return {
            "alerts": self.alerts,
            "seen": self.seen,
            "alert_rate": self.alert_rate,
            "threshold": self.threshold,
            "residual_moments": self.residuals.as_dict(),
            "input_moments": self.inputs.as_dict(),
        }


def calibrate_threshold(
    residuals: np.ndarray,
    ceiling: float = ALERT_RATE_CEILING,
) -> float:
    """Smallest multiple of the residual spread whose stationary alert rate stays under ``ceiling``.

    The rule is calibrated on stationary data rather than chosen, which is what makes the
    one-per-thousand ceiling meaningful.
    """
    values = np.abs(np.asarray(residuals, dtype=np.float64).ravel())
    if values.size == 0:
        return float("inf")
    scale = float(values.std()) if float(values.std()) > 0 else 1.0
    quantile = 1.0 - ceiling
    return float(np.quantile(values, quantile) / max(scale, 1e-9))


def calibrate_and_monitor(
    reference_residuals: np.ndarray,
    stream_residuals: np.ndarray,
    stream_summaries: np.ndarray,
    ceiling: float = ALERT_RATE_CEILING,
) -> StandardisedDriftDetector:
    """Calibrate on a stationary window, then monitor a stream."""
    detector = StandardisedDriftDetector(
        threshold=calibrate_threshold(reference_residuals, ceiling)
    )
    for residual, summary in zip(stream_residuals, stream_summaries, strict=True):
        detector.observe(float(residual), float(summary))
    return detector


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Population stability index between two samples of one input summarised to a scalar.

    PSI complements the residual monitor: it detects a shift in the inputs before the
    residuals move.
    """
    base = np.asarray(reference, dtype=np.float64).ravel()
    other = np.asarray(current, dtype=np.float64).ravel()
    if base.size == 0 or other.size == 0:
        return float("nan")
    edges = np.quantile(base, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * 1e-12)
    base_counts, _ = np.histogram(base, bins=edges)
    other_counts, _ = np.histogram(other, bins=edges)
    base_share = np.clip(base_counts / max(base.size, 1), 1e-6, None)
    other_share = np.clip(other_counts / max(other.size, 1), 1e-6, None)
    return float(np.sum((other_share - base_share) * np.log(other_share / base_share)))
