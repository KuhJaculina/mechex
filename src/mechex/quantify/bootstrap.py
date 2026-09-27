"""Patient-level bootstrap resampling.

Ref: Sec. 3.8 and Sec. 3.9, uncertainty on correlation-type metrics from patient-level
bootstrap resampling within each site after averaging across repeats.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TAU = 1e-12


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile interval and the resampled distribution."""

    estimate: float
    lower: float
    upper: float
    samples: np.ndarray
    n_resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "n_resamples": self.n_resamples,
        }

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)


def cluster_resample_indices(
    clusters: np.ndarray,
    n_resamples: int,
    seed: int,
) -> list[np.ndarray]:
    """Resample whole clusters and return one index array per draw.

    Resampling patients rather than records keeps repeated records from the same patient
    inside one draw, which is what makes the interval a patient-level interval. A cluster
    drawn more than once contributes its records more than once, so the draws are ragged:
    forcing them into a fixed-width buffer would fail as soon as a cluster repeats.
    """
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = labels.size
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, n_clusters, size=(n_resamples, n_clusters))
    members: list[np.ndarray] = [np.flatnonzero(inverse == index) for index in range(n_clusters)]
    return [np.concatenate([members[index] for index in row]) for row in draws]


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_samples: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    clusters: np.ndarray | None = None,
) -> BootstrapResult:
    """Percentile bootstrap interval for ``statistic`` evaluated on index arrays."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    full_index = np.arange(n_samples)
    estimate = float(statistic(full_index))
    samples = np.empty(n_resamples, dtype=np.float64)
    if clusters is not None:
        for row, draw in enumerate(
            cluster_resample_indices(np.asarray(clusters), n_resamples, seed)
        ):
            samples[row] = float(statistic(draw))
    else:
        generator = np.random.default_rng(seed)
        draws = generator.integers(0, n_samples, size=(n_resamples, n_samples))
        for row in range(n_resamples):
            samples[row] = float(statistic(draws[row]))
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"), samples, n_resamples)
    lower = float(np.quantile(finite, alpha / 2.0))
    upper = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return BootstrapResult(estimate, lower, upper, samples, n_resamples)


def paired_bootstrap_difference(
    statistic: Callable[[np.ndarray], float],
    n_samples: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Interval for a paired difference, resampling cases once per draw.

    A paired comparison must draw the same cases for both arms; drawing them separately
    would remove the pairing and widen the interval.
    """
    index = np.arange(n_samples)
    estimate = float(statistic(index))
    generator = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for row in range(n_resamples):
        draw = generator.integers(0, n_samples, size=n_samples)
        samples[row] = float(statistic(draw))
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"), samples, n_resamples)
    return BootstrapResult(
        estimate,
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
        samples,
        n_resamples,
    )


def mean_over_repeats(values: np.ndarray) -> float:
    """Mean of a per-repeat metric, the summary the manuscript reports."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def interval_contains(lower: float, upper: float, value: float) -> bool:
    """Whether a value sits inside a closed interval."""
    return bool(lower - TAU <= value <= upper + TAU)


def overlap_fraction(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Fraction of the shorter interval covered by the intersection."""
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    if high <= low:
        return 0.0
    shorter = min(first[1] - first[0], second[1] - second[0])
    if shorter <= TAU:
        return 0.0
    return float((high - low) / shorter)
