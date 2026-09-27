"""Rank correlation and its permutation reference distribution.

Ref: Sec. 2.8, the rank correlation between the exposure ranking and the mechanism-only
ranking; Sec. 3.11, the rank correlation between predicted efflux propensity and the
measured accumulation ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.quantify.roc import rank_average


@dataclass(frozen=True)
class CorrelationResult:
    """Spearman rho with an optional permutation p-value."""

    rho: float
    p_value: float
    n: int
    n_permutations: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rho": self.rho,
            "p_value": self.p_value,
            "n": self.n,
            "n_permutations": self.n_permutations,
        }


def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman rank correlation with ties handled by average ranks."""
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("both vectors must have the same length")
    if a.size < 2:
        return float("nan")
    ra = rank_average(a)
    rb = rank_average(b)
    centred_a = ra - ra.mean()
    centred_b = rb - rb.mean()
    denominator = float(np.sqrt((centred_a**2).sum() * (centred_b**2).sum()))
    if denominator < 1e-12:
        return float("nan")
    return float((centred_a * centred_b).sum() / denominator)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    centred_a = a - a.mean()
    centred_b = b - b.mean()
    denominator = float(np.sqrt((centred_a**2).sum() * (centred_b**2).sum()))
    if denominator < 1e-12:
        return float("nan")
    return float((centred_a * centred_b).sum() / denominator)


def permutation_p_value(
    first: np.ndarray,
    second: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 0,
    two_sided: bool = True,
) -> float:
    """Permutation p-value for a rank correlation.

    Exact in the sense that the reference distribution is generated from the observed
    values themselves, which is the appropriate null when the sample is a small molecule
    panel rather than a random draw from a population.
    """
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    observed = abs(spearman_rho(a, b)) if two_sided else spearman_rho(a, b)
    if not np.isfinite(observed):
        return float("nan")
    generator = np.random.default_rng(seed)
    count = 0
    rank_a = rank_average(a)
    for _ in range(n_permutations):
        shuffled = generator.permutation(b)
        value = _pearson(rank_a, rank_average(shuffled))
        statistic = abs(value) if two_sided else value
        if statistic >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def spearman_with_permutation(
    first: np.ndarray,
    second: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 0,
) -> CorrelationResult:
    """Rank correlation with its permutation p-value attached."""
    rho = spearman_rho(first, second)
    p_value = permutation_p_value(first, second, n_permutations=n_permutations, seed=seed)
    return CorrelationResult(
        rho=rho,
        p_value=p_value,
        n=int(np.asarray(first).size),
        n_permutations=n_permutations,
    )


def rank_positions(values: np.ndarray, descending: bool = True) -> np.ndarray:
    """Competition ranks of a score vector, ties sharing the better position."""
    array = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(-array if descending else array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.int64)
    ranks[order] = np.arange(1, array.size + 1)
    return ranks


def rank_movement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Signed rank change between two rankings of the same items.

    Negative entries moved up the list (a better position) when the first ranking is the
    comparator.
    """
    movement: np.ndarray = rank_positions(second) - rank_positions(first)
    return movement


def moved_at_least(first: np.ndarray, second: np.ndarray, threshold: int = 5) -> int:
    """Number of items whose rank moved by at least ``threshold`` positions."""
    movement = rank_movement(first, second)
    return int(np.sum(np.abs(movement) >= threshold))
