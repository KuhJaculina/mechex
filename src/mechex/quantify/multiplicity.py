"""Multiplicity control for the stratified comparison.

Ref: Sec. 3.9, the multiple comparisons across strata are controlled with the
Benjamini-Hochberg procedure at a false-discovery rate of 0.05.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Benjamini-Hochberg step-up procedure.

    Returns the step-up adjusted p-values, the rejection mask and the largest rank that
    survives. Adjusted p-values are monotone by construction, which the step-up ordering
    guarantees but a naive per-rank rescaling does not.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return {"adjusted": p, "rejected": np.zeros(0, dtype=bool), "threshold_rank": -1}
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in the unit interval")
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    scaled = ranked * n / ranks
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    rejected = adjusted <= alpha
    surviving = np.flatnonzero(ranked <= alpha * ranks / n)
    threshold_rank = int(surviving.max()) if surviving.size else -1
    return {
        "adjusted": adjusted,
        "rejected": rejected,
        "threshold_rank": threshold_rank,
    }


def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Holm step-down procedure, reported alongside for comparison."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return {"adjusted": p, "rejected": np.zeros(0, dtype=bool)}
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    factors = np.arange(n, 0, -1, dtype=np.float64)
    scaled = np.clip(ranked * factors, 0.0, 1.0)
    adjusted = np.maximum.accumulate(scaled)
    restored = np.empty(n, dtype=np.float64)
    restored[order] = adjusted
    return {"adjusted": restored, "rejected": restored <= alpha}


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Bonferroni correction."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    adjusted = np.clip(p * max(p.size, 1), 0.0, 1.0)
    return {"adjusted": adjusted, "rejected": adjusted <= alpha}


def family_size(n_strata: int) -> int:
    """Size of the tested family for the stratified table."""
    return int(max(n_strata, 1))
