"""Abstention evaluation.

Ref: Sec. 3.9 and Sec. 4, coverage and retained AUROC after the abstained cases are
removed; removing the lowest-confidence tenth raises the retained AUROC.
"""

from __future__ import annotations

import numpy as np

from mechex.decision.abstention import abstention_mask, coverage_and_retained_auroc
from mechex.quantify.roc import auroc


def abstention_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    deviation: np.ndarray,
    quantiles: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30),
) -> list[dict[str, float]]:
    """Coverage, retained AUROC and referral rate across referral quantiles."""
    rows: list[dict[str, float]] = []
    for quantile in quantiles:
        summary = coverage_and_retained_auroc(scores, labels, deviation, quantile)
        rows.append({"quantile": float(quantile), **summary})
    return rows


def retention_at(
    scores: np.ndarray,
    labels: np.ndarray,
    deviation: np.ndarray,
    quantile: float = 0.10,
) -> dict[str, float]:
    """Retained AUROC at one referral quantile, with the gain over full coverage."""
    baseline = auroc(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64))
    summary = coverage_and_retained_auroc(scores, labels, deviation, quantile)
    return {
        "full_coverage_auroc": float(baseline),
        "retained_auroc": float(summary["retained_auroc"]),
        "gain": float(summary["retained_auroc"] - baseline),
        "coverage": float(summary["coverage"]),
        "referral_rate": float(summary["referral_rate"]),
        "n_referred": float(summary["referral_rate"] * np.asarray(scores).size),
    }


def referral_set(deviation: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """Boolean mask of the cases referred for direct measurement."""
    return abstention_mask(deviation, quantile)


def threshold_consistency(
    scores: np.ndarray,
    labels: np.ndarray,
    deviation: np.ndarray,
    quantile: float = 0.10,
) -> dict[str, float]:
    """Check that abstention raises discrimination and that the gain is positive."""
    summary = retention_at(scores, labels, deviation, quantile)
    return {**summary, "gain_positive": float(summary["gain"] > 0.0)}
