"""DeLong test for two correlated ROC curves.

Ref: Sec. 3.9, the DeLong test for differences in AUROC, and the paired comparison
between the proposed model and the strongest reconstructed baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeLongResult:
    """Difference in AUROC with its standard error, z statistic and two-sided p-value."""

    auc_a: float
    auc_b: float
    difference: float
    standard_error: float
    z_statistic: float
    p_value: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
        }


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered = values[order]
    size = ordered.size
    ranks = np.zeros(size, dtype=np.float64)
    index = 0
    while index < size:
        stop = index
        while stop + 1 < size and ordered[stop + 1] == ordered[index]:
            stop += 1
        ranks[index : stop + 1] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    restored = np.empty(size, dtype=np.float64)
    restored[order] = ranks
    return restored


def _fast_delong(
    predictions: np.ndarray,
    n_positive: int,
) -> tuple[np.ndarray, np.ndarray]:
    """AUROCs and their covariance for a set of classifiers on shared cases.

    ``predictions`` is ``[n_classifiers, n_cases]`` with the positive cases first.
    """
    m = int(n_positive)
    total = predictions.shape[1]
    n = total - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    n_classifiers = predictions.shape[0]
    tx = np.empty((n_classifiers, m), dtype=np.float64)
    ty = np.empty((n_classifiers, n), dtype=np.float64)
    tz = np.empty((n_classifiers, total), dtype=np.float64)
    for row in range(n_classifiers):
        tx[row] = _midrank(positive[row])
        ty[row] = _midrank(negative[row])
        tz[row] = _midrank(predictions[row])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, covariance


def delong_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> DeLongResult:
    """Two-sided DeLong comparison of two classifiers evaluated on the same cases."""
    from scipy import stats

    truth = np.asarray(labels, dtype=np.int64).ravel()
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    if not (truth.size == a.size == b.size):
        raise ValueError("labels and both score vectors must share a length")
    order = np.argsort(-truth, kind="mergesort")
    n_positive = int(truth.sum())
    if n_positive == 0 or n_positive == truth.size:
        raise ValueError("the DeLong test needs both classes present")
    stacked = np.vstack([a[order], b[order]])
    aucs, covariance = _fast_delong(stacked, n_positive)
    contrast = np.array([1.0, -1.0])
    difference = float(contrast @ aucs)
    variance = float(contrast @ covariance @ contrast)
    if variance <= 0:
        return DeLongResult(
            auc_a=float(aucs[0]),
            auc_b=float(aucs[1]),
            difference=difference,
            standard_error=0.0,
            z_statistic=float("nan"),
            p_value=float("nan"),
        )
    standard_error = float(np.sqrt(variance))
    z = difference / standard_error
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    return DeLongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        difference=difference,
        standard_error=standard_error,
        z_statistic=float(z),
        p_value=p_value,
    )
