"""Rank-based discrimination statistics.

Ref: Sec. 3.9, the AUROC for the adequate/inadequate classification and the DeLong test
for differences between two correlated ROC curves.
"""

from __future__ import annotations

import numpy as np

TAU = 1e-12


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties sharing the mean of their positions."""
    array = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    index = 0
    while index < array.size:
        stop = index
        while stop + 1 < array.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        average = 0.5 * (index + stop) + 1.0
        ranks[order[index : stop + 1]] = average
        index = stop + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve by the Mann-Whitney rank statistic.

    Ties contribute a half-credit through the average rank, which keeps the statistic
    exact for the discretised exposure index.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.int64).ravel()
    if values.size != truth.size:
        raise ValueError("scores and labels must have the same length")
    positives = truth == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_average(values)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-positive rate, true-positive rate and thresholds at every cut."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_truth = truth[order]
    distinct = np.where(np.diff(sorted_values))[0]
    cut_points = np.r_[distinct, sorted_values.size - 1]
    true_positive = np.cumsum(sorted_truth)[cut_points]
    false_positive = 1 + cut_points - true_positive
    n_pos = float(sorted_truth.sum())
    n_neg = float(sorted_truth.size - sorted_truth.sum())
    tpr = true_positive / max(n_pos, TAU)
    fpr = false_positive / max(n_neg, TAU)
    thresholds = sorted_values[cut_points]
    return (
        np.r_[0.0, fpr],
        np.r_[0.0, tpr],
        np.r_[np.inf, thresholds],
    )


def operating_point_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    budget: float = 0.05,
) -> dict[str, float]:
    """Highest-sensitivity cut whose false-positive rate stays inside ``budget``."""
    fpr, tpr, thresholds = roc_curve(scores, labels)
    admissible = fpr <= budget + TAU
    if not admissible.any():
        return {"threshold": float(np.max(scores)), "fpr": 0.0, "tpr": 0.0}
    index = int(np.argmax(np.where(admissible, tpr, -1.0)))
    return {
        "threshold": float(thresholds[index]),
        "fpr": float(fpr[index]),
        "tpr": float(tpr[index]),
    }


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Cut maximising sensitivity minus false-positive rate."""
    fpr, tpr, thresholds = roc_curve(scores, labels)
    statistic = tpr - fpr
    index = int(np.argmax(statistic))
    return {
        "threshold": float(thresholds[index]),
        "fpr": float(fpr[index]),
        "tpr": float(tpr[index]),
        "youden": float(statistic[index]),
    }


def confusion_at(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Confusion counts for a cut on the score."""
    predicted = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    return {
        "true_positive": int(np.sum((predicted == 1) & (truth == 1))),
        "false_positive": int(np.sum((predicted == 1) & (truth == 0))),
        "true_negative": int(np.sum((predicted == 0) & (truth == 0))),
        "false_negative": int(np.sum((predicted == 0) & (truth == 1))),
    }
