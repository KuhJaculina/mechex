"""Abstention policy and the harm tiering that fixes its threshold.

Ref: Sec. 3.6, the decision layer abstains when the predictive deviation exceeds the
policy limit; Sec. 3.10, an incorrect adequate call is the harmful direction, so the
deployed threshold bounds the false-positive rate below 5%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.decision.threshold import ADEQUATE, INADEQUATE, ThresholdDecision


@dataclass(frozen=True)
class HarmTiering:
    """Expected cost of each error direction.

    A false-adequate call diverts a repurposing programme onto a molecule that will not
    reach the tissue, which the manuscript treats as the costlier direction.
    """

    false_adequate_cost: float = 5.0
    false_inadequate_cost: float = 1.0


def abstention_mask(deviation: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """Flag the highest-deviation fraction of records for referral."""
    values = np.asarray(deviation, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    fraction = float(np.clip(quantile, 0.0, 1.0))
    if fraction <= 0.0:
        return np.zeros(values.shape[0], dtype=bool)
    cut = np.quantile(values, 1.0 - fraction)
    return values >= cut


def coverage_and_retained_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    deviation: np.ndarray,
    quantile: float = 0.10,
) -> dict[str, float]:
    """Coverage and AUROC after the abstained records leave the evaluation.

    Ref: Sec. 3.9 and Sec. 4, removing the lowest-confidence tenth raises the retained
    AUROC while one case in ten is referred for direct measurement.
    """
    from mechex.quantify.roc import auroc

    mask = abstention_mask(deviation, quantile)
    retained = ~mask
    coverage = float(retained.mean()) if retained.size else 0.0
    if retained.sum() < 2 or len(np.unique(np.asarray(labels)[retained])) < 2:
        retained_auroc = float("nan")
    else:
        retained_auroc = auroc(np.asarray(scores)[retained], np.asarray(labels)[retained])
    return {
        "coverage": coverage,
        "retained_auroc": retained_auroc,
        "referral_rate": 1.0 - coverage,
        "n_retained": float(retained.sum()),
    }


def expected_cost(
    predicted: np.ndarray,
    truth: np.ndarray,
    tiering: HarmTiering | None = None,
) -> float:
    """Expected harm of a labelling at the harm tiering."""
    costs = tiering or HarmTiering()
    predicted = np.asarray(predicted, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    false_adequate = np.sum((predicted == ADEQUATE) & (truth == INADEQUATE))
    false_inadequate = np.sum((predicted == INADEQUATE) & (truth == ADEQUATE))
    return float(
        false_adequate * costs.false_adequate_cost + false_inadequate * costs.false_inadequate_cost
    )


def select_boundary(
    scores: np.ndarray,
    labels: np.ndarray,
    tiering: HarmTiering | None = None,
    false_positive_budget: float = 0.05,
    grid: int = 200,
) -> dict[str, float]:
    """Choose the exposure boundary that minimises expected cost inside an FPR budget.

    The manuscript fixes the deployed boundary on the development split and bounds the
    false-positive rate below 5%; this routine reproduces that selection rule.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    negatives = int(np.sum(truth == INADEQUATE))
    candidates = np.quantile(values, np.linspace(0.01, 0.99, grid))
    best: dict[str, float] | None = None
    for boundary in np.unique(candidates):
        predicted = (values >= boundary).astype(np.int64)
        fpr = float(np.sum((predicted == ADEQUATE) & (truth == INADEQUATE))) / max(negatives, 1)
        if fpr > false_positive_budget:
            continue
        cost = expected_cost(predicted, truth, tiering)
        if best is None or cost < best["expected_cost"]:
            best = {"boundary": float(boundary), "expected_cost": cost, "false_positive_rate": fpr}
    if best is None:
        fallback = ThresholdDecision(0.0)
        rate, _, _ = fallback.decision_curve_point(values, truth)
        return {
            "boundary": float(np.max(values)),
            "expected_cost": float("inf"),
            "false_positive_rate": rate,
        }
    return best


def cost_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    tiering: HarmTiering | None = None,
    grid: int = 200,
) -> list[dict[str, float]]:
    """Expected cost at every operating point, the auditable form of the choice."""
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    negatives = int(np.sum(truth == INADEQUATE))
    positives = int(np.sum(truth == ADEQUATE))
    rows: list[dict[str, float]] = []
    for boundary in np.unique(np.quantile(values, np.linspace(0.01, 0.99, grid))):
        predicted = (values >= boundary).astype(np.int64)
        fpr = float(np.sum((predicted == ADEQUATE) & (truth == INADEQUATE))) / max(negatives, 1)
        fnr = float(np.sum((predicted == INADEQUATE) & (truth == ADEQUATE))) / max(positives, 1)
        rows.append(
            {
                "boundary": float(boundary),
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "expected_cost": expected_cost(predicted, truth, tiering),
            }
        )
    return rows
