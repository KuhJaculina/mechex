"""Operating-point selection under the harm asymmetry.

Ref: Sec. 3.10, the deployed threshold bounds the false-positive rate below 5% at the cost
of a higher false-negative rate, because a false-adequate call is the harmful direction
for a repurposing decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.decision.abstention import HarmTiering
from mechex.quantify.roc import confusion_at


@dataclass(frozen=True)
class OperatingPoint:
    """One candidate cut with its error rates and net benefit."""

    threshold: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    true_negative_rate: float
    net_benefit: float
    expected_cost: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "true_positive_rate": self.true_positive_rate,
            "true_negative_rate": self.true_negative_rate,
            "net_benefit": self.net_benefit,
            "expected_cost": self.expected_cost,
        }


def evaluate_point(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    tiering: HarmTiering | None = None,
) -> OperatingPoint:
    """Error rates, net benefit and expected cost at one cut."""
    from mechex.decision.abstention import expected_cost
    from mechex.quantify.netbenefit import net_benefit

    counts = confusion_at(scores, labels, threshold)
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["true_negative"] + counts["false_positive"]
    false_positive_rate = counts["false_positive"] / max(negatives, 1)
    false_negative_rate = counts["false_negative"] / max(positives, 1)
    predicted = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    return OperatingPoint(
        threshold=float(threshold),
        false_positive_rate=float(false_positive_rate),
        false_negative_rate=float(false_negative_rate),
        true_positive_rate=counts["true_positive"] / max(positives, 1),
        true_negative_rate=counts["true_negative"] / max(negatives, 1),
        net_benefit=float(
            net_benefit(np.asarray(scores, dtype=np.float64), truth, float(threshold))
        ),
        expected_cost=float(expected_cost(predicted, truth, tiering)),
    )


def operating_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    grid: int = 100,
    tiering: HarmTiering | None = None,
) -> list[OperatingPoint]:
    """Every operating point on a quantile grid, so the choice is auditable."""
    values = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(np.quantile(values, np.linspace(0.01, 0.99, grid)))
    return [evaluate_point(values, labels, float(cut), tiering) for cut in thresholds]


def select_operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    false_positive_budget: float = 0.05,
    tiering: HarmTiering | None = None,
    grid: int = 100,
) -> OperatingPoint:
    """Lowest-cost point inside the false-positive budget.

    When no point satisfies the budget the least-cost point on the curve is returned with
    its own rates, so the caller sees that the budget was unreachable instead of a
    silently substituted threshold.
    """
    curve = operating_curve(scores, labels, grid, tiering)
    admissible = [
        point for point in curve if point.false_positive_rate <= false_positive_budget + 1e-12
    ]
    if admissible:
        return min(admissible, key=lambda point: point.expected_cost)
    return min(curve, key=lambda point: point.expected_cost)


def harm_tradeoff(
    curve: list[OperatingPoint], tiering: HarmTiering | None = None
) -> dict[str, float]:
    """Summary of the asymmetry across the curve."""
    costs = tiering or HarmTiering()
    return {
        "false_adequate_cost": costs.false_adequate_cost,
        "false_inadequate_cost": costs.false_inadequate_cost,
        "minimum_false_positive_rate": float(min(point.false_positive_rate for point in curve)),
        "maximum_false_positive_rate": float(max(point.false_positive_rate for point in curve)),
        "best_net_benefit": float(max(point.net_benefit for point in curve)),
    }
