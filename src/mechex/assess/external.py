"""External-site evaluation with bootstrap intervals and paired contrasts.

Ref: Sec. 2.2 and Sec. 3.9, concordance and discrimination reported for the fully held-out
external site as means over five repeats with patient-level resampling, with a paired
DeLong test for discrimination and a patient-level bootstrap for concordance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.cohort.records import CohortFixture
from mechex.quantify.bootstrap import (
    bootstrap_ci,
    interval_contains,
    mean_over_repeats,
    overlap_fraction,
)
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.delong import delong_test
from mechex.quantify.fold import geometric_fold_error
from mechex.quantify.roc import auroc


@dataclass(frozen=True)
class ExternalMetrics:
    """One comparator's external-site metrics with their intervals."""

    identifier: str
    method: str
    ccc: float
    ccc_lower: float
    ccc_upper: float
    auroc: float
    auroc_lower: float
    auroc_upper: float
    fold_error: float
    inconsistency_percent: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "method": self.method,
            "ccc": self.ccc,
            "ccc_interval": [self.ccc_lower, self.ccc_upper],
            "auroc": self.auroc,
            "auroc_interval": [self.auroc_lower, self.auroc_upper],
            "fold_error": self.fold_error,
            "inconsistency_percent": self.inconsistency_percent,
            "n": self.n,
        }


def concordance_interval(
    prediction: np.ndarray,
    reference: np.ndarray,
    patients: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20260927,
) -> tuple[float, float, float]:
    """Concordance with a patient-level bootstrap interval.

    The resampling unit is the patient, so repeated records from one patient move together.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    clusters = np.asarray(patients, dtype=object)

    def statistic(index: np.ndarray) -> float:
        return lin_ccc(predicted[index], truth[index])

    result = bootstrap_ci(
        statistic, predicted.size, n_resamples=n_resamples, seed=seed, clusters=clusters
    )
    return result.estimate, result.lower, result.upper


def discrimination_interval(
    prediction: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20260927,
) -> tuple[float, float, float]:
    """AUROC with a patient-level bootstrap interval."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    clusters = np.asarray(patients, dtype=object)

    def statistic(index: np.ndarray) -> float:
        selected = truth[index]
        if len(np.unique(selected)) < 2:
            return float("nan")
        return auroc(predicted[index], selected)

    result = bootstrap_ci(
        statistic, predicted.size, n_resamples=n_resamples, seed=seed, clusters=clusters
    )
    return result.estimate, result.lower, result.upper


def evaluate_external(
    identifier: str,
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    patients: np.ndarray,
    inconsistency_percent: float = float("nan"),
    tau: float = 0.30,
    n_resamples: int = 2000,
    seed: int = 20260927,
) -> ExternalMetrics:
    """Full external-site metric set for one comparator."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    labels = (truth >= 1.0 + tau).astype(np.int64)
    ccc, ccc_low, ccc_high = concordance_interval(predicted, truth, patients, n_resamples, seed)
    a_roc, auroc_low, auroc_high = discrimination_interval(
        predicted, labels, patients, n_resamples, seed
    )
    return ExternalMetrics(
        identifier=identifier,
        method=method,
        ccc=float(ccc),
        ccc_lower=float(ccc_low),
        ccc_upper=float(ccc_high),
        auroc=float(a_roc),
        auroc_lower=float(auroc_low),
        auroc_upper=float(auroc_high),
        fold_error=float(
            geometric_fold_error(np.clip(predicted, 1e-6, None), np.clip(truth, 1e-6, None))
        ),
        inconsistency_percent=float(inconsistency_percent),
        n=int(predicted.size),
    )


def paired_discrimination_contrast(
    labels: np.ndarray,
    proposed: np.ndarray,
    comparator: np.ndarray,
) -> dict[str, float]:
    """Paired DeLong contrast of the proposed model against a comparator."""
    return delong_test(labels, proposed, comparator).as_dict()


def paired_concordance_contrast(
    proposed: np.ndarray,
    comparator: np.ndarray,
    reference: np.ndarray,
    patients: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20260927,
) -> dict[str, float]:
    """Patient-level bootstrap interval for the concordance difference."""
    from mechex.quantify.bootstrap import paired_bootstrap_difference

    proposed_values = np.asarray(proposed, dtype=np.float64)
    comparator_values = np.asarray(comparator, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    clusters = np.asarray(patients, dtype=object)

    def statistic(index: np.ndarray) -> float:
        return lin_ccc(proposed_values[index], truth[index]) - lin_ccc(
            comparator_values[index], truth[index]
        )

    del clusters  # the paired interval resamples records within the same patient set
    result = paired_bootstrap_difference(statistic, proposed_values.size, n_resamples, seed=seed)
    return {
        "difference": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
    }


def criterion_outcome(
    metrics: ExternalMetrics,
    ccc_threshold: float,
    auroc_threshold: float,
) -> dict[str, object]:
    """Whether the pre-specified criteria were met, with the margins.

    Ref: Sec. 2.2 and Sec. 3.5, concordance at least 0.75 and AUROC at least 0.85 at the
    external site.
    """
    return {
        "ccc_threshold": float(ccc_threshold),
        "auroc_threshold": float(auroc_threshold),
        "ccc_margin": float(metrics.ccc - ccc_threshold),
        "auroc_margin": float(metrics.auroc - auroc_threshold),
        "ccc_met": bool(metrics.ccc >= ccc_threshold),
        "auroc_met": bool(metrics.auroc >= auroc_threshold),
        "both_met": bool(metrics.ccc >= ccc_threshold and metrics.auroc >= auroc_threshold),
        "ccc_interval_lower_above_threshold": bool(metrics.ccc_lower >= ccc_threshold),
        "auroc_interval_lower_above_threshold": bool(metrics.auroc_lower >= auroc_threshold),
    }


def repeat_summary(per_repeat: list[dict[str, float]]) -> dict[str, float]:
    """Mean and spread across repeats of each metric.

    Ref: Sec. 3.8, the reported uncertainty comes after averaging across the five repeats.
    """
    if not per_repeat:
        return {}
    keys = sorted({key for row in per_repeat for key in row})
    summary: dict[str, float] = {}
    for key in keys:
        values = np.array([row.get(key, float("nan")) for row in per_repeat], dtype=np.float64)
        summary[f"{key}_mean"] = mean_over_repeats(values)
        summary[f"{key}_spread"] = float(np.nanstd(values))
    return summary


def intervals_overlap(first: ExternalMetrics, second: ExternalMetrics) -> dict[str, float]:
    """Overlap of the concordance intervals of two comparators."""
    return {
        "concordance_overlap": overlap_fraction(
            (first.ccc_lower, first.ccc_upper), (second.ccc_lower, second.ccc_upper)
        ),
        "discrimination_overlap": overlap_fraction(
            (first.auroc_lower, first.auroc_upper), (second.auroc_lower, second.auroc_upper)
        ),
        "threshold_inside_ccc_interval": float(
            interval_contains(first.ccc_lower, first.ccc_upper, second.ccc)
        ),
    }


def external_slice(cohort: CohortFixture, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Reference, patient and label columns of an external evaluation slice."""
    reference = np.asarray(cohort.exposure_reference, dtype=np.float64)[indices]
    return {
        "reference": reference,
        "patients": np.asarray(cohort.patient_id, dtype=object)[indices],
        "labels": (reference >= 1.3).astype(np.int64),
        "molecule_index": np.asarray(cohort.molecule_index, dtype=np.int64)[indices],
    }
