"""Tiered reference standard and threshold freezing.

Ref: Sec. 2.1 and Sec. 3.5, a three-tier reference standard with double adjudication at a
Cohen's kappa of 0.86, a calibration set withheld from the development split, and a
threshold fixed at the 30% minimum clinically important difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.cohort.records import (
    TIER_COMPOSITE,
    TIER_DIRECT_ASSAY,
    TIER_MICRODIALYSIS,
    CohortFixture,
)
from mechex.cohort.transforms import dichotomise
from mechex.quantify.agreement import cohen_kappa

TIER_NAMES = {
    TIER_DIRECT_ASSAY: "direct_assay",
    TIER_MICRODIALYSIS: "microdialysis",
    TIER_COMPOSITE: "composite_index",
}


@dataclass(frozen=True)
class ReferenceStandard:
    """The frozen adequacy boundary and its provenance."""

    tau: float
    boundary: float
    frozen_on_records: int
    tiers_used: tuple[int, ...]
    adjudication_kappa: float

    def as_dict(self) -> dict[str, object]:
        return {
            "tau": self.tau,
            "boundary": self.boundary,
            "frozen_on_records": self.frozen_on_records,
            "tiers_used": list(self.tiers_used),
            "adjudication_kappa": self.adjudication_kappa,
        }

    def label(self, exposure: np.ndarray) -> np.ndarray:
        """Dichotomise a relative exposure against the frozen boundary."""
        return dichotomise(np.asarray(exposure, dtype=np.float64), self.tau)


def tier_counts(cohort: CohortFixture) -> dict[str, int]:
    """Record count per reference tier."""
    values, counts = np.unique(
        np.asarray(cohort.reference_tier, dtype=np.int64), return_counts=True
    )
    return {
        TIER_NAMES.get(int(value), str(value)): int(count)
        for value, count in zip(values, counts, strict=True)
    }


def tier_subsets(cohort: CohortFixture) -> dict[str, CohortFixture]:
    """One cohort slice per tier, for the per-tier sensitivity analysis."""
    return {
        name: cohort.subset(np.asarray(cohort.reference_tier, dtype=np.int64) == tier)
        for tier, name in TIER_NAMES.items()
    }


def adjudicate(
    first_rating: np.ndarray,
    second_rating: np.ndarray,
    reported_kappa: float = 0.86,
) -> dict[str, float]:
    """Double adjudication of reference adequacy against the reported agreement.

    Returns both the observed kappa and the difference from the reported value, so a
    disagreement shows up as a reported quantity rather than as a silent pass.
    """
    result = cohen_kappa(first_rating, second_rating)
    return {
        "observed_kappa": float(result.kappa),
        "reported_kappa": float(reported_kappa),
        "absolute_difference": float(abs(result.kappa - reported_kappa)),
        "observed_agreement": float(result.observed),
        "n": int(result.n),
    }


def adjudication_fixture(
    cohort: CohortFixture,
    disagreement_rate: float = 0.07,
    seed: int = 20260927,
) -> tuple[np.ndarray, np.ndarray]:
    """Two rater columns consistent with the reported disagreement level.

    The second rater flips a fixed fraction of the first rater's calls, which is the
    mechanism that produces a kappa below one.
    """
    generator = np.random.default_rng(seed)
    first = np.asarray(cohort.exposure_reference, dtype=np.float64) >= np.median(
        cohort.exposure_reference
    )
    first = first.astype(np.int64)
    flip = generator.random(first.size) < disagreement_rate
    second = np.where(flip, 1 - first, first)
    return first, second


def freeze_threshold(
    development_exposure: np.ndarray,
    tau: float,
    adequacy_relative: float = 1.0,
) -> ReferenceStandard:
    """Freeze the adequacy boundary on the development split.

    Ref: Sec. 3.5. The boundary is returned with the record count it was fixed on so that
    a later evaluation can state that it was never re-tuned on the external site.
    """
    exposure = np.asarray(development_exposure, dtype=np.float64)
    return ReferenceStandard(
        tau=float(tau),
        boundary=float(adequacy_relative) * (1.0 + float(tau)),
        frozen_on_records=int(exposure.size),
        tiers_used=(TIER_DIRECT_ASSAY, TIER_MICRODIALYSIS, TIER_COMPOSITE),
        adjudication_kappa=0.86,
    )


def adequacy_rate(exposure: np.ndarray, standard: ReferenceStandard) -> float:
    """Fraction of records labelled adequate under the frozen standard."""
    labels = standard.label(exposure)
    if labels.size == 0:
        return float("nan")
    return float(labels.mean())


def tier_sensitivity(
    exposure: np.ndarray,
    labels: np.ndarray,
    reference_tier: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-tier concordance and discrimination of a fixed prediction.

    Ref: Sec. 3.5, the per-tier breakdown quantifies how much of the result rests on the
    directly assayed subgroup.
    """
    from mechex.quantify.concordance import lin_ccc
    from mechex.quantify.roc import auroc

    values = np.asarray(exposure, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    tiers = np.asarray(reference_tier, dtype=np.int64)
    report: dict[str, dict[str, float]] = {}
    for tier, name in TIER_NAMES.items():
        selected = tiers == tier
        if not selected.any():
            continue
        labels_selected = truth[selected]
        ccc = lin_ccc(values[selected], np.where(labels_selected == 1, 1.5, 0.5))
        discrimination = (
            auroc(values[selected], labels_selected)
            if len(np.unique(labels_selected)) > 1
            else float("nan")
        )
        report[name] = {
            "n": float(selected.sum()),
            "ccc": float(ccc),
            "auroc": float(discrimination),
        }
    return report
