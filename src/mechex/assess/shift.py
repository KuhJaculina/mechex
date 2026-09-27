"""Generalisation across region, scanner and time.

Ref: Sec. 2.5 and Table 4, four independent distribution shifts with degradation reported
as the external-minus-development difference and a pre-specified banding of graceful (at
most 5 points), moderate (5 to 10) and severe (10 to 20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.cohort.records import CohortFixture
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.roc import auroc
from mechex.quantify.shift import DEFAULT_BANDING, Banding, shift_row
from mechex.utils.coerce import as_float, as_text

SHIFT_NAMES = (
    "development",
    "cross_region",
    "cross_vendor",
    "temporal",
    "cross_vendor_temporal",
)


@dataclass(frozen=True)
class ShiftEvaluation:
    """Metrics of one shift condition."""

    name: str
    n: int
    ccc: float
    auroc: float

    def as_dict(self) -> dict[str, object]:
        return {"shift": self.name, "n": self.n, "ccc": self.ccc, "auroc": self.auroc}


def shift_masks(cohort: CohortFixture, split_external: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks of the four held-out populations.

    Each mask describes an independent held-out population: the cross-region condition is
    the external site, the vendor condition is every record outside the development vendor,
    the temporal condition is the later accrual window, and the combined condition is their
    intersection.
    """
    vendor = np.asarray(cohort.scanner_vendor, dtype=object)
    era = np.asarray(cohort.accrual_era, dtype=object)
    site = np.asarray(cohort.site, dtype=object)
    development = site != "C"
    cross_vendor = vendor != "vendor_1"
    temporal = era == "2024-2025"
    return {
        "development": development,
        "cross_region": np.asarray(split_external, dtype=bool),
        "cross_vendor": cross_vendor,
        "temporal": temporal,
        "cross_vendor_temporal": cross_vendor & temporal,
    }


def evaluate_shift(
    name: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    tau: float = 0.30,
) -> ShiftEvaluation:
    """Concordance and discrimination of one held-out population."""
    selected = np.asarray(mask, dtype=bool)
    predicted = np.asarray(prediction, dtype=np.float64)[selected]
    truth = np.asarray(reference, dtype=np.float64)[selected]
    if predicted.size == 0:
        return ShiftEvaluation(name=name, n=0, ccc=float("nan"), auroc=float("nan"))
    labels = (truth >= 1.0 + tau).astype(np.int64)
    discrimination = auroc(predicted, labels) if len(np.unique(labels)) > 1 else float("nan")
    return ShiftEvaluation(
        name=name,
        n=int(predicted.size),
        ccc=float(lin_ccc(predicted, truth)),
        auroc=float(discrimination),
    )


def generalisation_table(
    prediction: np.ndarray,
    reference: np.ndarray,
    masks: dict[str, np.ndarray],
    tau: float = 0.30,
    banding: Banding | None = None,
) -> list[dict[str, object]]:
    """The shift table with the banding applied to the AUROC loss."""
    bands = banding or DEFAULT_BANDING
    evaluations = {
        name: evaluate_shift(name, prediction, reference, masks[name], tau) for name in SHIFT_NAMES
    }
    reference_row = evaluations["development"]
    rows: list[dict[str, object]] = [
        {
            "shift": "Development (Site A + Site B)",
            "n": reference_row.n,
            "ccc": reference_row.ccc,
            "ccc_points": None,
            "auroc": reference_row.auroc,
            "auroc_points": None,
            "classification": "reference",
        }
    ]
    labels = {
        "cross_region": "Cross-region (Site C)",
        "cross_vendor": "Cross-vendor (different scanner generation)",
        "temporal": "Temporal (2019-2021 to 2024-2025)",
        "cross_vendor_temporal": "Cross-vendor + temporal",
    }
    for name in ("cross_region", "cross_vendor", "temporal", "cross_vendor_temporal"):
        row = shift_row(
            labels[name],
            reference_row.ccc,
            reference_row.auroc,
            evaluations[name].ccc,
            evaluations[name].auroc,
            bands,
        )
        row["n"] = evaluations[name].n
        rows.append(row)
    return rows


def inconsistency_across_shifts(
    residual: np.ndarray,
    implied: np.ndarray,
    fractions: np.ndarray,
    exposures: np.ndarray,
    masks: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, float]:
    """Inconsistency rate per shift condition.

    Ref: Sec. 2.5, the inconsistency rate stayed below 3.5% under every shift.
    """
    from mechex.conservation.inconsistency import inconsistency_flags, inconsistency_rate

    rates: dict[str, float] = {}
    for name, mask in masks.items():
        selected = np.asarray(mask, dtype=bool)
        flags = inconsistency_flags(
            np.asarray(residual)[selected],
            np.asarray(implied)[selected],
            np.asarray(fractions)[selected],
            np.asarray(exposures)[selected],
            tolerance,
        )
        rates[name] = inconsistency_rate(flags)
    return rates


def dominant_axis(rows: list[dict[str, object]]) -> dict[str, object]:
    """The shift with the largest AUROC loss, which the manuscript reports as temporal.

    Ref: Sec. 2.5 and Sec. 4, the temporal loss is the larger of the two single-axis
    losses, so re-validation on the time axis matters at least as much as on a new site.
    """
    scored = [row for row in rows if row["auroc_points"] is not None]
    if not scored:
        return {"shift": "", "auroc_points": 0.0}
    worst = min(scored, key=lambda row: as_float(row["auroc_points"], "auroc_points"))
    return {
        "shift": as_text(worst["shift"], "shift"),
        "auroc_points": as_float(worst["auroc_points"], "auroc_points"),
    }


def single_axis_ordering(rows: list[dict[str, object]]) -> dict[str, float]:
    """Absolute single-axis losses, used to compare the time and site penalties."""
    lookup = {as_text(row["shift"], "shift"): row for row in rows}
    site = lookup.get("Cross-region (Site C)", {})
    temporal = lookup.get("Temporal (2019-2021 to 2024-2025)", {})
    vendor = lookup.get("Cross-vendor (different scanner generation)", {})
    return {
        "region_points": abs(as_float(site.get("auroc_points", 0.0), "auroc_points")),
        "vendor_points": abs(as_float(vendor.get("auroc_points", 0.0), "auroc_points")),
        "temporal_points": abs(as_float(temporal.get("auroc_points", 0.0), "auroc_points")),
    }


def temporal_exceeds_region(rows: list[dict[str, object]]) -> bool:
    """Whether the temporal penalty exceeds the cross-region penalty."""
    ordering = single_axis_ordering(rows)
    return bool(ordering["temporal_points"] > ordering["region_points"])
