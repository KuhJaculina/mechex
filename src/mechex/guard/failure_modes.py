"""Five-category failure taxonomy of the worst decile.

Ref: Sec. 3.10, failure modes were enumerated by inspecting the worst decile of absolute
prediction error on the external site and grouping each case by mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAILURE_TYPES = {
    1: "dense_calcification_confounds_perfused_fraction",
    2: "delayed_phase_timing_heterogeneity_across_vendors",
    3: "very_low_systemic_availability_ratio_instability",
    4: "concurrent_vasoactive_medication_unmodelled",
    5: "severe_renal_impairment_alters_kinetics",
}

HARM_DIRECTION = {
    1: "false_adequate",
    2: "false_inadequate",
    3: "false_adequate",
    4: "false_inadequate",
    5: "false_inadequate",
}

MITIGATION = {
    1: "abstain when the calcification burden saturates the perfused-fraction estimate",
    2: "abstain when the delayed-phase acquisition falls outside the vendor window",
    3: "abstain when systemic availability is below the numerical floor",
    4: "flag concurrent vasoactive therapy at data entry rather than at inference",
    5: "abstain when the renal-function stage exceeds the modelled range",
}


@dataclass(frozen=True)
class FailureAssignment:
    """Per-record failure type, aggregated into the taxonomy table."""

    types: np.ndarray
    worst_decile_mask: np.ndarray

    def table(self) -> list[dict[str, object]]:
        """Frequency and harm direction per failure type."""
        rows: list[dict[str, object]] = []
        total = max(int(self.worst_decile_mask.sum()), 1)
        for failure_type, name in FAILURE_TYPES.items():
            selected = self.types == failure_type
            count = int(selected.sum())
            rows.append(
                {
                    "type": failure_type,
                    "mechanism": name,
                    "frequency": count,
                    "frequency_of_worst_decile": count / total,
                    "harm_direction": HARM_DIRECTION[failure_type],
                    "mitigation": MITIGATION[failure_type],
                    "unclassified": False,
                }
            )
        unclassified = int((self.types == 0).sum())
        if unclassified:
            rows.append(
                {
                    "type": 0,
                    "mechanism": "outside the taxonomy",
                    "frequency": unclassified,
                    "frequency_of_worst_decile": unclassified / total,
                    "harm_direction": "mixed",
                    "mitigation": (
                        "retained as an unexplained residual rather than forced into a category"
                    ),
                    "unclassified": True,
                }
            )
        return rows


def worst_decile(error: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Boolean mask of the largest absolute errors."""
    magnitudes = np.abs(np.asarray(error, dtype=np.float64).ravel())
    if magnitudes.size == 0:
        return np.zeros(0, dtype=bool)
    cut = float(np.quantile(magnitudes, 1.0 - fraction))
    return magnitudes >= cut


def classify_failures(
    error: np.ndarray,
    occlusion: np.ndarray,
    availability: np.ndarray,
    scanner_vendor: np.ndarray,
    vasoactive: np.ndarray | None = None,
    ckd_stage: np.ndarray | None = None,
    fraction: float = 0.10,
    calcification: np.ndarray | None = None,
) -> FailureAssignment:
    """Assign each worst-decile record to a failure mechanism.

    The rules are read off the mechanisms the manuscript names: a saturated calcified bed
    a vendor-specific delayed phase, an availability below the numerical floor, an
    unmodelled vasoactive drug and severe renal impairment. Records outside all five are
    left unclassified rather than forced into a category.
    """
    magnitudes = np.abs(np.asarray(error, dtype=np.float64).ravel())
    mask = worst_decile(magnitudes, fraction)
    types = np.zeros(magnitudes.shape[0], dtype=np.int64)
    high_occlusion = np.asarray(occlusion, dtype=np.float64) >= 0.80
    low_availability = np.asarray(availability, dtype=np.float64) <= np.quantile(availability, 0.10)
    vendor = np.asarray(scanner_vendor, dtype=object)
    minority_vendor = vendor == "vendor_3"
    dense = np.asarray(calcification, dtype=bool) if calcification is not None else high_occlusion
    vaso = (
        np.asarray(vasoactive, dtype=bool)
        if vasoactive is not None
        else np.zeros(magnitudes.shape[0], dtype=bool)
    )
    renal = (
        np.asarray(ckd_stage, dtype=np.int64) >= 4
        if ckd_stage is not None
        else np.zeros(magnitudes.shape[0], dtype=bool)
    )

    remaining = mask.copy()
    for failure_type, condition in (
        (3, low_availability),
        (5, renal),
        (4, vaso),
        (2, minority_vendor),
        (1, dense),
    ):
        selected = remaining & condition
        types[selected] = failure_type
        remaining = remaining & ~selected
    return FailureAssignment(types=types, worst_decile_mask=mask)


def taxonomy_table(assignment: FailureAssignment) -> list[dict[str, object]]:
    """The taxonomy table."""
    return assignment.table()


def unresolved_share(assignment: FailureAssignment) -> float:
    """Share of the worst decile that no mechanism explains."""
    total = max(int(assignment.worst_decile_mask.sum()), 1)
    unclassified = int(np.sum(assignment.worst_decile_mask & (assignment.types == 0)))
    return float(unclassified / total)
