"""Distribution-shift quantification.

Ref: Sec. 2.5 and Table 4, degradation as the external-minus-development difference with
a pre-specified banding of graceful (at most 5 points), moderate (5 to 10) and severe
(10 to 20).
"""

from __future__ import annotations

from dataclasses import dataclass

from mechex.utils.coerce import as_float


@dataclass(frozen=True)
class Banding:
    """Pre-specified degradation bands in percentage points."""

    graceful_upper: float = 5.0
    moderate_upper: float = 10.0
    severe_upper: float = 20.0

    def classify(self, magnitude_pp: float) -> str:
        """Band a magnitude in percentage points."""
        value = abs(float(magnitude_pp))
        if value <= self.graceful_upper:
            return "graceful"
        if value <= self.moderate_upper:
            return "moderate"
        if value <= self.severe_upper:
            return "severe"
        return "beyond_severe"


DEFAULT_BANDING = Banding()


def degradation(reference: float, shifted: float) -> float:
    """Shifted minus reference, in the metric's own units."""
    return float(shifted) - float(reference)


def degradation_points(reference: float, shifted: float) -> float:
    """Shifted-minus-reference difference in percentage points."""
    return float(100.0 * (float(shifted) - float(reference)))


def shift_row(
    name: str,
    reference_ccc: float,
    reference_auroc: float,
    shifted_ccc: float,
    shifted_auroc: float,
    banding: Banding | None = None,
) -> dict[str, object]:
    """One row of the shift table with the banding applied to the AUROC loss."""
    bands = banding or DEFAULT_BANDING
    ddd = degradation_points(reference_ccc, shifted_ccc)
    d_auroc = degradation_points(reference_auroc, shifted_auroc)
    return {
        "shift": name,
        "ccc": float(shifted_ccc),
        "ccc_points": float(ddd),
        "auroc": float(shifted_auroc),
        "auroc_points": float(d_auroc),
        "classification": bands.classify(d_auroc),
    }


def within_consistency_limit(reference: float, value: float, limit_points: float = 5.0) -> bool:
    """Whether a stratum stays inside the pre-specified consistency limit."""
    return bool(abs(degradation_points(reference, value)) <= limit_points)


def worst_case(rows: list[dict[str, object]], key: str = "auroc_points") -> dict[str, object]:
    """The row with the largest absolute degradation on ``key``."""
    if not rows:
        raise ValueError("no rows to compare")
    return max(rows, key=lambda row: abs(as_float(row[key], key)))
