"""Stratified evaluation across severity, drug class and demographic axes.

Ref: Sec. 2.4 and Table 3, each stratum reported with its size and the concordance
decrement relative to the overall external-site value, with a pre-specified decision
threshold of 0.75 and a 5-percentage-point consistency limit. The multiple comparisons are
controlled with the Benjamini-Hochberg procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.cohort.records import CohortFixture
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.multiplicity import benjamini_hochberg
from mechex.quantify.roc import auroc
from mechex.quantify.shift import degradation_points

STRATUM_AXES: tuple[tuple[str, str], ...] = (
    ("severity", "severity"),
    ("drug_class", "drug_class"),
    ("sex", "sex"),
    ("age_band", "age_band"),
    ("diabetes", "diabetes"),
    ("ckd_stage", "ckd_stage"),
)

PRESPECIFIED_THRESHOLD = 0.75
CONSISTENCY_LIMIT_POINTS = 5.0


@dataclass(frozen=True)
class StratumRow:
    """One row of the stratified table."""

    axis: str
    level: str
    n: int
    ccc: float
    auroc: float
    delta_ccc: float
    below_threshold: bool
    inside_consistency_limit: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "level": self.level,
            "n": self.n,
            "ccc": self.ccc,
            "auroc": self.auroc,
            "delta_ccc": self.delta_ccc,
            "below_threshold": self.below_threshold,
            "inside_consistency_limit": self.inside_consistency_limit,
        }


def derive_axes(cohort: CohortFixture, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Categorical axes derived from the stored columns."""
    age = np.asarray(cohort.age, dtype=np.int64)[indices]
    ckd = np.asarray(cohort.ckd_stage, dtype=np.int64)[indices]
    diabetes = np.asarray(cohort.diabetes, dtype=bool)[indices]
    sex = np.asarray(cohort.sex, dtype=object)[indices]
    severity = np.asarray(cohort.severity, dtype=object)[indices]
    drug_class = np.asarray(cohort.drug_class, dtype=object)[indices]
    return {
        "severity": severity,
        "drug_class": drug_class,
        "sex": sex,
        "age_band": np.where(age >= 75, "age_75_plus", "age_under_75").astype(object),
        "diabetes": np.where(diabetes, "diabetes", "no_diabetes").astype(object),
        "ckd_stage": np.where(ckd >= 4, "severe_ckd", "ckd_stage_under_4").astype(object),
    }


def _stratum_p_value(values: np.ndarray, labels: np.ndarray) -> float:
    """Permutation p-value for a stratum's discrimination above chance."""
    from mechex.quantify.roc import auroc as _auroc

    if len(np.unique(labels)) < 2:
        return 1.0
    observed = _auroc(values, labels)
    if not np.isfinite(observed):
        return 1.0
    generator = np.random.default_rng(20260927)
    exceed = 0
    for _ in range(200):
        shuffled = generator.permutation(labels)
        if _auroc(values, shuffled) >= observed:
            exceed += 1
    return float((exceed + 1) / 201)


def evaluate_strata(
    prediction: np.ndarray,
    reference: np.ndarray,
    axes: dict[str, np.ndarray],
    overall_ccc: float,
    tau: float = 0.30,
    fdr: float = 0.05,
) -> list[StratumRow]:
    """Concordance and discrimination per stratum, with the decisions attached."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    labels = (truth >= 1.0 + tau).astype(np.int64)
    rows: list[StratumRow] = []
    for axis_name, _column in STRATUM_AXES:
        column = axes[axis_name]
        levels = [level for level in np.unique(column) if level is not None]
        for level in levels:
            selected = column == level
            if selected.sum() < 5:
                continue
            stratum_labels = labels[selected]
            stratum_auroc = (
                auroc(predicted[selected], stratum_labels)
                if len(np.unique(stratum_labels)) > 1
                else float("nan")
            )
            ccc = lin_ccc(predicted[selected], truth[selected])
            delta = degradation_points(overall_ccc, ccc) / 100.0
            rows.append(
                StratumRow(
                    axis=axis_name,
                    level=str(level),
                    n=int(selected.sum()),
                    ccc=float(ccc),
                    auroc=float(stratum_auroc),
                    delta_ccc=float(delta),
                    below_threshold=bool(ccc < PRESPECIFIED_THRESHOLD),
                    inside_consistency_limit=bool(
                        abs(degradation_points(overall_ccc, ccc)) <= CONSISTENCY_LIMIT_POINTS
                    ),
                )
            )
    _ = fdr
    return rows


def multiplicity_adjustment(
    prediction: np.ndarray,
    reference: np.ndarray,
    axes: dict[str, np.ndarray],
    tau: float = 0.30,
    fdr: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Benjamini-Hochberg adjustment over the stratified comparisons."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    labels = (truth >= 1.0 + tau).astype(np.int64)
    p_values: list[float] = []
    labels_used: list[str] = []
    for axis_name, _column in STRATUM_AXES:
        column = axes[axis_name]
        for level in [level for level in np.unique(column) if level is not None]:
            selected = column == level
            if selected.sum() < 5:
                continue
            p_values.append(_stratum_p_value(predicted[selected], labels[selected]))
            labels_used.append(f"{axis_name}={level}")
    adjusted = benjamini_hochberg(np.asarray(p_values, dtype=np.float64), fdr)
    return {
        "p_values": np.asarray(p_values, dtype=np.float64),
        "adjusted": np.asarray(adjusted["adjusted"], dtype=np.float64),
        "rejected": np.asarray(adjusted["rejected"], dtype=bool),
        "family_size": float(len(p_values)),
        "alpha": float(fdr),
        "labels": np.asarray(labels_used, dtype=object),
    }


def consistency_summary(rows: list[StratumRow]) -> dict[str, float | str]:
    """Worst stratum by concordance and whether the limits held."""
    if not rows:
        return {"n_strata": 0.0, "worst_level": "", "worst_ccc": float("nan")}
    worst = min(rows, key=lambda row: row.ccc)
    return {
        "n_strata": float(len(rows)),
        "worst_level": f"{worst.axis}={worst.level}",
        "worst_ccc": float(worst.ccc),
        "worst_delta_points": float(worst.delta_ccc * 100.0),
        "all_above_threshold": bool(all(not row.below_threshold for row in rows)),
        "all_inside_consistency_limit": bool(all(row.inside_consistency_limit for row in rows)),
    }


def stratum_sizes(cohort: CohortFixture, indices: np.ndarray) -> dict[str, dict[str, int]]:
    """Record count per level of every axis, for the size column of the table."""
    axes = derive_axes(cohort, indices)
    sizes: dict[str, dict[str, int]] = {}
    for axis_name in axes:
        levels, counts = np.unique(axes[axis_name], return_counts=True)
        sizes[axis_name] = {
            str(level): int(count) for level, count in zip(levels, counts, strict=True)
        }
    return sizes
