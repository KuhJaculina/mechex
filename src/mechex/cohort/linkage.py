"""Deterministic linkage across the source tables.

Ref: Sec. 3.3, linkage is deterministic on a site-level de-identified study key joining
the CTA, laboratory, concentration and outcome tables so that every derived variable can
be traced to its source table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mechex.cohort.records import CohortFixture

TABLE_NAMES = ("imaging", "laboratory", "concentration", "outcome")


@dataclass
class LinkageReport:
    """Outcome of a linkage pass, including the traceability checks."""

    n_records: int
    key_unique: bool
    keys_without_imaging: int
    keys_without_concentration: int
    keys_without_outcome: int
    duplicate_keys: int
    orphan_references: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_records": self.n_records,
            "key_unique": self.key_unique,
            "keys_without_imaging": self.keys_without_imaging,
            "keys_without_concentration": self.keys_without_concentration,
            "keys_without_outcome": self.keys_without_outcome,
            "duplicate_keys": self.duplicate_keys,
            "orphan_references": list(self.orphan_references),
        }

    @property
    def is_consistent(self) -> bool:
        return (
            self.key_unique
            and self.keys_without_imaging == 0
            and self.keys_without_concentration == 0
            and self.keys_without_outcome == 0
            and self.duplicate_keys == 0
            and not self.orphan_references
        )


def study_key(site: str, patient_id: str, episode: int) -> str:
    """Compose the deterministic de-identified study key."""
    return f"{site}:{patient_id}:{int(episode):04d}"


def compose_keys(cohort: CohortFixture) -> np.ndarray:
    """Study key for every record."""
    keys = [
        study_key(str(site), str(patient), index)
        for index, (site, patient) in enumerate(zip(cohort.site, cohort.patient_id, strict=True))
    ]
    return np.asarray(keys, dtype=object)


def imaging_table(cohort: CohortFixture) -> dict[str, np.ndarray]:
    """Imaging-layer columns keyed by study key."""
    return {
        "key": compose_keys(cohort),
        "arterial_present": np.ones(len(cohort), dtype=bool),
        "delayed_present": np.ones(len(cohort), dtype=bool),
        "occlusion": cohort.occlusion,
        "volume_seed": cohort.volume_seed,
        "scanner_vendor": cohort.scanner_vendor,
    }


def laboratory_table(cohort: CohortFixture) -> dict[str, np.ndarray]:
    """Laboratory and clinical columns keyed by study key."""
    return {
        "key": compose_keys(cohort),
        "sex": cohort.sex,
        "age": cohort.age,
        "diabetes": cohort.diabetes,
        "ckd_stage": cohort.ckd_stage,
        "severity": cohort.severity,
    }


def concentration_table(cohort: CohortFixture) -> dict[str, np.ndarray]:
    """Concentration-time columns keyed by study key."""
    return {
        "key": compose_keys(cohort),
        "times": cohort.times,
        "concentrations": cohort.concentrations,
        "reference_tier": cohort.reference_tier,
    }


def outcome_table(cohort: CohortFixture) -> dict[str, np.ndarray]:
    """Outcome columns keyed by study key."""
    return {
        "key": compose_keys(cohort),
        "availability_reference": cohort.availability_reference,
        "exposure_reference": cohort.exposure_reference,
    }


def link(cohort: CohortFixture) -> tuple[dict[str, np.ndarray], LinkageReport]:
    """Join the four tables on the study key and report traceability.

    The join is positional after a key sort rather than a hash join, so a duplicated key
    shows up as a length mismatch instead of silently overwriting a row.
    """
    imaging = imaging_table(cohort)
    laboratory = laboratory_table(cohort)
    concentration = concentration_table(cohort)
    outcome = outcome_table(cohort)
    keys = imaging["key"]
    unique, counts = np.unique(keys, return_counts=True)
    duplicates = int(np.sum(counts > 1))
    order = np.argsort(keys, kind="mergesort")
    merged: dict[str, np.ndarray] = {}
    for name, table in (
        ("imaging", imaging),
        ("laboratory", laboratory),
        ("concentration", concentration),
        ("outcome", outcome),
    ):
        if not np.array_equal(table["key"][order], keys[order]):
            raise ValueError(f"the {name} table is not aligned on the study key")
        for column, values in table.items():
            if column == "key":
                continue
            merged[f"{name}.{column}"] = np.asarray(values)[order]
    merged["key"] = keys[order]
    merged["site"] = np.asarray(cohort.site)[order]
    merged["patient_id"] = np.asarray(cohort.patient_id)[order]
    merged["molecule_index"] = np.asarray(cohort.molecule_index)[order]
    merged["drug_class"] = np.asarray(cohort.drug_class)[order]
    merged["accrual_era"] = np.asarray(cohort.accrual_era)[order]

    orphans: list[str] = []
    for required in (
        "imaging.occlusion",
        "concentration.concentrations",
        "outcome.exposure_reference",
    ):
        if required not in merged:
            orphans.append(required)
    report = LinkageReport(
        n_records=int(keys.size),
        key_unique=bool(unique.size == keys.size),
        keys_without_imaging=int(
            np.sum(~imaging["arterial_present"] | ~imaging["delayed_present"])
        ),
        keys_without_concentration=int(
            np.sum(~np.isfinite(concentration["concentrations"]).all(axis=1))
        ),
        keys_without_outcome=int(np.sum(~np.isfinite(outcome["exposure_reference"]))),
        duplicate_keys=duplicates,
        orphan_references=orphans,
    )
    return merged, report


def excluded_records(
    cohort: CohortFixture, has_measurable_bed: np.ndarray | None = None
) -> dict[str, int]:
    """Counting form of the reported exclusion rules.

    Ref: Sec. 3.3, records leave the cohort when a prior above-knee amputation leaves no
    measurable delivery bed, when the CTA is non-diagnostic, or when the concentration
    record cannot be time-aligned to the imaging episode.
    """
    measurable = (
        np.ones(len(cohort), dtype=bool)
        if has_measurable_bed is None
        else np.asarray(has_measurable_bed, dtype=bool)
    )
    non_diagnostic = np.asarray(cohort.occlusion, dtype=np.float64) > 0.95
    unalignable = ~np.isfinite(np.asarray(cohort.concentrations, dtype=np.float64)).all(axis=1)
    return {
        "no_measurable_bed": int(np.sum(~measurable)),
        "non_diagnostic_cta": int(np.sum(non_diagnostic)),
        "unalignable_concentration": int(np.sum(unalignable)),
        "retained": int(np.sum(measurable & ~non_diagnostic & ~unalignable)),
    }


def merge_columns(candidate: np.ndarray, required: list[str]) -> bool:
    """Whether a merged table carries every required column."""
    return all(name in candidate for name in required)
