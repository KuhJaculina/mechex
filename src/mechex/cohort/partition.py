"""Patient-level splitting with leakage control.

Ref: Sec. 3.4, a 70/15/15 development, validation and test partition of Sites A and B with
a tenth of the development group withheld as a calibration set, and Site C held out as an
independent external site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mechex.cohort.records import CohortFixture
from mechex.settings import ProtocolConfig


@dataclass
class SplitIndices:
    """Positional indices of each partition."""

    development: np.ndarray
    calibration: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    external: np.ndarray

    def as_dict(self) -> dict[str, int]:
        return {
            "development": int(self.development.size),
            "calibration": int(self.calibration.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
            "external": int(self.external.size),
        }

    def mask(self, name: str, n_records: int) -> np.ndarray:
        """Boolean mask for one partition name."""
        selected = getattr(self, name)
        mask = np.zeros(n_records, dtype=bool)
        mask[selected] = True
        return mask

    def names(self) -> tuple[str, ...]:
        return ("development", "calibration", "validation", "test", "external")


@dataclass
class LeakageReport:
    """Overlap checks across partitions."""

    patient_overlap: int = 0
    record_overlap: int = 0
    external_patient_overlap: int = 0
    molecule_overlap_test_development: int = 0
    molecule_overlap_external_development: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            self.patient_overlap == 0
            and self.record_overlap == 0
            and self.external_patient_overlap == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "patient_overlap": self.patient_overlap,
            "record_overlap": self.record_overlap,
            "external_patient_overlap": self.external_patient_overlap,
            "molecule_overlap_test_development": self.molecule_overlap_test_development,
            "molecule_overlap_external_development": self.molecule_overlap_external_development,
            "clean": bool(self.clean),
            "details": list(self.details),
        }


def _split_patients(
    patients: np.ndarray,
    fractions: tuple[float, float, float],
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition unique patients into three groups."""
    unique = np.unique(patients)
    shuffled = generator.permutation(unique)
    n_total = shuffled.size
    n_development = round(fractions[0] * n_total)
    n_validation = round(fractions[1] * n_total)
    development = shuffled[:n_development]
    validation = shuffled[n_development : n_development + n_validation]
    test = shuffled[n_development + n_validation :]
    return development, validation, test


def partition(
    cohort: CohortFixture,
    protocol: ProtocolConfig | None = None,
    seed: int = 20260927,
) -> SplitIndices:
    """Build the five partitions, splitting on patients rather than records."""
    settings = protocol or ProtocolConfig()
    generator = np.random.default_rng(seed)
    site = np.asarray(cohort.site, dtype=object)
    patients = np.asarray(cohort.patient_id, dtype=object)
    development_sites = np.isin(site, np.asarray(settings.sites_development, dtype=object))
    external = np.flatnonzero(site == settings.site_external)
    development_pool = np.flatnonzero(development_sites)

    fractions = (
        settings.development_fraction,
        settings.validation_fraction,
        settings.test_fraction,
    )
    pool_patients = patients[development_pool]
    development_patients, validation_patients, test_patients = _split_patients(
        pool_patients, fractions, generator
    )
    development_mask = np.isin(patients, development_patients)
    validation_mask = np.isin(patients, validation_patients)
    test_mask = np.isin(patients, test_patients)
    development = development_pool[development_mask[development_pool]]
    validation = development_pool[validation_mask[development_pool]]
    test = development_pool[test_mask[development_pool]]

    development_unique = patients[development]
    n_calibration = round(settings.calibration_fraction * np.unique(development_unique).size)
    shuffled_development = generator.permutation(np.unique(development_unique))[:n_calibration]
    calibration_patients = set(shuffled_development.tolist())
    in_calibration = np.array([patient in calibration_patients for patient in development_unique])
    calibration = development[in_calibration]
    development = development[~in_calibration]

    return SplitIndices(
        development=np.sort(development),
        calibration=np.sort(calibration),
        validation=np.sort(validation),
        test=np.sort(test),
        external=np.sort(external),
    )


def check_leakage(cohort: CohortFixture, split: SplitIndices) -> LeakageReport:
    """Verify that no patient or record crosses a partition boundary."""
    patients = np.asarray(cohort.patient_id, dtype=object)
    molecules = np.asarray(cohort.molecule_index, dtype=np.int64)
    report = LeakageReport()

    development_patients = set(patients[split.development].tolist())
    for name in ("calibration", "validation", "test"):
        other = set(patients[getattr(split, name)].tolist())
        overlap = development_patients & other
        report.patient_overlap += len(overlap)
        if overlap:
            report.details.append(f"{name} shares {len(overlap)} patients with development")
    external_patients = set(patients[split.external].tolist())
    report.external_patient_overlap = len(external_patients & development_patients)
    if report.external_patient_overlap:
        report.details.append("the external site shares patients with development")

    record_sets = [set(getattr(split, name).tolist()) for name in split.names()]
    for index in range(len(record_sets)):
        for other_index in range(index + 1, len(record_sets)):
            report.record_overlap += len(record_sets[index] & record_sets[other_index])

    development_molecules = set(molecules[split.development].tolist())
    report.molecule_overlap_test_development = len(
        development_molecules & set(molecules[split.test].tolist())
    )
    report.molecule_overlap_external_development = len(
        development_molecules & set(molecules[split.external].tolist())
    )
    if report.molecule_overlap_external_development:
        report.details.append(
            "molecule identities recur across the external site by design; this is a shared "
            "panel, not a shared record, and is reported rather than treated as leakage"
        )
    return report


def partition_summary(cohort: CohortFixture, split: SplitIndices) -> dict[str, object]:
    """Partition sizes with the site composition of each."""
    summary: dict[str, object] = {"sizes": split.as_dict(), "sites": {}}
    site = np.asarray(cohort.site, dtype=object)
    for name in split.names():
        selected = getattr(split, name)
        levels, counts = np.unique(site[selected], return_counts=True)
        summary["sites"][name] = {  # type: ignore[index]
            str(level): int(count) for level, count in zip(levels, counts, strict=True)
        }
    return summary


def temporal_partition(
    cohort: CohortFixture,
    development_era: str = "2019-2021",
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the temporal shift.

    Ref: Table 4, development on 2019-2021 accruals and evaluation on 2024-2025 accruals.
    """
    era = np.asarray(cohort.accrual_era, dtype=object)
    return era == development_era, era != development_era


def vendor_partition(
    cohort: CohortFixture, development_vendor: str = "vendor_1"
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the cross-vendor shift."""
    vendor = np.asarray(cohort.scanner_vendor, dtype=object)
    return vendor == development_vendor, vendor != development_vendor
