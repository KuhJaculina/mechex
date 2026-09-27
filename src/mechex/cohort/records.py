"""Record and cohort containers for the tabular layer.

Ref: Sec. 3.2, the two data layers; the primary layer is the de-identified multi-site
clinical cohort and the auxiliary layer supplies only background development material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mechex.utils.io import ensure_dir

TIER_DIRECT_ASSAY = 1
TIER_MICRODIALYSIS = 2
TIER_COMPOSITE = 3

SITE_REGION = {"A": "Region I", "B": "Region I", "C": "Region II"}

SEVERITY_LEVELS = ("claudication", "rest_pain", "chronic_limb_threatening_ischaemia")
DRUG_CLASSES = ("lipid_lowering", "antiplatelet", "prostaglandin_analogue", "other")
SCANNER_VENDORS = ("vendor_1", "vendor_2", "vendor_3")
ACCRUAL_ERAS = ("2019-2021", "2024-2025")


@dataclass
class CohortFixture:
    """Column-oriented fixture of the primary cohort.

    Volumes are not stored: each record carries a generation seed so a CTA pair can be
    constructed on demand, which keeps the fixture small enough to ship and still lets the
    volumetric encoder see a real tensor.
    """

    record_id: np.ndarray
    patient_id: np.ndarray
    site: np.ndarray
    molecule_index: np.ndarray
    descriptors: np.ndarray
    times: np.ndarray
    concentrations: np.ndarray
    occlusion: np.ndarray
    availability_reference: np.ndarray
    exposure_reference: np.ndarray
    severity: np.ndarray
    drug_class: np.ndarray
    scanner_vendor: np.ndarray
    accrual_era: np.ndarray
    sex: np.ndarray
    age: np.ndarray
    diabetes: np.ndarray
    ckd_stage: np.ndarray
    reference_tier: np.ndarray
    volume_seed: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.record_id.shape[0])

    def subset(self, mask: np.ndarray) -> CohortFixture:
        """Take a boolean subset of the cohort, preserving every column."""
        selected = np.asarray(mask, dtype=bool)
        return CohortFixture(
            record_id=self.record_id[selected],
            patient_id=self.patient_id[selected],
            site=self.site[selected],
            molecule_index=self.molecule_index[selected],
            descriptors=self.descriptors[selected],
            times=self.times[selected],
            concentrations=self.concentrations[selected],
            occlusion=self.occlusion[selected],
            availability_reference=self.availability_reference[selected],
            exposure_reference=self.exposure_reference[selected],
            severity=self.severity[selected],
            drug_class=self.drug_class[selected],
            scanner_vendor=self.scanner_vendor[selected],
            accrual_era=self.accrual_era[selected],
            sex=self.sex[selected],
            age=self.age[selected],
            diabetes=self.diabetes[selected],
            ckd_stage=self.ckd_stage[selected],
            reference_tier=self.reference_tier[selected],
            volume_seed=self.volume_seed[selected],
            metadata=dict(self.metadata),
        )

    def save(self, path: Path) -> Path:
        """Write the fixture to a compressed npz container."""
        ensure_dir(path.parent)
        np.savez_compressed(
            path,
            record_id=self.record_id,
            patient_id=self.patient_id,
            site=self.site,
            molecule_index=self.molecule_index,
            descriptors=self.descriptors,
            times=self.times,
            concentrations=self.concentrations,
            occlusion=self.occlusion,
            availability_reference=self.availability_reference,
            exposure_reference=self.exposure_reference,
            severity=self.severity,
            drug_class=self.drug_class,
            scanner_vendor=self.scanner_vendor,
            accrual_era=self.accrual_era,
            sex=self.sex,
            age=self.age,
            diabetes=self.diabetes,
            ckd_stage=self.ckd_stage,
            reference_tier=self.reference_tier,
            volume_seed=self.volume_seed,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> CohortFixture:
        """Read a fixture written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                record_id=payload["record_id"],
                patient_id=payload["patient_id"],
                site=payload["site"],
                molecule_index=payload["molecule_index"],
                descriptors=payload["descriptors"],
                times=payload["times"],
                concentrations=payload["concentrations"],
                occlusion=payload["occlusion"],
                availability_reference=payload["availability_reference"],
                exposure_reference=payload["exposure_reference"],
                severity=payload["severity"],
                drug_class=payload["drug_class"],
                scanner_vendor=payload["scanner_vendor"],
                accrual_era=payload["accrual_era"],
                sex=payload["sex"],
                age=payload["age"],
                diabetes=payload["diabetes"],
                ckd_stage=payload["ckd_stage"],
                reference_tier=payload["reference_tier"],
                volume_seed=payload["volume_seed"],
            )

    def group_counts(self, column: str) -> dict[str, int]:
        """Count records by the level of one categorical column."""
        values = getattr(self, column)
        levels, counts = np.unique(values, return_counts=True)
        return {str(level): int(count) for level, count in zip(levels, counts, strict=True)}
