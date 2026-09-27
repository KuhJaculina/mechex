"""Top-level cohort construction and staging.

Ref: Sec. 3.2-3.5, the two data layers, the linkage, the reference standard and the split
assemble into the cohort the rest of the release reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mechex.cohort.cohort import CohortCollator, CohortTorchDataset
from mechex.cohort.fixture import MoleculePanel, build_cohort_fixture, build_molecule_panel
from mechex.cohort.linkage import LinkageReport, link
from mechex.cohort.partition import (
    LeakageReport,
    SplitIndices,
    check_leakage,
    partition,
    partition_summary,
)
from mechex.cohort.records import CohortFixture
from mechex.cohort.reference import (
    ReferenceStandard,
    freeze_threshold,
    tier_counts,
)
from mechex.cohort.transforms import molecule_reference, relative_exposure
from mechex.estimator import MechExBatch
from mechex.settings import EstimatorConfig
from mechex.utils.io import atomic_write_json, ensure_dir
from mechex.utils.logging import get_logger

_log = get_logger("cohort.construction")

STAGED_COHORT_NAME = "cohort_fixture.npz"
STAGED_SUMMARY_NAME = "cohort_fixture_summary.json"


@dataclass
class StagedCohort:
    """Everything downstream code needs from the cohort layer."""

    cohort: CohortFixture
    split: SplitIndices
    reference: ReferenceStandard
    linkage: LinkageReport
    leakage: LeakageReport
    panel: MoleculePanel
    normalisation: np.ndarray
    tables: dict[str, np.ndarray] = field(default_factory=dict)

    def dataset(self, partition_name: str, config: EstimatorConfig) -> CohortTorchDataset:
        """Torch dataset over one partition."""
        indices = getattr(self.split, partition_name)
        return CohortTorchDataset(
            self.cohort,
            indices,
            self.panel.graphs,
            config.vascular,
            tau=config.decision.tau,
        )

    def collator(self, partition_name: str, config: EstimatorConfig) -> CohortCollator:
        return CohortCollator(self.dataset(partition_name, config).molecule_batch)

    def summary(self) -> dict[str, object]:
        return {
            "n_records": len(self.cohort),
            "n_molecules": int(np.unique(self.cohort.molecule_index).size),
            "sites": self.cohort.group_counts("site"),
            "tiers": tier_counts(self.cohort),
            "split": partition_summary(self.cohort, self.split),
            "linkage": self.linkage.as_dict(),
            "leakage": self.leakage.as_dict(),
            "reference": self.reference.as_dict(),
        }


def build_cohort(config: EstimatorConfig, seed: int | None = None) -> StagedCohort:
    """Generate the fixture, link it, freeze the standard and split it."""
    resolved_seed = config.cohort.fixture_seed if seed is None else seed
    panel = build_molecule_panel(config.cohort.n_molecules, resolved_seed)
    raw = build_cohort_fixture(config.cohort)
    split = partition(raw, config.protocol, resolved_seed)

    development_mask = split.mask("development", len(raw))
    normalisation = molecule_reference(
        raw.exposure_reference,
        raw.molecule_index,
        config.cohort.n_molecules,
        development_mask,
    )
    relative = relative_exposure(raw.exposure_reference, raw.molecule_index, normalisation)
    cohort = CohortFixture(
        record_id=raw.record_id,
        patient_id=raw.patient_id,
        site=raw.site,
        molecule_index=raw.molecule_index,
        descriptors=raw.descriptors,
        times=raw.times,
        concentrations=raw.concentrations,
        occlusion=raw.occlusion,
        availability_reference=raw.availability_reference,
        exposure_reference=relative,
        severity=raw.severity,
        drug_class=raw.drug_class,
        scanner_vendor=raw.scanner_vendor,
        accrual_era=raw.accrual_era,
        sex=raw.sex,
        age=raw.age,
        diabetes=raw.diabetes,
        ckd_stage=raw.ckd_stage,
        reference_tier=raw.reference_tier,
        volume_seed=raw.volume_seed,
        metadata=raw.metadata
        | {
            "normalisation": normalisation.tolist(),
            "normalised_on": "development partition",
        },
    )
    reference = freeze_threshold(
        cohort.exposure_reference[split.development],
        config.decision.tau,
        config.decision.adequacy_relative,
    )
    tables, linkage = link(cohort)
    leakage = check_leakage(cohort, split)
    if not leakage.clean:
        _log.warning("split carries leakage: %s", leakage.details)
    _log.info(
        "cohort built: %d records, %d molecules, split %s",
        len(cohort),
        config.cohort.n_molecules,
        split.as_dict(),
    )
    return StagedCohort(
        cohort=cohort,
        split=split,
        reference=reference,
        linkage=linkage,
        leakage=leakage,
        panel=panel,
        normalisation=normalisation,
        tables=tables,
    )


def stage(staged: StagedCohort, root: Path) -> dict[str, Path]:
    """Write the fixture and its summary under ``root``."""
    ensure_dir(root)
    cohort_path = staged.cohort.save(root / STAGED_COHORT_NAME)
    summary_path = root / STAGED_SUMMARY_NAME
    atomic_write_json(summary_path, staged.summary())
    return {"cohort": cohort_path, "summary": summary_path}


def load_staged(root: Path) -> CohortFixture:
    """Read a staged fixture."""
    return CohortFixture.load(root / STAGED_COHORT_NAME)


def batch_for_training(
    staged: StagedCohort,
    config: EstimatorConfig,
    partition_name: str,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> MechExBatch:
    """Materialise one batch, used by the smoke path and the execution checks."""
    from torch.utils.data import DataLoader

    dataset = staged.dataset(partition_name, config)
    collator = CohortCollator(dataset.molecule_batch)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=None,
    )
    first: MechExBatch = next(iter(loader))
    return first


def cohort_fingerprint(staged: StagedCohort) -> dict[str, object]:
    """Shape and composition fingerprint recorded in the verification report."""
    cohort = staged.cohort
    return {
        "n_records": len(cohort),
        "n_patients": int(np.unique(cohort.patient_id).size),
        "n_molecules": int(np.unique(cohort.molecule_index).size),
        "sites": cohort.group_counts("site"),
        "severity": cohort.group_counts("severity"),
        "drug_class": cohort.group_counts("drug_class"),
        "tiers": tier_counts(cohort),
        "split_sizes": staged.split.as_dict(),
        "leakage_clean": staged.leakage.clean,
    }
