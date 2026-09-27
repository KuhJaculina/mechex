"""Shared fixtures.

The smoke overlay is the only configuration the suite runs on; every fixture is session
scoped so the fixture is generated once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechex.cohort.construction import StagedCohort, build_cohort  # noqa: E402
from mechex.settings import EstimatorConfig, load_smoke_config  # noqa: E402


@pytest.fixture(scope="session")
def release_root() -> Path:
    """Repository root of the release under test."""
    return ROOT


@pytest.fixture(scope="session")
def config() -> EstimatorConfig:
    """The smoke configuration."""
    return load_smoke_config(ROOT / "configs")


@pytest.fixture(scope="session")
def staged(config: EstimatorConfig) -> StagedCohort:
    """The fixture cohort with its split and reference standard."""
    return build_cohort(config, seed=config.cohort.fixture_seed)


@pytest.fixture(scope="session")
def restricted_staged(staged: StagedCohort):
    """The staged cohort with every partition truncated, for the training-loop tests."""
    from mechex.cohort.construction import StagedCohort
    from mechex.cohort.partition import SplitIndices

    def take(indices):
        return indices[: min(96, indices.size)]

    return StagedCohort(
        cohort=staged.cohort,
        split=SplitIndices(
            development=take(staged.split.development),
            calibration=take(staged.split.calibration),
            validation=take(staged.split.validation),
            test=take(staged.split.test),
            external=take(staged.split.external),
        ),
        reference=staged.reference,
        linkage=staged.linkage,
        leakage=staged.leakage,
        panel=staged.panel,
        normalisation=staged.normalisation,
        tables=staged.tables,
    )


def make_batch(config: EstimatorConfig, n_records: int = 4, seed: int = 20260927):
    """A small batch built from fixture volumes and graphs.

    Kept independent of the cohort so a model test does not depend on the split.
    """
    import numpy as np
    import torch

    from mechex.basis.chemistry import descriptor_matrix, molecule_fixture
    from mechex.basis.transformer import collate_graphs
    from mechex.cohort.cohort import construct_volume
    from mechex.estimator import MechExBatch

    generator = np.random.default_rng(seed)
    graphs = [molecule_fixture(12 + (index % 7), 700000 + index) for index in range(n_records)]
    molecule_batch = collate_graphs(graphs)
    volumes = torch.stack(
        [
            torch.as_tensor(
                construct_volume(
                    float(generator.uniform(0.1, 0.8)), 900000 + index, config.vascular
                ),
                dtype=torch.float32,
            )
            for index in range(n_records)
        ]
    )
    descriptors = torch.as_tensor(descriptor_matrix(graphs), dtype=torch.float32)
    reference = torch.as_tensor(generator.uniform(0.6, 1.6, size=n_records), dtype=torch.float32)
    availability = torch.as_tensor(generator.uniform(0.3, 1.4, size=n_records), dtype=torch.float32)
    return MechExBatch(
        atoms=molecule_batch.atoms,
        coords=molecule_batch.coords,
        atom_mask=molecule_batch.mask,
        bond_index=molecule_batch.bond_index,
        descriptors=descriptors,
        volume=volumes,
        availability_reference=availability,
        exposure_reference=reference,
        adequacy_label=(reference >= 1.3).to(torch.float32),
    )


@pytest.fixture(scope="session")
def batch(config: EstimatorConfig):
    """A small batch for the model tests."""
    return make_batch(config)
