"""Torch dataset over the tabular fixture.

Ref: Sec. 3.8, input volumes are resampled to isotropic spacing, windowed to the arterial
contrast range and intensity-normalised per scan; molecules are featurised with the
encoder's native featurisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from mechex.basis.graph import MoleculeGraph
from mechex.basis.transformer import MolecularBatch, collate_graphs
from mechex.cohort.records import CohortFixture
from mechex.estimator import MechExBatch
from mechex.settings import VascularEncoderConfig
from mechex.vascular.perfusion import phase_fixture
from mechex.vascular.volume import normalise_per_scan, window_to_contrast_range

LOG_FLOOR = 1e-6


class CohortItem(TypedDict):
    """One prepared record as the dataset hands it to a collator."""

    position: int
    molecule_index: int
    volume: torch.Tensor
    descriptors: torch.Tensor
    availability_reference: float
    exposure_reference: float
    adequacy_label: float
    occlusion: float
    patient_id: str
    reference_tier: int


def construct_volume(
    occlusion: float,
    seed: int,
    config: VascularEncoderConfig,
) -> np.ndarray:
    """Two-channel prepared CTA pair for one record.

    The pair is generated at the target grid so the preparation runs the reported window
    and per-scan normalisation without an unnecessary resampling pass.
    """
    arterial, delayed = phase_fixture(
        size=config.volume_shape, occlusion=float(occlusion), seed=int(seed)
    )
    windowed_arterial = window_to_contrast_range(arterial, *config.arterial_window_hu)
    windowed_delayed = window_to_contrast_range(delayed, *config.arterial_window_hu)
    return np.stack(
        [normalise_per_scan(windowed_arterial), normalise_per_scan(windowed_delayed)], axis=0
    )


class CohortTorchDataset(Dataset[CohortItem]):
    """Yields one prepared record per index, with the volume constructed on demand."""

    def __init__(
        self,
        cohort: CohortFixture,
        indices: np.ndarray,
        graphs: Sequence[MoleculeGraph],
        vascular_config: VascularEncoderConfig,
        tau: float = 0.30,
    ) -> None:
        self.cohort = cohort
        self.indices = np.asarray(indices, dtype=np.int64)
        self.graphs = list(graphs)
        self.vascular_config = vascular_config
        self.tau = float(tau)
        self.molecule_batch = collate_graphs(self.graphs)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> CohortItem:
        record = int(self.indices[position])
        volume = construct_volume(
            float(self.cohort.occlusion[record]),
            int(self.cohort.volume_seed[record]),
            self.vascular_config,
        )
        exposure = float(self.cohort.exposure_reference[record])
        return {
            "position": position,
            "molecule_index": int(self.cohort.molecule_index[record]),
            "volume": torch.as_tensor(volume, dtype=torch.float32),
            "descriptors": torch.as_tensor(self.cohort.descriptors[record], dtype=torch.float32),
            "availability_reference": float(self.cohort.availability_reference[record]),
            "exposure_reference": exposure,
            "adequacy_label": float(exposure >= (1.0 + self.tau)),
            "occlusion": float(self.cohort.occlusion[record]),
            "patient_id": str(self.cohort.patient_id[record]),
            "reference_tier": int(self.cohort.reference_tier[record]),
        }


class CohortCollator:
    """Turns dataset items into a :class:`MechExBatch`."""

    def __init__(self, molecule_batch: MolecularBatch) -> None:
        self.molecule_batch = molecule_batch

    def __call__(self, items: list[CohortItem]) -> MechExBatch:
        if not items:
            raise ValueError("the collator received an empty batch")
        molecule_index = torch.as_tensor(
            [int(item["molecule_index"]) for item in items], dtype=torch.long
        )
        volumes = torch.stack([item["volume"] for item in items])
        descriptors = torch.stack([item["descriptors"] for item in items])
        return MechExBatch(
            atoms=self.molecule_batch.atoms[molecule_index],
            coords=self.molecule_batch.coords[molecule_index],
            atom_mask=self.molecule_batch.mask[molecule_index],
            bond_index=self.molecule_batch.bond_index[molecule_index],
            descriptors=descriptors,
            volume=volumes,
            availability_reference=torch.as_tensor(
                [float(item["availability_reference"]) for item in items], dtype=torch.float32
            ).clamp_min(LOG_FLOOR),
            exposure_reference=torch.as_tensor(
                [float(item["exposure_reference"]) for item in items], dtype=torch.float32
            ).clamp_min(LOG_FLOOR),
            adequacy_label=torch.as_tensor(
                [item["adequacy_label"] for item in items], dtype=torch.float32
            ),
        )


def collator_for(dataset: CohortTorchDataset) -> CohortCollator:
    """Collator holding the shared molecule batch of ``dataset``."""
    return CohortCollator(dataset.molecule_batch)


def describe_batch(batch: MechExBatch) -> dict[str, object]:
    """Shape summary of a batch, used by the shape checks."""
    return {
        "batch_size": batch.batch_size,
        "atoms": list(batch.atoms.shape),
        "volume": list(batch.volume.shape),
        "descriptors": list(batch.descriptors.shape),
        "has_exposure_reference": batch.exposure_reference is not None,
    }
