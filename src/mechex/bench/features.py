"""Feature blocks shared by the reconstructed baselines.

Ref: Table 1 note, the reconstructed baselines were trained and evaluated in-house on the
identical split with identical preprocessing and an identical tuning budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.cohort.records import CohortFixture

MOLECULAR_BLOCK = "molecular"
IMAGING_BLOCK = "imaging"
CLINICAL_BLOCK = "clinical"

SEVERITY_ONE_HOT = ("claudication", "rest_pain", "chronic_limb_threatening_ischaemia")


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature blocks a baseline receives."""

    blocks: tuple[str, ...] = (MOLECULAR_BLOCK, IMAGING_BLOCK, CLINICAL_BLOCK)
    standardise: bool = True


def molecular_features(cohort: CohortFixture, indices: np.ndarray) -> np.ndarray:
    """The molecular descriptor block."""
    block: np.ndarray = np.asarray(cohort.descriptors, dtype=np.float64)[indices]
    return block


def imaging_features(cohort: CohortFixture, indices: np.ndarray) -> np.ndarray:
    """Imaging block: the CTA-derived perfused fraction and its phase ratio.

    A reconstructed baseline sees the same CTA in the same preprocessed form, so the block
    is the dimensionless delivery term plus the phase contrast that produced it.
    """
    occlusion = np.asarray(cohort.occlusion, dtype=np.float64)[indices]
    delivered = np.clip(1.0 - occlusion, 0.0, 1.0)
    contrast = delivered * (1.0 - delivered)
    return np.column_stack([delivered, contrast, occlusion])


def clinical_features(cohort: CohortFixture, indices: np.ndarray) -> np.ndarray:
    """Clinical block: age, sex, diabetes, renal stage and severity indicators."""
    age = np.asarray(cohort.age, dtype=np.float64)[indices]
    female = (np.asarray(cohort.sex, dtype=object)[indices] == "female").astype(np.float64)
    diabetes = np.asarray(cohort.diabetes, dtype=bool)[indices].astype(np.float64)
    ckd = np.asarray(cohort.ckd_stage, dtype=np.float64)[indices]
    severity = np.asarray(cohort.severity, dtype=object)[indices]
    severity_block = np.column_stack(
        [(severity == level).astype(np.float64) for level in SEVERITY_ONE_HOT]
    )
    return np.column_stack([age, female, diabetes, ckd, severity_block])


BLOCK_BUILDERS = {
    MOLECULAR_BLOCK: molecular_features,
    IMAGING_BLOCK: imaging_features,
    CLINICAL_BLOCK: clinical_features,
}


def build_features(
    cohort: CohortFixture,
    indices: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    """Stack the requested blocks into one design matrix."""
    blocks = [BLOCK_BUILDERS[name](cohort, indices) for name in spec.blocks]
    if not blocks:
        raise ValueError("a baseline needs at least one feature block")
    return np.column_stack(blocks)


def standardise_columns(
    train: np.ndarray,
    other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise ``other`` with the training moments.

    The moments come from the training block only; using the evaluation block's own
    moments would leak its distribution into the fit.
    """
    mean = train.mean(axis=0)
    scale = np.where(train.std(axis=0) > 1e-9, train.std(axis=0), 1.0)
    return (train - mean) / scale, (other - mean) / scale


def feature_names(spec: FeatureSpec) -> list[str]:
    """Names of the produced columns, in order."""
    names: list[str] = []
    for block in spec.blocks:
        if block == MOLECULAR_BLOCK:
            from mechex.basis.chemistry import DESCRIPTOR_NAMES

            names.extend(DESCRIPTOR_NAMES)
        elif block == IMAGING_BLOCK:
            names.extend(["delivered_fraction", "phase_contrast", "occlusion"])
        else:
            names.extend(["age", "female", "diabetes", "ckd_stage"])
            names.extend([f"severity_{level}" for level in SEVERITY_ONE_HOT])
    return names


def design_matrix(
    cohort: CohortFixture,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    spec: FeatureSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardised design matrices and their targets for a train and an evaluation block."""
    train = build_features(cohort, train_indices, spec)
    evaluation = build_features(cohort, evaluation_indices, spec)
    if spec.standardise:
        train, evaluation = standardise_columns(train, evaluation)
    return (
        train,
        np.asarray(cohort.exposure_reference, dtype=np.float64)[train_indices],
        evaluation,
        np.asarray(cohort.exposure_reference, dtype=np.float64)[evaluation_indices],
    )
