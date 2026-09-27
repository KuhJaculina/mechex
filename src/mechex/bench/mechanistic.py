"""Mechanistic and population-prior predictors.

Ref: Table 1, the mechanism-only repurposing score, the descriptor models, the clinical
logistic regression and the two PBPK variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.basis.chemistry import DESCRIPTOR_NAMES
from mechex.quantify.roc import rank_average

EFFLUX_INDEX = DESCRIPTOR_NAMES.index("efflux_substrate_propensity")
LIPOPHILICITY_INDEX = DESCRIPTOR_NAMES.index("lipophilicity")
SIZE_INDEX = DESCRIPTOR_NAMES.index("molecular_size")

# Rodgers-Rowland style tissue-composition priors for a muscle compartment; the
# manuscript does not state its PBPK parameterisation, so the composition values are the
# release's own engineering constants exposed here rather than buried in a call site.
MUSCLE_FRACTION_NEUTRAL_LIPID = 0.0238
MUSCLE_FRACTION_PHOSPHOLIPID = 0.0158
MUSCLE_FRACTION_WATER = 0.766
PLASMA_FRACTION_WATER = 0.945
FRACTION_UNBOUND_DEFAULT = 0.35


@dataclass(frozen=True)
class MechanismWeights:
    """Relative contribution of each descriptor axis to the mechanism score."""

    efflux: float = -0.55
    lipophilicity: float = 0.30
    size: float = -0.12


def mechanism_only_score(
    descriptors: np.ndarray,
    weights: MechanismWeights | None = None,
) -> np.ndarray:
    """Mechanism-only repurposing score.

    Ref: Table 1, B9. The score reads the same physicochemical axes a repurposing campaign
    would rank on, with no delivery term, which is what makes it the no-delivery comparator
    the manuscript reports.
    """
    matrix = np.asarray(descriptors, dtype=np.float64)
    resolved = weights or MechanismWeights()
    scale = np.where(matrix.std(axis=0) > 1e-9, matrix.std(axis=0), 1.0)
    standardised = (matrix - matrix.mean(axis=0)) / scale
    trend: np.ndarray = (
        resolved.efflux * standardised[:, EFFLUX_INDEX]
        + resolved.lipophilicity * standardised[:, LIPOPHILICITY_INDEX]
        + resolved.size * standardised[:, SIZE_INDEX]
    )
    return trend


def mechanism_to_exposure(
    fit_score: np.ndarray,
    fit_reference: np.ndarray,
    apply_score: np.ndarray,
) -> np.ndarray:
    """Map a mechanism score onto the exposure scale with an affine fit on the training block.

    A concordance coefficient is only defined between two quantities on the same scale, so
    the comparator is calibrated to the reference range before its concordance is read. The
    calibration is estimated on the training block and applied to the evaluation block;
    fitting it on the evaluation block itself would leak the target.
    """
    fit_values = np.asarray(fit_score, dtype=np.float64)
    target = np.asarray(fit_reference, dtype=np.float64)
    design = np.column_stack([fit_values, np.ones_like(fit_values)])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    apply_values = np.asarray(apply_score, dtype=np.float64)
    apply_design = np.column_stack([apply_values, np.ones_like(apply_values)])
    mapped: np.ndarray = apply_design @ coefficients
    clipped: np.ndarray = np.clip(mapped, 1e-4, None)
    return clipped


def rank_correlation_to_reference(
    score: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Rank agreement between a comparator and the reference exposure."""
    from mechex.quantify.correlation import spearman_rho

    return spearman_rho(score, reference)


def kp_rodgers_rowland(
    descriptors: np.ndarray,
    fraction_unbound: np.ndarray | float = FRACTION_UNBOUND_DEFAULT,
) -> np.ndarray:
    """Tissue-to-plasma partition coefficient from a muscle-compartment composition model.

    The neutral-lipid and phospholipid terms use the lipophilicity descriptor, so the
    predictor has no patient-specific information beyond the molecule.
    """
    matrix = np.asarray(descriptors, dtype=np.float64)
    lipophilicity = matrix[:, LIPOPHILICITY_INDEX]
    neutral_lipid_term = MUSCLE_FRACTION_NEUTRAL_LIPID * 10.0**lipophilicity
    phospholipid_term = MUSCLE_FRACTION_PHOSPHOLIPID * (0.3 * 10.0**lipophilicity + 0.7)
    water_term = MUSCLE_FRACTION_WATER / PLASMA_FRACTION_WATER
    unbound = np.asarray(fraction_unbound, dtype=np.float64)
    coefficient: np.ndarray = unbound * (neutral_lipid_term + phospholipid_term + water_term) + (
        1.0 - unbound
    )
    return coefficient


def kp_high_throughput(descriptors: np.ndarray) -> np.ndarray:
    """High-throughput PBPK prior: the same composition model with population-average binding.

    Ref: Table 1, B5. Everything is averaged over the population, so no patient-specific
    delivery information enters.
    """
    return kp_rodgers_rowland(descriptors, FRACTION_UNBOUND_DEFAULT)


def pbk_exposure(
    descriptors: np.ndarray,
    perfused_fraction: np.ndarray,
    population_perfusion: float = 0.62,
    use_patient_perfusion: bool = True,
) -> np.ndarray:
    """PBPK-style exposure on the reference scale.

    Ref: Table 1, B4 uses the population perfusion prior for every patient; the variant
    that substitutes the patient's own perfused fraction is the coupling the model adds.
    """
    coefficient = kp_rodgers_rowland(descriptors)
    perfusion = (
        np.asarray(perfused_fraction, dtype=np.float64)
        if use_patient_perfusion
        else np.full(len(coefficient), float(population_perfusion))
    )
    exposure: np.ndarray = coefficient * perfusion
    return exposure


def descriptor_ridge(
    train: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
    penalty: float = 1.0,
) -> np.ndarray:
    """Ridge regression on the descriptor block, the descriptor machine-learning baseline."""
    design = np.column_stack([train, np.ones(train.shape[0])])
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ np.asarray(target, dtype=np.float64))
    evaluation_design = np.column_stack([evaluation, np.ones(evaluation.shape[0])])
    prediction: np.ndarray = evaluation_design @ coefficients
    clipped: np.ndarray = np.clip(prediction, 1e-4, None)
    return clipped


def clinical_logistic(
    train: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
    steps: int = 200,
    learning_rate: float = 0.1,
    penalty: float = 1e-2,
) -> np.ndarray:
    """Logistic regression on the clinical block, with gradient descent on the log-loss."""
    design = np.column_stack([train, np.ones(train.shape[0])])
    labels = np.asarray(target, dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = design @ coefficients
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        gradient = design.T @ (probabilities - labels) / design.shape[0] + penalty * coefficients
        coefficients -= learning_rate * gradient
    evaluation_design = np.column_stack([evaluation, np.ones(evaluation.shape[0])])
    logits = evaluation_design @ coefficients
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))), 1e-4, 1.0 - 1e-4)


def graph_message_passing_features(
    descriptors: np.ndarray,
    adjacency: np.ndarray,
    rounds: int = 3,
) -> np.ndarray:
    """Descriptor features aggregated over a molecule-similarity graph.

    Ref: Table 1, B3. A geometric graph neural network propagates information between
    neighbouring molecules; the same propagation on the chemistry-similarity graph is the
    reconstruction used here, with a row-normalised adjacency.
    """
    features = np.asarray(descriptors, dtype=np.float64)
    weight = np.asarray(adjacency, dtype=np.float64)
    weight = weight + np.eye(weight.shape[0])
    degree = np.clip(weight.sum(axis=1, keepdims=True), 1e-9, None)
    normalised = weight / degree
    current = features
    for _ in range(rounds):
        current = 0.5 * current + 0.5 * (normalised @ current)
    return np.column_stack([features, current])


def similarity_adjacency(features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Gaussian kernel adjacency over standardised features, with a zero diagonal."""
    centred = features - features.mean(axis=0)
    scale = np.where(features.std(axis=0) > 1e-9, features.std(axis=0), 1.0)
    standardised = centred / scale
    squared = ((standardised[:, None, :] - standardised[None, :, :]) ** 2).sum(axis=-1)
    weight: np.ndarray = np.exp(-squared / (2.0 * temperature**2))
    np.fill_diagonal(weight, 0.0)
    return weight


def propaginate_by_molecule(
    values: np.ndarray,
    molecule_index: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    """Lift per-molecule values onto records through their molecule index."""
    return np.asarray(values, dtype=np.float64)[np.asarray(molecule_index, dtype=np.int64)]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks of a score vector, exposed for the ranking comparison."""
    return rank_average(values)
