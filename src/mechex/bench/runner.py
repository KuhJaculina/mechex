"""Run a reconstructed comparator on the fixture and score it.

Ref: Table 1 note, the reconstructed baselines were trained and evaluated in-house on the
identical split with identical preprocessing and an identical tuning budget.
"""

from __future__ import annotations

import numpy as np

from mechex.bench.features import (
    IMAGING_BLOCK,
    MOLECULAR_BLOCK,
    FeatureSpec,
    design_matrix,
    feature_names,
)
from mechex.bench.mechanistic import (
    clinical_logistic,
    descriptor_ridge,
    graph_message_passing_features,
    kp_high_throughput,
    kp_rodgers_rowland,
    mechanism_only_score,
    mechanism_to_exposure,
    similarity_adjacency,
)
from mechex.bench.registry import (
    BASELINES,
    FAMILY_FUSION,
    FAMILY_MECHANISTIC,
    FAMILY_TABULAR,
    BaselineSpec,
    baseline_spec,
)
from mechex.cohort.records import CohortFixture
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.fold import geometric_fold_error
from mechex.quantify.roc import auroc


def score_predictions(
    prediction: np.ndarray,
    reference: np.ndarray,
    tau: float = 0.30,
) -> dict[str, float]:
    """Concordance, discrimination and fold error of a prediction."""
    predicted = np.asarray(prediction, dtype=np.float64).ravel()
    truth = np.asarray(reference, dtype=np.float64).ravel()
    labels = (truth >= 1.0 + tau).astype(np.int64)
    discrimination = auroc(predicted, labels) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "ccc": float(lin_ccc(predicted, truth)),
        "auroc": float(discrimination),
        "fold_error": float(
            geometric_fold_error(np.clip(predicted, 1e-6, None), np.clip(truth, 1e-6, None))
        ),
        "n": float(predicted.size),
    }


def _tabular_prediction(
    spec: BaselineSpec,
    cohort: CohortFixture,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
) -> np.ndarray:
    design = FeatureSpec(blocks=spec.blocks)
    train, target_train, evaluation, _ = design_matrix(
        cohort, train_indices, evaluation_indices, design
    )
    if spec.estimator == "descriptor_ridge":
        return descriptor_ridge(train, target_train, evaluation)
    if spec.estimator == "clinical_logistic":
        return clinical_logistic(train, target_train, evaluation)
    if spec.estimator == "graph_propagation_ridge":
        adjacency = similarity_adjacency(
            np.asarray(cohort.descriptors, dtype=np.float64)[np.unique(cohort.molecule_index)]
        )
        propagated = graph_message_passing_features(
            np.asarray(cohort.descriptors, dtype=np.float64)[np.unique(cohort.molecule_index)],
            adjacency,
        )
        lookup = {
            int(molecule): propagated[position]
            for position, molecule in enumerate(np.unique(cohort.molecule_index))
        }
        train_propagated = np.vstack(
            [lookup[int(value)] for value in cohort.molecule_index[train_indices]]
        )
        evaluation_propagated = np.vstack(
            [lookup[int(value)] for value in cohort.molecule_index[evaluation_indices]]
        )
        return descriptor_ridge(train_propagated, target_train, evaluation_propagated)
    if spec.estimator == "radiomics_gradient_boosting":
        from mechex.bench.radiomics import gradient_boosting_fit

        return gradient_boosting_fit(train, target_train, evaluation)
    raise ValueError(f"unsupported tabular estimator: {spec.estimator}")


def _mechanistic_prediction(
    spec: BaselineSpec,
    cohort: CohortFixture,
    evaluation_indices: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    descriptors = np.asarray(cohort.descriptors, dtype=np.float64)[evaluation_indices]
    train_descriptors = np.asarray(cohort.descriptors, dtype=np.float64)[train_indices]
    target_train = np.asarray(cohort.exposure_reference, dtype=np.float64)[train_indices]
    delivered = np.clip(
        1.0 - np.asarray(cohort.occlusion, dtype=np.float64)[evaluation_indices], 0.0, 1.0
    )
    if spec.estimator == "mechanism_only_score":
        return mechanism_to_exposure(
            mechanism_only_score(train_descriptors),
            target_train,
            mechanism_only_score(descriptors),
        )
    if spec.estimator == "pbk_population":
        population: np.ndarray = kp_rodgers_rowland(descriptors) * delivered
        return population
    if spec.estimator == "pbk_high_throughput":
        throughput: np.ndarray = kp_high_throughput(descriptors) * delivered.mean()
        return throughput
    raise ValueError(f"unsupported mechanistic estimator: {spec.estimator}")


def _fusion_prediction(
    spec: BaselineSpec,
    cohort: CohortFixture,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    epochs: int = 40,
) -> np.ndarray:
    import torch
    from torch import nn

    from mechex.bench.fusion import build_fusion, predict_fusion, train_fusion
    from mechex.bench.single_branch import (
        SingleBranchRegressor,
        predict_single_branch,
        train_single_branch,
    )
    from mechex.seeding import set_seed

    set_seed(20260927)
    design = FeatureSpec(blocks=spec.blocks)
    train, target_train, evaluation, _ = design_matrix(
        cohort, train_indices, evaluation_indices, design
    )
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    evaluation_tensor = torch.as_tensor(evaluation, dtype=torch.float32)
    target_tensor = torch.as_tensor(target_train, dtype=torch.float32)
    if spec.estimator == "single_branch_mlp":
        branch: nn.Module = SingleBranchRegressor(train.shape[1])
        train_single_branch(branch, train_tensor, target_tensor, epochs=epochs)
        branch_prediction = predict_single_branch(branch, evaluation_tensor)
        return np.clip(branch_prediction.detach().cpu().numpy().astype(np.float64), 1e-4, None)
    molecular_train, imaging_train = _split_blocks(cohort, train_indices, spec)
    molecular_eval, imaging_eval = _split_blocks(cohort, evaluation_indices, spec)
    model = build_fusion(spec.estimator, molecular_train.shape[1], imaging_train.shape[1])
    train_fusion(
        model,
        torch.as_tensor(molecular_train, dtype=torch.float32),
        torch.as_tensor(imaging_train, dtype=torch.float32),
        target_tensor,
        epochs=epochs,
    )
    prediction = predict_fusion(
        model,
        torch.as_tensor(molecular_eval, dtype=torch.float32),
        torch.as_tensor(imaging_eval, dtype=torch.float32),
    )
    return np.clip(prediction.detach().cpu().numpy().astype(np.float64), 1e-4, None)


def _split_blocks(
    cohort: CohortFixture,
    indices: np.ndarray,
    spec: BaselineSpec,
) -> tuple[np.ndarray, np.ndarray]:
    from mechex.bench.features import build_features, standardise_columns

    molecular = build_features(cohort, indices, FeatureSpec(blocks=(MOLECULAR_BLOCK,)))
    imaging = build_features(cohort, indices, FeatureSpec(blocks=(IMAGING_BLOCK,)))
    # Each block is standardised with its own column moments: the two blocks describe
    # different quantities and share no columns to borrow moments from.
    molecular, _ = standardise_columns(molecular, molecular)
    imaging, _ = standardise_columns(imaging, imaging)
    return molecular, imaging


def run_baseline(
    identifier: str,
    cohort: CohortFixture,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    tau: float = 0.30,
) -> dict[str, object]:
    """Fit and score one reconstructed comparator."""
    spec = baseline_spec(identifier)
    if spec.family == FAMILY_TABULAR:
        prediction = _tabular_prediction(spec, cohort, train_indices, evaluation_indices)
    elif spec.family == FAMILY_MECHANISTIC:
        prediction = _mechanistic_prediction(spec, cohort, evaluation_indices, train_indices)
    elif spec.family == FAMILY_FUSION:
        prediction = _fusion_prediction(spec, cohort, train_indices, evaluation_indices)
    else:
        raise ValueError(f"unknown family: {spec.family}")
    reference = np.asarray(cohort.exposure_reference, dtype=np.float64)[evaluation_indices]
    metrics = score_predictions(prediction, reference, tau)
    return {
        "id": spec.identifier,
        "method": spec.name,
        "family": spec.family,
        "estimator": spec.estimator,
        "metrics": metrics,
        "feature_names": feature_names(FeatureSpec(blocks=spec.blocks)),
    }


def run_all_baselines(
    cohort: CohortFixture,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    identifiers: tuple[str, ...] | None = None,
    tau: float = 0.30,
) -> list[dict[str, object]]:
    """Fit and score several comparators, skipping any that raises."""
    from mechex.utils.logging import get_logger

    log = get_logger("bench.runner")
    selected = identifiers or tuple(spec.identifier for spec in BASELINES)
    rows: list[dict[str, object]] = []
    for identifier in selected:
        try:
            rows.append(run_baseline(identifier, cohort, train_indices, evaluation_indices, tau))
        except Exception as error:
            log.warning("baseline %s failed: %s", identifier, error)
            rows.append({"id": identifier, "metrics": {"error": str(error)}})
    return rows
