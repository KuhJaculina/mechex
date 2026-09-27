"""Safety, reconstructed baselines, quantisation and the in vitro arm."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.assay.accumulation import (
    COMPOUND_COUNT,
    MIN_REPLICATES,
    READOUT_METHODS,
    AssayDesign,
    accumulation_ratio,
    assay_fixture,
    design_summary,
    method_agreement,
    per_compound_ratio,
    required_sample_size,
    validate_efflux_hypothesis,
)
from mechex.bench.features import (
    CLINICAL_BLOCK,
    IMAGING_BLOCK,
    MOLECULAR_BLOCK,
    FeatureSpec,
    build_features,
    feature_names,
)
from mechex.bench.fusion import (
    ContrastiveAlignmentFusion,
    EarlyConcatenationFusion,
    UnconstrainedCrossAttentionFusion,
    build_fusion,
    predict_fusion,
    train_fusion,
)
from mechex.bench.mechanistic import (
    clinical_logistic,
    descriptor_ridge,
    graph_message_passing_features,
    kp_high_throughput,
    kp_rodgers_rowland,
    mechanism_only_score,
    pbk_exposure,
    similarity_adjacency,
)
from mechex.bench.radiomics import gradient_boosting_fit, radiomic_features, radiomic_matrix
from mechex.bench.registry import (
    BASELINES,
    baseline_spec,
    by_family,
    comparators,
    strongest_single_branch,
    strongest_unconstrained,
)
from mechex.bench.runner import run_all_baselines, run_baseline, score_predictions
from mechex.bench.single_branch import (
    FrozenHeadModel,
    SingleBranchRegressor,
    predict_single_branch,
    train_single_branch,
)
from mechex.compress.int8 import (
    INT8_MAX,
    evaluate_quantised_linear,
    memory_footprint_bytes,
    quantisation_report,
    quantise_linear_per_channel,
)
from mechex.guard.drift import (
    RunningMoments,
    StandardisedDriftDetector,
    calibrate_and_monitor,
    population_stability_index,
)
from mechex.guard.failure_modes import (
    FAILURE_TYPES,
    HARM_DIRECTION,
    classify_failures,
    taxonomy_table,
    unresolved_share,
    worst_decile,
)
from mechex.guard.operating_point import (
    evaluate_point,
    harm_tradeoff,
    operating_curve,
    select_operating_point,
)
from mechex.guard.perturbation import (
    DEFAULT_PERTURBATIONS,
    apply_attenuation,
    apply_dropout,
    apply_gain,
    apply_perturbation,
    apply_phase_shift,
    response_matrix,
    worst_relative_change,
)


def test_running_moments_match_direct_statistics() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    moments = RunningMoments()
    for value in values:
        moments.update(value)
    assert moments.mean == pytest.approx(float(np.mean(values)))
    assert moments.std == pytest.approx(float(np.std(values, ddof=1)))


def test_drift_threshold_calibration_bounds_the_alert_rate() -> None:
    generator = np.random.default_rng(0)
    reference = generator.normal(size=4000)
    stream = generator.normal(size=15000)
    detector = calibrate_and_monitor(reference, stream, stream, ceiling=1e-3)
    assert detector.alert_rate <= 1e-3
    assert detector.as_dict()["seen"] == stream.size


def test_detector_silent_before_enough_observations() -> None:
    detector = StandardisedDriftDetector(threshold=2.0)
    assert not detector.observe(5.0, 1.0)


def test_population_stability_index_is_zero_for_the_same_sample() -> None:
    values = np.random.default_rng(1).normal(size=500)
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_worst_decile_selects_the_largest_errors() -> None:
    error = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    mask = worst_decile(error, 0.5)
    assert int(mask.sum()) == 5
    assert mask[0] and not mask[-1]


def test_failure_taxonomy_assigns_the_intended_mechanisms() -> None:
    """Each crafted record carries exactly one mechanism, so the assignment is unambiguous."""
    error = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    occlusion = np.array([0.9, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    availability = np.array([0.5, 0.5, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    vendor = np.array(["vendor_1", "vendor_3"] + ["vendor_1"] * 8, dtype=object)
    ckd = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 2])
    assignment = classify_failures(
        error, occlusion, availability, vendor, ckd_stage=ckd, fraction=0.4
    )
    assert assignment.types.tolist() == [1, 2, 3, 5, 0, 0, 0, 0, 0, 0]
    table = taxonomy_table(assignment)
    assert len(table) == len(FAILURE_TYPES) + 1
    assert table[-1]["unclassified"] is True
    assert 0.0 <= unresolved_share(assignment) <= 1.0


def test_failure_taxonomy_priority_is_declared() -> None:
    """A record carrying several mechanisms goes to the first rule that claims it.

    The order runs from the numerical-instability mechanisms to the imaging ones, so a
    record with an availability below the floor is classified there even when its bed is
    also densely calcified.
    """
    error = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    occlusion = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    availability = np.array([0.02, 0.5, 0.5, 0.5, 0.5])
    vendor = np.array(["vendor_1"] * 5, dtype=object)
    assignment = classify_failures(error, occlusion, availability, vendor, fraction=0.2)
    assert assignment.types[0] == 3


def test_taxonomy_harm_directions_are_declared() -> None:
    assert set(HARM_DIRECTION) == set(FAILURE_TYPES)
    assert all(value in {"false_adequate", "false_inadequate"} for value in HARM_DIRECTION.values())


def test_operating_point_evaluation_and_selection() -> None:
    generator = np.random.default_rng(2)
    labels = np.r_[np.ones(200), np.zeros(200)].astype(np.int64)
    scores = np.r_[generator.normal(1.1, 1.0, 200), generator.normal(0.4, 1.0, 200)]
    point = evaluate_point(scores, labels, 0.6)
    assert 0.0 <= point.false_positive_rate <= 1.0
    choice = select_operating_point(scores, labels, 0.05)
    assert choice.false_positive_rate <= 0.05 + 1e-9
    curve = operating_curve(scores, labels, grid=30)
    assert len(curve) > 5
    assert harm_tradeoff(curve)["best_net_benefit"] >= 0.0


def test_perturbations_scale_and_shift_as_declared() -> None:
    volume = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.5, 2.5], [3.5, 4.5]]])
    assert float(apply_gain(volume, 0.1)[0].mean()) == pytest.approx(2.5 * 1.1)
    assert float(apply_attenuation(volume, 0.1)[0].mean()) == pytest.approx(2.5 * 0.9)
    shifted = apply_phase_shift(volume, 4.0)
    assert not np.allclose(shifted[1], volume[1])
    dropped = apply_dropout(volume, 0.5)
    assert dropped.shape == volume.shape
    assert apply_perturbation(volume, DEFAULT_PERTURBATIONS[0]).shape == volume.shape


def test_perturbation_dispatch_rejects_an_unknown_name() -> None:
    from mechex.guard.perturbation import Perturbation

    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((2, 4, 4)), Perturbation("not_a_perturbation", 0.1))


def test_response_matrix_records_each_perturbation() -> None:
    volume = np.array([[[1.0, 2.0]], [[1.5, 2.5]]])
    matrix = response_matrix(lambda array: float(array[0].mean()), volume)
    assert len(matrix) == len(DEFAULT_PERTURBATIONS)
    assert worst_relative_change(matrix) >= 0.0


def test_baseline_registry_is_consistent_with_the_reported_table() -> None:
    assert len(BASELINES) == 12
    assert strongest_single_branch().identifier == "B6"
    assert strongest_unconstrained().identifier == "B11"
    assert baseline_spec("B9").family == "mechanistic"
    assert by_family("fusion")
    assert all("reported" in row for row in comparators())


def test_feature_blocks_and_names() -> None:
    from mechex.cohort.fixture import build_cohort_fixture
    from mechex.settings import CohortConfig

    cohort = build_cohort_fixture(CohortConfig(fixture_seed=21))
    indices = np.arange(6)
    for block in (MOLECULAR_BLOCK, IMAGING_BLOCK, CLINICAL_BLOCK):
        spec = FeatureSpec(blocks=(block,))
        matrix = build_features(cohort, indices, spec)
        assert matrix.shape[0] == 6
        assert matrix.shape[1] == len(feature_names(spec))


def test_mechanistic_predictors_run() -> None:
    generator = np.random.default_rng(3)
    descriptors = np.abs(generator.normal(size=(20, 8)))
    delivered = generator.uniform(0.1, 0.9, size=20)
    assert np.all(np.isfinite(kp_rodgers_rowland(descriptors)))
    assert np.all(np.isfinite(kp_high_throughput(descriptors)))
    assert np.all(np.isfinite(pbk_exposure(descriptors, delivered)))
    assert mechanism_only_score(descriptors).shape == (20,)


def test_mechanism_only_score_falls_with_efflux_propensity() -> None:
    low = np.zeros(8)
    high = np.zeros(8)
    low[0] = 0.2
    high[0] = 0.9
    scores = mechanism_only_score(np.vstack([low, high]))
    assert scores[1] < scores[0]


def test_descriptor_ridge_and_clinical_logistic_run() -> None:
    generator = np.random.default_rng(4)
    train = generator.normal(size=(40, 6))
    target = train[:, 0] * 0.5 + 1.0
    evaluation = generator.normal(size=(10, 6))
    ridge = descriptor_ridge(train, target, evaluation)
    logistic = clinical_logistic(train, target / np.abs(target).max(), evaluation)
    assert ridge.shape == (10,) and np.all(ridge > 0)
    assert logistic.min() > 0.0 and logistic.max() <= 1.0


def test_graph_propagation_and_adjacency() -> None:
    generator = np.random.default_rng(5)
    features = generator.normal(size=(8, 5))
    adjacency = similarity_adjacency(features)
    assert np.allclose(np.diag(adjacency), 0.0)
    propagated = graph_message_passing_features(features, adjacency, rounds=2)
    assert propagated.shape == (8, 10)


def test_radiomic_features_and_boosting() -> None:
    from mechex.cohort.cohort import construct_volume
    from mechex.settings import VascularEncoderConfig

    config = VascularEncoderConfig(latent_dim=8, stem_channels=(4, 8), volume_shape=(12, 12, 8))
    volume = construct_volume(0.3, 1, config)
    summary = radiomic_features(volume)
    assert summary.values.shape == (7,)
    matrix = radiomic_matrix([volume, volume])
    assert matrix.shape[0] == 2
    generator = np.random.default_rng(6)
    target = generator.normal(size=2)
    boosted = gradient_boosting_fit(matrix, target, matrix)
    assert boosted.shape == (2,)
    assert np.all(np.isfinite(boosted))


def test_fusion_baselines_train_and_predict() -> None:
    torch.manual_seed(0)
    molecular = torch.randn(24, 5)
    imaging = torch.randn(24, 4)
    target = molecular[:, 0] + 0.5 * imaging[:, 0]
    for model in (
        EarlyConcatenationFusion(5, 4),
        UnconstrainedCrossAttentionFusion(5, 4, width=16, heads=2),
        ContrastiveAlignmentFusion(5, 4, width=16),
    ):
        history = train_fusion(model, molecular, imaging, target, epochs=5)
        assert history[-1] <= history[0] or len(history) == 1
        prediction = predict_fusion(model, molecular, imaging)
        assert prediction.shape == (24,)


def test_fusion_builder_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_fusion("not_a_fusion", 4, 4)


def test_single_branch_and_frozen_head() -> None:
    torch.manual_seed(1)
    features = torch.randn(20, 5)
    target = features[:, 0]
    model = SingleBranchRegressor(5)
    train_single_branch(model, features, target, epochs=5)
    assert predict_single_branch(model, features).shape == (20,)
    frozen = FrozenHeadModel(5, width=8, frozen=True)
    assert all(not parameter.requires_grad for parameter in frozen.trunk.parameters())
    assert float(frozen(features).shape[0]) == 20.0


def test_score_predictions_metrics() -> None:
    prediction = np.array([1.4, 0.8, 1.5, 0.7])
    reference = np.array([1.3, 0.9, 1.4, 0.8])
    metrics = score_predictions(prediction, reference)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["fold_error"] >= 1.0


def test_baselines_run_on_the_fixture(staged) -> None:
    train = staged.split.development[:80]
    evaluation = staged.split.external[:80]
    for identifier in ("B9", "B2", "B8", "B4"):
        row = run_baseline(identifier, staged.cohort, train, evaluation)
        metrics = row["metrics"]
        assert np.isfinite(float(metrics["ccc"]))
    rows = run_all_baselines(staged.cohort, train, evaluation, identifiers=("B9", "B2"))
    assert len(rows) == 2


def test_per_channel_quantisation_matches_a_hand_computation() -> None:
    torch.manual_seed(2)
    layer = torch.nn.Linear(12, 6)
    quantised = quantise_linear_per_channel(layer, "probe")
    scales = layer.weight.detach().abs().amax(dim=1) / INT8_MAX
    hand = torch.round(layer.weight.detach() / scales.view(-1, 1)).clamp(-INT8_MAX, INT8_MAX)
    assert torch.equal(quantised.integer_weights, hand.to(torch.int8))
    residual = (layer.weight.detach() - quantised.dequantise()).abs()
    assert bool((residual <= scales.view(-1, 1) / 2 + 1e-6).all())
    assert quantised.original_shape == tuple(layer.weight.shape)


def test_quantisation_report_and_memory() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    report = quantisation_report(module)
    assert report["n_layers"] == 2
    assert float(report["mean_relative_frobenius_error"]) < 0.1
    footprint = memory_footprint_bytes(module)
    assert footprint["quantised_bytes"] < footprint["float32_bytes"]


def test_quantised_linear_deviation() -> None:
    torch.manual_seed(3)
    layer = torch.nn.Linear(16, 8)
    report = evaluate_quantised_linear(layer, torch.randn(5, 16))
    assert report["relative_deviation"] < 0.05


def test_assay_design_is_a_full_factorial() -> None:
    design = AssayDesign()
    summary = design_summary()
    assert summary["compounds"] == float(COMPOUND_COUNT)
    assert summary["min_replicates"] == float(MIN_REPLICATES)
    assert summary["readout_methods"] == list(READOUT_METHODS)
    assert design.n_conditions == COMPOUND_COUNT * 3 * 2 * len(READOUT_METHODS)


def test_accumulation_ratio_arithmetic() -> None:
    values = accumulation_ratio(np.array([3.0]), np.array([1.0]))
    assert values.tolist() == [3.0]
    with pytest.raises(ValueError):
        accumulation_ratio(np.zeros(2), np.zeros(3))


def test_assay_fixture_and_hypothesis_direction() -> None:
    propensity = np.array([0.2, 0.45, 0.7, 0.95])
    table = assay_fixture(propensity)
    ratios = per_compound_ratio(table)
    assert np.all(np.diff(ratios) > 0)
    report = validate_efflux_hypothesis(propensity, ratios, n_permutations=200, seed=1)
    assert report["rho"] > 0.5
    agreement = method_agreement(table)
    assert agreement["n"] > 0


def test_required_sample_size_grows_as_the_effect_shrinks() -> None:
    strong = required_sample_size(0.87)
    weak = required_sample_size(0.30)
    assert weak > strong > 3.0
    with pytest.raises(ValueError):
        required_sample_size(0.0)


def test_assay_table_rejects_a_wrong_panel_size() -> None:
    with pytest.raises(ValueError):
        assay_fixture(np.array([0.5, 0.7]))
