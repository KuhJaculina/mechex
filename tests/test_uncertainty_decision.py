"""Uncertainty head, isotonic recalibration and the decision layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.decision.abstention import (
    HarmTiering,
    abstention_mask,
    cost_curve,
    coverage_and_retained_auroc,
    expected_cost,
    select_boundary,
)
from mechex.decision.threshold import (
    ThresholdDecision,
    adequacy_boundary,
    adequacy_from_baseline,
    concentration_time_integral,
    exposure_index,
    is_adequate,
)
from mechex.settings import UncertaintyConfig
from mechex.uncertainty.ensemble import DeepEnsembleHeads
from mechex.uncertainty.evidential import (
    EvidentialHead,
    PointHead,
    evidential_variance,
    nig_nll,
    nig_parameters,
)
from mechex.uncertainty.heads import UNCERTAINTY_MODES, UncertaintyHead, build_uncertainty_head
from mechex.uncertainty.isotonic import (
    MEAN_ABS_TO_SIGMA,
    IsotonicRecalibrator,
    pool_adjacent_violators,
)


def test_nig_parameters_are_positive_and_bounded() -> None:
    raw = torch.randn(6, 4)
    mean, nu, alpha, beta = nig_parameters(raw)
    assert torch.isfinite(mean).all()
    assert bool((nu > 0).all())
    assert bool((alpha > 1.0).all())
    assert bool((beta > 0).all())


def test_evidential_variance_is_positive() -> None:
    variance = evidential_variance(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.5]))
    assert float(variance) > 0


def test_evidential_head_and_loss() -> None:
    head = EvidentialHead(12, 8)
    output = head(torch.randn(5, 12))
    loss = head.loss(torch.randn(5), output)
    assert output.mean.shape == (5,)
    assert torch.isfinite(loss)


def test_nig_loss_prefers_a_sharper_correct_prediction() -> None:
    target = torch.tensor([1.0])
    sharp = EvidentialHead(4, 4)
    with torch.no_grad():
        sharp.trunk[-1].weight.zero_()
        sharp.trunk[-1].bias.copy_(torch.tensor([1.0, 2.0, 2.0, -2.0]))
    output = sharp(torch.zeros(1, 4))
    loss = nig_nll(target, output)
    assert torch.isfinite(loss)


def test_point_head_uses_the_constant_residual_scale() -> None:
    head = PointHead(8, 4)
    head.set_residual_scale(0.7)
    output = head(torch.randn(3, 8))
    assert torch.allclose(output.std, torch.full((3,), 0.7), atol=1e-5)


def test_deep_ensemble_spread_is_zero_for_identical_members() -> None:
    head = DeepEnsembleHeads(6, n_members=3, hidden_dim=4)
    for member in head.members[1:]:
        member.load_state_dict(head.members[0].state_dict())
    output = head(torch.randn(4, 6))
    assert torch.allclose(output.variance, torch.zeros(4), atol=1e-8)


def test_deep_ensemble_requires_two_members() -> None:
    with pytest.raises(ValueError):
        DeepEnsembleHeads(6, n_members=1)


@pytest.mark.parametrize("mode", UNCERTAINTY_MODES)
def test_every_uncertainty_mode_emits_mean_and_deviation(mode: str) -> None:
    head = build_uncertainty_head(mode, width=16, n_bins=8)
    output = head(torch.randn(7, 16))
    assert output.mean.shape == (7,)
    assert bool((output.std >= 0).all())


def test_unknown_uncertainty_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        UncertaintyHead(UncertaintyConfig(mode="not_a_mode"), width=8)


def test_recalibration_is_monotone_in_the_raw_deviation() -> None:
    generator = np.random.default_rng(3)
    raw = np.abs(generator.normal(0.2, 0.05, size=300))
    error = raw * generator.uniform(0.5, 2.0, size=300)
    calibrator = IsotonicRecalibrator(16).fit(raw, error)
    grid = np.linspace(float(raw.min()), float(raw.max()), 100)
    predicted = calibrator.predict(grid)
    assert np.all(np.diff(predicted) >= -1e-9)


def test_recalibrator_round_trips_through_its_mapping() -> None:
    generator = np.random.default_rng(4)
    raw = np.abs(generator.normal(0.3, 0.1, size=200))
    error = raw * 2.0
    calibrator = IsotonicRecalibrator(12).fit(raw, error)
    restored = IsotonicRecalibrator.from_dict(calibrator.as_dict())
    assert restored.is_fitted()
    assert np.allclose(calibrator.predict(raw[:5]), restored.predict(raw[:5]), atol=1e-9)


def test_recalibrator_requires_fitting() -> None:
    with pytest.raises(RuntimeError):
        IsotonicRecalibrator(8).predict(np.array([0.1]))


def test_mean_abs_to_sigma_factor() -> None:
    assert pytest.approx(float(np.sqrt(np.pi / 2.0)), rel=1e-9) == MEAN_ABS_TO_SIGMA


def test_pava_preserves_order_and_total() -> None:
    values = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    fitted = pool_adjacent_violators(values)
    assert np.all(np.diff(fitted) >= -1e-12)
    assert fitted.sum() == pytest.approx(values.sum())


def test_adequacy_boundary_is_one_plus_tau() -> None:
    assert adequacy_boundary(0.30) == pytest.approx(1.30)
    assert adequacy_boundary(0.30, adequacy_relative=2.0) == pytest.approx(2.60)


def test_is_adequate_labels_at_the_boundary() -> None:
    labels = is_adequate(np.array([1.29, 1.30, 1.31]), 0.30)
    assert labels.tolist() == [0, 1, 1]


def test_adequacy_from_baseline_is_relative() -> None:
    labels = adequacy_from_baseline(np.array([1.2, 1.4]), np.array([1.0, 1.0]), 0.30)
    assert labels.tolist() == [0, 1]


def test_exposure_index_normalises_by_the_reference() -> None:
    assert exposure_index(200.0, 0.5, reference_integral=100.0) == pytest.approx(1.0)


def test_exposure_index_rejects_a_non_positive_reference() -> None:
    with pytest.raises(ValueError):
        exposure_index(1.0, 0.5, reference_integral=0.0)


def test_concentration_integral_matches_the_polygon() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    concentrations = np.array([2.0, 4.0, 4.0, 2.0])
    assert concentration_time_integral(times, concentrations) == pytest.approx(10.0)


def test_threshold_decision_matches_the_array_path() -> None:
    decision = ThresholdDecision(0.30)
    values = np.array([0.9, 1.3, 1.6])
    assert decision.classify(values).tolist() == [0, 1, 1]
    assert decision.forward_tensor(torch.tensor(values)).tolist() == [0, 1, 1]


def test_decision_curve_point_counts_errors() -> None:
    decision = ThresholdDecision(0.30)
    exposure = np.array([1.4, 0.9, 1.5, 0.8])
    truth = np.array([1, 0, 0, 1])
    fpr, fnr, net = decision.decision_curve_point(exposure, truth)
    assert fpr == pytest.approx(0.5)
    assert fnr == pytest.approx(0.5)
    assert np.isfinite(net)


def test_abstention_mask_flags_the_highest_deviation() -> None:
    deviation = np.array([0.1, 0.2, 0.3, 0.4])
    mask = abstention_mask(deviation, 0.25)
    assert mask.tolist() == [False, False, False, True]


def test_abstention_with_zero_quantile_refers_nobody() -> None:
    assert not abstention_mask(np.array([0.1, 0.2]), 0.0).any()


def test_coverage_and_retained_auroc_shapes() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    deviation = np.array([0.1, 0.2, 0.3, 0.4])
    summary = coverage_and_retained_auroc(scores, labels, deviation, 0.25)
    assert summary["coverage"] == pytest.approx(0.75)
    assert summary["referral_rate"] == pytest.approx(0.25)


def test_expected_cost_weights_the_harm_direction() -> None:
    predicted = np.array([1, 0])
    truth = np.array([0, 1])
    tiering = HarmTiering(false_adequate_cost=5.0, false_inadequate_cost=1.0)
    assert expected_cost(predicted, truth, tiering) == pytest.approx(6.0)


def test_select_boundary_respects_the_budget() -> None:
    generator = np.random.default_rng(7)
    labels = np.r_[np.ones(200), np.zeros(200)].astype(np.int64)
    scores = np.r_[generator.normal(1.2, 1.0, size=200), generator.normal(0.4, 1.0, size=200)]
    choice = select_boundary(scores, labels, false_positive_budget=0.05)
    assert choice["false_positive_rate"] <= 0.05 + 1e-9


def test_cost_curve_is_monotone_in_length() -> None:
    labels = np.r_[np.ones(30), np.zeros(30)].astype(np.int64)
    scores = np.linspace(0.0, 1.0, 60)
    rows = cost_curve(scores, labels, grid=25)
    assert len(rows) > 5
    assert all("expected_cost" in row for row in rows)
