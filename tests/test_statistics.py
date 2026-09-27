"""Statistics against independent references and closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from mechex.quantify.agreement import (
    cohen_kappa,
    correctness,
    kappa_with_ci,
    mcnemar_test,
)
from mechex.quantify.bootstrap import (
    bootstrap_ci,
    cluster_resample_indices,
    interval_contains,
    mean_over_repeats,
    overlap_fraction,
    paired_bootstrap_difference,
)
from mechex.quantify.calibration import (
    PlattCalibrator,
    brier_score,
    describe_bins,
    expected_calibration_error,
    exposure_probabilities,
    limits_for,
    reliability_curve,
)
from mechex.quantify.concordance import (
    ccc_components,
    ccc_decrement,
    lin_ccc,
    lin_ccc_ci,
)
from mechex.quantify.correlation import (
    moved_at_least,
    permutation_p_value,
    rank_movement,
    rank_positions,
    spearman_rho,
    spearman_with_permutation,
)
from mechex.quantify.delong import delong_test
from mechex.quantify.fold import (
    fold_error_distribution,
    fold_error_report,
    geometric_fold_error,
    zone_position,
)
from mechex.quantify.mixed import mixed_effects_logistic, skill_gradient
from mechex.quantify.multiplicity import (
    benjamini_hochberg,
    bonferroni,
    family_size,
    holm_bonferroni,
)
from mechex.quantify.netbenefit import (
    decision_curve,
    maximum_net_benefit_advantage,
    net_benefit,
    stable_around,
    treat_all_net_benefit,
)
from mechex.quantify.roc import (
    auroc,
    confusion_at,
    operating_point_at_fpr,
    rank_average,
    roc_curve,
    youden_threshold,
)
from mechex.quantify.shift import (
    DEFAULT_BANDING,
    Banding,
    degradation,
    degradation_points,
    shift_row,
    within_consistency_limit,
    worst_case,
)


def test_rank_average_shares_tied_positions() -> None:
    assert rank_average(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_auroc_matches_scikit_learn() -> None:
    generator = np.random.default_rng(1)
    scores = generator.normal(size=300)
    labels = (scores + generator.normal(0.7, 1.0, size=300) > 0.4).astype(np.int64)
    assert auroc(scores, labels) == pytest.approx(float(roc_auc_score(labels, scores)), abs=1e-10)


def test_auroc_is_undefined_without_both_classes() -> None:
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_roc_curve_endpoints() -> None:
    fpr, tpr, thresholds = roc_curve(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1]))
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0)
    assert thresholds[0] == np.inf


def test_operating_point_respects_the_budget() -> None:
    labels = np.r_[np.ones(50), np.zeros(50)].astype(np.int64)
    scores = np.linspace(0.0, 1.0, 100)
    point = operating_point_at_fpr(scores, labels, 0.10)
    assert point["fpr"] <= 0.10 + 1e-9


def test_youden_and_confusion_counts() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    worst = confusion_at(scores, labels, 1.0)
    assert worst == {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 2,
    }
    assert 0.0 <= youden_threshold(scores, labels)["youden"] <= 1.0


def test_lin_ccc_hand_formula() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.1, 1.9, 3.2, 3.8])
    mean_x, mean_y = x.mean(), y.mean()
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    denominator = float(
        ((x - mean_x) ** 2).mean() + ((y - mean_y) ** 2).mean() + (mean_x - mean_y) ** 2
    )
    assert lin_ccc(x, y) == pytest.approx(2.0 * covariance / denominator)


def test_lin_ccc_of_identical_vectors_is_one() -> None:
    values = np.linspace(0.0, 1.0, 25)
    assert lin_ccc(values, values) == pytest.approx(1.0, abs=1e-12)


def test_ccc_components_decompose() -> None:
    values = np.linspace(0.0, 1.0, 20)
    components = ccc_components(values, values + 0.5)
    assert components["ccc"] < 1.0
    assert components["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert ccc_decrement(0.8, 0.5) == pytest.approx(0.3)


def test_ccc_interval_brackets_the_estimate() -> None:
    generator = np.random.default_rng(2)
    x = generator.normal(size=50)
    y = x + generator.normal(0.0, 0.3, size=50)
    interval = lin_ccc_ci(x, y)
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]


def test_delong_recovers_both_aurocs() -> None:
    generator = np.random.default_rng(3)
    labels = np.r_[np.ones(70), np.zeros(90)].astype(np.int64)
    first = generator.normal(0.7, 1.0, size=160)
    second = generator.normal(0.3, 1.0, size=160)
    result = delong_test(labels, first, second)
    assert result.auc_a == pytest.approx(auroc(first, labels), abs=1e-10)
    assert result.auc_b == pytest.approx(auroc(second, labels), abs=1e-10)
    assert 0.0 <= result.p_value <= 1.0


def test_delong_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        delong_test(np.ones(4, dtype=np.int64), np.zeros(4), np.ones(4))


def test_bootstrap_interval_brackets_the_estimate() -> None:
    values = np.random.default_rng(4).normal(size=120)
    result = bootstrap_ci(lambda index: float(values[index].mean()), values.size, 200, seed=1)
    assert result.lower <= result.estimate <= result.upper
    assert result.as_dict()["n_resamples"] == 200


def test_cluster_resample_keeps_clusters_together() -> None:
    clusters = np.array([0, 0, 1, 2, 2, 2])
    draws = cluster_resample_indices(clusters, 5, seed=2)
    assert len(draws) == 5
    membership: dict[int, set[int]] = {}
    for index, label in enumerate(clusters.tolist()):
        membership.setdefault(int(label), set()).add(index)
    for row in draws:
        selected = set(row.tolist())
        assert selected <= set(range(clusters.size))
        for members in membership.values():
            assert members <= selected or not (members & selected)


def test_paired_bootstrap_difference_interval() -> None:
    generator = np.random.default_rng(5)
    values = generator.normal(size=80)

    def statistic(index: np.ndarray) -> float:
        return float(values[index].mean()) - 0.1

    result = paired_bootstrap_difference(statistic, values.size, 150, seed=3)
    assert result.lower <= result.upper


def test_interval_helpers() -> None:
    assert interval_contains(0.1, 0.4, 0.3)
    assert not interval_contains(0.1, 0.4, 0.5)
    assert overlap_fraction((0.0, 1.0), (0.5, 1.5)) == pytest.approx(0.5)
    assert overlap_fraction((0.0, 1.0), (2.0, 3.0)) == 0.0
    assert mean_over_repeats(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)


def test_expected_calibration_error_hand_computation() -> None:
    probabilities = np.array([0.05, 0.15, 0.85, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probabilities, labels, 2) == pytest.approx(0.10)


def test_reliability_curve_and_bins() -> None:
    probabilities = np.linspace(0.0, 1.0, 40)
    labels = (probabilities > 0.5).astype(np.int64)
    curve = reliability_curve(probabilities, labels, 5)
    assert len(curve) == 5
    described = describe_bins(probabilities, labels, 5)
    assert described["count"].sum() == 40
    assert np.allclose(limits_for(5), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_brier_and_platt_and_exposure_probabilities() -> None:
    assert brier_score(np.array([0.0, 1.0]), np.array([0, 1])) == pytest.approx(0.0)
    generator = np.random.default_rng(6)
    scores = generator.normal(size=200)
    labels = (scores > 0).astype(np.int64)
    calibrator = PlattCalibrator().fit(scores, labels)
    probabilities = calibrator.predict(scores)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    mapped = exposure_probabilities(scores, 0.0)
    assert mapped.shape == scores.shape


def test_geometric_fold_error_hand_value() -> None:
    assert geometric_fold_error(np.array([2.0]), np.array([1.0])) == pytest.approx(2.0)
    assert geometric_fold_error(np.array([0.5]), np.array([1.0])) == pytest.approx(2.0)


def test_fold_distribution_and_zone() -> None:
    predicted = np.array([1.0, 2.0, 0.5, 4.0])
    reference = np.array([1.0, 1.0, 1.0, 1.0])
    distribution = fold_error_distribution(predicted, reference)
    assert 0.0 <= distribution["within_two_fold"] <= 1.0
    assert zone_position(1.0) == "below"
    assert zone_position(2.0) == "inside"
    assert zone_position(5.0) == "above"
    assert fold_error_report(predicted, reference)["zone_position"] in {"below", "inside", "above"}


def test_kappa_matches_scikit_learn() -> None:
    first = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    second = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    assert cohen_kappa(first, second).kappa == pytest.approx(
        float(cohen_kappa_score(first, second)), abs=1e-12
    )


def test_kappa_interval_and_correctness() -> None:
    first = np.random.default_rng(7).integers(0, 2, size=200)
    second = np.where(np.random.default_rng(8).uniform(size=200) < 0.3, 1 - first, first)
    interval = kappa_with_ci(first, second)
    assert interval["lower"] <= interval["kappa"] <= interval["upper"]
    correct = correctness(np.array([0.9, 0.1]), np.array([1, 0]), 0.5)
    assert correct.tolist() == [True, True]


def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    first = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    second = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    result = mcnemar_test(first, second, exact=True)
    discordant = result.discordant_a + result.discordant_b
    hand = float(
        min(
            1.0,
            2.0 * stats.binom.cdf(min(result.discordant_a, result.discordant_b), discordant, 0.5),
        )
    )
    assert result.p_value == pytest.approx(hand)


def test_mcnemar_without_discordance() -> None:
    flags = np.array([True, False, True])
    result = mcnemar_test(flags, flags)
    assert result.p_value == 1.0


def test_benjamini_hochberg_and_holm() -> None:
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.075, 0.2, 0.31, 0.42])
    bh = benjamini_hochberg(p_values, 0.05)
    adjusted = np.asarray(bh["adjusted"])
    ordered = adjusted[np.argsort(p_values)]
    assert np.all(np.diff(ordered) >= -1e-12)
    assert bonferroni(p_values, 0.05)["adjusted"][0] == pytest.approx(0.01)
    assert holm_bonferroni(p_values, 0.05)["adjusted"].min() == pytest.approx(0.01, abs=1e-9)
    assert family_size(0) == 1


def test_multiplicity_rejects_an_invalid_interval() -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([1.5]))


def test_spearman_matches_scipy() -> None:
    generator = np.random.default_rng(9)
    first = generator.normal(size=50)
    second = 0.4 * first + generator.normal(size=50)
    assert spearman_rho(first, second) == pytest.approx(
        float(stats.spearmanr(first, second).statistic)
    )


def test_permutation_p_value_is_bounded() -> None:
    generator = np.random.default_rng(10)
    first = generator.normal(size=20)
    second = 0.9 * first
    p_value = permutation_p_value(first, second, n_permutations=400, seed=1)
    assert 0.0 < p_value <= 1.0
    result = spearman_with_permutation(first, second, n_permutations=200, seed=2)
    assert result.n == 20


def test_rank_positions_and_movement() -> None:
    assert rank_positions(np.array([0.2, 0.9, 0.5])).tolist() == [3, 1, 2]
    assert rank_movement(np.array([0.2, 0.9, 0.5]), np.array([0.9, 0.2, 0.5])).tolist() == [
        -2,
        2,
        0,
    ]
    assert moved_at_least(np.array([0.2, 0.9, 0.5]), np.array([0.9, 0.2, 0.5]), 2) == 2


def test_mixed_effects_agrees_with_a_plain_logistic_on_clustered_data() -> None:
    """With a small true random effect the contrast lands near the pooled logistic fit.

    The reference is an independent Newton fit on the same records, so the comparison does
    not reuse the mixed-effects likelihood.
    """
    generator = np.random.default_rng(11)
    n_readers, per_reader = 12, 25
    readers = np.repeat(np.arange(n_readers), per_reader)
    intercepts = generator.normal(0.0, 0.25, size=n_readers)[readers]
    predictor = generator.normal(size=n_readers * per_reader)
    logits = 0.3 + 1.0 * predictor + intercepts
    outcome = (generator.uniform(size=logits.size) < 1.0 / (1.0 + np.exp(-logits))).astype(
        np.float64
    )
    result = mixed_effects_logistic(predictor, outcome, readers)
    design = np.column_stack([np.ones(logits.size), predictor])
    coefficients = np.zeros(2)
    for _ in range(200):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(design @ coefficients, -30.0, 30.0)))
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = design.T @ (probabilities - outcome) / logits.size
        hessian = design.T @ (design * weights[:, None]) / logits.size + 1e-6 * np.eye(2)
        coefficients -= np.linalg.solve(hessian, gradient)
    assert abs(result.coefficient - float(coefficients[1])) < 0.25
    assert 0.0 < result.random_effect_sd < 2.0
    assert result.n_groups == n_readers


def test_skill_gradient_reports_three_gains() -> None:
    correctness = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=np.float64)
    assisted = np.array([False, True, False, True, False, True, False, True])
    skill = np.linspace(0.0, 1.0, 8)
    gradient = skill_gradient(correctness, assisted, skill)
    assert set(gradient) == {"lowest_third_gain", "highest_third_gain", "overall_gain"}


def test_net_benefit_and_decision_curve() -> None:
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2])
    assert net_benefit(probabilities, labels, 0.5) > 0
    curve = decision_curve(probabilities, labels)
    assert len(curve) == 99
    assert all(row["treat_none"] == 0.0 for row in curve)
    assert treat_all_net_benefit(labels, 0.5) == pytest.approx(0.0)
    assert maximum_net_benefit_advantage(probabilities, labels) > 0
    assert set(stable_around(probabilities, labels, 0.5)) == {
        "lower_threshold",
        "upper_threshold",
        "minimum_net_benefit",
        "maximum_net_benefit",
        "net_benefit_range",
    }


def test_shift_banding() -> None:
    assert DEFAULT_BANDING.classify(4.1) == "graceful"
    assert DEFAULT_BANDING.classify(5.8) == "moderate"
    assert DEFAULT_BANDING.classify(15.0) == "severe"
    assert DEFAULT_BANDING.classify(25.0) == "beyond_severe"
    assert degradation(0.9, 0.85) == pytest.approx(-0.05)
    assert degradation_points(0.9, 0.85) == pytest.approx(-5.0)


def test_shift_row_and_consistency_limit() -> None:
    row = shift_row("Temporal", 0.826, 0.900, 0.786, 0.842)
    assert row["classification"] == "moderate"
    assert row["ccc_points"] == pytest.approx(-4.0)
    assert within_consistency_limit(0.826, 0.800)
    assert not within_consistency_limit(0.826, 0.700)
    rows = [
        {"shift": "a", "auroc_points": -1.0},
        {"shift": "b", "auroc_points": -7.0},
    ]
    assert worst_case(rows)["shift"] == "b"
    assert Banding().classify(0.0) == "graceful"
