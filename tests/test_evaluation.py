"""Evaluation layer: external site, strata, shifts, abstention, readers, ranking, deployment."""

from __future__ import annotations

import numpy as np
import pytest

from mechex.assess.abstain import (
    abstention_curve,
    referral_set,
    retention_at,
    threshold_consistency,
)
from mechex.assess.external import (
    criterion_outcome,
    evaluate_external,
    external_slice,
    intervals_overlap,
    paired_concordance_contrast,
    paired_discrimination_contrast,
    repeat_summary,
)
from mechex.assess.human_model import (
    ARMS,
    MODEL_ARM_IDENTIFIER,
    arm_accuracy,
    arms_summary,
    inter_reader_agreement,
    reader_fixture,
    skill_gradient,
    team_contrast,
)
from mechex.assess.ranking import (
    compare_rankings,
    direction_of_movement,
    movement_table,
    per_molecule_exposure,
    ranking_summary,
    shortlist_effect,
)
from mechex.assess.shift import (
    SHIFT_NAMES,
    dominant_axis,
    evaluate_shift,
    generalisation_table,
    inconsistency_across_shifts,
    shift_masks,
    single_axis_ordering,
    temporal_exceeds_region,
)
from mechex.assess.silent_mode import (
    PRESPECIFIED_ENDPOINTS,
    deployment_fixture,
    endpoint_coverage,
    missed_candidates,
    per_site_adoption,
    shortlist_agreement,
    summarise_deployment,
    weekly_adoption,
)
from mechex.assess.strata import (
    PRESPECIFIED_THRESHOLD,
    STRATUM_AXES,
    consistency_summary,
    derive_axes,
    evaluate_strata,
    multiplicity_adjustment,
    stratum_sizes,
)


def _fixture_external(staged):
    indices = staged.split.external
    cohort = staged.cohort
    reference = np.asarray(cohort.exposure_reference, dtype=np.float64)[indices]
    generator = np.random.default_rng(0)
    prediction = reference * np.exp(generator.normal(0.0, 0.15, size=reference.size))
    patients = np.asarray(cohort.patient_id, dtype=object)[indices]
    deviation = np.abs(generator.normal(0.1, 0.02, size=reference.size))
    return prediction, reference, patients, deviation


def test_external_metrics_are_finite_and_bracketed(staged) -> None:
    prediction, reference, patients, _ = _fixture_external(staged)
    metrics = evaluate_external("X", "probe", prediction, reference, patients, n_resamples=100)
    assert np.isfinite(metrics.ccc) and np.isfinite(metrics.auroc)
    assert metrics.ccc_lower <= metrics.ccc <= metrics.ccc_upper
    assert metrics.n == reference.size


def test_external_metric_dict_shape(staged) -> None:
    prediction, reference, patients, _ = _fixture_external(staged)
    payload = evaluate_external(
        "X", "probe", prediction, reference, patients, n_resamples=60
    ).as_dict()
    assert set(payload) == {
        "id",
        "method",
        "ccc",
        "ccc_interval",
        "auroc",
        "auroc_interval",
        "fold_error",
        "inconsistency_percent",
        "n",
    }


def test_criterion_outcome_places_the_margins() -> None:
    from mechex.assess.external import ExternalMetrics

    metrics = ExternalMetrics("X", "probe", 0.82, 0.80, 0.84, 0.90, 0.88, 0.92, 1.4, 2.0, 100)
    outcome = criterion_outcome(metrics, 0.75, 0.85)
    assert outcome["both_met"]
    assert outcome["ccc_margin"] == pytest.approx(0.07, abs=1e-9)
    assert outcome["auroc_margin"] == pytest.approx(0.05, abs=1e-9)


def test_paired_contrasts_are_consistent(staged) -> None:
    prediction, reference, patients, _ = _fixture_external(staged)
    labels = (reference >= 1.3).astype(np.int64)
    delong = paired_discrimination_contrast(labels, prediction, prediction * 0.9)
    assert delong["difference"] == pytest.approx(0.0, abs=1e-9)
    bootstrap = paired_concordance_contrast(
        prediction, prediction, reference, patients, n_resamples=80
    )
    assert abs(bootstrap["difference"]) < 1e-9


def test_intervals_overlap_and_repeat_summary(staged) -> None:
    prediction, reference, patients, _ = _fixture_external(staged)
    first = evaluate_external("A", "a", prediction, reference, patients, n_resamples=60)
    second = evaluate_external("B", "b", prediction * 0.7, reference, patients, n_resamples=60)
    overlap = intervals_overlap(first, second)
    assert 0.0 <= overlap["concordance_overlap"] <= 1.0
    summary = repeat_summary([{"ccc": 0.8}, {"ccc": 0.82}])
    assert summary["ccc_mean"] == pytest.approx(0.81)


def test_external_slice_columns(staged) -> None:
    slice_columns = external_slice(staged.cohort, staged.split.external)
    assert set(slice_columns) == {"reference", "patients", "labels", "molecule_index"}
    assert slice_columns["labels"].size == slice_columns["reference"].size


def test_stratified_pipeline_reproduces_reported_sizes(staged) -> None:
    indices = staged.split.external
    reference = np.asarray(staged.cohort.exposure_reference, dtype=np.float64)[indices]
    axes = derive_axes(staged.cohort, indices)
    rows = evaluate_strata(reference, reference, axes, overall_ccc=1.0)
    assert len(rows) >= 8
    sizes = stratum_sizes(staged.cohort, indices)
    assert sum(sizes["severity"].values()) == 1598
    summary = consistency_summary(rows)
    assert summary["all_above_threshold"] in {True, False}


def test_multiplicity_adjustment_over_strata(staged) -> None:
    indices = staged.split.external
    reference = np.asarray(staged.cohort.exposure_reference, dtype=np.float64)[indices]
    axes = derive_axes(staged.cohort, indices)
    adjusted = multiplicity_adjustment(reference, reference, axes, fdr=0.05)
    assert adjusted["family_size"] == float(len(adjusted["p_values"]))
    assert np.all(np.asarray(adjusted["adjusted"]) <= 1.0)


def test_declared_constant_is_used_by_the_strata_layer() -> None:
    assert PRESPECIFIED_THRESHOLD == 0.75
    assert "severity" in {name for name, _ in STRATUM_AXES}


def test_shift_masks_and_table(staged) -> None:
    masks = shift_masks(staged.cohort, staged.split.mask("external", len(staged.cohort)))
    assert set(masks) == set(SHIFT_NAMES)
    reference = np.asarray(staged.cohort.exposure_reference, dtype=np.float64)
    rows = generalisation_table(reference, reference, masks)
    assert len(rows) == 5
    assert rows[0]["classification"] == "reference"
    assert all(
        row["classification"] in {"reference", "graceful", "moderate", "severe"} for row in rows
    )


def test_shift_evaluation_and_ordering(staged) -> None:
    masks = shift_masks(staged.cohort, staged.split.mask("external", len(staged.cohort)))
    reference = np.asarray(staged.cohort.exposure_reference, dtype=np.float64)
    evaluation = evaluate_shift("cross_region", reference, reference, masks["cross_region"])
    assert evaluation.n > 0
    rows = generalisation_table(reference, reference, masks)
    assert single_axis_ordering(rows)["region_points"] >= 0.0
    assert isinstance(temporal_exceeds_region(rows), bool)
    assert "shift" in dominant_axis(rows)


def test_inconsistency_across_shifts(staged) -> None:
    n = len(staged.cohort)
    masks = shift_masks(staged.cohort, staged.split.mask("external", n))
    residual = np.full(n, 0.05)
    implied = np.full(n, 0.5)
    fractions = np.linspace(0.0, 1.0, n)
    exposures = np.linspace(0.0, 1.0, n)
    rates = inconsistency_across_shifts(residual, implied, fractions, exposures, masks, 0.2)
    assert set(rates) == set(SHIFT_NAMES)
    assert all(value >= 0.0 for value in rates.values())


def test_abstention_gain_is_positive_on_sorted_signal() -> None:
    labels = np.r_[np.ones(200), np.zeros(200)].astype(np.int64)
    generator = np.random.default_rng(1)
    scores = np.r_[generator.normal(1.0, 1.0, 200), generator.normal(0.0, 1.0, 200)]
    deviation = np.r_[generator.normal(0.4, 0.05, 200), generator.normal(1.4, 0.05, 200)]
    summary = retention_at(scores, labels, deviation, 0.10)
    assert summary["referral_rate"] == pytest.approx(0.10, abs=0.02)
    assert summary["gain"] > 0
    curve = abstention_curve(scores, labels, deviation)
    assert len(curve) == 5
    assert referral_set(deviation, 0.10).dtype == bool
    assert threshold_consistency(scores, labels, deviation, 0.10)["gain_positive"] == 1.0


def test_ranking_comparison_counts_moves() -> None:
    exposure = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    mechanism = exposure[::-1].copy()
    comparison = compare_rankings(exposure, mechanism)
    assert comparison.movement.tolist() == [-5, -3, -1, 1, 3, 5]
    summary = ranking_summary(comparison, movement_threshold=5)
    assert summary["moved_at_least"] == 2
    assert len(movement_table(comparison)) == 6
    assert set(direction_of_movement(comparison)) == {"largest_upward", "largest_downward"}


def test_shortlist_effect_matches_a_hand_enumeration() -> None:
    exposure = np.array([3.0, 2.5, 2.2, 0.1, 0.1])
    mechanism = np.array([0.1, 0.1, 0.1, 9.0, 8.0])
    effect = shortlist_effect(exposure, mechanism, shortlist_size=2, adequacy_threshold=1.0)
    assert effect["excluded_from_mechanism_top"] == 2
    assert effect["promoted_from_outside_mechanism_top"] == 2


def test_per_molecule_exposure_averages() -> None:
    values = per_molecule_exposure(np.array([1.0, 3.0, 5.0]), np.array([0, 0, 1]), 2)
    assert values[0] == pytest.approx(2.0)
    assert values[1] == pytest.approx(5.0)


def test_reader_table_and_arms(staged) -> None:
    table = reader_fixture(n_cases=40)
    accuracy = arm_accuracy(table)
    assert set(accuracy) == set(ARMS)
    assert int(np.unique(table.reader).size) == 13  # twelve readers plus the model arm
    assert MODEL_ARM_IDENTIFIER in set(table.reader.tolist())
    summary = arms_summary(table)
    assert -1.0 <= float(summary["agreement"]["reader_only"]["mean_kappa"]) <= 1.0


def test_reader_contrast_and_gradient() -> None:
    table = reader_fixture(n_cases=50)
    contrast = team_contrast(table)
    assert "coefficient" in contrast and np.isfinite(contrast["coefficient"])
    gradient = skill_gradient(table)
    assert gradient["lowest_third_gain"] > gradient["highest_third_gain"] - 1.0
    assert inter_reader_agreement(table, "reader_only")["n_pairs"] > 0


def test_silent_mode_endpoints_match_a_hand_computation() -> None:
    log = deployment_fixture(n_assessments=600)
    report = summarise_deployment(log)
    assert report.n_assessments == 600
    assert report.consulted_percent == pytest.approx(
        100.0 * float(np.mean(log.recommendation_consulted)), abs=1e-9
    )
    assert report.time_reduction_percent > 0
    assert len(weekly_adoption(log)) == report.n_weeks
    assert len(per_site_adoption(log)) == report.n_sites
    coverage = endpoint_coverage(report)
    assert set(coverage) == set(PRESPECIFIED_ENDPOINTS)


def test_silent_mode_shortlist_agreement_and_missed() -> None:
    agreement = shortlist_agreement({1, 2, 3}, {1, 9}, {1, 2, 3, 4})
    assert agreement["model_agreement"] == pytest.approx(0.75)
    assert agreement["mechanism_agreement"] == pytest.approx(0.25)
    assert missed_candidates({1, 2, 3, 4}, {1, 2}, {1, 2, 3, 4}) == 2
