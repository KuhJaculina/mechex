"""Cohort layer: fixture, transforms, linkage, reference standard and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from mechex.cohort.construction import build_cohort, cohort_fingerprint, stage
from mechex.cohort.fixture import (
    EXTERNAL_CLASS_COUNTS,
    EXTERNAL_SEVERITY_COUNTS,
    SITE_SIZES,
    build_cohort_fixture,
    build_molecule_panel,
    strata_counts,
)
from mechex.cohort.linkage import excluded_records, imaging_table, link, study_key
from mechex.cohort.partition import (
    check_leakage,
    partition,
    partition_summary,
    temporal_partition,
    vendor_partition,
)
from mechex.cohort.reference import (
    ReferenceStandard,
    adequacy_rate,
    adjudicate,
    adjudication_fixture,
    freeze_threshold,
    tier_counts,
    tier_sensitivity,
    tier_subsets,
)
from mechex.cohort.transforms import (
    dichotomise,
    exposure_index,
    molecule_reference,
    perfused_fraction_to_delivery,
    relative_exposure,
    time_align,
    transform_chain,
    unit_normalise_concentrations,
)
from mechex.settings import CohortConfig, ProtocolConfig


def test_fixture_carries_the_reported_shape() -> None:
    cohort = build_cohort_fixture(CohortConfig())
    assert len(cohort) == sum(SITE_SIZES.values())
    assert cohort.group_counts("site") == SITE_SIZES
    assert int(np.unique(cohort.molecule_index).size) == 27


def test_fixture_is_deterministic() -> None:
    first = build_cohort_fixture(CohortConfig(fixture_seed=5))
    second = build_cohort_fixture(CohortConfig(fixture_seed=5))
    assert np.allclose(first.exposure_reference, second.exposure_reference)
    assert np.array_equal(first.molecule_index, second.molecule_index)


def test_external_site_marginals_match_the_reported_strata() -> None:
    cohort = build_cohort_fixture(CohortConfig())
    severity = strata_counts(cohort, "severity")
    drug_class = strata_counts(cohort, "drug_class")
    assert severity == EXTERNAL_SEVERITY_COUNTS
    assert drug_class == EXTERNAL_CLASS_COUNTS


def test_molecule_panel_is_consistent() -> None:
    panel = build_molecule_panel()
    assert len(panel.graphs) == 27
    assert panel.descriptor_matrix.shape[0] == 27
    assert len(panel.drug_class) == 27
    assert bool((panel.partition_truth > 0).all())


def test_unit_normalisation_scales_concentrations() -> None:
    values = np.array([1000.0, 2000.0])
    assert unit_normalise_concentrations(values).tolist() == [1.0, 2.0]


def test_unit_normalisation_rejects_a_bad_scale() -> None:
    with pytest.raises(ValueError):
        unit_normalise_concentrations(np.array([1.0]), unit_scale=0.0)


def test_time_alignment_shifts_and_sorts() -> None:
    times = np.array([4.0, 1.0, 2.0])
    concentrations = np.array([3.0, 1.0, 2.0])
    shifted, ordered = time_align(times, concentrations, 1.0)
    assert shifted.tolist() == [0.0, 1.0, 3.0]
    assert ordered.tolist() == [1.0, 2.0, 3.0]


def test_delivery_term_clips_to_the_admissible_interval() -> None:
    values = perfused_fraction_to_delivery(np.array([0.0, 0.5, 1.0]))
    assert values.tolist() == [0.05, 0.5, 0.95]


def test_exposure_index_is_relative_and_linear_in_delivery() -> None:
    assert exposure_index(np.array([200.0]), np.array([0.5]), 100.0).tolist() == [1.0]
    assert exposure_index(np.array([200.0]), np.array([1.0]), 100.0).tolist() == [2.0]


def test_dichotomise_uses_the_frozen_threshold() -> None:
    assert dichotomise(np.array([1.29, 1.31]), 0.30).tolist() == [0, 1]


def test_transform_chain_reproduces_a_hand_computation() -> None:
    times = np.array([0.0, 1.0, 2.0])
    concentrations = np.array([1000.0, 2000.0, 1000.0])
    chain = transform_chain(times, concentrations, np.array([0.5]), 0.0, np.array([1.0]))
    hand_integral = float(np.trapezoid(concentrations / 1000.0, times))
    assert float(chain["integral"][0]) == pytest.approx(hand_integral)
    assert float(chain["exposure"][0]) == pytest.approx(hand_integral * 0.5)


def test_molecule_reference_uses_only_the_development_split() -> None:
    exposure = np.array([1.0, 3.0, 5.0, 7.0])
    molecules = np.array([0, 0, 1, 1])
    development = np.array([True, False, True, False])
    reference = molecule_reference(exposure, molecules, 2, development)
    assert reference.tolist() == [1.0, 5.0]


def test_relative_exposure_divides_by_the_reference() -> None:
    values = relative_exposure(np.array([2.0, 4.0]), np.array([0, 1]), np.array([2.0, 8.0]))
    assert values.tolist() == [1.0, 0.5]


def test_tables_are_joinable_and_consistent() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=11))
    tables, report = link(cohort)
    assert report.is_consistent
    assert report.duplicate_keys == 0
    assert "outcome.exposure_reference" in tables
    assert imaging_table(cohort)["key"][0] == study_key(
        str(cohort.site[0]), str(cohort.patient_id[0]), 0
    )


def test_exclusion_rules_count_by_brute_force() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=12))
    counts = excluded_records(cohort)
    occlusion = np.asarray(cohort.occlusion, dtype=np.float64)
    assert counts["non_diagnostic_cta"] == int(np.sum(occlusion > 0.95))
    assert counts["retained"] == len(cohort)


def test_partition_is_patient_level_and_clean() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=13))
    split = partition(cohort, ProtocolConfig(), seed=13)
    report = check_leakage(cohort, split)
    assert report.clean
    assert split.external.size == SITE_SIZES["C"]


def test_partition_summary_reports_site_composition() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=14))
    split = partition(cohort, ProtocolConfig(), seed=14)
    summary = partition_summary(cohort, split)
    assert "sites" in summary
    sizes = summary["sizes"]
    assert sizes["external"] == SITE_SIZES["C"]


def test_temporal_and_vendor_partitions_are_complementary() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=15))
    development, shifted = temporal_partition(cohort)
    assert int(development.sum()) + int(shifted.sum()) == len(cohort)
    development_v, shifted_v = vendor_partition(cohort)
    assert int(development_v.sum()) + int(shifted_v.sum()) == len(cohort)


def test_tier_counts_and_subsets() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=16))
    counts = tier_counts(cohort)
    assert set(counts) == {"direct_assay", "microdialysis", "composite_index"}
    subsets = tier_subsets(cohort)
    assert sum(len(item) for item in subsets.values()) == len(cohort)


def test_adjudication_recovers_the_reported_kappa() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=17))
    first, second = adjudication_fixture(cohort, disagreement_rate=0.07)
    report = adjudicate(first, second, reported_kappa=0.86)
    assert 0.7 < report["observed_kappa"] <= 1.0
    assert report["n"] == len(cohort)


def test_threshold_freezing_and_adequacy_rate() -> None:
    exposure = np.array([0.5, 1.0, 1.5, 2.0])
    standard = freeze_threshold(exposure, 0.30)
    assert isinstance(standard, ReferenceStandard)
    assert standard.boundary == pytest.approx(1.30)
    assert adequacy_rate(exposure, standard) == pytest.approx(0.5)


def test_tier_sensitivity_reports_every_tier() -> None:
    cohort = build_cohort_fixture(CohortConfig(fixture_seed=18))
    exposure = np.asarray(cohort.exposure_reference, dtype=np.float64)
    labels = (exposure >= 1.3).astype(np.int64)
    report = tier_sensitivity(exposure, labels, np.asarray(cohort.reference_tier, dtype=np.int64))
    assert report
    for entry in report.values():
        assert entry["n"] > 0


def test_build_cohort_is_reproducible_and_reports_its_fingerprint() -> None:
    config = build_cohort.__module__ and None
    del config
    from pathlib import Path

    from mechex.settings import load_smoke_config

    smoke = load_smoke_config(Path(__file__).resolve().parents[1] / "configs")
    first = build_cohort(smoke, seed=smoke.cohort.fixture_seed)
    second = build_cohort(smoke, seed=smoke.cohort.fixture_seed)
    assert np.allclose(first.normalisation, second.normalisation)
    fingerprint = cohort_fingerprint(first)
    assert fingerprint["n_records"] == sum(SITE_SIZES.values())
    assert fingerprint["leakage_clean"] is True


def test_staging_writes_the_fixture(tmp_path) -> None:
    from pathlib import Path

    from mechex.settings import load_smoke_config

    smoke = load_smoke_config(Path(__file__).resolve().parents[1] / "configs")
    staged = build_cohort(smoke, seed=smoke.cohort.fixture_seed)
    written = stage(staged, tmp_path)
    assert written["cohort"].is_file()
    assert written["summary"].is_file()
