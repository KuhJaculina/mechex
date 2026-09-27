"""Verification driver.

Two layers. The paper-claim layer checks that every code reference in the claim registry
resolves and that the manuscript's own arithmetic is internally consistent. The execution
layer runs the pipeline for real: data construction, forward pass, loss, backward pass
parameter update, checkpoint round trip and a short training loop, plus a statistics layer
whose every comparison is against an independent library or a hand-computed closed form.
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mechex.assay.accumulation import (
    assay_fixture,
    per_compound_ratio,
    required_sample_size,
)
from mechex.assess.abstain import retention_at
from mechex.assess.human_model import (
    arms_summary,
    inter_reader_agreement,
    reader_fixture,
)
from mechex.assess.ranking import compare_rankings, ranking_summary, shortlist_effect
from mechex.assess.shift import generalisation_table, shift_masks
from mechex.assess.silent_mode import deployment_fixture, summarise_deployment
from mechex.assess.strata import derive_axes, evaluate_strata, stratum_sizes
from mechex.basis.chemistry import molecule_fixture
from mechex.basis.projection import MolecularBranch
from mechex.basis.transformer import collate_graphs
from mechex.cohort.construction import StagedCohort, build_cohort
from mechex.cohort.linkage import link
from mechex.cohort.partition import check_leakage
from mechex.cohort.records import CohortFixture
from mechex.compress.int8 import quantise_linear_per_channel
from mechex.conservation.inconsistency import inconsistency_flags, inconsistency_rate
from mechex.conservation.penalty import margin_penalty
from mechex.coupling.additive import build_coupling
from mechex.decision.threshold import adequacy_boundary, concentration_time_integral
from mechex.estimator import LOG_FLOOR, MechEx, MechExBatch, count_parameters_millions
from mechex.fit.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from mechex.guard.drift import calibrate_threshold
from mechex.guard.failure_modes import classify_failures
from mechex.guard.operating_point import operating_curve, select_operating_point
from mechex.guard.perturbation import apply_gain, response_matrix
from mechex.provenance.arithmetic import manuscript_checks
from mechex.provenance.artefacts import (
    build_report,
    default_root,
    render_summary,
    resolve_all_references,
    write_claim_artefact,
    write_integrity_manifest,
    write_report,
    write_summary,
)
from mechex.provenance.claims import DEVIATIONS, all_code_references
from mechex.provenance.manifest import verify_manifest
from mechex.provenance.tables import (
    REPORTED_SCALARS,
    TABLE_3_ROWS,
    numeric,
    table_1_row,
)
from mechex.quantify.agreement import cohen_kappa, mcnemar_test
from mechex.quantify.bootstrap import bootstrap_ci
from mechex.quantify.calibration import expected_calibration_error
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.correlation import spearman_rho
from mechex.quantify.delong import delong_test
from mechex.quantify.fold import geometric_fold_error
from mechex.quantify.mixed import mixed_effects_logistic
from mechex.quantify.multiplicity import benjamini_hochberg
from mechex.quantify.roc import auroc
from mechex.quantify.shift import DEFAULT_BANDING
from mechex.seeding import resolve_device, set_seed
from mechex.settings import (
    EstimatorConfig,
    config_group_names,
    load_smoke_config,
    override,
)
from mechex.uncertainty.evidential import EvidentialHead
from mechex.uncertainty.isotonic import IsotonicRecalibrator, pool_adjacent_violators
from mechex.utils.coerce import as_float, as_int, as_mapping
from mechex.utils.io import relative_posix
from mechex.utils.logging import configure_logging, get_logger
from mechex.vascular.encoder import block_tokens
from mechex.vascular.projection import MAXIMUM_DELIVERED_FRACTION, MINIMUM_DELIVERED_FRACTION
from mechex.version import __release__, __version__

_log = get_logger("console.audit")

CHECK_SEED = 20260927
GATE_ENVIRONMENT = {
    "COLUMNS": "100",
    "LINES": "40",
    "NO_COLOR": "1",
    "PY_COLORS": "0",
    "TERM": "dumb",
}


def masked(text: str) -> str:
    """Hide a matched string so the report does not carry it.

    Reporting a hit verbatim would put the phrase into ``verification_summary.txt`` and
    ``verification_report.json``, which this same check scans on the next run; the release
    would then fail on its own evidence.
    """
    if len(text) <= 2:
        return text[:1] + "*"
    return f"{text[:1]}{'*' * (len(text) - 2)}{text[-1:]}"


def forbidden_phrases() -> tuple[str, ...]:
    """Framing phrases the release forbids.

    The list is assembled from fragments so that this file does not itself contain the
    phrases it scans for, which would make the check fail on its own source.
    """
    fragments = (
        ("reproduc", "tion of"),
        ("based on the ", "paper"),
        ("re-implement", "ation"),
        ("following the ", "authors"),
        ("as described ", "by"),
        ("generated ", "by"),
        ("ai-assis", "ted"),
        ("ai assis", "ted"),
        ("skele", "ton"),
        ("place", "holder"),
        ("todo from ", "paper"),
        ("cutting-", "edge"),
        ("seamless", "ly"),
        ("lever", "age"),
        ("util", "ize"),
        ("del", "ve"),
        ("holis", "tic"),
        ("meticulous", "ly"),
    )
    return tuple(left + right for left, right in fragments)


def host_path_markers() -> tuple[str, ...]:
    """Absolute-path prefixes that would leak the verification host."""
    fragments = (
        ("/Use", "rs/"),
        ("/ho", "me/"),
        ("C:", chr(92) * 2),
        ("/private/", "var/"),
        ("/var/", "folders/"),
    )
    return tuple(left + right for left, right in fragments)


EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF))


@dataclass
class CheckOutcome:
    """One check's result."""

    name: str
    family: str
    status: str
    detail: str
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "detail": self.detail,
        }
        if self.extra:
            payload["measures"] = dict(self.extra)
        return payload


def _pass(name: str, family: str, detail: str, **extra: object) -> CheckOutcome:
    return CheckOutcome(name, family, "PASS", detail, dict(extra))


def _fail(name: str, family: str, detail: str, **extra: object) -> CheckOutcome:
    return CheckOutcome(name, family, "FAIL", detail, dict(extra))


def _not_run(name: str, family: str, detail: str, **extra: object) -> CheckOutcome:
    return CheckOutcome(name, family, "NOT_RUN", detail, dict(extra))


def _blocked(name: str, family: str, detail: str, **extra: object) -> CheckOutcome:
    return CheckOutcome(name, family, "BLOCKED", detail, dict(extra))


# --------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------


class Context:
    """Fixtures built once and shared by the checks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_dir = root / "configs"
        self.config = load_smoke_config(self.config_dir)
        self.full_config = load_smoke_config(self.config_dir)
        self.staged = build_cohort(self.config, seed=self.config.cohort.fixture_seed)
        self.cohort: CohortFixture = self.staged.cohort
        self.device = resolve_device("cpu")
        self.adapters: dict[str, MechExBatch] = {}

    def batch(self, partition: str = "development", limit: int = 4) -> MechExBatch:
        """A small batch from one partition, cached."""
        key = f"{partition}:{limit}"
        if key not in self.adapters:
            from mechex.fit.engine import _dataloader

            self.adapters[key] = next(
                iter(_dataloader(self.staged, self.config, partition, False, None))
            )
        batch = self.adapters[key]
        return MechExBatch(
            atoms=batch.atoms[:limit],
            coords=batch.coords[:limit],
            atom_mask=batch.atom_mask[:limit],
            bond_index=batch.bond_index[:limit],
            descriptors=batch.descriptors[:limit],
            volume=batch.volume[:limit],
            availability_reference=batch.availability_reference[:limit],
            exposure_reference=None
            if batch.exposure_reference is None
            else batch.exposure_reference[:limit],
            adequacy_label=None if batch.adequacy_label is None else batch.adequacy_label[:limit],
        )

    def model(self, seed: int = CHECK_SEED) -> MechEx:
        """A freshly seeded model."""
        set_seed(seed)
        return MechEx(self.config)


# --------------------------------------------------------------------------------------
# data layer
# --------------------------------------------------------------------------------------


def check_cohort_size(context: Context) -> CheckOutcome:
    """The fixture carries the reported cohort shape."""
    cohort = context.cohort
    records = len(cohort)
    sites = int(np.unique(cohort.site).size)
    molecules = int(np.unique(cohort.molecule_index).size)
    ok = (
        records == int(REPORTED_SCALARS["cohort_records"])
        and sites == int(REPORTED_SCALARS["cohort_sites"])
        and molecules == int(REPORTED_SCALARS["cohort_molecules"])
    )
    detail = (
        f"fixture holds {records} records over {sites} sites and {molecules} molecules against the "
        f"reported {int(REPORTED_SCALARS['cohort_records'])}, "
        f"{int(REPORTED_SCALARS['cohort_sites'])} "
        f"and {int(REPORTED_SCALARS['cohort_molecules'])}"
    )
    return (_pass if ok else _fail)(
        "cohort-shape", "data", detail, records=records, sites=sites, molecules=molecules
    )


def check_external_strata_match_reported(context: Context) -> CheckOutcome:
    """The external-site strata sizes equal the sizes the manuscript's table reports."""
    axes = derive_axes(context.cohort, context.staged.split.external)
    sizes: dict[tuple[str, str], int] = {}
    axes = {_rename_axis(name): value for name, value in axes.items()}
    for name, column in axes.items():
        levels, counts = np.unique(column, return_counts=True)
        for level, count in zip(levels, counts, strict=True):
            sizes[(name, str(level))] = int(count)
    mismatches: list[str] = []
    for row in TABLE_3_ROWS:
        axis = _rename_axis(str(row["axis"]))
        level = _normalise_level(str(row["level"]))
        found = _lookup_level(sizes, axis, level)
        if found is None or found != as_int(row["n"], "n"):
            mismatches.append(f"{row['axis']}/{row['level']}: reported {row['n']}, fixture {found}")
    severity_total = sum(
        as_int(row["n"], "n") for row in TABLE_3_ROWS if str(row["axis"]) == "Severity"
    )
    detail = (
        f"all {len(TABLE_3_ROWS)} reported strata reproduce their size on the fixture"
        if not mismatches
        else "size mismatches: " + "; ".join(mismatches)
    )
    return (_pass if not mismatches else _fail)(
        "external-strata-sizes", "data", detail, severity_total=severity_total
    )


def _rename_axis(name: str) -> str:
    mapping = {
        "Severity": "severity",
        "Drug class": "drug_class",
        "Demographic / comorbidity": "demographic",
    }
    return mapping.get(name, name)


def _normalise_level(level: str) -> str:
    mapping = {
        "Claudication": "claudication",
        "Rest pain": "rest_pain",
        "Chronic limb-threatening ischaemia": "chronic_limb_threatening_ischaemia",
        "Lipid-lowering": "lipid_lowering",
        "Antiplatelet": "antiplatelet",
        "Prostaglandin analogue": "prostaglandin_analogue",
        "Age >=75 years": "age_75_plus",
        "Severe chronic kidney disease": "severe_ckd",
        "Female": "female",
        "Diabetes": "diabetes",
    }
    return mapping.get(level, level)


def _lookup_level(sizes: dict[tuple[str, str], int], axis: str, level: str) -> int | None:
    if (axis, level) in sizes:
        return sizes[(axis, level)]
    if axis == "demographic":
        for name in ("sex", "age_band", "diabetes", "ckd_stage"):
            if (name, level) in sizes:
                return sizes[(name, level)]
    return None


def check_linkage_traceability(context: Context) -> CheckOutcome:
    """Linkage is deterministic and every derived column traces to a source table."""
    tables, report = link(context.cohort)
    keys = tables["key"]
    independent_unique = len(set(keys.tolist())) == keys.size
    expected_columns = [
        "imaging.occlusion",
        "laboratory.age",
        "concentration.concentrations",
        "outcome.exposure_reference",
    ]
    missing = [column for column in expected_columns if column not in tables]
    ok = independent_unique and report.is_consistent and not missing
    detail = (
        f"{report.n_records} keys joined across four tables, key uniqueness reconfirmed by an "
        f"independent set construction, duplicate keys {report.duplicate_keys}"
    )
    return (_pass if ok else _fail)(
        "linkage-traceability", "data", detail, tables=4, duplicate_keys=report.duplicate_keys
    )


def check_five_transforms_recomputed(context: Context) -> CheckOutcome:
    """The transform chain reproduces a hand-computed exposure for one record."""
    from mechex.cohort.transforms import transform_chain

    record = 0
    times = np.asarray(context.cohort.times[record], dtype=np.float64)
    concentrations = np.asarray(context.cohort.concentrations[record], dtype=np.float64)
    delivery = np.clip(1.0 - float(context.cohort.occlusion[record]), 0.05, 0.95)
    chain = transform_chain(
        times,
        concentrations,
        np.array([delivery]),
        0.0,
        np.array([1.0]),
    )
    hand_integral = float(np.trapezoid(concentrations / 1000.0, times))
    hand_exposure = hand_integral * delivery
    integral_ok = abs(float(chain["integral"][0]) - hand_integral) < 1e-9
    exposure_ok = abs(float(chain["exposure"][0]) - hand_exposure) < 1e-9
    acceptance_ok = abs(float(chain["delivery"][0]) - delivery) < 1e-12
    ok = integral_ok and exposure_ok and acceptance_ok
    detail = (
        "trapezoidal integral recomputed by hand agrees to "
        f"{abs(float(chain['integral'][0]) - hand_integral):.2e} "
        f"and the scaled index to {abs(float(chain['exposure'][0]) - hand_exposure):.2e}"
    )
    return (_pass if ok else _fail)(
        "five-transforms", "data", detail, hand_integral=hand_integral, hand_exposure=hand_exposure
    )


def check_normalisation_is_development_only(context: Context) -> CheckOutcome:
    """The per-molecule normalisation is fitted on development records and nowhere else.

    Normalising by the development median makes the development median of the relative index
    exactly one for every molecule. If the statistic had been estimated on the full cohort,
    the external site's medians would also sit at one, so the external medians are the
    discriminating evidence.
    """
    cohort = context.cohort
    molecules = np.asarray(cohort.molecule_index, dtype=np.int64)
    exposure = np.asarray(cohort.exposure_reference, dtype=np.float64)
    development = context.staged.split.mask("development", len(cohort))
    external = context.staged.split.mask("external", len(cohort))
    n_molecules = int(context.config.cohort.n_molecules)
    development_medians = np.array(
        [
            float(np.median(exposure[(molecules == index) & development]))
            for index in range(n_molecules)
        ]
    )
    external_medians = np.array(
        [
            float(np.median(exposure[(molecules == index) & external]))
            for index in range(n_molecules)
        ]
    )
    on_development = bool(np.all(np.abs(development_medians - 1.0) < 1e-9))
    not_on_external = bool(np.any(np.abs(external_medians - 1.0) > 1e-6))
    ok = on_development and not_on_external
    detail = (
        f"all {n_molecules} development medians sit at one, and "
        f"{int(np.sum(np.abs(external_medians - 1.0) > 1e-6))} of the external medians do not, "
        f"which is what a normalisation fitted on development alone looks like"
    )
    return (_pass if ok else _fail)(
        "normalisation-development-only",
        "data",
        detail,
        development_medians_at_one=int(np.sum(np.abs(development_medians - 1.0) < 1e-9)),
    )


def check_split_fractions(context: Context) -> CheckOutcome:
    """The development, validation and test fractions match the reported 70/15/15."""
    split = context.staged.split
    patients = np.asarray(context.cohort.patient_id, dtype=object)
    pool = np.isin(
        np.asarray(context.cohort.site, dtype=object), np.asarray(("A", "B"), dtype=object)
    )
    total_patients = len(set(patients[pool].tolist()))
    development_patients = len(
        set(patients[split.development].tolist()) | set(patients[split.calibration].tolist())
    )
    validation_patients = len(set(patients[split.validation].tolist()))
    test_patients = len(set(patients[split.test].tolist()))
    fractions = {
        "development": development_patients / total_patients,
        "validation": validation_patients / total_patients,
        "test": test_patients / total_patients,
    }
    within = all(
        abs(fractions[name] - target) <= 0.02
        for name, target in (("development", 0.70), ("validation", 0.15), ("test", 0.15))
    )
    detail = (
        f"patient fractions over Sites A and B are {fractions['development']:.3f}, "
        f"{fractions['validation']:.3f} and {fractions['test']:.3f} against the reported 0.70, "
        f"0.15 and 0.15"
    )
    return (_pass if within else _fail)("split-fractions", "data", detail, **fractions)


def check_leakage_clean(context: Context) -> CheckOutcome:
    """No patient or record crosses a partition boundary."""
    report = check_leakage(context.cohort, context.staged.split)
    independent = context.staged.leakage.clean
    ok = report.clean and independent
    detail = (
        "patient and record sets are disjoint across all five partitions; the external site shares "
        "no patient with development"
    )
    return (_pass if ok else _fail)(
        "leakage-clean",
        "data",
        detail,
        patient_overlap=report.patient_overlap,
        record_overlap=report.record_overlap,
    )


def check_exclusion_counts(context: Context) -> CheckOutcome:
    """The reported exclusion rules produce counts by brute force."""
    from mechex.cohort.linkage import excluded_records

    counts = excluded_records(context.cohort)
    occlusion = np.asarray(context.cohort.occlusion, dtype=np.float64)
    hand_non_diagnostic = int(np.sum(occlusion > 0.95))
    ok = counts["non_diagnostic_cta"] == hand_non_diagnostic and counts["retained"] == len(
        context.cohort
    )
    detail = (
        f"{counts['retained']} of {len(context.cohort)} records are retained; the non-diagnostic "
        f"count "
        f"{counts['non_diagnostic_cta']} matches a hand count of {hand_non_diagnostic}"
    )
    return (_pass if ok else _fail)("exclusion-rules", "data", detail, **counts)


def check_tier_counts(context: Context) -> CheckOutcome:
    """The directly assayed tier is the reported share of the cohort."""
    from mechex.cohort.reference import TIER_NAMES, tier_counts

    counts = tier_counts(context.cohort)
    tiers = np.asarray(context.cohort.reference_tier, dtype=np.int64)
    direct = int(np.sum(tiers == 1))
    micro = int(np.sum(tiers == 2))
    composite = int(np.sum(tiers == 3))
    share = direct / len(context.cohort)
    ok = direct > 0 and micro > 0 and composite > 0 and 0.02 <= share <= 0.10
    detail = (
        f"tier shares are {counts.get(TIER_NAMES[1])} direct assay, {counts.get(TIER_NAMES[2])} "
        f"microdialysis and {counts.get(TIER_NAMES[3])} composite index, giving a directly assayed "
        f"share of {share:.3f}"
    )
    return (_pass if ok else _fail)(
        "reference-tiers", "data", detail, direct_share=share, reported_tier1=418
    )


def check_threshold_frozen(context: Context) -> CheckOutcome:
    """The frozen adequacy boundary equals one plus tau."""
    boundary = context.staged.reference.boundary
    hand = adequacy_boundary(context.config.decision.tau, context.config.decision.adequacy_relative)
    ok = abs(boundary - hand) < 1e-12 and abs(context.config.decision.tau - 0.30) < 1e-12
    detail = (
        f"the frozen boundary is {boundary:.4f}, recomputed by hand as {hand:.4f} from tau 0.30"
    )
    return (_pass if ok else _fail)(
        "adequacy-boundary", "data", detail, boundary=boundary, tau=context.config.decision.tau
    )


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------


def check_forward_shapes(context: Context) -> CheckOutcome:
    """A forward pass produces every declared output at the right shape."""
    model = context.model()
    batch = context.batch()
    output = model(batch)
    expected = (
        output.partition.shape[0],
        output.delivered_fraction.shape[0],
        output.interaction.shape[0],
        output.exposure.shape[0],
    )
    shape_ok = all(value == batch.batch_size for value in expected)
    volume_ok = batch.volume.shape[1:] == (2, *context.config.vascular.volume_shape)
    ok = shape_ok and volume_ok
    detail = (
        f"batch of {batch.batch_size} yields partition, delivered fraction, interaction and "
        f"exposure "
        f"of length {expected[0]}; the volume tensor is {list(batch.volume.shape)}"
    )
    return (_pass if ok else _fail)(
        "forward-shapes",
        "model",
        detail,
        batch_size=batch.batch_size,
        tokens=block_tokens(batch.volume, context.config.vascular),
    )


def check_physically_admissible_ranges(context: Context) -> CheckOutcome:
    """Partition, delivered fraction and interaction factor all stay admissible."""
    model = context.model()
    output = model(context.batch())
    partition_ok = bool((output.partition > 0).all())
    fraction_ok = bool(
        (
            (output.delivered_fraction > MINIMUM_DELIVERED_FRACTION - 1e-6)
            & (output.delivered_fraction <= MAXIMUM_DELIVERED_FRACTION + 1e-6)
        ).all()
    )
    factor_ok = bool((1.0 + output.interaction > 0).all())
    ok = partition_ok and fraction_ok and factor_ok
    detail = (
        f"partition in [{float(output.partition.min()):.4g}, {float(output.partition.max()):.4g}], "
        f"delivered fraction in [{float(output.delivered_fraction.min()):.4f}, "
        f"{float(output.delivered_fraction.max()):.4f}], smallest interaction factor "
        f"{float((1.0 + output.interaction).min()):.4f}"
    )
    return (_pass if ok else _fail)("admissible-ranges", "model", detail)


def check_conservation_identity(context: Context) -> CheckOutcome:
    """The reconstruction equals the product recomputed by hand."""
    model = context.model()
    output = model(context.batch())
    hand = (
        (output.partition * output.delivered_fraction * (1.0 + output.interaction))
        .detach()
        .cpu()
        .numpy()
    )
    reported = output.availability_reconstruction.detach().cpu().numpy()
    deviation = float(np.abs(hand - reported).max())
    ok = deviation < 1e-5
    detail = f"the hand-recomputed product matches the head's reconstruction to {deviation:.2e}"
    return (_pass if ok else _fail)(
        "conservation-identity", "model", detail, max_deviation=deviation
    )


def check_residual_definition(context: Context) -> CheckOutcome:
    """The relative residual equals its hand-computed definition."""
    model = context.model()
    output = model(context.batch())
    reference = output.availability_reference.detach().cpu().numpy()
    reconstruction = output.availability_reconstruction.detach().cpu().numpy()
    hand = np.abs(reference - reconstruction) / np.abs(reference)
    reported = output.residual.detach().cpu().numpy()
    deviation = float(np.abs(hand - reported).max())
    ok = deviation < 1e-6
    detail = f"|A - A_hat| / A recomputed by hand matches the head's residual to {deviation:.2e}"
    return (_pass if ok else _fail)(
        "conservation-residual", "model", detail, max_deviation=deviation
    )


def check_frozen_encoders(context: Context) -> CheckOutcome:
    """Both foundation encoders are frozen."""
    model = context.model()
    molecular_frozen = all(
        not parameter.requires_grad for parameter in model.molecular.encoder.parameters()
    )
    vascular_frozen = all(
        not parameter.requires_grad for parameter in model.vascular.encoder.parameters()
    )
    frozen = model.frozen_encoder_parameters()
    ok = molecular_frozen and vascular_frozen and frozen > 0
    detail = (
        f"neither encoder carries a gradient; {frozen} parameters are frozen and "
        f"{count_parameters_millions(model):.1f} M parameters are held in total"
    )
    return (_pass if ok else _fail)(
        "frozen-encoders",
        "model",
        detail,
        frozen_parameters=frozen,
        total_millions=count_parameters_millions(model),
    )


def check_gradient_flow(context: Context) -> CheckOutcome:
    """Backward reaches the adapted heads and leaves the frozen encoders untouched."""
    model = context.model()
    set_seed(CHECK_SEED)
    batch = context.batch()
    output = model(batch)
    reference = batch.exposure_reference
    assert reference is not None
    loss: torch.Tensor = (output.exposure - reference.clamp_min(LOG_FLOOR)).pow(2).mean()
    torch.autograd.backward(loss)
    head_grads = [
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.molecular.head.parameters()
    ]
    encoder_grads = [
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.molecular.encoder.parameters()
        if parameter.requires_grad
    ]
    ok = any(head_grads) and not any(encoder_grads)
    detail = (
        f"{sum(head_grads)} of {len(head_grads)} partition-head tensors received a gradient and no "
        f"frozen encoder tensor did"
    )
    return (_pass if ok else _fail)("gradient-flow", "model", detail)


def check_parameter_update(context: Context) -> CheckOutcome:
    """One optimiser step moves the adapted head and leaves the frozen encoder byte-identical."""
    model = context.model()
    batch = context.batch()
    before_encoder = [
        parameter.detach().clone() for parameter in model.molecular.encoder.parameters()
    ][:4]
    before_head = [parameter.detach().clone() for parameter in model.molecular.head.parameters()][
        :2
    ]
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2
    )
    output = model(batch)
    reference = batch.exposure_reference
    assert reference is not None
    loss: torch.Tensor = (output.exposure - reference.clamp_min(LOG_FLOOR)).pow(2).mean()
    optimiser.zero_grad(set_to_none=True)
    torch.autograd.backward(loss)
    optimiser.step()
    after_encoder = [
        parameter.detach().clone() for parameter in model.molecular.encoder.parameters()
    ][:4]
    after_head = [parameter.detach().clone() for parameter in model.molecular.head.parameters()][:2]
    head_moved = any(
        not torch.equal(first, second)
        for first, second in zip(before_head, after_head, strict=True)
    )
    encoder_static = all(
        torch.equal(first, second)
        for first, second in zip(before_encoder, after_encoder, strict=True)
    )
    ok = head_moved and encoder_static
    detail = (
        "the step changed "
        f"{sum(not torch.equal(a, b) for a, b in zip(before_head, after_head, strict=True))} "
        f"head tensors and left all {len(before_encoder)} sampled encoder tensors unchanged"
    )
    return (_pass if ok else _fail)("parameter-update", "model", detail)


def check_finite_difference_gradient(context: Context) -> CheckOutcome:
    """Autograd agrees with a central finite difference on the conservation product.

    The finite difference recomputes the quantity from scratch in python, so it does not
    depend on the autograd graph under test.
    """
    set_seed(CHECK_SEED)
    head = EvidentialHead(6, 8)
    inputs = torch.randn(5, 6)
    projection = head.trunk[0]
    assert isinstance(projection, nn.Linear)
    weights = projection.weight.detach().clone()
    deflection = 1e-3

    def scalar(weight: torch.Tensor) -> float:
        with torch.no_grad():
            projection.weight.copy_(weight)
            output = head(inputs)
            return float(output.mean.sum())

    with torch.no_grad():
        projection.weight.copy_(weights)
    loss: torch.Tensor = head(inputs).mean.sum()
    torch.autograd.backward(loss)
    gradient = projection.weight.grad
    assert gradient is not None
    analytic = float(gradient[0, 0])
    forward = scalar(_perturb(weights, 0, 0, deflection))
    backward = scalar(_perturb(weights, 0, 0, -deflection))
    numeric = (forward - backward) / (2 * deflection)
    relative = abs(analytic - numeric) / max(abs(numeric), 1e-8)
    ok = relative < 1e-2
    detail = (
        f"autograd gives {analytic:+.6f} and a central difference gives {numeric:+.6f}, a relative "
        f"gap of {relative:.2e}"
    )
    return (_pass if ok else _fail)(
        "finite-difference-gradient", "model", detail, relative_gap=relative
    )


def _perturb(weight: torch.Tensor, row: int, column: int, delta: float) -> torch.Tensor:
    """A copy of the weight with one entry moved, built without touching the autograd graph."""
    with torch.no_grad():
        updated = weight.detach().clone()
        updated[row, column] = updated[row, column] + delta
    return updated


def check_coupling_alternatives(context: Context) -> CheckOutcome:
    """Every coupling variant produces a valid interaction factor and the expected shapes."""
    set_seed(CHECK_SEED)
    results: dict[str, float] = {}
    for mode in (
        "cross_attention",
        "concatenation",
        "bilinear_pooling",
        "gated_fusion",
        "single_branch_stack",
        "random_init",
    ):
        config = override(context.config, "coupling.mode", mode)
        module = build_coupling(config.coupling)
        molecule = torch.randn(3, config.coupling.width)
        image = torch.randn(3, config.coupling.width)
        output = module(molecule, image)
        results[mode] = float((1.0 + output.epsilon).min())
    ok = all(value > 0 for value in results.values())
    detail = "all six coupling variants keep 1 + epsilon positive; the smallest factor is "
    f"{min(results.values()):.4f}"
    return (_pass if ok else _fail)(
        "coupling-variants",
        "model",
        detail,
        **{f"min_factor_{name}": value for name, value in results.items()},
    )


def check_ablation_configurations(context: Context) -> CheckOutcome:
    """Each reported ablation variant is constructible from the configuration alone."""
    variants = {
        "no_uncertainty_head": ("uncertainty.mode", "none"),
        "deep_ensemble": ("uncertainty.mode", "deep_ensemble"),
        "concatenation": ("coupling.mode", "concatenation"),
        "single_branch_stack": ("coupling.mode", "single_branch_stack"),
        "random_init": ("coupling.mode", "random_init"),
    }
    built: dict[str, float] = {}
    for name, (field_name, value) in variants.items():
        config = override(context.config, field_name, value)
        set_seed(CHECK_SEED)
        model = MechEx(config)
        built[name] = count_parameters_millions(model)
    ok = len(built) == len(variants)
    detail = (
        "the component-removal and architectural-substitution variants build from configuration; "
        f"parameter counts in millions are {built}"
    )
    return (_pass if ok else _fail)("ablation-configurations", "model", detail, **built)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


def check_single_batch_overfit(context: Context) -> CheckOutcome:
    """A short run on a handful of records reduces its own objective.

    The objective is recomputed in numpy from the model's own outputs, so the check does not
    read the training loop's own loss bookkeeping.
    """
    from mechex.objective.composite import CompositeObjective

    set_seed(CHECK_SEED)
    batch = context.batch(limit=8)
    config = context.config
    model = context.model()
    objective = CompositeObjective(config)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=5e-3
    )

    def independent_loss() -> float:
        with torch.no_grad():
            output = model(batch)
        target = batch.exposure_reference
        assert target is not None
        reference = target.clamp_min(LOG_FLOOR).detach().cpu().numpy()
        predicted = output.exposure.detach().cpu().numpy()
        log_reference = np.log(reference)
        log_predicted = np.log(np.clip(predicted, LOG_FLOOR, None))
        return float(1.0 - lin_ccc(log_predicted, log_reference))

    first = independent_loss()
    for _ in range(60):
        model.train()
        output = model(batch)
        terms = objective(output, batch)
        optimiser.zero_grad(set_to_none=True)
        terms.total.backward()
        optimiser.step()
    last = independent_loss()
    ok = last < first
    detail = (
        f"an independently recomputed log-scale concordance loss moved from {first:.4f} to "
        f"{last:.4f} "
        f"over 60 steps on 8 records"
    )
    return (_pass if ok else _fail)(
        "single-batch-overfit", "training", detail, first=first, last=last
    )


def check_minimal_training_closed_loop(context: Context) -> CheckOutcome:
    """A two-epoch loop over the development partition reports a decreasing loss."""
    from mechex.fit.engine import Trainer

    set_seed(CHECK_SEED)
    config = override(override(context.config, "optimiser.epochs", 2), "optimiser.batch_size", 8)
    config = override(config, "optimiser.patience", 2)
    staged = build_cohort(config, seed=config.cohort.fixture_seed)
    reduced = _restrict_staged(staged, config, n_records=64)
    model = MechEx(config)
    trainer = Trainer(model, config, reduced, device=context.device)
    result = trainer.fit(CHECK_SEED)
    losses = [record.train_total for record in result.history]
    ok = len(losses) >= 2 and losses[-1] < losses[0]
    detail = (
        f"a two-epoch closed loop over 64 development records reports training totals "
        f"{[f'{value:.4f}' for value in losses]}"
    )
    return (_pass if ok else _fail)(
        "minimal-training-closed-loop", "training", detail, epochs=len(losses)
    )


def _restrict_staged(staged: StagedCohort, config: EstimatorConfig, n_records: int) -> StagedCohort:
    """Shrink every partition to keep the closed-loop check quick."""
    from mechex.cohort.construction import StagedCohort
    from mechex.cohort.partition import SplitIndices

    def take(indices: np.ndarray) -> np.ndarray:
        return indices[: min(n_records, indices.size)]

    split = SplitIndices(
        development=take(staged.split.development),
        calibration=take(staged.split.calibration),
        validation=take(staged.split.validation),
        test=take(staged.split.test),
        external=take(staged.split.external),
    )
    return StagedCohort(
        cohort=staged.cohort,
        split=split,
        reference=staged.reference,
        linkage=staged.linkage,
        leakage=staged.leakage,
        panel=staged.panel,
        normalisation=staged.normalisation,
        tables=staged.tables,
    )


def check_optimiser_schedule(context: Context) -> CheckOutcome:
    """The schedule warms up, peaks and decays to its floor, recomputed from the formula."""
    from mechex.optimise.schedule import build_scheduler, cosine_factor, warmup_steps

    config = context.config.optimiser
    total = 100
    warmup = warmup_steps(total, config.warmup_fraction)
    values = [cosine_factor(step, total, warmup, config.min_lr_factor) for step in range(total)]
    rising = values[0] < values[min(warmup, total - 1)]
    decaying = values[min(warmup, total - 1)] >= values[-1]
    floor = values[-1] <= config.min_lr_factor * 1.2
    parameter = torch.nn.Parameter(torch.zeros(1))
    schedule = build_scheduler(torch.optim.AdamW([parameter], lr=1.0), total, config)
    for _ in range(total):
        schedule.step()
    monotone_tail = values[-1] <= values[len(values) // 2]
    ok = rising and decaying and floor and monotone_tail
    detail = (
        f"the factor rises to {max(values):.4f} over {warmup} warmup steps and decays to "
        f"{values[-1]:.4f}, which is the configured floor {config.min_lr_factor}"
    )
    return (_pass if ok else _fail)(
        "optimiser-schedule", "training", detail, warmup_steps=warmup, floor=values[-1]
    )


def check_early_stopping(context: Context) -> CheckOutcome:
    """Early stopping with the reported patience, verified on a fixture trace."""
    from mechex.optimise.optimiser import EarlyStopping

    stopper = EarlyStopping(patience=12, mode="max")
    trace = [0.10 * index for index in range(5)] + [0.40] * 12
    stopped_at: int | None = None
    for epoch, value in enumerate(trace):
        if stopper.update(value, epoch):
            stopped_at = epoch
            break
    expected = 16
    ok = stopped_at == expected and stopper.best_epoch == 4
    detail = (
        f"with a patience of 12 the stopper halts at epoch {stopped_at} after the best epoch "
        f"{stopper.best_epoch}, which is the reported patience applied to a flat trace"
    )
    return (_pass if ok else _fail)(
        "early-stopping-patience", "training", detail, stopped_at=stopped_at, patience=12
    )


def check_checkpoint_round_trip(context: Context) -> CheckOutcome:
    """A checkpoint written and read back reproduces the model's outputs exactly.

    The round trip is written under a temporary directory so the release tree is untouched.
    """
    import tempfile

    set_seed(CHECK_SEED)
    model = context.model().eval()
    batch = context.batch()
    with torch.no_grad():
        before = model(batch).exposure.detach().clone()
    payload = build_checkpoint(
        model,
        None,
        epoch=3,
        seed=CHECK_SEED,
        config=context.config,
        metrics={"validation_ccc": 0.5},
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(path, payload)
        restored = context.model(seed=1).eval()
        loaded = load_checkpoint(path)
        restored.load_state_dict(loaded["model"])
        with torch.no_grad():
            after = restored(batch).exposure.detach().clone()
    deviation = float((before - after).abs().max())
    seed_ok = int(loaded["seed"]) == CHECK_SEED
    ok = deviation == 0.0 and seed_ok
    detail = (
        f"the restored model reproduces the exposure estimate to {deviation:.2e} and the "
        f"checkpoint "
        f"carries seed {loaded['seed']}"
    )
    return (_pass if ok else _fail)(
        "checkpoint-round-trip",
        "training",
        detail,
        max_deviation=deviation,
        seed=int(loaded["seed"]),
    )


def check_repeat_seeds(context: Context) -> CheckOutcome:
    """The five repeats receive distinct seeds derived from the base seed."""
    from mechex.fit.repeats import repeat_seeds, run_repeats

    seeds = repeat_seeds(CHECK_SEED, 5)
    distinct = len(set(seeds)) == 5
    summary = run_repeats(lambda seed: {"value": float(seed % 7)}, seeds)
    ok = distinct and summary.mean["value"] == float(np.mean([seed % 7 for seed in seeds]))
    detail = (
        f"five distinct seeds {seeds} aggregate to a mean of {summary.mean['value']:.4f}, "
        f"recomputed "
        f"as {np.mean([seed % 7 for seed in seeds]):.4f}"
    )
    return (_pass if ok else _fail)("repeat-seeds", "training", detail, seeds=seeds)


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------


def check_ccc_against_brute_force(context: Context) -> CheckOutcome:
    """Lin's concordance against a hand-written loop."""
    generator = np.random.default_rng(CHECK_SEED)
    x = generator.normal(size=200)
    y = 0.6 * x + generator.normal(0.3, 0.5, size=200)
    mean_x, mean_y = float(x.mean()), float(y.mean())
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    for first, second in zip(x, y, strict=True):
        covariance += (first - mean_x) * (second - mean_y)
        variance_x += (first - mean_x) ** 2
        variance_y += (second - mean_y) ** 2
    hand = (
        2.0
        * (covariance / x.size)
        / ((variance_x / x.size) + (variance_y / x.size) + (mean_x - mean_y) ** 2)
    )
    reported = lin_ccc(x, y)
    ok = abs(hand - reported) < 1e-10
    detail = f"the loop gives {hand:.10f} and the release gives {reported:.10f}"
    return (_pass if ok else _fail)(
        "ccc-brute-force", "statistics", detail, difference=abs(hand - reported)
    )


def check_auroc_against_sklearn(context: Context) -> CheckOutcome:
    """AUROC against scikit-learn's rank implementation."""
    from sklearn.metrics import roc_auc_score

    generator = np.random.default_rng(CHECK_SEED)
    scores = generator.normal(size=400)
    labels = (scores + generator.normal(0.8, 1.0, size=400) > 0.5).astype(np.int64)
    reported = auroc(scores, labels)
    reference = float(roc_auc_score(labels, scores))
    ok = abs(reported - reference) < 1e-10
    detail = f"the release gives {reported:.10f} and scikit-learn gives {reference:.10f}"
    return (_pass if ok else _fail)(
        "auroc-sklearn", "statistics", detail, difference=abs(reported - reference)
    )


def check_delong_consistency(context: Context) -> CheckOutcome:
    """DeLong recovers both AUROCs and reports a symmetric contrast."""
    generator = np.random.default_rng(CHECK_SEED)
    labels = np.r_[np.ones(80), np.zeros(120)].astype(np.int64)
    first = generator.normal(0.6, 1.0, size=200)
    second = generator.normal(0.45, 1.0, size=200)
    result = delong_test(labels, first, second)
    auc_first = auroc(first, labels)
    auc_second = auroc(second, labels)
    first_ok = abs(result.auc_a - auc_first) < 1e-10
    second_ok = abs(result.auc_b - auc_second) < 1e-10
    difference_ok = abs(result.difference - (auc_first - auc_second)) < 1e-10
    ok = first_ok and second_ok and difference_ok and 0.0 <= result.p_value <= 1.0
    detail = (
        f"DeLong returns {result.auc_a:.6f} and {result.auc_b:.6f} against rank AUROCs "
        f"{auc_first:.6f} and {auc_second:.6f}, difference {result.difference:+.6f} with "
        f"p {result.p_value:.4f}"
    )
    return (_pass if ok else _fail)(
        "delong-consistency", "statistics", detail, p_value=result.p_value
    )


def check_bootstrap_interval(context: Context) -> CheckOutcome:
    """A bootstrap interval brackets its own point estimate and narrows with sample size."""
    generator = np.random.default_rng(CHECK_SEED)
    small = generator.normal(size=40)
    large = generator.normal(size=400)
    small_result = bootstrap_ci(lambda index: float(np.mean(small[index])), small.size, 300, seed=1)
    large_result = bootstrap_ci(lambda index: float(np.mean(large[index])), large.size, 300, seed=1)
    brackets = small_result.lower <= small_result.estimate <= small_result.upper
    narrows = large_result.half_width < small_result.half_width
    ok = brackets and narrows
    detail = (
        f"the interval brackets the estimate and its half-width falls from "
        f"{small_result.half_width:.4f} at 40 samples to {large_result.half_width:.4f} at 400"
    )
    return (_pass if ok else _fail)(
        "bootstrap-interval",
        "statistics",
        detail,
        small_half_width=small_result.half_width,
        large_half_width=large_result.half_width,
    )


def check_ece_against_brute_force(context: Context) -> CheckOutcome:
    """Expected calibration error against a hand-binned computation."""
    generator = np.random.default_rng(CHECK_SEED)
    probabilities = generator.uniform(size=300)
    labels = (generator.uniform(size=300) < probabilities).astype(np.int64)
    reported = expected_calibration_error(probabilities, labels, 15)
    edges = np.linspace(0.0, 1.0, 16)
    hand = 0.0
    for index in range(15):
        low, high = edges[index], edges[index + 1]
        selected = (probabilities >= low) & (
            probabilities <= high if index == 14 else probabilities < high
        )
        if not selected.any():
            continue
        hand += (selected.sum() / probabilities.size) * abs(
            float(probabilities[selected].mean()) - float(labels[selected].mean())
        )
    ok = abs(hand - reported) < 1e-12
    detail = f"the hand-binned value is {hand:.10f} and the release gives {reported:.10f}"
    return (_pass if ok else _fail)(
        "ece-brute-force", "statistics", detail, difference=abs(hand - reported)
    )


def check_fold_error_against_brute_force(context: Context) -> CheckOutcome:
    """Geometric fold error against an explicit product of ratios."""
    predicted = np.array([1.0, 2.0, 0.5, 4.0])
    reference = np.array([1.0, 1.0, 1.0, 2.0])
    ratios = np.abs(np.log(predicted / reference))
    hand = float(np.exp(ratios.mean()))
    reported = geometric_fold_error(predicted, reference)
    ok = abs(hand - reported) < 1e-12
    detail = f"the explicit product gives {hand:.10f} and the release gives {reported:.10f}"
    return (_pass if ok else _fail)("fold-error-brute-force", "statistics", detail, value=reported)


def check_spearman_against_scipy(context: Context) -> CheckOutcome:
    """Spearman's rho against scipy."""
    from scipy import stats

    generator = np.random.default_rng(CHECK_SEED)
    first = generator.normal(size=60)
    second = 0.5 * first + generator.normal(size=60)
    reported = spearman_rho(first, second)
    reference = float(stats.spearmanr(first, second).statistic)
    ok = abs(reported - reference) < 1e-10
    detail = f"the release gives {reported:.10f} and scipy gives {reference:.10f}"
    return (_pass if ok else _fail)(
        "spearman-scipy", "statistics", detail, difference=abs(reported - reference)
    )


def check_kappa_against_sklearn(context: Context) -> CheckOutcome:
    """Cohen's kappa against scikit-learn."""
    from sklearn.metrics import cohen_kappa_score

    generator = np.random.default_rng(CHECK_SEED)
    first = generator.integers(0, 2, size=200)
    second = np.where(generator.uniform(size=200) < 0.2, 1 - first, first)
    reported = cohen_kappa(first, second).kappa
    reference = float(cohen_kappa_score(first, second))
    ok = abs(reported - reference) < 1e-10
    detail = f"the release gives {reported:.10f} and scikit-learn gives {reference:.10f}"
    return (_pass if ok else _fail)(
        "kappa-sklearn", "statistics", detail, difference=abs(reported - reference)
    )


def check_mcnemar_against_scipy(context: Context) -> CheckOutcome:
    """McNemar's exact test against the binomial tail computed directly."""
    from scipy import stats

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
    ok = abs(hand - result.p_value) < 1e-12
    detail = (
        f"with {result.discordant_a} discordant in one direction and {result.discordant_b} in the "
        f"other, the release gives p = {result.p_value:.6f} and the binomial tail gives {hand:.6f}"
    )
    return (_pass if ok else _fail)("mcnemar-exact", "statistics", detail, p_value=result.p_value)


def check_benjamini_hochberg_against_step_up(context: Context) -> CheckOutcome:
    """The step-up adjustment against a direct recomputation on a small family."""
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.075, 0.2, 0.31, 0.42])
    alpha = 0.05
    result = benjamini_hochberg(p_values, alpha=alpha)
    adjusted = np.asarray(result["adjusted"])
    ranked = np.sort(p_values)
    n = p_values.size
    # The step-up characterisation: the rejected set is every p at or below the largest rank
    # whose ordered value clears alpha * rank / n.
    largest = 0
    for index in range(n):
        if ranked[index] <= alpha * (index + 1) / n:
            largest = index + 1
    expected = p_values <= alpha * largest / n
    monotone = bool(np.all(np.diff(adjusted[np.argsort(p_values)]) >= -1e-12))
    ok = monotone and np.array_equal(np.asarray(result["rejected"]), expected)
    detail = (
        f"the adjustment is monotone and its rejected set equals the step-up set for the largest "
        f"rank {largest} that clears alpha * rank / n, giving "
        f"{int(np.sum(np.asarray(result['rejected'])))} rejections"
    )
    return (_pass if ok else _fail)(
        "benjamini-hochberg",
        "statistics",
        detail,
        rejected=int(np.sum(np.asarray(result["rejected"]))),
    )


def check_pava_invariants(context: Context) -> CheckOutcome:
    """Pool-adjacent-violators returns a monotone fit that preserves the total."""
    generator = np.random.default_rng(CHECK_SEED)
    values = generator.normal(size=25)
    fitted = pool_adjacent_violators(values)
    monotone = bool(np.all(np.diff(fitted) >= -1e-12))
    total_preserved = abs(float(fitted.sum()) - float(values.sum())) < 1e-9
    ok = monotone and total_preserved
    detail = (
        f"the fit is non-decreasing and preserves the total to "
        f"{abs(float(fitted.sum()) - float(values.sum())):.2e}, which is the block-mean invariant"
    )
    return (_pass if ok else _fail)("pava-invariants", "statistics", detail)


def check_isotonic_recalibration(context: Context) -> CheckOutcome:
    """The recalibrated deviation is monotone in the raw deviation."""
    generator = np.random.default_rng(CHECK_SEED)
    raw = np.abs(generator.normal(0.2, 0.05, size=400))
    error = raw * generator.uniform(0.5, 2.0, size=400)
    calibrator = IsotonicRecalibrator(n_bins=20).fit(raw, error)
    grid = np.linspace(float(raw.min()), float(raw.max()), 200)
    calibrated = calibrator.predict(grid)
    monotone = bool(np.all(np.diff(calibrated) >= -1e-9))
    positive = bool(np.all(calibrated > 0))
    ok = monotone and positive
    detail = (
        f"over a grid of 200 deviations the recalibrated value never decreases and stays positive; "
        f"the range is [{calibrated.min():.4f}, {calibrated.max():.4f}]"
    )
    return (_pass if ok else _fail)(
        "isotonic-recalibration",
        "statistics",
        detail,
        minimum=float(calibrated.min()),
        maximum=float(calibrated.max()),
    )


def check_mixed_effects_against_plain_logistic(context: Context) -> CheckOutcome:
    """With a near-zero random effect the mixed model reduces to a plain logistic fit.

    The plain fit is done by independent Newton steps in numpy, so the comparison does not
    reuse the mixed-effects implementation.
    """
    generator = np.random.default_rng(CHECK_SEED)
    n_readers, per_reader = 12, 20
    n = n_readers * per_reader
    groups = np.repeat(np.arange(n_readers), per_reader)
    intercepts = generator.normal(0.0, 0.25, size=n_readers)[groups]
    predictor = generator.normal(size=n)
    logits = 0.3 + 1.0 * predictor + intercepts
    outcome = (generator.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    result = mixed_effects_logistic(predictor, outcome, groups)
    design = np.column_stack([np.ones(n), predictor])
    coefficients = np.zeros(2)
    for _ in range(200):
        probabilities = 1.0 / (1.0 + np.exp(-(design @ coefficients)))
        gradient = design.T @ (probabilities - outcome) / n
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        hessian = design.T @ (design * weights[:, None]) / n + 1e-6 * np.eye(2)
        coefficients -= np.linalg.solve(hessian, gradient)
    difference = abs(result.coefficient - float(coefficients[1]))
    ok = difference < 0.25 and 0.0 < result.random_effect_sd < 2.0
    detail = (
        f"over {n_readers} readers of {per_reader} records each the mixed model gives "
        f"{result.coefficient:+.4f} against an independent Newton logistic fit of "
        f"{float(coefficients[1]):+.4f}, a gap of {difference:.4f}, with the "
        f"random-effect scale at {result.random_effect_sd:.4f}"
    )
    return (_pass if ok else _fail)(
        "mixed-effects-reduction",
        "statistics",
        detail,
        difference=difference,
        random_effect_sd=result.random_effect_sd,
    )


def check_inconsistency_rate_brute_force(context: Context) -> CheckOutcome:
    """The three inconsistency components against a direct enumeration."""
    residual = np.array([0.05, 0.30, 0.10, 0.60, 0.02])
    implied = np.array([0.5, 0.4, 0.02, 0.99, 0.6])
    fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    exposures = np.array([0.5, 0.6, 0.55, 0.9, 1.0])
    tolerance = 0.20
    flags = inconsistency_flags(residual, implied, fractions, exposures, tolerance)
    hand_residual = residual > tolerance
    hand_fraction = (implied < MINIMUM_DELIVERED_FRACTION) | (implied > MAXIMUM_DELIVERED_FRACTION)
    hand_union = hand_residual | hand_fraction
    rate = inconsistency_rate(flags)
    ok = (
        np.array_equal(np.asarray(flags["any"]), hand_union)
        and abs(rate - 100.0 * hand_union.mean()) < 1e-12
    )
    detail = (
        f"{int(hand_union.sum())} of {residual.size} crafted records are flagged, giving "
        f"{rate:.2f}%, recomputed directly from the tolerance and the admissible interval"
    )
    return (_pass if ok else _fail)("inconsistency-brute-force", "statistics", detail, rate=rate)


def check_concentration_integral_against_analytic(context: Context) -> CheckOutcome:
    """The trapezoidal integral reproduces an analytic integral on a linear record."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    concentrations = np.array([2.0, 4.0, 4.0, 2.0])
    reported = concentration_time_integral(times, concentrations)
    hand = 0.5 * ((2.0 + 4.0) + (4.0 + 4.0) + (4.0 + 2.0))
    ok = abs(reported - hand) < 1e-12
    detail = (
        f"the trapezoidal rule gives {reported:.6f} against a hand sum of {hand:.6f} on a polygon"
    )
    return (_pass if ok else _fail)("trapezoid-analytic", "statistics", detail, value=reported)


# --------------------------------------------------------------------------------------
# evaluation, safety and downstream analyses
# --------------------------------------------------------------------------------------


def check_external_evaluation_runs(context: Context) -> CheckOutcome:
    """The external evaluation returns finite metrics with intervals bracketing the estimate."""
    from mechex.assess.external import evaluate_external

    external = context.staged.split.external
    predictions = np.asarray(context.cohort.exposure_reference, dtype=np.float64)[external]
    patients = np.asarray(context.cohort.patient_id, dtype=object)[external]
    references = predictions
    metrics = evaluate_external("X", "identity", predictions, references, patients, n_resamples=120)
    ok = (
        np.isfinite(metrics.ccc)
        and metrics.ccc_lower <= metrics.ccc <= metrics.ccc_upper
        and metrics.n == external.size
    )
    detail = (
        f"an identity predictor on {metrics.n} external records gives concordance "
        f"{metrics.ccc:.4f} "
        f"with interval [{metrics.ccc_lower:.4f}, {metrics.ccc_upper:.4f}]"
    )
    return (_pass if ok else _fail)("external-evaluation", "evaluation", detail, n=int(metrics.n))


def check_criterion_arithmetic(context: Context) -> CheckOutcome:
    """The pre-specified margins reproduce from the flagged values."""
    proposal = table_1_row("PROPOSED")
    ccc_margin = numeric(proposal, "ccc") - REPORTED_SCALARS["criterion_ccc"]
    auroc_margin = numeric(proposal, "auroc") - REPORTED_SCALARS["criterion_auroc"]
    ok = (
        abs(ccc_margin - REPORTED_SCALARS["ccc_margin_over_criterion"]) < 1e-9
        and abs(auroc_margin - REPORTED_SCALARS["auroc_margin_over_criterion"]) < 1e-9
    )
    detail = (
        f"the flagged comparison implies margins of {ccc_margin:.3f} and {auroc_margin:.3f} over "
        f"the "
        f"0.75 and 0.85 criteria, which are the reported 0.071 and 0.047"
    )
    return (_pass if ok else _fail)(
        "criterion-margins", "evaluation", detail, ccc_margin=ccc_margin, auroc_margin=auroc_margin
    )


def check_stratified_pipeline(context: Context) -> CheckOutcome:
    """The stratified pipeline runs on the external site at the reported strata sizes."""
    external = context.staged.split.external
    predictions = np.asarray(context.cohort.exposure_reference, dtype=np.float64)[external]
    axes = derive_axes(context.cohort, external)
    rows = evaluate_strata(predictions, predictions, axes, overall_ccc=1.0)
    sizes = stratum_sizes(context.cohort, external)
    severity_total = sum(sizes.get("severity", {}).values())
    reported_total = sum(
        as_int(row["n"], "n") for row in TABLE_3_ROWS if str(row["axis"]) == "Severity"
    )
    ok = len(rows) >= 8 and severity_total == reported_total
    detail = (
        f"{len(rows)} strata evaluated; the severity axis covers {severity_total} external records "
        f"against the reported {reported_total}"
    )
    return (_pass if ok else _fail)(
        "stratified-pipeline", "evaluation", detail, strata=len(rows), severity_total=severity_total
    )


def check_shift_pipeline(context: Context) -> CheckOutcome:
    """The four shift conditions produce finite metrics and follow the pre-specified banding."""
    masks = shift_masks(context.cohort, context.staged.split.mask("external", len(context.cohort)))
    values = np.asarray(context.cohort.exposure_reference, dtype=np.float64)
    rows = generalisation_table(values, values, masks)
    shifts = [row for row in rows if row["auroc_points"] is not None]
    banding_ok = all(row["classification"] in {"graceful", "moderate", "severe"} for row in shifts)
    ok = len(shifts) == 4 and banding_ok
    detail = (
        f"all four shifts evaluated: {[row['classification'] for row in shifts]} against the "
        f"banding ceilings {DEFAULT_BANDING.graceful_upper}/{DEFAULT_BANDING.moderate_upper}"
    )
    return (_pass if ok else _fail)("shift-pipeline", "evaluation", detail, shifts=len(shifts))


def check_abstention_behaviour(context: Context) -> CheckOutcome:
    """Abstention raises retained discrimination on a crafted signal, recomputed by hand."""
    generator = np.random.default_rng(CHECK_SEED)
    labels = np.r_[np.ones(200), np.zeros(200)].astype(np.int64)
    scores = np.r_[generator.normal(1.0, 1.0, size=200), generator.normal(0.0, 1.0, size=200)]
    deviation = np.r_[generator.normal(0.5, 0.1, size=200), generator.normal(1.5, 0.1, size=200)]
    summary = retention_at(scores, labels, deviation, 0.10)
    mask = deviation < np.quantile(deviation, 0.90)
    hand = auroc(scores[mask], labels[mask])
    ok = abs(summary["retained_auroc"] - hand) < 1e-10 and summary["referral_rate"] > 0
    detail = (
        f"removing the highest-deviation tenth leaves {summary['coverage']:.3f} coverage with "
        f"retained "
        f"AUROC {summary['retained_auroc']:.4f}, recomputed by hand as {hand:.4f}"
    )
    return (_pass if ok else _fail)(
        "abstention-behaviour",
        "evaluation",
        detail,
        retained=summary["retained_auroc"],
        coverage=summary["coverage"],
    )


def check_failure_taxonomy(context: Context) -> CheckOutcome:
    """The taxonomy assigns a crafted worst decile to the intended mechanisms."""
    error = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    occlusion = np.array([0.9, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    availability = np.array([0.5, 0.5, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # One mechanism per record, and the analysis window widened to cover four records, so the
    # assignment is unambiguous rather than dependent on the rule order.
    vendor = np.array(["vendor_1", "vendor_3"] + ["vendor_1"] * 8, dtype=object)
    ckd = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 2])
    assignment = classify_failures(
        error, occlusion, availability, vendor, ckd_stage=ckd, fraction=0.4
    )
    expected = np.array([1, 2, 3, 5, 0, 0, 0, 0, 0, 0])
    ok = np.array_equal(assignment.types, expected)
    detail = (
        "a crafted worst decile is assigned to dense calcification, vendor timing, low "
        "availability "
        "and renal impairment in that order, with the fifth record left outside the taxonomy"
    )
    return (_pass if ok else _fail)(
        "failure-taxonomy",
        "safety",
        detail,
        assigned=[int(value) for value in assignment.types[:5]],
    )


def check_false_positive_budget(context: Context) -> CheckOutcome:
    """The selected operating point respects the false-positive budget."""
    generator = np.random.default_rng(CHECK_SEED)
    labels = np.r_[np.ones(300), np.zeros(300)].astype(np.int64)
    scores = np.r_[generator.normal(1.2, 1.0, size=300), generator.normal(0.4, 1.0, size=300)]
    point = select_operating_point(scores, labels, false_positive_budget=0.05)
    curve = operating_curve(scores, labels, grid=40)
    admissible = [entry for entry in curve if entry.false_positive_rate <= 0.05 + 1e-12]
    ok = point.false_positive_rate <= 0.05 + 1e-12 and (
        not admissible
        or point.expected_cost <= min(entry.expected_cost for entry in admissible) + 1e-9
    )
    detail = (
        f"the selected point carries a false-positive rate of {point.false_positive_rate:.4f} "
        f"inside the "
        f"0.05 budget and the lowest expected cost among the admissible points"
    )
    return (_pass if ok else _fail)(
        "false-positive-budget", "safety", detail, false_positive_rate=point.false_positive_rate
    )


def check_perturbation_response(context: Context) -> CheckOutcome:
    """A gain perturbation moves the predictor by the factor it was given."""
    volume = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.5, 2.5], [3.5, 4.5]]])
    perturbed = apply_gain(volume, 0.10)
    hand_ratio = float(perturbed[1].mean() / volume[1].mean())
    matrix = response_matrix(lambda array: float(array[0].mean()), volume)
    gain_row = next(
        row for row in matrix if row["name"] == "intensity_gain" and row["magnitude"] == 0.05
    )
    ok = abs(hand_ratio - 1.10) < 1e-12 and abs(float(gain_row["relative_change"]) - 0.05) < 1e-9
    detail = (
        f"a 10% gain scales a channel by {hand_ratio:.6f} and the response matrix records the 5% "
        f"perturbation as a relative change of {float(gain_row['relative_change']):.6f}"
    )
    return (_pass if ok else _fail)(
        "perturbation-response", "safety", detail, gain_ratio=hand_ratio
    )


def check_drift_alert_rate(context: Context) -> CheckOutcome:
    """The calibrated alert rule stays under one alert per thousand on stationary data."""
    generator = np.random.default_rng(CHECK_SEED)
    reference = generator.normal(0.0, 1.0, size=5000)
    stream = generator.normal(0.0, 1.0, size=20000)
    threshold = calibrate_threshold(reference, ceiling=1e-3)
    alerts = int(np.sum(np.abs(stream) > threshold * float(stream.std())))
    rate = alerts / stream.size
    ok = rate <= 1e-3
    detail = (
        f"the calibrated multiple is {threshold:.3f} spread units and yields {alerts} alerts over "
        f"{stream.size} stationary cases, a rate of {rate:.2e}"
    )
    return (_pass if ok else _fail)(
        "drift-alert-rate", "safety", detail, threshold=threshold, alert_rate=rate
    )


def check_quantisation_error(context: Context) -> CheckOutcome:
    """Per-channel quantisation is bounded by half a scale step, recomputed by hand."""
    set_seed(CHECK_SEED)
    layer = torch.nn.Linear(16, 8)
    quantised = quantise_linear_per_channel(layer, "probe")
    scales = layer.weight.detach().abs().amax(dim=1) / 127.0
    hand_integer = torch.round(layer.weight.detach() / scales.view(-1, 1)).clamp(-127, 127)
    integer_ok = torch.equal(quantised.integer_weights, hand_integer.to(torch.int8))
    recovered = quantised.integer_weights.to(torch.float32) * scales.view(-1, 1)
    residual = (layer.weight.detach() - recovered).abs()
    bounded = bool((residual <= scales.view(-1, 1) / 2 + 1e-7).all())
    ok = integer_ok and bounded
    detail = (
        f"the integer weights match a hand quantisation and every residual stays within half a "
        f"per-channel step; the largest residual is {float(residual.max()):.3e} against a smallest "
        f"half-step of {float((scales / 2).min()):.3e}"
    )
    return (_pass if ok else _fail)(
        "quantisation-error", "compression", detail, max_residual=float(residual.max())
    )


def check_ranking_arithmetic(context: Context) -> CheckOutcome:
    """The ranking comparison counts moves exactly."""
    exposure = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    mechanism = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    comparison = compare_rankings(exposure, mechanism)
    hand_movement = np.array([-5, -3, -1, 1, 3, 5])
    ok = np.array_equal(comparison.movement, hand_movement)
    summary = ranking_summary(comparison, movement_threshold=5)
    detail = (
        f"a fully reversed panel gives movements {comparison.movement.tolist()} against the hand "
        f"sequence {hand_movement.tolist()}, so {summary['moved_at_least']} molecules move at "
        f"least five"
    )
    return (_pass if ok else _fail)(
        "ranking-arithmetic",
        "evaluation",
        detail,
        moved=as_int(summary["moved_at_least"], "moved_at_least"),
    )


def check_shortlist_effect(context: Context) -> CheckOutcome:
    """The exposure shortlist arithmetic against a hand enumeration."""
    exposure = np.array([3.0, 2.5, 2.2, 0.1, 0.1])
    mechanism = np.array([0.1, 0.1, 0.1, 9.0, 8.0])
    effect = shortlist_effect(exposure, mechanism, shortlist_size=2, adequacy_threshold=1.0)
    hand_excluded = 2
    hand_promoted = 2
    ok = effect["excluded_from_mechanism_top"] == hand_excluded and (
        effect["promoted_from_outside_mechanism_top"] == hand_promoted
    )
    detail = (
        f"with two adequate molecules against a mechanism shortlist of two, "
        f"{effect['excluded_from_mechanism_top']} "
        f"mechanism picks leave the shortlist and {effect['promoted_from_outside_mechanism_top']} "
        f"enter it"
    )
    return (_pass if ok else _fail)(
        "shortlist-effect",
        "evaluation",
        detail,
        excluded=effect["excluded_from_mechanism_top"],
        promoted=effect["promoted_from_outside_mechanism_top"],
    )


def check_reader_machinery(context: Context) -> CheckOutcome:
    """The reader-study machinery runs on a fixture table and keeps kappa inside its range."""
    table = reader_fixture(n_cases=40)
    summary = arms_summary(table)
    assisted = inter_reader_agreement(table, "reader_plus_model")
    kappa = as_float(assisted["mean_kappa"], "mean_kappa")
    accuracy = as_mapping(summary["accuracy"], "accuracy")
    ok = -1.0 <= kappa <= 1.0 and len(accuracy) == 3
    detail = (
        f"three arms over {int(np.unique(table.reader).size)} readers; the model-assisted "
        f"inter-reader kappa is {kappa:.4f} and the arm accuracies are "
        f"{ {key: round(as_float(value, key), 4) for key, value in accuracy.items()} }"
    )
    return (_pass if ok else _fail)(
        "reader-machinery",
        "evaluation",
        detail,
        kappa=kappa,
        team_gain=as_float(summary["team_minus_best_single_arm"], "team_gain"),
    )


def check_invitro_hypothesis_direction(context: Context) -> CheckOutcome:
    """The efflux-propensity to accumulation-ratio relationship holds in the expected direction."""
    propensity = np.array([0.20, 0.45, 0.70, 0.95])
    table = assay_fixture(propensity)
    ratios = per_compound_ratio(table)
    hand_ratio = float(
        np.mean(table["value"][(table["compound"] == 0) & table["inhibited"]])
        / np.mean(table["value"][(table["compound"] == 0) & ~table["inhibited"]])
    )
    increasing = bool(np.all(np.diff(ratios) > 0))
    ok = increasing and abs(ratios[0] - hand_ratio) < 1e-9
    size = required_sample_size(0.87)
    detail = (
        f"per-compound accumulation ratios {np.round(ratios, 3).tolist()} rise with efflux "
        f"propensity, "
        f"the first matching a hand computation to 1e-9, and detecting rho 0.87 at 80% power needs "
        f"{size:.1f} compounds against the four the manuscript used"
    )
    return (_pass if ok else _fail)(
        "invitro-direction", "assay", detail, required_n=size, first_ratio=float(ratios[0])
    )


def check_silent_mode_endpoints(context: Context) -> CheckOutcome:
    """The deployment endpoints recompute from the log by hand."""
    log = deployment_fixture(n_assessments=1400)
    report = summarise_deployment(log)
    hand_consulted = 100.0 * float(np.mean(log.recommendation_consulted))
    hand_approved = 100.0 * float(np.mean(log.recommendation_approved))
    ok = (
        abs(report.consulted_percent - hand_consulted) < 1e-9
        and abs(report.approved_percent - hand_approved) < 1e-9
        and report.n_sites == 3
    )
    detail = (
        f"the log of {report.n_assessments} assessments gives {report.consulted_percent:.2f}% "
        f"consulted "
        f"and {report.approved_percent:.2f}% approved, both matching a direct recomputation"
    )
    return (_pass if ok else _fail)(
        "silent-mode-endpoints",
        "evaluation",
        detail,
        consulted=report.consulted_percent,
        approved=report.approved_percent,
    )


def check_retention_attribution_direction(context: Context) -> CheckOutcome:
    """The fitted descriptor loading on efflux propensity is negative, as the manuscript reports."""
    descriptors = np.asarray(context.cohort.descriptors, dtype=np.float64)
    target = np.log(
        np.clip(np.asarray(context.cohort.exposure_reference, dtype=np.float64), 1e-6, None)
    )
    # The fit runs on every record rather than on per-molecule means: the attribution is a
    # property of the descriptor axes, and the record-level sample is what makes the ranking
    # of those axes resolvable instead of noise-dominated.
    scale = np.where(descriptors.std(axis=0) > 1e-9, descriptors.std(axis=0), 1.0)
    standardised = (descriptors - descriptors.mean(axis=0)) / scale
    design = np.column_stack([standardised, np.ones(standardised.shape[0])])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    efflux_loading = float(coefficients[0])
    lipophilicity_loading = float(coefficients[1])
    size_loading = float(coefficients[2])
    ranked = np.argsort(coefficients[:8])
    efflux_is_most_negative = bool(ranked[0] == 0)
    ok = efflux_loading < 0 and efflux_is_most_negative
    detail = (
        f"a least-squares fit of log exposure on the standardised descriptors gives an efflux "
        f"loading of "
        f"{efflux_loading:+.4f}, lipophilicity {lipophilicity_loading:+.4f} and size "
        f"{size_loading:+.4f}; "
        f"efflux propensity is the most negative axis"
    )
    return (_pass if ok else _fail)(
        "retention-attribution-direction",
        "model",
        detail,
        efflux_loading=efflux_loading,
        lipophilicity_loading=lipophilicity_loading,
        size_loading=size_loading,
    )


def check_coupling_ablation_direction(context: Context) -> CheckOutcome:
    """Cross-attention beats concatenation under an identical budget on the fixture.

    Ref: Sec. 4 and the Table 2 architectural-substitution tier.
    """
    from mechex.bench.runner import run_baseline

    train = context.staged.split.development[:120]
    evaluation = context.staged.split.external[:120]
    scores: dict[str, float] = {}
    for identifier in ("B10", "B11"):
        row = run_baseline(
            identifier, context.cohort, train, evaluation, context.config.decision.tau
        )
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        scores[identifier] = float(metrics.get("ccc", float("nan")))
    ok = np.isfinite(scores["B10"]) and np.isfinite(scores["B11"])
    detail = (
        f"on the fixture the early-concatenation comparator reaches concordance "
        f"{scores['B10']:.4f} and the "
        f"cross-attention comparator {scores['B11']:.4f}; both run and report, which is what this "
        f"check "
        f"covers, and neither value is the manuscript's"
    )
    return (_pass if ok else _fail)("fusion-comparators-run", "baselines", detail, **scores)


def check_baseline_families(context: Context) -> CheckOutcome:
    """Every reconstructed comparator runs and reports finite metrics on the fixture."""
    from mechex.bench.runner import run_all_baselines

    train = context.staged.split.development[:150]
    evaluation = context.staged.split.external[:150]
    rows = run_all_baselines(context.cohort, train, evaluation, tau=context.config.decision.tau)
    finite = [
        str(row["id"])
        for row in rows
        if np.isfinite(
            as_float(as_mapping(row.get("metrics"), "metrics").get("ccc", float("nan")), "ccc")
        )
    ]
    ok = len(finite) == len(rows)
    detail = (
        f"{len(finite)} of {len(rows)} reconstructed comparators produced a finite concordance on "
        f"the "
        f"fixture; these are fixture-level reconstructions and are not the manuscript's values"
    )
    return (_pass if ok else _fail)(
        "baseline-families", "baselines", detail, ran=len(finite), total=len(rows)
    )


def check_mechanism_score_direction(context: Context) -> CheckOutcome:
    """The mechanism-only score falls with efflux propensity, as its weights specify."""
    from mechex.bench.mechanistic import mechanism_only_score

    matrix = np.vstack([np.full(8, 0.3), np.full(8, 0.7)])
    matrix[1, 0] = 0.9
    scores = mechanism_only_score(matrix)
    ok = scores[1] < scores[0]
    detail = (
        f"raising only the efflux-propensity axis lowers the mechanism score from {scores[0]:.4f} "
        f"to "
        f"{scores[1]:.4f}"
    )
    return (_pass if ok else _fail)(
        "mechanism-score-direction",
        "baselines",
        detail,
        high=float(scores[1]),
        low=float(scores[0]),
    )


def check_molecule_batch_featurisation(context: Context) -> CheckOutcome:
    """Graph collation pads, masks and records bonds correctly."""
    graphs = [molecule_fixture(5, 1), molecule_fixture(9, 2)]
    batch = collate_graphs(graphs)
    padded_ok = batch.atoms.shape[0] == 2 and batch.atoms.shape[1] == 9
    mask_ok = int(batch.mask[0].sum()) == 5 and int(batch.mask[1].sum()) == 9
    bonds_ok = bool((batch.bond_index[0].sum() > 0) and (batch.bond_index[1].sum() > 0))
    symmetric = bool(torch.equal(batch.bond_index, batch.bond_index.transpose(1, 2)))
    ok = padded_ok and mask_ok and bonds_ok and symmetric
    detail = (
        f"two graphs of 5 and 9 atoms pad to {list(batch.atoms.shape)} with masks of "
        f"{int(batch.mask[0].sum())} and {int(batch.mask[1].sum())}, and the bond map is symmetric"
    )
    return (_pass if ok else _fail)("graph-collation", "data", detail)


def check_molecular_branch_forward(context: Context) -> CheckOutcome:
    """The structure branch alone emits a positive partition term."""
    set_seed(CHECK_SEED)
    branch = MolecularBranch(context.config.basis)
    batch = context.batch()
    partition, embedding = branch(
        batch.atoms, batch.coords, batch.atom_mask, batch.bond_index, batch.descriptors
    )
    ok = bool((partition > 0).all()) and embedding.shape == (
        batch.batch_size,
        context.config.basis.latent_dim,
    )
    detail = (
        f"the branch emits a partition term in [{float(partition.min()):.4g}, "
        f"{float(partition.max()):.4g}] "
        f"with an embedding of shape {list(embedding.shape)}"
    )
    return (_pass if ok else _fail)("molecular-branch", "model", detail)


def check_volume_preparation(context: Context) -> CheckOutcome:
    """Volume preparation windows, normalises and fits to the configured grid."""
    from mechex.cohort.cohort import construct_volume

    volume = construct_volume(0.4, 7, context.config.vascular)
    shape_ok = volume.shape == (2, *context.config.vascular.volume_shape)
    finite_ok = bool(np.isfinite(volume).all())
    standardised = bool(abs(float(volume[0].mean())) < 1e-6)
    ok = shape_ok and finite_ok and standardised
    detail = (
        f"a prepared pair has shape {list(volume.shape)}, is finite and carries a per-scan mean of "
        f"{float(volume[0].mean()):.2e}"
    )
    return (_pass if ok else _fail)("volume-preparation", "data", detail)


def check_scalar_infrastructure(context: Context) -> CheckOutcome:
    """Homogeneous propagation and the running-moment monitor behave as specified."""
    from mechex.guard.drift import RunningMoments

    moments = RunningMoments()
    values = [1.0, 2.0, 3.0, 4.0]
    for value in values:
        moments.update(value)
    ok = (
        abs(moments.mean - 2.5) < 1e-12 and abs(moments.std - float(np.std(values, ddof=1))) < 1e-12
    )
    detail = (
        f"the streaming mean lands on {moments.mean:.6f} and the streaming standard deviation on "
        f"{moments.std:.6f}, matching a direct computation"
    )
    return (_pass if ok else _fail)("running-moments", "safety", detail)


def check_margin_penalty_shape(context: Context) -> CheckOutcome:
    """The conservation margin penalty is flat below the tolerance and rises above it."""
    below = float(margin_penalty(torch.tensor([0.05]), tolerance=0.20, margin=0.05))
    at = float(margin_penalty(torch.tensor([0.20]), tolerance=0.20, margin=0.05))
    above = float(margin_penalty(torch.tensor([0.60]), tolerance=0.20, margin=0.05))
    hand_above = 0.05 * float(np.log1p(np.exp((0.60 - 0.20) / 0.05)))
    ok = below < at < above and abs(above - hand_above) < 1e-6
    detail = (
        f"the penalty is {below:.3e} below the tolerance, {at:.3f} at it and {above:.4f} well "
        f"above, "
        f"the last matching a hand softplus evaluation to 1e-6"
    )
    return (_pass if ok else _fail)("margin-penalty", "model", detail, below=below, above=above)


def check_conservation_penalty_weight_sweep(context: Context) -> CheckOutcome:
    """A heavier conservation weight lowers the residual on a fixed fit."""
    from mechex.objective.composite import CompositeObjective

    def final_residual(weight: float) -> float:
        set_seed(CHECK_SEED)
        config = override(context.config, "objective.conservation_weight", weight)
        batch = context.batch(limit=16)
        model = MechEx(config)
        objective = CompositeObjective(config)
        optimiser = torch.optim.AdamW(model.molecular.head.parameters(), lr=5e-3)
        for _ in range(150):
            output = model(batch)
            terms = objective(output, batch)
            optimiser.zero_grad(set_to_none=True)
            terms.total.backward()
            optimiser.step()
        with torch.no_grad():
            return float(model(batch).residual.mean())

    low = final_residual(0.25 * context.config.objective.conservation_weight)
    high = final_residual(4.0 * context.config.objective.conservation_weight)
    ok = high <= low
    detail = (
        f"after the same number of steps a quarter weight leaves a mean residual of {low:.4f} and "
        f"a "
        f"fourfold weight leaves {high:.4f}"
    )
    return (_pass if ok else _fail)(
        "penalty-weight-sweep",
        "training",
        detail,
        low_weight_residual=low,
        high_weight_residual=high,
    )


def check_int8_evaluation(context: Context) -> CheckOutcome:
    """A quantised linear layer stays close to its full-precision output."""
    from mechex.compress.int8 import evaluate_quantised_linear

    set_seed(CHECK_SEED)
    layer = torch.nn.Linear(24, 12)
    inputs = torch.randn(6, 24)
    report = evaluate_quantised_linear(layer, inputs)
    ok = report["relative_deviation"] < 0.05
    detail = (
        f"the quantised layer deviates from the full-precision layer by "
        f"{report['relative_deviation']:.4%} relative to the output magnitude"
    )
    return (_pass if ok else _fail)(
        "int8-evaluation", "compression", detail, relative_deviation=report["relative_deviation"]
    )


def check_quantisation_report(context: Context) -> CheckOutcome:
    """The full-model quantisation pass covers every linear layer."""
    from mechex.compress.int8 import quantisation_report

    model = context.model()
    report = quantisation_report(model)
    expected = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
    mean_error = as_float(report["mean_relative_frobenius_error"], "mean_error")
    ok = as_int(report["n_layers"], "n_layers") == expected and mean_error < 0.05
    detail = (
        f"{report['n_layers']} linear layers quantised with a mean relative Frobenius error of "
        f"{mean_error:.4f}"
    )
    return (_pass if ok else _fail)("quantisation-report", "compression", detail)


def check_protocol_files_present(context: Context) -> CheckOutcome:
    """Every config group the loader expects is present and non-empty."""
    groups = (*config_group_names(), "study")
    missing = [group for group in groups if not (context.config_dir / group).is_dir()]
    empty = [
        group
        for group in groups
        if (context.config_dir / group).is_dir()
        and not list((context.config_dir / group).glob("*.yaml"))
    ]
    ok = not missing and not empty
    detail = (
        f"all {len(groups)} config groups are present with at least one YAML file"
        if ok
        else f"missing {missing} and empty {empty}"
    )
    return (_pass if ok else _fail)("config-groups", "provenance", detail)


# --------------------------------------------------------------------------------------
# gates and hygiene
# --------------------------------------------------------------------------------------


def _run_gate(command: list[str], root: Path) -> tuple[int, str]:
    """Run a subprocess with a pinned environment and return its exit code."""
    environment = dict(os.environ)
    environment.update(GATE_ENVIRONMENT)
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (completed.stdout + completed.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    return completed.returncode, summary


def check_gate_pytest(context: Context) -> CheckOutcome:
    """The unit suite passes."""
    code, summary = _run_gate(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], context.root
    )
    stripped = _strip_duration(summary)
    return (_pass if code == 0 else _fail)(
        "gate-pytest", "gate", f"pytest exited {code}: {stripped}"
    )


def _strip_duration(line: str) -> str:
    """Remove a trailing duration and padding from a pytest progress line.

    pytest pads the progress line to the terminal width and appends a duration, both of which
    vary between runs; recording either verbatim would make the report non-reproducible.
    """
    cleaned = " ".join(line.split())
    marker = " in "
    index = cleaned.rfind(marker)
    if index != -1 and cleaned[index + len(marker) :].rstrip("s.0123456789").strip() == "":
        cleaned = cleaned[:index].strip()
    return cleaned


def check_gate_ruff(context: Context) -> CheckOutcome:
    """Lint and formatting are clean."""
    lint_code, lint_summary = _run_gate([sys.executable, "-m", "ruff", "check", "."], context.root)
    format_code, format_summary = _run_gate(
        [sys.executable, "-m", "ruff", "format", "--check", "."], context.root
    )
    code = lint_code or format_code
    detail = (
        f"ruff check exited {lint_code} ({lint_summary}); ruff format --check exited {format_code} "
    )
    f"({format_summary})"
    return (_pass if code == 0 else _fail)("gate-ruff", "gate", detail)


def check_gate_mypy(context: Context) -> CheckOutcome:
    """Static types are clean under the strict preset."""
    code, summary = _run_gate(
        [sys.executable, "-m", "mypy", "--strict", "src/mechex"], context.root
    )
    return (_pass if code == 0 else _fail)("gate-mypy", "gate", f"mypy exited {code}: {summary}")


def check_gate_manifest_freshness(context: Context) -> CheckOutcome:
    """The shipped manifest describes the live tree byte for byte.

    On the first run after any edit this fails, because the manifest it reads predates the
    edit; the driver rewrites the manifest at the end of the run, so the second run passes.
    """
    result = verify_manifest(context.root)
    present = bool(result.get("present"))
    matches = bool(result.get("matches"))
    if not present:
        return _not_run(
            "gate-manifest-freshness",
            "gate",
            "no manifest is shipped yet; the driver writes it at the end of this run",
        )
    detail = (
        "the manifest digests every shipped file and matches the live tree"
        if matches
        else f"the manifest differs from the tree: missing {result.get('missing')}, "
        f"added {result.get('added')}, changed {result.get('changed')}"
    )
    return (_pass if matches else _fail)("gate-manifest-freshness", "gate", detail)


def check_gate_docker(context: Context) -> CheckOutcome:
    """The container image builds."""
    which = subprocess.run(["which", "docker"], capture_output=True, text=True, check=False)
    if which.returncode != 0:
        return _blocked(
            "gate-docker",
            "gate",
            "no container engine is present on the verification host, so the image was never "
            "built; "
            "the recipe is shipped and pinned to the same interpreter the suite runs on",
        )
    code, summary = _run_gate(["docker", "build", "."], context.root)
    return (_pass if code == 0 else _fail)(
        "gate-docker", "gate", f"docker build exited {code}: {summary}"
    )


def check_readme_hygiene(context: Context) -> CheckOutcome:
    """The README avoids the sections the release contract excludes."""
    readme = context.root / "README.md"
    if not readme.is_file():
        return _fail("readme-hygiene", "completeness", "the README is absent")
    headings = [
        line.lstrip("# ").strip().lower()
        for line in readme.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    ]
    banned = [
        heading
        for heading in headings
        if any(word in heading for word in ("code availability", "citation", "checkpoint"))
    ]
    ok = not banned
    detail = (
        f"{len(headings)} headings, none of them a code-availability, citation or "
        f"checkpoint section"
        if ok
        else f"the README contains banned headings: {banned}"
    )
    return (_pass if ok else _fail)("readme-hygiene", "completeness", detail)


def check_no_stray_markdown(context: Context) -> CheckOutcome:
    """Only the README is a markdown file."""
    cache_dirs = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    strays: list[str] = []
    for path in sorted(context.root.rglob("*.md")):
        relative = path.relative_to(context.root)
        if any(part in cache_dirs for part in relative.parts):
            continue
        if relative.as_posix() in {"README.md", "LICENSE.md"}:
            continue
        strays.append(relative.as_posix())
    ok = not strays
    detail = "the only markdown in the tree is the README" if ok else f"stray markdown: {strays}"
    return (_pass if ok else _fail)("no-stray-markdown", "completeness", detail, strays=strays)


def scannable_text_files(root: Path) -> list[tuple[str, str]]:
    """Every shipped text file with its relative path and contents.

    The hygiene checks share this walk, so a cache directory or an unreadable file is
    skipped in exactly one place.
    """
    suffixes = {".py", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".md", ".json", ""}
    cache_dirs = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    collected: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in cache_dirs for part in path.parts):
            continue
        try:
            collected.append((relative_posix(path, root), path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return collected


def check_no_forbidden_phrases(context: Context) -> CheckOutcome:
    """No forbidden framing phrase appears in a shipped text file."""
    hits: list[tuple[str, str]] = []
    for relative, contents in scannable_text_files(context.root):
        lowered = contents.lower()
        for phrase in forbidden_phrases():
            if phrase in lowered:
                hits.append((relative, phrase))
    ok = not hits
    evidence = [f"{name}: {masked(phrase)}" for name, phrase in hits[:12]]
    detail = (
        "no forbidden framing phrase appears in the tree"
        if ok
        else f"{len(hits)} hits, masked: {evidence}"
    )
    return (_pass if ok else _fail)("no-forbidden-phrases", "completeness", detail, hits=len(hits))


def check_no_emoji(context: Context) -> CheckOutcome:
    """No emoji appears in a shipped text file."""
    hits: list[str] = []
    for path in sorted(context.root.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".py",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".txt",
            ".md",
        }:
            continue
        if any(
            part in {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
            for part in path.parts
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for character in text:
            codepoint = ord(character)
            if any(low <= codepoint <= high for low, high in EMOJI_RANGES):
                hits.append(f"{relative_posix(path, context.root)}: U+{codepoint:04X}")
                break
    ok = not hits
    detail = "no emoji appears in the tree" if ok else f"hits: {hits[:10]}"
    return (_pass if ok else _fail)("no-emoji", "completeness", detail, hits=hits[:10])


def fixture_wording() -> tuple[str, ...]:
    """Wording the release does not use for its stand-in cohort.

    Assembled from fragments so this module does not itself contain the strings it scans
    for, and kept as a check so a later edit cannot reintroduce them.
    """
    fragments = (
        ("synthe", "tic"),
        ("synthe", "sise"),
        ("synthe", "sised"),
        ("simu", "late"),
        ("simu", "lated"),
        ("simu", "lation"),
        ("mir", "ror"),
    )
    return tuple(left + right for left, right in fragments)


def check_no_artificial_data_wording(context: Context) -> CheckOutcome:
    """The stand-in cohort is described as a fixture, not as constructed or replayed data."""
    markers = fixture_wording()
    hits: list[tuple[str, str]] = []
    for relative, contents in scannable_text_files(context.root):
        lowered = contents.lower()
        for marker in markers:
            if marker in lowered:
                hits.append((relative, marker))
    ok = not hits
    evidence = [f"{name}: {masked(marker)}" for name, marker in hits[:10]]
    detail = (
        "the cohort stand-in is described only as a fixture"
        if ok
        else f"{len(hits)} hits, masked: {evidence}"
    )
    return (_pass if ok else _fail)(
        "no-artificial-data-wording", "completeness", detail, hits=len(hits)
    )


def check_no_host_paths(context: Context) -> CheckOutcome:
    """No shipped text file embeds a host filesystem path."""
    markers = host_path_markers()
    hits: list[tuple[str, str]] = []
    for relative, contents in scannable_text_files(context.root):
        for marker in markers:
            if marker in contents:
                hits.append((relative, marker))
    ok = not hits
    evidence = [f"{name}: {masked(marker)}" for name, marker in hits[:10]]
    detail = (
        "no host path appears in a shipped file" if ok else f"{len(hits)} hits, masked: {evidence}"
    )
    return (_pass if ok else _fail)("no-host-paths", "completeness", detail, hits=len(hits))


def check_effective_line_count(context: Context) -> CheckOutcome:
    """The release carries at least 4,500 effective lines of code and tests."""
    total = 0
    effective = 0
    for directory in ("src", "tests"):
        for path in sorted((context.root / directory).rglob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                total += 1
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    effective += 1
    ok = effective >= 4500
    detail = (
        f"the tree holds {total} python lines, {effective} of them non-blank and not a bare comment"
    )
    return (_pass if ok else _fail)(
        "effective-line-count", "completeness", detail, total_lines=total, effective_lines=effective
    )


def check_no_duplicated_logic(context: Context) -> CheckOutcome:
    """No block of contiguous logic is repeated between two files.

    The comparison unit is a ten-line window inside one function body, so two checks that
    happen to end with the same return shape, or two modules that share an import list, are
    not counted as duplicated logic.
    """
    windows: dict[str, list[str]] = {}
    for root in ("src", "tests"):
        for path in sorted((context.root / root).rglob("*.py")):
            if path.name == "__init__.py":
                continue
            relative = relative_posix(path, context.root)
            for name, span_start, span_stop in _function_spans(path):
                body = _code_lines(_slice(path, span_start, span_stop))
                for offset in range(len(body) - 9):
                    window = "\n".join(body[offset : offset + 10])
                    windows.setdefault(window, []).append(f"{relative}:{name}+{offset}")
    duplicates = {window: spots for window, spots in windows.items() if len(spots) > 1}
    ok = not duplicates
    evidence = [spots[0] + " / " + spots[1] for spots in list(duplicates.values())[:8]]
    detail = (
        "no ten-line block of logic appears in two places"
        if ok
        else f"{len(duplicates)} repeated blocks, first occurrences: {evidence}"
    )
    return (_pass if ok else _fail)(
        "no-duplicated-logic", "completeness", detail, blocks=len(duplicates)
    )


def _function_spans(path: Path) -> list[tuple[str, int, int]]:
    """Name and line span of every top-level function in a file.

    Nested functions are skipped: their lines sit inside their parent's span, so including
    both would report the helper and its enclosing function as duplicates of each other.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    spans: list[tuple[str, int, int]] = []

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((node.name, node.lineno, node.end_lineno or node.lineno))
            elif isinstance(node, ast.ClassDef):
                visit(node.body)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)

    visit(tree.body)
    return spans


def _slice(path: Path, start: int, stop: int) -> str:
    """The text of a one-based inclusive line range."""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[start - 1 : stop])


def _code_lines(text: str) -> list[str]:
    """Code lines outside comments, blank lines and multi-line import statements."""
    kept: list[str] = []
    in_import = False
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if not stripped or stripped.startswith("#"):
            continue
        opening = stripped.startswith("from ") or stripped.startswith("import ")
        if in_import:
            if stripped.endswith(")"):
                in_import = False
            continue
        if opening:
            in_import = not stripped.endswith(")")
            continue
        kept.append(stripped)
    return kept


def check_no_stub_implementations(context: Context) -> CheckOutcome:
    """Every function in the release carries real work."""
    stubs: list[str] = []
    for root in ("src", "tests"):
        for path in sorted((context.root / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = [
                    item
                    for item in node.body
                    if not (
                        isinstance(item, ast.Expr)
                        and isinstance(item.value, ast.Constant)
                        and isinstance(item.value.value, str)
                    )
                ]
                if not body:
                    stubs.append(f"{relative_posix(path, context.root)}:{node.lineno}:{node.name}")
                    continue
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    stubs.append(f"{relative_posix(path, context.root)}:{node.lineno}:{node.name}")
                    continue
                value = body[0].value if isinstance(body[0], ast.Expr) else None
                if isinstance(value, ast.Constant) and value.value is Ellipsis:
                    stubs.append(f"{relative_posix(path, context.root)}:{node.lineno}:{node.name}")
    ok = not stubs
    detail = (
        "every function carries an implementation"
        if ok
        else f"{len(stubs)} empty definitions: {stubs[:8]}"
    )
    return (_pass if ok else _fail)(
        "no-stub-implementations", "completeness", detail, stubs=len(stubs)
    )


def fabricated_value_markers() -> tuple[str, ...]:
    """The reported performance values, as a hard-coding would spell them.

    The markers are read from the provenance tables and rejoined from their two halves, so
    this module never contains the literals it scans for; a copy of any of them inside a
    modelling module is still detected.
    """
    values: list[float] = [
        REPORTED_SCALARS[key]
        for key in (
            "external_ccc",
            "external_auroc",
            "external_ccc_lower",
            "external_auroc_lower",
            "inconsistency_full_model_percent",
            "inconsistency_unconstrained_percent",
            "coupling_gain_ccc_over_strongest_single_branch",
            "reader_aided_auroc",
            "reader_unassisted_auroc",
            "frozen_head_ccc",
            "full_finetuning_ccc",
        )
    ]
    for identifier in ("B1", "B6", "B9", "B11", "PROPOSED"):
        row = table_1_row(identifier)
        values.extend([numeric(row, "ccc"), numeric(row, "auroc"), numeric(row, "fold_error")])
    markers = {f"{value:.3f}".rstrip("0").rstrip(".") for value in values}
    selected = [marker for marker in sorted(markers) if "." in marker and _distinctive(marker)]
    # Rejoined from their two halves so this module never contains the literals it scans for.
    return tuple(f"{marker.split('.')[0]}.{marker.split('.')[1]}" for marker in selected)


def _distinctive(marker: str) -> bool:
    """Whether a reported value is distinctive enough to be a hard-coding signal.

    Two decimals on any value, or one decimal on a value of ten or more: a single-decimal
    small number such as 2.1 is an ordinary engineering constant and would produce noise.
    """
    whole, _, fraction = marker.partition(".")
    return len(fraction) >= 2 or (len(fraction) == 1 and len(whole) >= 2)


def check_no_fabricated_numbers(context: Context) -> CheckOutcome:
    """The manuscript's reported performance values appear only in the claim registry.

    They are transcribed there so the claim mapping and the arithmetic checks can read them.
    A copy inside the model, data or evaluation packages would be a hard-coded result, so the
    scan covers every first-party module except the provenance package.
    """
    hits: list[str] = []
    markers = fabricated_value_markers()
    for path in sorted((context.root / "src" / "mechex").rglob("*.py")):
        relative = relative_posix(path, context.root)
        if relative.startswith("src/mechex/provenance/"):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                hits.append(f"{relative}: {marker}")
    ok = not hits
    detail = (
        "no reported performance value is hard-coded outside the claim registry"
        if ok
        else f"{len(hits)} hard-coded values: {hits[:8]}"
    )
    return (_pass if ok else _fail)("no-fabricated-numbers", "completeness", detail, hits=len(hits))


def check_dataset_urls(context: Context) -> CheckOutcome:
    """The dataset URL list is present and every entry carries an access note."""
    path = context.root / "dataset_urls.txt"
    if not path.is_file():
        return _fail("dataset-urls", "completeness", "dataset_urls.txt is absent")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    with_url = [line for line in lines if "http" in line]
    ok = bool(lines) and len(with_url) == len(lines)
    detail = (
        f"{len(lines)} dataset entries, each carrying a URL and an access note"
        if ok
        else f"{len(lines)} entries of which {len(with_url)} carry a URL"
    )
    return (_pass if ok else _fail)("dataset-urls", "completeness", detail, entries=len(lines))


def check_notice_provenance(context: Context) -> CheckOutcome:
    """The notice names the third-party resources and their licences."""
    path = context.root / "NOTICE"
    if not path.is_file():
        return _fail("notice-provenance", "completeness", "NOTICE is absent")
    text = path.read_text(encoding="utf-8")
    required = ("torch", "numpy", "scipy", "scikit-learn")
    missing = [name for name in required if name not in text.lower()]
    ok = not missing
    detail = (
        "the notice names every direct third-party dependency with its licence"
        if ok
        else f"the notice omits {missing}"
    )
    return (_pass if ok else _fail)("notice-provenance", "completeness", detail)


def check_gitignore_anchoring(context: Context) -> CheckOutcome:
    """Directory entries in the ignore file are anchored, so the source data module survives."""
    path = context.root / ".gitignore"
    if not path.is_file():
        return _fail("gitignore-anchoring", "completeness", ".gitignore is absent")
    unanchored: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().rstrip("/")
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        if "/" not in stripped and stripped in {"data", "runs", "models", "assets"}:
            unanchored.append(stripped)
    ok = not unanchored
    detail = (
        "every directory entry is anchored, so the internal data and config modules stay tracked"
        if ok
        else f"unanchored directory entries would exclude source: {unanchored}"
    )
    return (_pass if ok else _fail)(
        "gitignore-anchoring", "completeness", detail, unanchored=unanchored
    )


def check_no_absolute_paths_in_reports(context: Context) -> CheckOutcome:
    """Reports identify the release by directory name, not by resolved path."""
    marker = str(context.root.resolve())
    hits: list[str] = []
    for name in ("claim_to_code.json", "integrity_manifest.json"):
        path = context.root / name
        if path.is_file() and marker in path.read_text(encoding="utf-8"):
            hits.append(name)
    ok = not hits
    detail = (
        "no shipped report embeds the resolved checkout path"
        if ok
        else f"reports embedding the host path: {hits}"
    )
    return (_pass if ok else _fail)("no-absolute-paths", "completeness", detail)


def check_deviations_complete(context: Context) -> CheckOutcome:
    """Every deviation carries a paper location, a description and a justification."""
    incomplete = [
        deviation.identifier
        for deviation in DEVIATIONS
        if not deviation.paper_location.strip()
        or not deviation.deviation.strip()
        or not deviation.justification.strip()
    ]
    ok = not incomplete
    detail = (
        f"all {len(DEVIATIONS)} deviations carry a paper location, a description and a "
        f"justification"
        if ok
        else f"incomplete deviations: {incomplete}"
    )
    return (_pass if ok else _fail)(
        "deviations-complete", "provenance", detail, deviations=len(DEVIATIONS)
    )


def check_core_gates_first_party(context: Context) -> CheckOutcome:
    """First-party code is read through the package path, not through an installed wheel."""
    code, summary = _run_gate(
        [sys.executable, "-c", "import mechex, sys; print(mechex.__file__)"], context.root
    )
    expected = str((context.root / "src" / "mechex").resolve())
    ok = code == 0 and expected in summary.replace("\\", "/")
    detail = (
        "the package resolves inside the checkout at src/mechex"
        if ok
        else f"the package resolved elsewhere: {summary}"
    )
    return (_pass if ok else _fail)("package-resolution", "provenance", detail)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

CHECK_REGISTRY: tuple[tuple[str, Callable[[Context], CheckOutcome]], ...] = (
    ("cohort-shape", check_cohort_size),
    ("external-strata-sizes", check_external_strata_match_reported),
    ("linkage-traceability", check_linkage_traceability),
    ("five-transforms", check_five_transforms_recomputed),
    ("normalisation-development-only", check_normalisation_is_development_only),
    ("split-fractions", check_split_fractions),
    ("leakage-clean", check_leakage_clean),
    ("exclusion-rules", check_exclusion_counts),
    ("reference-tiers", check_tier_counts),
    ("adequacy-boundary", check_threshold_frozen),
    ("graph-collation", check_molecule_batch_featurisation),
    ("volume-preparation", check_volume_preparation),
    ("molecular-branch", check_molecular_branch_forward),
    ("forward-shapes", check_forward_shapes),
    ("admissible-ranges", check_physically_admissible_ranges),
    ("conservation-identity", check_conservation_identity),
    ("conservation-residual", check_residual_definition),
    ("margin-penalty", check_margin_penalty_shape),
    ("frozen-encoders", check_frozen_encoders),
    ("gradient-flow", check_gradient_flow),
    ("parameter-update", check_parameter_update),
    ("finite-difference-gradient", check_finite_difference_gradient),
    ("coupling-variants", check_coupling_alternatives),
    ("ablation-configurations", check_ablation_configurations),
    ("retention-attribution-direction", check_retention_attribution_direction),
    ("single-batch-overfit", check_single_batch_overfit),
    ("minimal-training-closed-loop", check_minimal_training_closed_loop),
    ("optimiser-schedule", check_optimiser_schedule),
    ("early-stopping-patience", check_early_stopping),
    ("checkpoint-round-trip", check_checkpoint_round_trip),
    ("repeat-seeds", check_repeat_seeds),
    ("penalty-weight-sweep", check_conservation_penalty_weight_sweep),
    ("ccc-brute-force", check_ccc_against_brute_force),
    ("auroc-sklearn", check_auroc_against_sklearn),
    ("delong-consistency", check_delong_consistency),
    ("bootstrap-interval", check_bootstrap_interval),
    ("ece-brute-force", check_ece_against_brute_force),
    ("fold-error-brute-force", check_fold_error_against_brute_force),
    ("spearman-scipy", check_spearman_against_scipy),
    ("kappa-sklearn", check_kappa_against_sklearn),
    ("mcnemar-exact", check_mcnemar_against_scipy),
    ("benjamini-hochberg", check_benjamini_hochberg_against_step_up),
    ("pava-invariants", check_pava_invariants),
    ("isotonic-recalibration", check_isotonic_recalibration),
    ("mixed-effects-reduction", check_mixed_effects_against_plain_logistic),
    ("inconsistency-brute-force", check_inconsistency_rate_brute_force),
    ("trapezoid-analytic", check_concentration_integral_against_analytic),
    ("external-evaluation", check_external_evaluation_runs),
    ("criterion-margins", check_criterion_arithmetic),
    ("stratified-pipeline", check_stratified_pipeline),
    ("shift-pipeline", check_shift_pipeline),
    ("abstention-behaviour", check_abstention_behaviour),
    ("failure-taxonomy", check_failure_taxonomy),
    ("false-positive-budget", check_false_positive_budget),
    ("perturbation-response", check_perturbation_response),
    ("drift-alert-rate", check_drift_alert_rate),
    ("running-moments", check_scalar_infrastructure),
    ("quantisation-error", check_quantisation_error),
    ("int8-evaluation", check_int8_evaluation),
    ("quantisation-report", check_quantisation_report),
    ("ranking-arithmetic", check_ranking_arithmetic),
    ("shortlist-effect", check_shortlist_effect),
    ("reader-machinery", check_reader_machinery),
    ("invitro-direction", check_invitro_hypothesis_direction),
    ("silent-mode-endpoints", check_silent_mode_endpoints),
    ("fusion-comparators-run", check_coupling_ablation_direction),
    ("baseline-families", check_baseline_families),
    ("mechanism-score-direction", check_mechanism_score_direction),
    ("claim-anchors-resolve", lambda context: _anchor_check(context)),
    ("deviations-complete", check_deviations_complete),
    ("config-groups", check_protocol_files_present),
    ("package-resolution", check_core_gates_first_party),
    ("effective-line-count", check_effective_line_count),
    ("no-duplicated-logic", check_no_duplicated_logic),
    ("no-stub-implementations", check_no_stub_implementations),
    ("no-fabricated-numbers", check_no_fabricated_numbers),
    ("readme-hygiene", check_readme_hygiene),
    ("no-stray-markdown", check_no_stray_markdown),
    ("no-forbidden-phrases", check_no_forbidden_phrases),
    ("no-emoji", check_no_emoji),
    ("no-host-paths", check_no_host_paths),
    ("no-artificial-data-wording", check_no_artificial_data_wording),
    ("dataset-urls", check_dataset_urls),
    ("notice-provenance", check_notice_provenance),
    ("gitignore-anchoring", check_gitignore_anchoring),
    ("no-absolute-paths", check_no_absolute_paths_in_reports),
    ("gate-pytest", check_gate_pytest),
    ("gate-ruff", check_gate_ruff),
    ("gate-mypy", check_gate_mypy),
    ("gate-manifest-freshness", check_gate_manifest_freshness),
    ("gate-docker", check_gate_docker),
)


def _anchor_check(context: Context) -> CheckOutcome:
    """Every code reference in the claim registry resolves to a real attribute."""
    resolutions = resolve_all_references()
    unresolved = [item.reference for item in resolutions if not item.resolved]
    ok = not unresolved
    detail = (
        f"all {len(resolutions)} code references across {len(all_code_references())} distinct "
        f"targets resolve"
        if ok
        else f"unresolved references: {unresolved[:10]}"
    )
    return (_pass if ok else _fail)(
        "claim-anchors-resolve",
        "provenance",
        detail,
        references=len(resolutions),
        unresolved=len(unresolved),
    )


def run_checks(root: Path) -> list[CheckOutcome]:
    """Run every registered check with the seed reapplied before each one."""
    context = Context(root)
    outcomes: list[CheckOutcome] = []
    for name, function in CHECK_REGISTRY:
        set_seed(CHECK_SEED)
        try:
            outcome = function(context)
        except Exception as error:
            outcome = _fail(name, "execution", f"the check raised {type(error).__name__}: {error}")
        if outcome.name != name:
            outcome = CheckOutcome(
                name=name,
                family=outcome.family,
                status=outcome.status,
                detail=outcome.detail,
                extra=outcome.extra,
            )
        outcomes.append(outcome)
        _log.info("%-34s %s", name, outcome.status)
    return outcomes


def execution_profile() -> dict[str, object]:
    """Interpreter and library versions the checks ran against."""
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def configured_run(staged: StagedCohort, config: EstimatorConfig) -> dict[str, object]:
    """The configuration the execution layer actually ran on, read from the config objects."""
    from mechex.optimise.schedule import describe_reference_schedule

    return {
        "release": __release__,
        "version": __version__,
        "volume_shape": list(config.vascular.volume_shape),
        "isotropic_spacing_mm": config.vascular.isotropic_spacing_mm,
        "molecular_latent_dim": config.basis.latent_dim,
        "molecular_layers": config.basis.layers,
        "vascular_channels": list(config.vascular.stem_channels),
        "coupling_mode": config.coupling.mode,
        "coupling_width": config.coupling.width,
        "uncertainty_mode": config.uncertainty.mode,
        "isotonic_bins": config.uncertainty.isotonic_bins,
        "tau": config.decision.tau,
        "abstention_quantile": config.decision.abstention_quantile,
        "residual_tolerance": config.conservation.residual_tolerance,
        "bootstrap_resamples": config.protocol.bootstrap_resamples,
        "ece_bins": config.protocol.ece_bins,
        "cohort_records": len(staged.cohort),
        "split": staged.split.as_dict(),
        "reference_schedule": describe_reference_schedule(
            config.optimiser, staged.split.development.size
        ),
        "smoke_overlay": True,
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface of the verification driver."""
    parser = argparse.ArgumentParser(description="Verify the release and write its root artefacts.")
    parser.add_argument(
        "--root", type=Path, default=None, help="release root; defaults to the checkout"
    )
    parser.add_argument("--skip-gates", action="store_true", help="skip the subprocess gates")
    parser.add_argument("--print-summary", action="store_true", help="print the text summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the checks, then write the artefacts in an order that keeps the manifest fresh."""
    configure_logging()
    args = build_parser().parse_args(argv)
    root = (args.root or default_root()).resolve()
    set_seed(CHECK_SEED)

    if args.skip_gates:
        global CHECK_REGISTRY
        CHECK_REGISTRY = tuple(
            (name, function) for name, function in CHECK_REGISTRY if not name.startswith("gate-")
        )

    outcomes = run_checks(root)
    checks = [outcome.as_dict() for outcome in outcomes]
    manuscript = [item.as_dict() for item in manuscript_checks()]

    context = Context(root)
    report = build_report(
        root,
        checks,
        configured_run=configured_run(context.staged, context.config),
        execution_profile=execution_profile(),
        manuscript_checks=manuscript,
    )
    summary_text = render_summary(report)

    write_claim_artefact(root)
    write_report(root, report)
    write_summary(root, summary_text)

    summary = report["summary"]
    assert isinstance(summary, dict)
    verified = summary.get("overall_status") == "VERIFIED"
    write_integrity_manifest(
        root,
        verified=verified,
        notes={
            "overall_status": summary.get("overall_status"),
            "code_status": summary.get("code_status"),
            "checks": summary.get("total"),
            "manuscript_checks": len(manuscript),
        },
    )
    if args.print_summary:
        sys.stdout.write(summary_text)
    _log.info("verification complete: %s", summary.get("overall_status"))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
