"""Numeric tables transcribed from the manuscript.

Ref: Table 1, Table 2, Table 3 and the reported quantities in the abstract, Sec. 2.2,
Sec. 2.3, Sec. 2.4, Sec. 2.5, Sec. 2.6, Sec. 2.8, Sec. 2.9, Sec. 3.10, Sec. 3.11 and
Sec. 4.

These are the manuscript's own numbers, kept in one place so that the code-to-claim
mapping, the reconstructed comparators and the arithmetic checks all read the same source
rather than restating a value in three modules.
"""

from __future__ import annotations

from mechex.utils.coerce import as_float, as_text

# --- Table 1: main comparison on the shared patient-level test set ---------------------
TABLE_1_HEADER = (
    "id",
    "method",
    "ccc",
    "auroc",
    "fold_error",
    "params_millions",
    "inconsistency_percent",
)

TABLE_1_ROWS: tuple[dict[str, object], ...] = (
    {
        "id": "B12",
        "method": "Contrastive-alignment fusion",
        "ccc": 0.744,
        "auroc": 0.847,
        "fold_error": 1.84,
        "params_millions": 77.4,
        "inconsistency_percent": 12.5,
    },
    {
        "id": "B11",
        "method": "Cross-attention fusion (unconstrained)",
        "ccc": 0.813,
        "auroc": 0.901,
        "fold_error": 1.51,
        "params_millions": 81.2,
        "inconsistency_percent": 18.7,
    },
    {
        "id": "B10",
        "method": "Early-concatenation fusion",
        "ccc": 0.762,
        "auroc": 0.858,
        "fold_error": 1.79,
        "params_millions": 79.8,
        "inconsistency_percent": 15.3,
    },
    {
        "id": "B6",
        "method": "Volumetric CTA encoder (imaging only)",
        "ccc": 0.688,
        "auroc": 0.806,
        "fold_error": 2.11,
        "params_millions": 38.4,
        "inconsistency_percent": 9.8,
    },
    {
        "id": "B7",
        "method": "CTA radiomics + gradient boosting",
        "ccc": 0.604,
        "auroc": 0.742,
        "fold_error": 2.66,
        "params_millions": 0.2,
        "inconsistency_percent": 7.1,
    },
    {
        "id": "B1",
        "method": "Molecular foundation transformer (molecular only)",
        "ccc": 0.641,
        "auroc": 0.771,
        "fold_error": 2.34,
        "params_millions": 42.6,
        "inconsistency_percent": 5.4,
    },
    {
        "id": "B3",
        "method": "Geometric graph neural network",
        "ccc": 0.612,
        "auroc": 0.752,
        "fold_error": 2.58,
        "params_millions": 31.2,
        "inconsistency_percent": 4.9,
    },
    {
        "id": "B2",
        "method": "Descriptor machine learning",
        "ccc": 0.588,
        "auroc": 0.734,
        "fold_error": 2.71,
        "params_millions": 0.4,
        "inconsistency_percent": 3.8,
    },
    {
        "id": "B4",
        "method": "Standard PBPK (population priors)",
        "ccc": 0.612,
        "auroc": 0.761,
        "fold_error": 2.62,
        "params_millions": 0.0,
        "inconsistency_percent": 6.4,
    },
    {
        "id": "B5",
        "method": "High-throughput PBPK",
        "ccc": 0.566,
        "auroc": 0.718,
        "fold_error": 2.94,
        "params_millions": 0.0,
        "inconsistency_percent": 5.1,
    },
    {
        "id": "B8",
        "method": "Clinical logistic regression",
        "ccc": 0.552,
        "auroc": 0.703,
        "fold_error": 3.05,
        "params_millions": 0.0,
        "inconsistency_percent": 2.9,
    },
    {
        "id": "B9",
        "method": "Mechanism-only repurposing score",
        "ccc": 0.431,
        "auroc": 0.628,
        "fold_error": 3.62,
        "params_millions": 0.0,
        "inconsistency_percent": 1.2,
    },
    {
        "id": "PROPOSED",
        "method": "MechEx",
        "ccc": 0.821,
        "auroc": 0.897,
        "fold_error": 1.42,
        "params_millions": 84.3,
        "inconsistency_percent": 2.1,
    },
)

# --- Table 2: ablation across three tiers --------------------------------------------
TABLE_2_HEADER = ("tier", "variant", "ccc", "auroc", "inconsistency_percent", "ece")

TABLE_2_ROWS: tuple[dict[str, object], ...] = (
    {
        "tier": "All",
        "variant": "Full MechEx",
        "ccc": 0.821,
        "auroc": 0.897,
        "inconsistency_percent": 2.1,
        "ece": 0.031,
    },
    {
        "tier": "1",
        "variant": "No molecular branch",
        "ccc": 0.688,
        "auroc": 0.806,
        "inconsistency_percent": 9.8,
        "ece": 0.042,
    },
    {
        "tier": "1",
        "variant": "No imaging branch",
        "ccc": 0.641,
        "auroc": 0.771,
        "inconsistency_percent": 5.4,
        "ece": 0.038,
    },
    {
        "tier": "1",
        "variant": "No uncertainty head",
        "ccc": 0.818,
        "auroc": 0.894,
        "inconsistency_percent": 2.3,
        "ece": 0.089,
    },
    {
        "tier": "1",
        "variant": "No decision layer (regression only)",
        "ccc": 0.819,
        "auroc": None,
        "inconsistency_percent": 2.2,
        "ece": 0.033,
    },
    {
        "tier": "2",
        "variant": "Concatenation instead of cross-attention",
        "ccc": 0.779,
        "auroc": 0.865,
        "inconsistency_percent": 11.4,
        "ece": 0.047,
    },
    {
        "tier": "2",
        "variant": "Single-branch-then-stack backbone",
        "ccc": 0.735,
        "auroc": 0.829,
        "inconsistency_percent": 13.8,
        "ece": 0.052,
    },
    {
        "tier": "2",
        "variant": "Both encoders randomly initialised",
        "ccc": 0.712,
        "auroc": 0.815,
        "inconsistency_percent": 15.1,
        "ece": 0.061,
    },
    {
        "tier": "2",
        "variant": "Molecular encoder frozen without adaptation",
        "ccc": 0.786,
        "auroc": 0.869,
        "inconsistency_percent": 8.7,
        "ece": 0.044,
    },
    {
        "tier": "3",
        "variant": "No conservation constraint",
        "ccc": 0.829,
        "auroc": 0.901,
        "inconsistency_percent": 18.7,
        "ece": 0.034,
    },
    {
        "tier": "3",
        "variant": "Constraint weight x0.25",
        "ccc": 0.826,
        "auroc": 0.900,
        "inconsistency_percent": 9.4,
        "ece": 0.032,
    },
    {
        "tier": "3",
        "variant": "Constraint weight x4",
        "ccc": 0.806,
        "auroc": 0.884,
        "inconsistency_percent": 1.3,
        "ece": 0.036,
    },
    {
        "tier": "3",
        "variant": "Uncertainty head replaced by deep ensemble",
        "ccc": 0.819,
        "auroc": 0.896,
        "inconsistency_percent": 2.2,
        "ece": 0.038,
    },
    {
        "tier": "I-1+I-2",
        "variant": "Interaction (synergy) excess",
        "ccc": 0.078,
        "auroc": 0.053,
        "inconsistency_percent": 2.1,
        "ece": 0.031,
    },
)

TABLE_2_CONSTRAINT_WEIGHTS = {"x0.25": 0.25, "x1": 1.0, "x4": 4.0}

# --- Table 3: stratified performance at the external site -----------------------------
TABLE_3_OVERALL_CCC = 0.814
TABLE_3_OVERALL_AUROC = 0.893
TABLE_3_PRESPECIFIED_THRESHOLD = 0.75
TABLE_3_CONSISTENCY_LIMIT_POINTS = 5.0

TABLE_3_HEADER = ("axis", "level", "n", "ccc", "auroc", "delta_ccc")
TABLE_3_ROWS: tuple[dict[str, object], ...] = (
    {
        "axis": "Severity",
        "level": "Claudication",
        "n": 612,
        "ccc": 0.838,
        "auroc": 0.908,
        "delta_ccc": 0.024,
    },
    {
        "axis": "Severity",
        "level": "Rest pain",
        "n": 508,
        "ccc": 0.816,
        "auroc": 0.896,
        "delta_ccc": 0.002,
    },
    {
        "axis": "Severity",
        "level": "Chronic limb-threatening ischaemia",
        "n": 478,
        "ccc": 0.792,
        "auroc": 0.882,
        "delta_ccc": -0.022,
    },
    {
        "axis": "Drug class",
        "level": "Lipid-lowering",
        "n": 396,
        "ccc": 0.844,
        "auroc": 0.910,
        "delta_ccc": 0.030,
    },
    {
        "axis": "Drug class",
        "level": "Antiplatelet",
        "n": 352,
        "ccc": 0.829,
        "auroc": 0.902,
        "delta_ccc": 0.015,
    },
    {
        "axis": "Drug class",
        "level": "Prostaglandin analogue",
        "n": 214,
        "ccc": 0.779,
        "auroc": 0.874,
        "delta_ccc": -0.035,
    },
    {
        "axis": "Demographic / comorbidity",
        "level": "Female",
        "n": 566,
        "ccc": 0.821,
        "auroc": 0.899,
        "delta_ccc": 0.007,
    },
    {
        "axis": "Demographic / comorbidity",
        "level": "Age >=75 years",
        "n": 431,
        "ccc": 0.808,
        "auroc": 0.891,
        "delta_ccc": -0.006,
    },
    {
        "axis": "Demographic / comorbidity",
        "level": "Diabetes",
        "n": 624,
        "ccc": 0.812,
        "auroc": 0.893,
        "delta_ccc": -0.002,
    },
    {
        "axis": "Demographic / comorbidity",
        "level": "Severe chronic kidney disease",
        "n": 188,
        "ccc": 0.782,
        "auroc": 0.878,
        "delta_ccc": -0.032,
    },
)

# --- Table 4: generalisation under four independent distribution shifts ---------------
TABLE_4_HEADER = ("shift", "ccc", "ccc_points", "auroc", "auroc_points", "classification")
TABLE_4_ROWS: tuple[dict[str, object], ...] = (
    {
        "shift": "Development (Site A + Site B)",
        "ccc": 0.826,
        "ccc_points": None,
        "auroc": 0.900,
        "auroc_points": None,
        "classification": "reference",
    },
    {
        "shift": "Cross-region (Site C)",
        "ccc": 0.814,
        "ccc_points": -1.2,
        "auroc": 0.893,
        "auroc_points": -0.7,
        "classification": "graceful",
    },
    {
        "shift": "Cross-vendor (different scanner generation)",
        "ccc": 0.797,
        "ccc_points": -2.9,
        "auroc": 0.859,
        "auroc_points": -4.1,
        "classification": "graceful",
    },
    {
        "shift": "Temporal (2019-2021 to 2024-2025)",
        "ccc": 0.786,
        "ccc_points": -4.0,
        "auroc": 0.842,
        "auroc_points": -5.8,
        "classification": "moderate",
    },
    {
        "shift": "Cross-vendor + temporal",
        "ccc": 0.776,
        "ccc_points": -5.0,
        "auroc": 0.828,
        "auroc_points": -7.2,
        "classification": "moderate",
    },
)

SHIFT_INCONSISTENCY_CEILING = 3.5

# --- Reported scalar quantities -------------------------------------------------------
REPORTED_SCALARS: dict[str, float] = {
    "cohort_records": 8333.0,
    "cohort_sites": 3.0,
    "cohort_regions": 2.0,
    "cohort_molecules": 27.0,
    "tier1_assay_records": 418.0,
    "reference_adjudication_kappa": 0.86,
    "threshold_tau": 0.30,
    "criterion_ccc": 0.75,
    "criterion_auroc": 0.85,
    "external_ccc": 0.821,
    "external_ccc_lower": 0.806,
    "external_ccc_upper": 0.835,
    "external_auroc": 0.897,
    "external_auroc_lower": 0.881,
    "external_auroc_upper": 0.912,
    "ccc_margin_over_criterion": 0.071,
    "auroc_margin_over_criterion": 0.047,
    "coupling_gain_ccc_over_strongest_single_branch": 0.133,
    "coupling_gain_auroc_over_strongest_single_branch": 0.091,
    "preset_nonsignificant_margin_auroc": 0.08,
    "unconstrained_fusion_delta_ccc": 0.008,
    "unconstrained_fusion_delta_auroc": -0.004,
    "inconsistency_full_model_percent": 2.1,
    "inconsistency_unconstrained_percent": 18.7,
    "interaction_excess_ccc": 0.078,
    "interaction_excess_auroc": 0.053,
    "abstention_removed_fraction": 0.10,
    "abstention_retained_auroc": 0.918,
    "reader_unassisted_auroc": 0.812,
    "reader_unassisted_lower": 0.794,
    "reader_unassisted_upper": 0.830,
    "reader_aided_auroc": 0.934,
    "reader_aided_lower": 0.921,
    "reader_aided_upper": 0.946,
    "reader_team_gain_over_best_arm": 0.037,
    "reader_kappa_unassisted": 0.58,
    "reader_kappa_assisted": 0.71,
    "reader_gain_least_skilled": 0.051,
    "reader_gain_most_skilled": 0.021,
    "repurposing_rank_correlation": 0.31,
    "repurposing_molecules_moved_five_ranks": 14.0,
    "repurposing_top_ten_excluded": 9.0,
    "repurposing_outside_top_ten_promoted": 6.0,
    "silent_weeks": 14.0,
    "silent_sites": 3.0,
    "silent_consulted_percent": 76.3,
    "silent_approved_percent": 68.9,
    "silent_adoption_all_assessments_percent": 52.6,
    "silent_median_minutes_before": 14.2,
    "silent_median_minutes_after": 6.8,
    "silent_time_reduction_percent": 52.1,
    "silent_shortlist_agreement_model_percent": 84.1,
    "silent_shortlist_agreement_mechanism_percent": 61.2,
    "silent_missed_candidates_before": 12.0,
    "silent_missed_candidates_after": 4.0,
    "silent_relevance_percent": 71.4,
    "invitro_rho": 0.87,
    "invitro_p": 0.002,
    "invitro_accumulation_ratio_low": 1.08,
    "invitro_accumulation_ratio_high": 3.42,
    "invitro_compounds": 4.0,
    "invitro_concentrations": 3.0,
    "invitro_min_replicates": 3.0,
    "frozen_head_ccc": 0.819,
    "full_finetuning_ccc": 0.784,
    "coupling_concat_ccc": 0.779,
    "coupling_bilinear_ccc": 0.789,
    "coupling_gated_ccc": 0.768,
    "ece_isotonic": 0.031,
    "ece_uncorrected": 0.062,
    "ece_deep_ensemble": 0.038,
    "false_positive_budget": 0.05,
    "drift_alert_ceiling": 0.001,
    "ckd_largest_decrement_points": 3.2,
    "subgroup_deviation_points": 0.7,
    "fold_error_declared_low": 1.5,
    "fold_error_declared_high": 3.0,
}

# --- Reported text quantities whose rendering is unrecoverable ------------------------
UNRECOVERABLE_TEXT_QUANTITIES: tuple[dict[str, str], ...] = (
    {
        "quantity": "silent-mode assessment count",
        "rendered": "417 2 assessments",
        "anchor": "Sec. 2.9",
        "note": "the released PDF renders the count with an interleaved space, so neither 417 nor "
        "4172 can be read from the text layer; the surrounding percentages are recorded instead",
    },
    {
        "quantity": "reader experience range",
        "rendered": "30-39 19 8 years",
        "anchor": "Sec. 2.6",
        "note": "three numbers run together; the reader composition (six surgeons, four "
        "interventional radiologists, two vascular medicine physicians) is readable and is "
        "checked instead",
    },
    {
        "quantity": "in vitro accumulation ratios",
        "rendered": "1.08 and 3.42",
        "anchor": "Sec. 4 / Sec. 3.11",
        "note": "four compounds at three concentrations were run but only two accumulation ratios "
        "are given, so the per-compound table cannot be reconstructed",
    },
)

# --- Supplementary tables the manuscript cites ----------------------------------------
SUPPLEMENTARY_TABLES: dict[str, str] = {
    "SA-3": "per-tier breakdown of the main analysis",
    "SA-4": "reader-arm concordance, cited only as the source of the frozen external-site numbers",
    "SA-5": "net-benefit stability within +/-30% of the fixed threshold",
    "SB-5": "design options supported against alternative designs",
    "SC-1": "harm tiering of the decision layer (Sec. 3.6) and, separately, the "
    "perturbation-response matrix (Sec. 3.10)",
    "SC-2": "five-category failure taxonomy with frequency and harm direction",
    "SD-2": "auxiliary public resources with access level, licence and distribution clauses",
}

SUPPLEMENTARY_TABLE_COLLISIONS: tuple[dict[str, str], ...] = (
    {
        "table": "SC-1",
        "first_anchor": "Sec. 3.6, decision layer harm tiering",
        "second_anchor": "Sec. 3.10, perturbation-response matrix",
        "note": "the same supplementary table number is cited for two unrelated artefacts",
    },
)


def table_1_row(identifier: str) -> dict[str, object]:
    """One row of Table 1 by its identifier."""
    for row in TABLE_1_ROWS:
        if row["id"] == identifier:
            return dict(row)
    raise KeyError(identifier)


def table_2_row(variant: str) -> dict[str, object]:
    """One row of Table 2 by its variant label."""
    for row in TABLE_2_ROWS:
        if row["variant"] == variant:
            return dict(row)
    raise KeyError(variant)


def baseline_identifiers() -> tuple[str, ...]:
    """The reconstructed-baseline identifiers in Table 1 order."""
    return tuple(str(row["id"]) for row in TABLE_1_ROWS if row["id"] != "PROPOSED")


def numeric(row: dict[str, object], key: str) -> float:
    """Read a numeric field from a transcribed table row."""
    return as_float(row[key], key)


def text(row: dict[str, object], key: str) -> str:
    """Read a text field from a transcribed table row."""
    return as_text(row[key], key)


def reported(key: str) -> float:
    """One reported scalar by key."""
    return REPORTED_SCALARS[key]
