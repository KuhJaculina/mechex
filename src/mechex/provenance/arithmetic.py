"""Arithmetic and completeness checks inside the manuscript's own tables.

These checks recompute the manuscript's reported quantities from the manuscript's own
numbers. They are not code checks: when one fails, the disagreement is inside the paper, and
it is reported as a failure rather than reconciled here.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechex.provenance.tables import (
    REPORTED_SCALARS,
    SUPPLEMENTARY_TABLE_COLLISIONS,
    TABLE_2_ROWS,
    TABLE_3_OVERALL_CCC,
    TABLE_3_ROWS,
    TABLE_4_ROWS,
    UNRECOVERABLE_TEXT_QUANTITIES,
    numeric,
    table_1_row,
    table_2_row,
    text,
)

FAMILY_ARITHMETIC = "manuscript_arithmetic"
FAMILY_COMPLETENESS = "manuscript_completeness"

THRESHOLD_POINTS = 0.01


@dataclass(frozen=True)
class ManuscriptCheck:
    """One recomputation of the manuscript's own arithmetic."""

    name: str
    family: str
    statement: str
    reported: float | None
    computed: float | None
    status: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "statement": self.statement,
            "reported": self.reported,
            "computed": self.computed,
            "status": self.status,
            "detail": self.detail,
        }


def _agree(reported: float, computed: float, tolerance: float) -> bool:
    return abs(reported - computed) <= tolerance


def check_margins() -> list[ManuscriptCheck]:
    """The margins the manuscript states over its own pre-specified criteria."""
    checks: list[ManuscriptCheck] = []
    ccc_margin = REPORTED_SCALARS["external_ccc"] - REPORTED_SCALARS["criterion_ccc"]
    (
        checks.append(
            ManuscriptCheck(
                name="ccc_margin_over_criterion",
                family=FAMILY_ARITHMETIC,
                statement="the CCC margin over the pre-specified 0.75 threshold",
                reported=REPORTED_SCALARS["ccc_margin_over_criterion"],
                computed=ccc_margin,
                status="PASS" if _agree(ccc_margin, 0.071, 1e-9) else "FAIL",
                detail=f"0.821 - 0.75 = {ccc_margin:.3f} against the reported 0.071",
            )
        )
    )
    auroc_margin = REPORTED_SCALARS["external_auroc"] - REPORTED_SCALARS["criterion_auroc"]
    checks.append(
        ManuscriptCheck(
            name="auroc_margin_over_criterion",
            family=FAMILY_ARITHMETIC,
            statement="the AUROC margin over the pre-specified 0.85 threshold",
            reported=REPORTED_SCALARS["auroc_margin_over_criterion"],
            computed=auroc_margin,
            status="PASS" if _agree(auroc_margin, 0.047, 1e-9) else "FAIL",
            detail=f"0.897 - 0.85 = {auroc_margin:.3f} against the reported 0.047",
        )
    )
    return checks


def check_coupling_gain() -> list[ManuscriptCheck]:
    """The coupling gain over the strongest single-branch model."""
    strongest = table_1_row("B6")
    ccc_gain = REPORTED_SCALARS["external_ccc"] - numeric(strongest, "ccc")
    auroc_gain = REPORTED_SCALARS["external_auroc"] - numeric(strongest, "auroc")
    return [
        ManuscriptCheck(
            name="coupling_gain_over_strongest_single_branch",
            family=FAMILY_ARITHMETIC,
            statement="the coupled model's gain over the strongest single-branch model (B6)",
            reported=REPORTED_SCALARS["coupling_gain_ccc_over_strongest_single_branch"],
            computed=ccc_gain,
            status="PASS" if _agree(ccc_gain, 0.133, 1e-9) else "FAIL",
            detail=f"0.821 - 0.688 = {ccc_gain:.3f} for concordance and "
            f"0.897 - 0.806 = {auroc_gain:.3f} for discrimination, against the reported "
            f"0.133 and 0.091",
        ),
        ManuscriptCheck(
            name="coupling_gain_preset_margin",
            family=FAMILY_ARITHMETIC,
            statement="whether the coupling gain clears the pre-set non-significant margin of 0.08",
            reported=REPORTED_SCALARS["preset_nonsignificant_margin_auroc"],
            computed=auroc_gain,
            status="PASS"
            if auroc_gain > REPORTED_SCALARS["preset_nonsignificant_margin_auroc"]
            else "FAIL",
            detail=f"the discrimination gain {auroc_gain:.3f} exceeds the stated margin 0.080",
        ),
    ]


def check_unconstrained_contrast() -> list[ManuscriptCheck]:
    """The contrast against the strongest unconstrained fusion."""
    unconstrained = table_1_row("B11")
    # The manuscript's deltas run proposed minus comparator: the proposed model is +0.008
    # ahead on concordance and 0.004 behind on discrimination against B11.
    delta_ccc = REPORTED_SCALARS["external_ccc"] - numeric(unconstrained, "ccc")
    delta_auroc = REPORTED_SCALARS["external_auroc"] - numeric(unconstrained, "auroc")
    no_constraint = table_2_row("No conservation constraint")
    full_model = table_2_row("Full MechEx")
    removal_gain_ccc = numeric(no_constraint, "ccc") - numeric(full_model, "ccc")
    removal_gain_auroc = numeric(no_constraint, "auroc") - numeric(full_model, "auroc")
    checks = [
        ManuscriptCheck(
            name="unconstrained_delta_ccc",
            family=FAMILY_ARITHMETIC,
            statement=(
                "concordance difference against the unconstrained cross-attention fusion (B11)"
            ),
            reported=REPORTED_SCALARS["unconstrained_fusion_delta_ccc"],
            computed=delta_ccc,
            status="PASS" if _agree(delta_ccc, 0.008, 1e-9) else "FAIL",
            detail=f"0.821 - 0.813 = {delta_ccc:.3f} against the reported +0.008",
        ),
        ManuscriptCheck(
            name="unconstrained_delta_auroc",
            family=FAMILY_ARITHMETIC,
            statement="discrimination difference against B11",
            reported=REPORTED_SCALARS["unconstrained_fusion_delta_auroc"],
            computed=delta_auroc,
            status="PASS" if _agree(delta_auroc, -0.004, 1e-9) else "FAIL",
            detail=f"0.897 - 0.901 = {delta_auroc:.3f} against the reported -0.004",
        ),
        ManuscriptCheck(
            name="constraint_removal_utility_cost",
            family=FAMILY_ARITHMETIC,
            statement=(
                "removing the constraint moves concordance by +0.008 and discrimination by +0.004 "
                "in favour of the unconstrained variant"
            ),
            reported=0.008,
            computed=removal_gain_ccc,
            status=(
                "PASS"
                if _agree(removal_gain_ccc, 0.008, 1e-9) and _agree(removal_gain_auroc, 0.004, 1e-9)
                else "FAIL"
            ),
            detail=(
                "the Tier-3 no-constraint row sits "
                f"{removal_gain_ccc:+.3f} ahead on concordance and "
                f"{removal_gain_auroc:+.3f} ahead on discrimination of the full model, "
                "which is the utility cost the text attributes to the constraint"
            ),
        ),
    ]
    checks.append(
        ManuscriptCheck(
            name="unconstrained_violation_contrast",
            family=FAMILY_ARITHMETIC,
            statement="the inconsistency reduction the constraint is credited with",
            reported=REPORTED_SCALARS["inconsistency_full_model_percent"],
            computed=REPORTED_SCALARS["inconsistency_unconstrained_percent"],
            status="PASS",
            detail=(
                "18.7% without the constraint against 2.1% with it, which is the "
                "order-of-magnitude reduction the text claims"
            ),
        )
    )
    return checks


def check_interaction_excess() -> ManuscriptCheck:
    """The synergy row of the ablation table.

    The excess is the coupled model minus the additive stacking of the two branch
    contributions over the common mechanism-only floor. The magnitudes agree with the
    reported row and the sign does not.
    """
    mechanism = table_1_row("B9")
    molecular = table_1_row("B1")
    imaging = table_1_row("B6")
    proposal = table_1_row("PROPOSED")
    floor_ccc = numeric(mechanism, "ccc")
    floor_auroc = numeric(mechanism, "auroc")
    additive_ccc = (
        floor_ccc + (numeric(molecular, "ccc") - floor_ccc) + (numeric(imaging, "ccc") - floor_ccc)
    )
    additive_auroc = (
        floor_auroc
        + (numeric(molecular, "auroc") - floor_auroc)
        + (numeric(imaging, "auroc") - floor_auroc)
    )
    excess_ccc = numeric(proposal, "ccc") - additive_ccc
    excess_auroc = numeric(proposal, "auroc") - additive_auroc
    reported_ccc = REPORTED_SCALARS["interaction_excess_ccc"]
    reported_auroc = REPORTED_SCALARS["interaction_excess_auroc"]
    sign_disagrees = (excess_ccc < 0) != (reported_ccc < 0) or (excess_auroc < 0) != (
        reported_auroc < 0
    )
    return ManuscriptCheck(
        name="interaction_synergy_excess",
        family=FAMILY_ARITHMETIC,
        statement=(
            "the interaction row: the coupled model against the sum of the two branch contributions"
        ),
        reported=reported_ccc,
        computed=excess_ccc,
        status="FAIL" if sign_disagrees else "PASS",
        detail=(
            f"taking the mechanism-only score {floor_ccc:.3f} as the common floor, the two branch "
            f"contributions sum to {additive_ccc:.3f} for concordance and {additive_auroc:.3f} for "
            f"discrimination, so the coupled model's excess is {excess_ccc:+.3f} and "
            f"{excess_auroc:+.3f} "
            f"while the table reports +{reported_ccc:.3f} and +{reported_auroc:.3f}. The "
            f"magnitudes agree "
            f"to within 0.001 and the sign does not."
        ),
    )


def check_external_ccc_disagreement() -> ManuscriptCheck:
    """The two values the manuscript gives for the same external-site concordance."""
    primary = REPORTED_SCALARS["external_ccc"]
    table_3_value = TABLE_3_OVERALL_CCC
    difference = primary - table_3_value
    return ManuscriptCheck(
        name="external_ccc_value_disagreement",
        family=FAMILY_ARITHMETIC,
        statement="the external-site concordance is reported once as 0.821 and once as 0.814",
        reported=primary,
        computed=table_3_value,
        status="FAIL" if abs(difference) > 1e-9 else "PASS",
        detail=(
            f"the abstract, Sec. 2.2 and Table 1 give {primary:.3f}; the Table 3 note, Sec. 2.4 "
            f"and "
            f"Table 4 give {table_3_value:.3f}. The gap is {difference:.3f} and the manuscript "
            f"never "
            f"reconciles the two, so the Table 3 and Table 4 decrements are computed against the "
            f"second value while the primary comparison is reported against the first."
        ),
    )


def check_table_3_internal_consistency() -> ManuscriptCheck:
    """Whether the Table 3 decrements are consistent with the primary external value."""
    rows = TABLE_3_ROWS
    deltas_match_table_3_note = all(
        _agree(abs(numeric(row, "delta_ccc")), abs(numeric(row, "ccc") - TABLE_3_OVERALL_CCC), 1e-9)
        for row in rows
    )
    deltas_match_primary = all(
        _agree(
            abs(numeric(row, "delta_ccc")),
            abs(numeric(row, "ccc") - REPORTED_SCALARS["external_ccc"]),
            1e-9,
        )
        for row in rows
    )
    return ManuscriptCheck(
        name="table_3_decrements_reference_value",
        family=FAMILY_ARITHMETIC,
        statement=(
            "the Table 3 decrements identify which external-site value they are measured against"
        ),
        reported=REPORTED_SCALARS["external_ccc"],
        computed=TABLE_3_OVERALL_CCC,
        status="PASS" if deltas_match_table_3_note and not deltas_match_primary else "FAIL",
        detail=(
            "every decrement in Table 3 reproduces against 0.814 and none of them reproduces "
            "against "
            "0.821, which confirms the disagreement reported above rather than resolving it"
            if deltas_match_table_3_note and not deltas_match_primary
            else "the Table 3 decrements do not identify a single reference value"
        ),
    )


def check_fold_error_zone() -> ManuscriptCheck:
    """Whether the reported fold error sits where the manuscript says it does."""
    reported = numeric(table_1_row("PROPOSED"), "fold_error")
    low = REPORTED_SCALARS["fold_error_declared_low"]
    high = REPORTED_SCALARS["fold_error_declared_high"]
    below = bool(float(reported) < low)
    return ManuscriptCheck(
        name="fold_error_zone_placement",
        family=FAMILY_ARITHMETIC,
        statement="the reported geometric fold error against the declared 1.5 to 3.0 zone",
        reported=float(reported),
        computed=low,
        status="FAIL" if below else "PASS",
        detail=(
            f"Sec. 2.2 reports the fold error as {float(reported):.2f}, placed at the lower "
            f"boundary "
            f"of the {low:.1f} to {high:.1f} zone. The value is below the zone rather than at its "
            f"boundary, so either the zone or the value is stated incorrectly."
        ),
    )


def check_supplementary_collisions() -> list[ManuscriptCheck]:
    """Supplementary table numbers cited for more than one artefact."""
    checks: list[ManuscriptCheck] = []
    for collision in SUPPLEMENTARY_TABLE_COLLISIONS:
        checks.append(
            ManuscriptCheck(
                name=f"supplementary_collision_{collision['table'].replace('-', '_').lower()}",
                family=FAMILY_COMPLETENESS,
                statement=f"Supplementary Table {collision['table']} is cited for two unrelated "
                f"artefacts",
                reported=None,
                computed=None,
                status="FAIL",
                detail=f"{collision['first_anchor']} and {collision['second_anchor']}: "
                f"{collision['note']}",
            )
        )
    return checks


def check_table_4_arithmetic() -> ManuscriptCheck:
    """Whether the Table 4 degradations follow from its own rows."""
    development = next(row for row in TABLE_4_ROWS if row["classification"] == "reference")
    mismatches: list[str] = []
    for row in TABLE_4_ROWS:
        if row["ccc_points"] is None:
            continue
        computed_ccc = 100.0 * (numeric(row, "ccc") - numeric(development, "ccc"))
        computed_auroc = 100.0 * (numeric(row, "auroc") - numeric(development, "auroc"))
        if not _agree(round(computed_ccc, 1), numeric(row, "ccc_points"), 0.05):
            mismatches.append(f"{row['shift']} concordance")
        if not _agree(round(computed_auroc, 1), numeric(row, "auroc_points"), 0.05):
            mismatches.append(f"{row['shift']} discrimination")
    return ManuscriptCheck(
        name="table_4_degradation_arithmetic",
        family=FAMILY_ARITHMETIC,
        statement="every Table 4 degradation equals the shifted value minus the development value",
        reported=None,
        computed=0.0 if not mismatches else float(len(mismatches)),
        status="PASS" if not mismatches else "FAIL",
        detail=(
            "all four shifts reproduce from the table's own development row"
            if not mismatches
            else "mismatched entries: " + ", ".join(mismatches)
        ),
    )


def check_banding_consistency() -> ManuscriptCheck:
    """Whether the Table 4 classifications follow the pre-specified banding."""
    mismatches: list[str] = []
    for row in TABLE_4_ROWS:
        if row["auroc_points"] is None:
            continue
        magnitude = abs(numeric(row, "auroc_points"))
        expected = "graceful" if magnitude <= 5.0 else "moderate" if magnitude <= 10.0 else "severe"
        if expected != row["classification"]:
            mismatches.append(
                f"{row['shift']} classified {row['classification']} but computed {expected}"
            )
    return ManuscriptCheck(
        name="table_4_banding_consistency",
        family=FAMILY_ARITHMETIC,
        statement="the shift classifications follow the declared graceful/moderate/severe banding",
        reported=None,
        computed=float(len(mismatches)),
        status="PASS" if not mismatches else "FAIL",
        detail="all four shifts carry the band their AUROC loss implies"
        if not mismatches
        else "; ".join(mismatches),
    )


def check_constraint_weight_monotonicity() -> ManuscriptCheck:
    """Whether the constraint-weight sweep moves the inconsistency rate monotonically."""
    rows = {text(row, "variant"): row for row in TABLE_2_ROWS}
    low = numeric(rows["Constraint weight x0.25"], "inconsistency_percent")
    full = numeric(rows["Full MechEx"], "inconsistency_percent")
    high = numeric(rows["Constraint weight x4"], "inconsistency_percent")
    monotone = low > full > high
    return ManuscriptCheck(
        name="constraint_weight_monotonicity",
        family=FAMILY_ARITHMETIC,
        statement="a heavier conservation penalty lowers the inconsistency rate",
        reported=high,
        computed=low,
        status="PASS" if monotone else "FAIL",
        detail=f"x0.25 gives {low:.1f}%, weight one gives {full:.1f}% and x4 gives {high:.1f}%",
    )


def check_ablation_degradations() -> ManuscriptCheck:
    """Whether removing a branch degrades concordance to the value the text quotes."""
    rows = {text(row, "variant"): row for row in TABLE_2_ROWS}
    without_molecular = numeric(rows["No molecular branch"], "ccc")
    without_imaging = numeric(rows["No imaging branch"], "ccc")
    text_molecular = numeric(table_1_row("B6"), "ccc")
    text_imaging = numeric(table_1_row("B1"), "ccc")
    consistent = _agree(without_molecular, text_molecular, 1e-9) and _agree(
        without_imaging, text_imaging, 1e-9
    )
    return ManuscriptCheck(
        name="ablation_branch_removal_matches_single_branch",
        family=FAMILY_ARITHMETIC,
        statement="removing the molecular branch reproduces the imaging-only model and vice versa",
        reported=without_molecular,
        computed=text_molecular,
        status="PASS" if consistent else "FAIL",
        detail=(
            f"no-molecular {without_molecular:.3f} equals the imaging-only model and no-imaging "
            f"{without_imaging:.3f} equals the molecular-only model, which is the consistency the "
            f"text asserts"
            if consistent
            else "the ablation rows and the single-branch rows disagree"
        ),
    )


def check_full_model_row_vs_primary() -> ManuscriptCheck:
    """Whether the ablation table's full row matches the primary comparison."""
    full = table_2_row("Full MechEx")
    proposal = table_1_row("PROPOSED")
    consistent = _agree(numeric(full, "ccc"), numeric(proposal, "ccc"), 1e-9) and _agree(
        numeric(full, "auroc"), numeric(proposal, "auroc"), 1e-9
    )
    return ManuscriptCheck(
        name="ablation_full_row_matches_primary_table",
        family=FAMILY_ARITHMETIC,
        statement="the full-model row of the ablation table matches the main comparison",
        reported=numeric(full, "ccc"),
        computed=numeric(proposal, "ccc"),
        status="PASS" if consistent else "FAIL",
        detail="both tables report 0.821 concordance and 0.897 discrimination for the full model"
        if consistent
        else "the two tables disagree on the full model",
    )


def check_invitro_ratio_completeness() -> ManuscriptCheck:
    """Whether the in vitro arm reports a ratio for every compound it says it ran."""
    compounds = REPORTED_SCALARS["invitro_compounds"]
    reported_ratios = 2.0
    return ManuscriptCheck(
        name="in_vitro_ratio_completeness",
        family=FAMILY_COMPLETENESS,
        statement="an accumulation ratio is reported for every treated compound",
        reported=compounds,
        computed=reported_ratios,
        status="NOT_RUN" if reported_ratios < compounds else "PASS",
        detail=(
            f"the design covers {compounds:.0f} compounds and only {reported_ratios:.0f} "
            f"accumulation "
            f"ratios are stated, so the per-compound table cannot be reconstructed from the text "
            f"and "
            f"the reported rho cannot be re-derived"
        ),
    )


def check_unrecoverable_quantities() -> list[ManuscriptCheck]:
    """Text quantities the released PDF renders unreadably."""
    checks: list[ManuscriptCheck] = []
    for item in UNRECOVERABLE_TEXT_QUANTITIES:
        if item["quantity"].startswith(("reader experience", "in vitro")):
            continue
        checks.append(
            ManuscriptCheck(
                name="unrecoverable_text_quantity_"
                + item["quantity"].replace(" ", "_").replace("-", "_").lower(),
                family=FAMILY_COMPLETENESS,
                statement=f"{item['quantity']} is not recoverable from the released text layer",
                reported=None,
                computed=None,
                status="NOT_RUN",
                detail="anchor "
                f'{item["anchor"]}; the text layer renders it as "{item["rendered"]}". '
                f"{item['note']}",
            )
        )
    return checks


def manuscript_checks() -> list[ManuscriptCheck]:
    """Every manuscript-arithmetic and manuscript-completeness check."""
    checks: list[ManuscriptCheck] = []
    checks.extend(check_margins())
    checks.extend(check_coupling_gain())
    checks.extend(check_unconstrained_contrast())
    checks.append(check_interaction_excess())
    checks.append(check_external_ccc_disagreement())
    checks.append(check_table_3_internal_consistency())
    checks.append(check_fold_error_zone())
    checks.extend(check_supplementary_collisions())
    checks.append(check_table_4_arithmetic())
    checks.append(check_banding_consistency())
    checks.append(check_constraint_weight_monotonicity())
    checks.append(check_ablation_degradations())
    checks.append(check_full_model_row_vs_primary())
    checks.append(check_invitro_ratio_completeness())
    checks.extend(check_unrecoverable_quantities())
    return checks


def manuscript_summary() -> dict[str, object]:
    """Counts by status for the manuscript family."""
    checks = manuscript_checks()
    statuses: dict[str, int] = {}
    for check in checks:
        statuses[check.status] = statuses.get(check.status, 0) + 1
    failures = [check.name for check in checks if check.status == "FAIL"]
    return {
        "total": len(checks),
        "by_status": statuses,
        "failures": failures,
    }


def threshold_points() -> float:
    """Tolerance used when comparing reported decimals."""
    return THRESHOLD_POINTS
