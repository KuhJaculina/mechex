"""Claim-to-code registry.

Every claim the release answers to carries the manuscript location it came from, the module
and callable that implements it, and the reported values attached to it where the
manuscript states any. The registry is the single source for the release's
``claim_to_code.json`` and for the anchor-resolution check in the verification pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLAIM_KINDS = ("method", "design", "protocol", "reported_value", "tooling")

# Where the anchor came from on the released PDF, in the release's own page numbering.
LOCATION_NOTES = {
    "abstract": "Abstract, p. 1",
    "intro": "Sec. 1, pp. 1-3",
    "results_cohort": "Sec. 2.1, pp. 3-4",
    "results_exposure": "Sec. 2.2, pp. 4-5",
    "results_constraint": "Sec. 2.3, pp. 5-6",
    "results_strata": "Sec. 2.4, p. 6",
    "results_shift": "Sec. 2.5, pp. 6-7",
    "results_readers": "Sec. 2.6, pp. 7-8",
    "results_ranking": "Sec. 2.8, p. 8",
    "results_silent": "Sec. 2.9, p. 8",
    "methods_design": "Sec. 3.1, p. 9",
    "methods_layers": "Sec. 3.2, p. 9",
    "methods_cohort": "Sec. 3.3, pp. 9-10",
    "methods_split": "Sec. 3.4, p. 10",
    "methods_reference": "Sec. 3.5, p. 10",
    "methods_architecture": "Sec. 3.6, pp. 10-11",
    "methods_training": "Sec. 3.8, p. 12",
    "methods_statistics": "Sec. 3.9, p. 12",
    "methods_safety": "Sec. 3.10, pp. 12-13",
    "methods_invitro": "Sec. 3.11, p. 13",
    "methods_silent": "Sec. 3.12, p. 14",
    "discussion": "Sec. 4, pp. 14-19",
    "table_1": "Table 1, p. 5",
    "table_2": "Table 2, p. 6",
    "table_3": "Table 3, p. 7",
    "table_4": "Table 4, p. 7",
}


@dataclass(frozen=True)
class Claim:
    """One paper statement and the code that answers it."""

    identifier: str
    statement: str
    anchor: str
    location: str
    kind: str
    code: tuple[str, ...]
    reported: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "statement": self.statement,
            "anchor": self.anchor,
            "location": LOCATION_NOTES.get(self.location, self.location),
            "kind": self.kind,
            "code": list(self.code),
            "reported": dict(self.reported),
            "note": self.note,
        }


@dataclass(frozen=True)
class Deviation:
    """One intentional departure from the manuscript, or one engineering default."""

    identifier: str
    paper_location: str
    deviation: str
    justification: str
    kind: str = "engineering_default"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "paper_location": self.paper_location,
            "deviation": self.deviation,
            "justification": self.justification,
            "kind": self.kind,
        }


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "c01",
        "The structure branch maps a molecule to a partition term through a 3D "
        "molecular foundation transformer with a frozen encoder and a modified projection head.",
        "Sec. 3.6, molecular derivation",
        "methods_architecture",
        "method",
        (
            "mechex.basis.projection:MolecularBranch",
            "mechex.basis.transformer:MolecularBasisEncoder",
        ),
    ),
    Claim(
        "c02",
        "Atom attention is geometry-aware: interatomic distance and bond type bias the "
        "attention logits.",
        "Sec. 3.6, 3D molecular basis transformer",
        "methods_architecture",
        "method",
        (
            "mechex.basis.transformer:MolecularAttention",
            "mechex.basis.transformer:RadialDistanceBias",
            "mechex.basis.transformer:BondTypeBias",
        ),
    ),
    Claim(
        "c03",
        "The molecular encoder is frozen and the branch projects with the encoder's own "
        "featurisation.",
        "Sec. 3.6 and Sec. 3.8",
        "methods_training",
        "method",
        ("mechex.basis.transformer:MolecularBasisEncoder", "mechex.basis.graph:MoleculeGraph"),
    ),
    Claim(
        "c04",
        "The partition term is produced by the adapted projection head.",
        "Sec. 3.6, modified projection head",
        "methods_architecture",
        "method",
        ("mechex.basis.projection:PartitionHead",),
    ),
    Claim(
        "c05",
        "Retention attribution reads physicochemical descriptors including "
        "efflux-substrate propensity, lipophilicity and molecular size.",
        "Sec. 4, third conclusion",
        "discussion",
        "reported_value",
        ("mechex.basis.chemistry:descriptors", "mechex.basis.projection:PartitionHead.attribution"),
        {"descriptor_count": 8.0},
    ),
    Claim(
        "c06",
        "The delivery branch encodes the two-phase CTA into a delivery-capacity "
        "embedding with a frozen volumetric CT encoder and an adapted head.",
        "Sec. 3.6, vascular line",
        "methods_architecture",
        "method",
        (
            "mechex.vascular.projection:VascularBranch",
            "mechex.vascular.encoder:VolumetricCTEncoder",
        ),
    ),
    Claim(
        "c07",
        "The volumetric encoder is a convolutional stem followed by token attention over "
        "the volume.",
        "Sec. 3.6, volumetric CT base encoder",
        "methods_architecture",
        "method",
        ("mechex.vascular.encoder:ConvStem", "mechex.vascular.encoder:TokenAttention"),
    ),
    Claim(
        "c08",
        "The delivered fraction is confined to the physically admissible open interval.",
        "Sec. 3.6, Eq. (1) and the admissible range",
        "methods_architecture",
        "method",
        ("mechex.vascular.projection:DeliveredFractionHead",),
        {"minimum": 0.05, "maximum": 0.95},
    ),
    Claim(
        "c09",
        "The perfused fraction is estimated from the arterial and delayed phases and "
        "converted into a dimensionless delivery term.",
        "Sec. 3.3, third transform",
        "methods_cohort",
        "method",
        ("mechex.vascular.perfusion:perfused_fraction", "mechex.vascular.perfusion:delivery_term"),
    ),
    Claim(
        "c10",
        "Input volumes are resampled to isotropic 1.0 mm spacing, windowed to the arterial "
        "contrast range and intensity-normalised per scan.",
        "Sec. 3.8, input preparation",
        "methods_training",
        "method",
        (
            "mechex.vascular.volume:resample_isotropic",
            "mechex.vascular.volume:window_to_contrast_range",
            "mechex.vascular.volume:normalise_per_scan",
        ),
        {"isotropic_spacing_mm": 1.0},
    ),
    Claim(
        "c11",
        "The exposure-coupling module applies bidirectional cross-attention between the "
        "two embeddings and emits the interaction term.",
        "Sec. 3.6, exposure-coupling module",
        "methods_architecture",
        "method",
        ("mechex.coupling.bidirectional:BidirectionalCrossAttention",),
    ),
    Claim(
        "c12",
        "Concatenation, bilinear pooling, gated fusion and a single-branch stack are the "
        "architectural alternatives compared against cross-attention.",
        "Sec. 4, design options",
        "discussion",
        "method",
        ("mechex.coupling.additive:build_coupling",),
    ),
    Claim(
        "c13",
        "Systemic availability is the product of the partition term, the delivered "
        "fraction and the interaction factor.",
        "Sec. 3.6, Eq. (1)",
        "methods_architecture",
        "method",
        ("mechex.conservation.law:systemic_availability",),
    ),
    Claim(
        "c14",
        "The conservation residual is the relative difference between the "
        "concentration-derived availability and the model reconstruction.",
        "Sec. 3.6, Eq. (1)",
        "methods_architecture",
        "method",
        ("mechex.conservation.residual:relative_residual",),
    ),
    Claim(
        "c15",
        "The conservation head applies a differential cost only once the residual passes "
        "a threshold.",
        "Sec. 3.6, conservation head",
        "methods_architecture",
        "method",
        (
            "mechex.conservation.penalty:margin_penalty",
            "mechex.conservation.residual:ConservationHead",
        ),
    ),
    Claim(
        "c16",
        "A prediction is physically inconsistent when the residual exceeds tolerance, the "
        "implied delivered fraction leaves the admissible range, or exposure falls with perfused "
        "fraction.",
        "Sec. 2.3, definition of the physical-inconsistency rate",
        "results_constraint",
        "method",
        (
            "mechex.conservation.inconsistency:inconsistency_flags",
            "mechex.conservation.inconsistency:inconsistency_rate",
        ),
    ),
    Claim(
        "c17",
        "The uncertainty head has an evidential output combined with a post-hoc isotonic "
        "recalibration to give a calibrated predictive standard deviation.",
        "Sec. 3.6, uncertainty head",
        "methods_architecture",
        "method",
        (
            "mechex.uncertainty.evidential:EvidentialHead",
            "mechex.uncertainty.isotonic:IsotonicRecalibrator",
            "mechex.uncertainty.heads:UncertaintyHead",
        ),
    ),
    Claim(
        "c18",
        "A deep ensemble is the reported alternative to the evidential head.",
        "Table 2, Tier 3",
        "table_2",
        "method",
        ("mechex.uncertainty.ensemble:DeepEnsembleHeads",),
    ),
    Claim(
        "c19",
        "The no-uncertainty-head variant fits a constant residual scale instead of a "
        "learned evidence model.",
        "Table 2, Tier 1",
        "table_2",
        "method",
        ("mechex.uncertainty.evidential:PointHead",),
    ),
    Claim(
        "c20",
        "The exposure index is the concentration-time integral normalised per molecule and "
        "scaled by the delivery term.",
        "Sec. 3.3, fourth transform",
        "methods_cohort",
        "method",
        ("mechex.decision.threshold:exposure_index", "mechex.cohort.transforms:relative_exposure"),
    ),
    Claim(
        "c21",
        "The adequacy boundary is the minimum clinically important difference of 30% fixed "
        "on the development split.",
        "Sec. 3.5, threshold",
        "methods_reference",
        "reported_value",
        ("mechex.decision.threshold:adequacy_boundary", "mechex.cohort.reference:freeze_threshold"),
        {"tau": 0.30},
    ),
    Claim(
        "c22",
        "The decision layer dichotomises against the fixed boundary and abstains when the "
        "predictive deviation exceeds the policy limit.",
        "Sec. 3.6, decision layer",
        "methods_architecture",
        "method",
        (
            "mechex.decision.abstention:abstention_mask",
            "mechex.decision.threshold:ThresholdDecision",
        ),
    ),
    Claim(
        "c23",
        "The abstention boundary is selected to bound the false-positive rate below 5% "
        "because a false-adequate call is the harmful direction.",
        "Sec. 3.10, operating point",
        "methods_safety",
        "reported_value",
        (
            "mechex.decision.abstention:select_boundary",
            "mechex.guard.operating_point:select_operating_point",
        ),
        {"false_positive_budget": 0.05},
    ),
    Claim(
        "c24",
        "MechEx assembles the molecular branch, the vascular branch, the coupling, the "
        "conservation head, the uncertainty head and the decision layer.",
        "Sec. 3.6, six components",
        "methods_architecture",
        "method",
        ("mechex.estimator:MechEx",),
    ),
    Claim(
        "c25",
        "Cohort construction runs five explicit transforms from the source fields to the "
        "exposure label.",
        "Sec. 3.3, five transforms",
        "methods_cohort",
        "method",
        (
            "mechex.cohort.transforms:transform_chain",
            "mechex.cohort.transforms:unit_normalise_concentrations",
            "mechex.cohort.transforms:time_align",
            "mechex.cohort.transforms:perfused_fraction_to_delivery",
            "mechex.cohort.transforms:exposure_index",
            "mechex.cohort.transforms:dichotomise",
        ),
    ),
    Claim(
        "c26",
        "The primary cohort comprises 8,333 records across three sites in two regions "
        "with 27 molecules.",
        "Sec. 2.1, study cohort",
        "results_cohort",
        "reported_value",
        ("mechex.cohort.fixture:build_cohort_fixture", "mechex.cohort.records:CohortFixture"),
        {"records": 8333.0, "sites": 3.0, "regions": 2.0, "molecules": 27.0},
    ),
    Claim(
        "c27",
        "The reference standard is tiered with 418 directly assayed records and a "
        "composite index for the remainder.",
        "Sec. 2.1 and Sec. 3.5, reference standard",
        "results_cohort",
        "reported_value",
        ("mechex.cohort.reference:tier_counts", "mechex.cohort.reference:tier_sensitivity"),
        {"tier1_records": 418.0},
    ),
    Claim(
        "c28",
        "Reference adequacy is double-adjudicated at a Cohen's kappa of 0.86.",
        "Sec. 2.1, reference adequacy",
        "results_cohort",
        "reported_value",
        ("mechex.cohort.reference:adjudicate", "mechex.quantify.agreement:cohen_kappa"),
        {"kappa": 0.86},
    ),
    Claim(
        "c29",
        "Linkage is deterministic on a site-level de-identified study key so every "
        "derived variable traces to a source table.",
        "Sec. 3.3, linkage",
        "methods_cohort",
        "method",
        ("mechex.cohort.linkage:link", "mechex.cohort.linkage:study_key"),
    ),
    Claim(
        "c30",
        "Records leave the cohort for an absent delivery bed, a non-diagnostic CTA or an "
        "unalignable concentration record.",
        "Sec. 3.3, exclusion rules",
        "methods_cohort",
        "method",
        ("mechex.cohort.linkage:excluded_records",),
    ),
    Claim(
        "c31",
        "Sites A and B are split 70/15/15 by patient into development, validation and "
        "test with a tenth of development withheld as a calibration set, and Site C is held out "
        "entirely.",
        "Sec. 3.4, splitting",
        "methods_split",
        "reported_value",
        ("mechex.cohort.partition:partition",),
        {"development": 0.70, "validation": 0.15, "test": 0.15, "calibration": 0.10},
    ),
    Claim(
        "c32",
        "Leakage control verifies that no patient or record crosses a partition boundary.",
        "Sec. 3.4, leakage control",
        "methods_split",
        "method",
        ("mechex.cohort.partition:check_leakage",),
    ),
    Claim(
        "c33",
        "Missing concentration values are not imputed for the primary endpoint.",
        "Sec. 3.3, missingness",
        "methods_cohort",
        "method",
        ("mechex.cohort.linkage:excluded_records",),
    ),
    Claim(
        "c34",
        "Training minimises a composite of a concordance-oriented regression loss, a "
        "binary cross-entropy term on the adequacy label and the conservation penalty.",
        "Sec. 3.8, training",
        "methods_training",
        "method",
        ("mechex.objective.composite:CompositeObjective",),
    ),
    Claim(
        "c35",
        "The regression term is concordance-oriented.",
        "Sec. 3.8, composite objective",
        "methods_training",
        "method",
        (
            "mechex.objective.concordance:log_concordance_loss",
            "mechex.objective.concordance:concordance_ccc",
        ),
    ),
    Claim(
        "c36",
        "The adequacy term is a binary cross-entropy on the adequate/inadequate label.",
        "Sec. 3.8, composite objective",
        "methods_training",
        "method",
        ("mechex.objective.calibration:adequacy_loss",),
    ),
    Claim(
        "c37",
        "Optimisation uses AdamW with cosine decay.",
        "Sec. 3.8, optimisation",
        "methods_training",
        "method",
        ("mechex.optimise.optimiser:build_optimizer", "mechex.optimise.schedule:build_scheduler"),
    ),
    Claim(
        "c38",
        "Early stopping monitors the validation partition with a patience of twelve epochs.",
        "Sec. 3.8, early stopping",
        "methods_training",
        "reported_value",
        ("mechex.optimise.optimiser:EarlyStopping",),
        {"patience": 12.0},
    ),
    Claim(
        "c39",
        "The batch size is 64 patients.",
        "Sec. 3.8, batch size",
        "methods_training",
        "reported_value",
        ("mechex.settings:OptimiserConfig",),
        {"batch_size": 64.0},
    ),
    Claim(
        "c40",
        "Five independent repeats with different random initialisations are run.",
        "Sec. 3.8, repeats",
        "methods_training",
        "reported_value",
        ("mechex.fit.repeats:run_repeats", "mechex.fit.repeats:repeat_seeds"),
        {"repeats": 5.0},
    ),
    Claim(
        "c41",
        "Uncertainty on correlation-type metrics comes from patient-level bootstrap "
        "resampling with 2,000 resamples after averaging across repeats.",
        "Sec. 3.8 and Sec. 3.9, resampling",
        "methods_statistics",
        "reported_value",
        (
            "mechex.quantify.bootstrap:bootstrap_ci",
            "mechex.quantify.bootstrap:cluster_resample_indices",
        ),
        {"resamples": 2000.0},
    ),
    Claim(
        "c42",
        "Concordance is Lin's concordance correlation coefficient.",
        "Sec. 3.9, concordance",
        "methods_statistics",
        "method",
        ("mechex.quantify.concordance:lin_ccc",),
    ),
    Claim(
        "c43",
        "AUROC differences are tested with the DeLong test.",
        "Sec. 3.9, discrimination",
        "methods_statistics",
        "method",
        ("mechex.quantify.delong:delong_test",),
    ),
    Claim(
        "c44",
        "Classification agreement is tested with McNemar's test.",
        "Sec. 3.9, agreement",
        "methods_statistics",
        "method",
        ("mechex.quantify.agreement:mcnemar_test",),
    ),
    Claim(
        "c45",
        "Net benefit comes from decision-curve analysis.",
        "Sec. 3.9, net benefit",
        "methods_statistics",
        "method",
        ("mechex.quantify.netbenefit:decision_curve",),
    ),
    Claim(
        "c46",
        "Reader differences are tested with a mixed-effects logistic model carrying a "
        "random intercept per reader.",
        "Sec. 2.6, reader study",
        "results_readers",
        "method",
        ("mechex.quantify.mixed:mixed_effects_logistic",),
    ),
    Claim(
        "c47",
        "Stratified comparisons are controlled with the Benjamini-Hochberg procedure at a "
        "false-discovery rate of 0.05.",
        "Sec. 3.9, multiplicity",
        "methods_statistics",
        "reported_value",
        ("mechex.quantify.multiplicity:benjamini_hochberg",),
        {"fdr": 0.05},
    ),
    Claim(
        "c48",
        "Calibration is reported as expected calibration error over fifteen equal-width bins.",
        "Sec. 3.9, calibration",
        "methods_statistics",
        "reported_value",
        ("mechex.quantify.calibration:expected_calibration_error",),
        {"bins": 15.0},
    ),
    Claim(
        "c49",
        "Results are reported for each site rather than pooled, with the external site "
        "fully held out.",
        "Sec. 3.9, per-site reporting",
        "methods_statistics",
        "method",
        ("mechex.assess.external:evaluate_external",),
    ),
    Claim(
        "c50",
        "Failure modes form a five-category taxonomy over the worst decile of absolute "
        "error, each with its frequency and harm direction.",
        "Sec. 3.10, failure taxonomy",
        "methods_safety",
        "reported_value",
        (
            "mechex.guard.failure_modes:classify_failures",
            "mechex.guard.failure_modes:FAILURE_TYPES",
        ),
        {"categories": 5.0},
    ),
    Claim(
        "c51",
        "The perturbation-response matrix reports the response of the estimate to input "
        "perturbation.",
        "Sec. 3.10, robustness",
        "methods_safety",
        "method",
        (
            "mechex.guard.perturbation:response_matrix",
            "mechex.guard.perturbation:DEFAULT_PERTURBATIONS",
        ),
    ),
    Claim(
        "c52",
        "Silent-failure monitoring is a drift detector over the input and residual "
        "distributions, calibrated below one alert per thousand stationary cases.",
        "Sec. 3.10, silent-failure monitoring",
        "methods_safety",
        "reported_value",
        ("mechex.guard.drift:calibrate_threshold", "mechex.guard.drift:StandardisedDriftDetector"),
        {"alert_ceiling": 0.001},
    ),
    Claim(
        "c53",
        "The int8 variant uses post-training quantisation with per-channel scales and is "
        "evaluated on the identical test set.",
        "Sec. 3.8, quantisation",
        "methods_training",
        "method",
        (
            "mechex.compress.int8:quantise_linear_per_channel",
            "mechex.compress.int8:quantise_module",
        ),
    ),
    Claim(
        "c54",
        "The in vitro arm treats four compounds at three concentrations with and without "
        "efflux inhibition, at least three replicates, read by two independent methods.",
        "Sec. 3.11, in vitro arm",
        "methods_invitro",
        "reported_value",
        ("mechex.assay.accumulation:AssayDesign", "mechex.assay.accumulation:method_agreement"),
        {"compounds": 4.0, "concentrations": 3.0, "min_replicates": 3.0},
    ),
    Claim(
        "c55",
        "Efflux-substrate propensity is the dominant negative determinant of arterial-wall "
        "retention, corroborated in vitro.",
        "Sec. 4 and Sec. 3.11",
        "discussion",
        "reported_value",
        (
            "mechex.assay.accumulation:validate_efflux_hypothesis",
            "mechex.basis.projection:PartitionHead.attribution",
        ),
        {"rho": 0.87, "p": 0.002, "ratio_low": 1.08, "ratio_high": 3.42},
    ),
    Claim(
        "c56",
        "Exposure-guided ranking changes the candidate shortlist: fourteen molecules move "
        "at least five ranks and nine of the mechanism top ten leave the shortlist.",
        "Sec. 2.8, ranking",
        "results_ranking",
        "reported_value",
        ("mechex.assess.ranking:ranking_summary", "mechex.assess.ranking:shortlist_effect"),
        {"moved_five_ranks": 14.0, "rho": 0.31, "excluded": 9.0, "promoted": 6.0},
    ),
    Claim(
        "c57",
        "Silent-mode deployment recorded integration, time and adoption endpoints over "
        "fourteen weeks at three sites.",
        "Sec. 2.9 and Sec. 3.12, deployment",
        "results_silent",
        "reported_value",
        (
            "mechex.assess.silent_mode:summarise_deployment",
            "mechex.assess.silent_mode:weekly_adoption",
        ),
        {"weeks": 14.0, "sites": 3.0},
    ),
    Claim(
        "c58",
        "Twelve readers assessed the same test set unaided and aided, with the team's "
        "gain tested against the best isolated arm.",
        "Sec. 2.6, reader study",
        "results_readers",
        "reported_value",
        (
            "mechex.assess.human_model:team_contrast",
            "mechex.assess.human_model:inter_reader_agreement",
        ),
        {"readers": 12.0, "team_gain": 0.037, "kappa_unassisted": 0.58, "kappa_assisted": 0.71},
    ),
    Claim(
        "c59",
        "Performance is stratified by severity, drug class and demographic and "
        "comorbidity axes, each with its size and concordance decrement.",
        "Sec. 2.4 and Table 3, stratification",
        "results_strata",
        "method",
        ("mechex.assess.strata:evaluate_strata", "mechex.assess.strata:stratum_sizes"),
    ),
    Claim(
        "c60",
        "Generalisation is assessed under four independent distribution shifts with a "
        "pre-specified banding of graceful, moderate and severe.",
        "Sec. 2.5 and Table 4, shifts",
        "results_shift",
        "method",
        ("mechex.assess.shift:generalisation_table", "mechex.quantify.shift:Banding"),
    ),
    Claim(
        "c61",
        "The pre-specified success criteria at the external site are concordance of at "
        "least 0.75 and AUROC of at least 0.85.",
        "Sec. 2.2 and Sec. 3.5, criteria",
        "results_exposure",
        "reported_value",
        ("mechex.assess.external:criterion_outcome",),
        {"ccc_threshold": 0.75, "auroc_threshold": 0.85},
    ),
    Claim(
        "c62",
        "Fold error is geometric.",
        "Sec. 2.2, fold error",
        "results_exposure",
        "method",
        ("mechex.quantify.fold:geometric_fold_error",),
    ),
    Claim(
        "c63",
        "Every baseline in the main comparison was reconstructed and trained in-house on "
        "the identical split with identical preprocessing and tuning budget.",
        "Table 1, note",
        "table_1",
        "method",
        ("mechex.bench.runner:run_baseline", "mechex.bench.registry:BASELINES"),
        {"comparators": 12.0},
    ),
    Claim(
        "c64",
        "The mechanism-only repurposing score is the current computational default and "
        "carries no delivery term.",
        "Table 1, B9",
        "table_1",
        "reported_value",
        ("mechex.bench.mechanistic:mechanism_only_score",),
        {"ccc": 0.431, "auroc": 0.628},
    ),
    Claim(
        "c65",
        "Coupling adds concordance over the strongest single-branch model and the "
        "difference against the strongest unconstrained fusion is inside sampling noise while "
        "inconsistency falls by an order of magnitude.",
        "Abstract and Sec. 2.2-2.3",
        "results_constraint",
        "reported_value",
        (
            "mechex.assess.external:paired_concordance_contrast",
            "mechex.conservation.inconsistency:inconsistency_rate",
        ),
        {
            "ccc_gain": 0.133,
            "auroc_gain": 0.091,
            "delta_ccc": 0.008,
            "delta_auroc": -0.004,
            "inconsistency_full": 2.1,
            "inconsistency_unconstrained": 18.7,
        },
    ),
    Claim(
        "c66",
        "Removing the lowest-confidence tenth of cases raises the retained AUROC.",
        "Sec. 4, abstention",
        "discussion",
        "reported_value",
        ("mechex.assess.abstain:retention_at",),
        {"quantile": 0.10, "retained_auroc": 0.918},
    ),
    Claim(
        "c67",
        "Frozen encoders with modified heads were preferred to full fine-tuning under an "
        "identical budget.",
        "Sec. 3.6 and Sec. 4, design options",
        "discussion",
        "reported_value",
        ("mechex.bench.single_branch:FrozenHeadModel",),
        {"frozen_ccc": 0.819, "finetuned_ccc": 0.784},
    ),
    Claim(
        "c68",
        "The interaction term carries information neither branch expresses alone, so the "
        "coupled model exceeds the sum of the two branch contributions.",
        "Table 2, interaction row",
        "table_2",
        "reported_value",
        ("mechex.conservation.law:interaction_factor",),
        {"excess_ccc": 0.078, "excess_auroc": 0.053},
    ),
)

DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        "dev1-private-cohort",
        "Sec. 2.1; Sec. 3.1-3.5; Data availability",
        "The three-site two-region de-identified cohort is not distributed, so the shipped "
        "pipeline reads a schema-compatible cohort fixture written by the release. Every "
        "cohort-level quantity in the manuscript stays out of reach of this checkout.",
        "The manuscript states the raw clinical records are available only from the "
        "corresponding author on reasonable request. A fixture that reproduces the schema, "
        "the strata sizes and the causal structure lets every mechanism run without "
        "inventing a single clinical result.",
        "unavailable_input",
    ),
    Deviation(
        "dev2-conservation-denominator",
        "Sec. 3.6, Eq. (1) and the sentence that follows it",
        "Eq. (1) writes the residual against the model product while the next sentence calls "
        "that quantity concentration-derived. The release takes the denominator as the "
        "concentration-derived reference.",
        "A relative residual needs the measured quantity in the denominator; against the fit "
        "itself the ratio is a self-consistency identity rather than an error.",
        "manuscript_ambiguity",
    ),
    Deviation(
        "dev3-physics-feature",
        "Sec. 3.3, fourth transform; Sec. 3.6, Eq. (1)",
        "The exposure estimate is produced by the uncertainty head from the coupling "
        "embedding together with the log of the conservation product scaled by the delivery "
        "term, rather than being read off the product directly.",
        "The manuscript defines the exposure index as the concentration-time integral scaled "
        "by the delivery term while Eq. (1) already puts the delivered fraction inside the "
        "availability. Feeding the product in as a feature keeps both statements live "
        "without double-counting the delivery term in the reported index.",
        "manuscript_ambiguity",
    ),
    Deviation(
        "dev4-threshold-normalisation",
        "Sec. 3.5, threshold; Sec. 3.3, fifth transform",
        "The adequacy boundary is applied to a per-molecule normalised exposure, with "
        "adequate defined as a relative exposure of at least 1 + tau.",
        "The manuscript fixes one threshold for every molecule, which is only meaningful on "
        "a dimensionless index. The normalisation statistic is estimated on the development "
        "split alone.",
        "engineering_default",
    ),
    Deviation(
        "dev5-composite-loss-weights",
        "Sec. 3.8, training",
        "The composite weights are not reported, so the concordance, adequacy, conservation "
        "and evidential terms enter at weight one, exposed in configs/objective/ and the "
        "train group.",
        "The manuscript names the four terms and the three constraint settings but not their "
        "relative weights. Exposing them keeps the choice auditable rather than implicit.",
        "engineering_default",
    ),
    Deviation(
        "dev6-optimiser-schedule",
        "Sec. 3.8, optimiser",
        "AdamW with cosine decay is implemented as reported; the learning rate, weight decay, "
        "epoch count, warmup fraction, minimum learning rate and gradient-clipping norm are "
        "engineering defaults exposed in configs/train/.",
        "The manuscript reports the optimiser family, the schedule and the batch size but not "
        "these values.",
        "engineering_default",
    ),
    Deviation(
        "dev7-precision-and-world-size",
        "Sec. 3.8, training",
        "Precision defaults to fp32 with gradient accumulation one and world size one; the "
        "int8 variant is applied after training with per-channel scales as reported.",
        "The manuscript does not state the mixed-precision setting or the process count. The "
        "release defaults to the setting that reproduces exactly on a single host.",
        "engineering_default",
    ),
    Deviation(
        "dev8-encoder-widths",
        "Sec. 3.6, architecture",
        "Both branches use hidden width 128, four molecular transformer layers and a "
        "convolutional stem with a token-attention stack; the coupling uses width 128 with "
        "four heads. All of these are engineering defaults.",
        "The manuscript names the encoder families and the frozen-head configuration but no "
        "widths or depths.",
        "engineering_default",
    ),
    Deviation(
        "dev9-arterial-contrast-window",
        "Sec. 3.8, input preparation",
        "The arterial contrast window is 100 to 700 Hounsfield units, an engineering default "
        "in configs/vascular/.",
        "The manuscript states that volumes are windowed to the arterial contrast range "
        "without giving the bounds.",
        "engineering_default",
    ),
    Deviation(
        "dev10-volume-grid",
        "Sec. 3.8, input preparation",
        "Volumes are resampled to isotropic 1.0 mm spacing as reported and then cropped or "
        "padded to a fixed 64x64x48 grid; the grid extent is an engineering default.",
        "The manuscript reports the isotropic spacing but not the tensor the network receives.",
        "engineering_default",
    ),
    Deviation(
        "dev11-residual-tolerance",
        "Sec. 2.3, definition of the physical-inconsistency rate",
        "The pre-specified tolerance on the conservation residual is 0.20 and the margin "
        "smoothing is 0.05, both engineering defaults exposed in configs/conservation/.",
        "The manuscript defines the inconsistency rate relative to a pre-specified tolerance "
        "without stating its value.",
        "engineering_default",
    ),
    Deviation(
        "dev12-descriptor-definitions",
        "Sec. 4 and Sec. 3.6, molecular derivation",
        "The eight physicochemical descriptors, their fragment contributions and the efflux "
        "composite are the release's own engineering definitions. The attribution reports "
        "the fitted loadings of these descriptors, not a re-measurement of the manuscript's.",
        "The manuscript reports the ranking of descriptor families rather than the descriptor "
        "recipe, so the recipe has to be stated somewhere; it lives in the chemistry module "
        "rather than in a config because it belongs to the descriptor definition.",
        "engineering_default",
    ),
    Deviation(
        "dev13-constructed-molecule-graphs",
        "Sec. 3.6, molecular derivation",
        "Atom and bond graphs are schema stand-ins that exercise the native featurisation; no "
        "source structure, SMILES string or catalogue identifier is redistributed.",
        "The manuscript does not release structures for the 27 molecules, and redistributing "
        "them would be a third-party claim the release cannot make.",
        "unavailable_input",
    ),
    Deviation(
        "dev14-interaction-scalar-map",
        "Sec. 3.6, Eq. (1)",
        "The interaction term comes from a softplus head shifted by -0.5, which keeps "
        "1 + epsilon strictly positive.",
        "The manuscript constrains the product to be physically admissible but does not give "
        "a parameterisation; a strictly positive factor is the minimal way to respect it.",
        "engineering_default",
    ),
    Deviation(
        "dev15-reader-study-not-reproduced",
        "Sec. 2.6 and Supplementary Table SA-4",
        "The twelve-reader, three-arm study is human data the release does not hold. The "
        "mixed-effects contrast, the kappa agreement and the skill gradient are implemented "
        "and exercised on a reader fixture; the reported values stay NOT_RUN.",
        "Reader assessments are human data collected at clinical sites and are not part of "
        "any distributed artefact.",
        "unavailable_input",
    ),
    Deviation(
        "dev16-in-vitro-arm-not-reproduced",
        "Sec. 3.11, in vitro arm",
        "The efflux arm is a wet-laboratory study. The accumulation-ratio, method-agreement "
        "and rank-correlation machinery is implemented and exercised on a fixture assay "
        "table; the reported rho and P stay NOT_RUN.",
        "The assays need purchased human vascular cells and a laboratory, which a code "
        "checkout cannot supply.",
        "unavailable_input",
    ),
    Deviation(
        "dev17-silent-mode-not-reproduced",
        "Sec. 2.9 and Sec. 3.12, deployment",
        "The fourteen-week observational deployment is a clinical workflow study recorded at "
        "three sites. The endpoint machinery runs on a deployment fixture; the "
        "reported endpoints stay NOT_RUN.",
        "Recommendation logs are site data that the release does not hold.",
        "unavailable_input",
    ),
    Deviation(
        "dev18-location-of-the-model-arm",
        "Sec. 2.6, model-alone arm",
        "The model-alone arm reuses the frozen external-site numbers rather than resampling, "
        "following the manuscript's own statement that a second modelling number with "
        "inconsistent numbers is not introduced.",
        "Keeping one modelling number for the model arm is what the manuscript asks for; "
        "producing a second would create exactly the inconsistency it avoids.",
        "manuscript_decision",
    ),
)


def claims_by_kind(kind: str) -> list[Claim]:
    """Claims of one kind."""
    return [claim for claim in CLAIMS if claim.kind == kind]


def claim(identifier: str) -> Claim:
    """One claim by identifier."""
    for item in CLAIMS:
        if item.identifier == identifier:
            return item
    raise KeyError(identifier)


def all_anchors() -> list[str]:
    """Every distinct paper anchor referenced by a claim."""
    return sorted({item.anchor for item in CLAIMS})


def all_code_references() -> list[str]:
    """Every distinct ``module:qualname`` reference, deduplicated and sorted."""
    references: set[str] = set()
    for item in CLAIMS:
        references.update(item.code)
    return sorted(references)


def claim_index() -> dict[str, object]:
    """Summary of the registry, written into the release artefacts."""
    return {
        "claim_count": len(CLAIMS),
        "anchor_count": len(all_anchors()),
        "code_reference_count": len(all_code_references()),
        "kinds": {kind: len(claims_by_kind(kind)) for kind in CLAIM_KINDS},
        "deviation_count": len(DEVIATIONS),
    }
