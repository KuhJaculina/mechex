"""Statistics reported by the manuscript.

Every routine here is a plain-numpy or scipy implementation with an independent
closed-form or brute-force counterpart in the test suite.
"""

from mechex.quantify.agreement import (
    KappaResult,
    McNemarResult,
    cohen_kappa,
    correctness,
    kappa_with_ci,
    mcnemar_test,
)
from mechex.quantify.bootstrap import (
    BootstrapResult,
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
    expected_calibration_error,
    exposure_probabilities,
    reliability_curve,
)
from mechex.quantify.concordance import (
    ccc_components,
    ccc_decrement,
    lin_ccc,
    lin_ccc_ci,
)
from mechex.quantify.correlation import (
    CorrelationResult,
    moved_at_least,
    permutation_p_value,
    rank_movement,
    rank_positions,
    spearman_rho,
    spearman_with_permutation,
)
from mechex.quantify.delong import DeLongResult, delong_test
from mechex.quantify.fold import (
    fold_error_distribution,
    fold_error_report,
    geometric_fold_error,
    zone_position,
)
from mechex.quantify.mixed import MixedEffectResult, mixed_effects_logistic, skill_gradient
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

__all__ = [
    "DEFAULT_BANDING",
    "Banding",
    "BootstrapResult",
    "CorrelationResult",
    "DeLongResult",
    "KappaResult",
    "McNemarResult",
    "MixedEffectResult",
    "PlattCalibrator",
    "auroc",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_ci",
    "brier_score",
    "ccc_components",
    "ccc_decrement",
    "cluster_resample_indices",
    "cohen_kappa",
    "confusion_at",
    "correctness",
    "decision_curve",
    "degradation",
    "degradation_points",
    "delong_test",
    "expected_calibration_error",
    "exposure_probabilities",
    "family_size",
    "fold_error_distribution",
    "fold_error_report",
    "geometric_fold_error",
    "holm_bonferroni",
    "interval_contains",
    "kappa_with_ci",
    "lin_ccc",
    "lin_ccc_ci",
    "maximum_net_benefit_advantage",
    "mcnemar_test",
    "mean_over_repeats",
    "mixed_effects_logistic",
    "moved_at_least",
    "net_benefit",
    "operating_point_at_fpr",
    "overlap_fraction",
    "paired_bootstrap_difference",
    "permutation_p_value",
    "rank_average",
    "rank_movement",
    "rank_positions",
    "reliability_curve",
    "roc_curve",
    "shift_row",
    "skill_gradient",
    "spearman_rho",
    "spearman_with_permutation",
    "stable_around",
    "treat_all_net_benefit",
    "within_consistency_limit",
    "worst_case",
    "youden_threshold",
    "zone_position",
]
