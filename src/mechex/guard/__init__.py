"""Safety and robustness: failure taxonomy, operating points, perturbation and drift."""

from mechex.guard.drift import (
    ALERT_RATE_CEILING,
    RunningMoments,
    StandardisedDriftDetector,
    calibrate_and_monitor,
    calibrate_threshold,
    population_stability_index,
)
from mechex.guard.failure_modes import (
    FAILURE_TYPES,
    HARM_DIRECTION,
    MITIGATION,
    FailureAssignment,
    classify_failures,
    taxonomy_table,
    unresolved_share,
    worst_decile,
)
from mechex.guard.operating_point import (
    OperatingPoint,
    evaluate_point,
    harm_tradeoff,
    operating_curve,
    select_operating_point,
)
from mechex.guard.perturbation import (
    DEFAULT_PERTURBATIONS,
    Perturbation,
    apply_attenuation,
    apply_dropout,
    apply_gain,
    apply_perturbation,
    apply_phase_shift,
    response_matrix,
    worst_relative_change,
)

__all__ = [
    "ALERT_RATE_CEILING",
    "DEFAULT_PERTURBATIONS",
    "FAILURE_TYPES",
    "HARM_DIRECTION",
    "MITIGATION",
    "FailureAssignment",
    "OperatingPoint",
    "Perturbation",
    "RunningMoments",
    "StandardisedDriftDetector",
    "apply_attenuation",
    "apply_dropout",
    "apply_gain",
    "apply_perturbation",
    "apply_phase_shift",
    "calibrate_and_monitor",
    "calibrate_threshold",
    "classify_failures",
    "evaluate_point",
    "harm_tradeoff",
    "operating_curve",
    "population_stability_index",
    "response_matrix",
    "select_operating_point",
    "taxonomy_table",
    "unresolved_share",
    "worst_decile",
    "worst_relative_change",
]
