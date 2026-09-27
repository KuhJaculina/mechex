"""Uncertainty quantification: evidential head, isotonic recalibration, deep ensemble."""

from mechex.uncertainty.ensemble import DeepEnsembleHeads
from mechex.uncertainty.evidential import (
    EvidentialHead,
    EvidentialOutput,
    PointHead,
    evidential_variance,
    nig_nll,
    nig_parameters,
)
from mechex.uncertainty.heads import (
    UNCERTAINTY_MODES,
    UncertaintyHead,
    build_uncertainty_head,
)
from mechex.uncertainty.isotonic import (
    MEAN_ABS_TO_SIGMA,
    IsotonicRecalibrator,
    pool_adjacent_violators,
)

__all__ = [
    "MEAN_ABS_TO_SIGMA",
    "UNCERTAINTY_MODES",
    "DeepEnsembleHeads",
    "EvidentialHead",
    "EvidentialOutput",
    "IsotonicRecalibrator",
    "PointHead",
    "UncertaintyHead",
    "build_uncertainty_head",
    "evidential_variance",
    "nig_nll",
    "nig_parameters",
    "pool_adjacent_violators",
]
