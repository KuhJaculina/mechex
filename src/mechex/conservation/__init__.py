"""Conservation constraint: the law, its residual, its penalty and its violation rate."""

from mechex.conservation.inconsistency import (
    inadmissible_fraction,
    inconsistency_flags,
    inconsistency_rate,
    monotonicity_violation,
    residual_from_tensors,
    residue_exceedance,
)
from mechex.conservation.law import (
    implied_delivered_fraction,
    interaction_factor,
    systemic_availability,
)
from mechex.conservation.penalty import conservation_penalty, margin_penalty, penalty_scale
from mechex.conservation.residual import (
    ConservationHead,
    ConservationRecord,
    relative_residual,
)

__all__ = [
    "ConservationHead",
    "ConservationRecord",
    "conservation_penalty",
    "implied_delivered_fraction",
    "inadmissible_fraction",
    "inconsistency_flags",
    "inconsistency_rate",
    "interaction_factor",
    "margin_penalty",
    "monotonicity_violation",
    "penalty_scale",
    "relative_residual",
    "residual_from_tensors",
    "residue_exceedance",
    "systemic_availability",
]
