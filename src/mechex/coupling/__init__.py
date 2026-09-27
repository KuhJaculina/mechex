"""Exposure-coupling module: bidirectional cross-attention and its alternatives."""

from mechex.coupling.additive import (
    COUPLING_MODES,
    BilinearPoolingCoupling,
    ConcatenationCoupling,
    GatedFusionCoupling,
    RandomInitCoupling,
    StackedBranchCoupling,
    build_coupling,
)
from mechex.coupling.bidirectional import (
    BidirectionalCrossAttention,
    CouplingOutput,
    CrossAttentionCoupling,
)

__all__ = [
    "COUPLING_MODES",
    "BidirectionalCrossAttention",
    "BilinearPoolingCoupling",
    "ConcatenationCoupling",
    "CouplingOutput",
    "CrossAttentionCoupling",
    "GatedFusionCoupling",
    "RandomInitCoupling",
    "StackedBranchCoupling",
    "build_coupling",
]
