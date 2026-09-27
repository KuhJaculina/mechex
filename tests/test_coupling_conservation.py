"""Coupling and the conservation constraint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
from mechex.conservation.residual import ConservationHead, relative_residual
from mechex.coupling.additive import (
    COUPLING_MODES,
    BilinearPoolingCoupling,
    ConcatenationCoupling,
    GatedFusionCoupling,
    RandomInitCoupling,
    StackedBranchCoupling,
    build_coupling,
)
from mechex.coupling.bidirectional import BidirectionalCrossAttention, CrossAttentionCoupling
from mechex.settings import ConservationConfig, CouplingConfig


def test_conservation_law_is_the_product() -> None:
    partition = torch.tensor([2.0, 3.0])
    delivered = torch.tensor([0.5, 0.25])
    interaction = torch.tensor([0.1, -0.2])
    product = systemic_availability(partition, delivered, interaction)
    expected = partition * delivered * interaction_factor(interaction)
    assert torch.allclose(product, expected)


def test_interaction_factor_is_positive_for_the_offset_config() -> None:
    config = CouplingConfig(width=16, heads=2)
    module = BidirectionalCrossAttention(config)
    molecule = torch.randn(8, 16)
    image = torch.randn(8, 16)
    _, epsilon = module(molecule, image)
    assert bool((interaction_factor(epsilon) > 0).all())


def test_implied_delivered_fraction_inverts_the_law() -> None:
    partition = torch.tensor([2.0])
    delivered = torch.tensor([0.4])
    interaction = torch.tensor([0.05])
    availability = systemic_availability(partition, delivered, interaction)
    implied = implied_delivered_fraction(availability, partition, interaction)
    assert implied.item() == pytest.approx(0.4, abs=1e-6)


def test_relative_residual_against_hand_definition() -> None:
    reference = torch.tensor([10.0, 20.0])
    reconstruction = torch.tensor([9.0, 25.0])
    residual = relative_residual(reference, reconstruction)
    assert torch.allclose(residual, torch.tensor([0.1, 0.25]))


def test_margin_penalty_is_flat_then_rises() -> None:
    below = float(margin_penalty(torch.tensor([0.05]), 0.2, 0.05))
    at = float(margin_penalty(torch.tensor([0.20]), 0.2, 0.05))
    above = float(margin_penalty(torch.tensor([0.60]), 0.2, 0.05))
    assert below < at < above
    hand = 0.05 * float(np.log1p(np.exp((0.6 - 0.2) / 0.05)))
    assert above == pytest.approx(hand, abs=1e-6)


def test_margin_penalty_rejects_zero_margin() -> None:
    with pytest.raises(ValueError):
        margin_penalty(torch.tensor([0.5]), 0.2, 0.0)


def test_conservation_penalty_scales_with_weight() -> None:
    residual = torch.tensor([0.5, 0.6])
    light = conservation_penalty(residual, 0.2, 0.05, weight=0.25)
    heavy = conservation_penalty(residual, 0.2, 0.05, weight=4.0)
    assert float(heavy) == pytest.approx(float(light) * 16.0, rel=1e-5)


def test_penalty_scale_is_the_exceedance_share() -> None:
    residual = torch.tensor([0.1, 0.3, 0.5, 0.05])
    assert float(penalty_scale(residual, 0.2)) == pytest.approx(0.5)


def test_conservation_head_record_shapes() -> None:
    head = ConservationHead(ConservationConfig())
    record = head(
        torch.tensor([2.0, 3.0]),
        torch.tensor([0.5, 0.4]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([1.05, 1.4]),
    )
    assert record.residual.shape == (2,)
    assert record.implied_fraction.shape == (2,)
    assert torch.isfinite(record.mean_residual())


def test_inconsistency_components_by_hand() -> None:
    residual = np.array([0.05, 0.30, 0.10, 0.60, 0.02])
    implied = np.array([0.5, 0.4, 0.02, 0.99, 0.6])
    fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    exposures = np.array([0.5, 0.6, 0.55, 0.9, 1.0])
    flags = inconsistency_flags(residual, implied, fractions, exposures, 0.2)
    assert np.array_equal(flags["residual"], residue_exceedance(residual, 0.2))
    assert np.array_equal(flags["implied_fraction"], inadmissible_fraction(implied))
    assert inconsistency_rate(flags) == pytest.approx(60.0)


def test_monotonicity_flags_both_members_of_a_step() -> None:
    flags = monotonicity_violation(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.5, 1.2]))
    assert int(flags.sum()) == 2


def test_residual_from_tensors_matches_the_module() -> None:
    reference = torch.tensor([4.0, 5.0])
    reconstruction = torch.tensor([3.0, 5.0])
    assert np.allclose(
        residual_from_tensors(reference, reconstruction),
        relative_residual(reference, reconstruction).numpy(),
    )


@pytest.mark.parametrize("mode", COUPLING_MODES)
def test_every_coupling_mode_returns_the_declared_outputs(mode: str) -> None:
    config = CouplingConfig(mode=mode, width=16, heads=2)
    module = build_coupling(config)
    molecule = torch.randn(5, 16)
    image = torch.randn(5, 16)
    output = module(molecule, image)
    assert output.embedding.shape == (5, 16)
    assert output.epsilon.shape == (5,)
    assert bool((interaction_factor(output.epsilon) > 0).all())


def test_unknown_coupling_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_coupling(CouplingConfig(mode="not_a_mode"))


def test_cross_attention_coupling_wrapper() -> None:
    module = CrossAttentionCoupling(CouplingConfig(width=16, heads=2))
    embedding, epsilon = module(torch.randn(3, 16), torch.randn(3, 16)).as_tuple()
    assert embedding.shape == (3, 16)
    assert epsilon.shape == (3,)


def test_additive_variants_share_the_interface() -> None:
    config = CouplingConfig(width=16, heads=2)
    modules = (
        ConcatenationCoupling(config),
        BilinearPoolingCoupling(config),
        GatedFusionCoupling(config),
        StackedBranchCoupling(config),
        RandomInitCoupling(config),
    )
    for module in modules:
        output = module(torch.randn(4, 16), torch.randn(4, 16))
        assert output.epsilon.shape == (4,)


def test_stacked_branch_coupling_is_additive() -> None:
    torch.manual_seed(0)
    module = StackedBranchCoupling(CouplingConfig(width=8, heads=2))
    average = torch.randn(6, 8)
    first = module(average, average).epsilon
    second = module(average * 2.0, average * 2.0).epsilon
    assert not torch.allclose(first, second, atol=1e-6)


def test_bilinear_pooling_uses_a_product() -> None:
    config = CouplingConfig(width=16, heads=2)
    module = BilinearPoolingCoupling(config, rank=4)
    with torch.no_grad():
        merged = module.merge(torch.zeros(2, 16), torch.ones(2, 16))
    assert float(merged.abs().sum()) < 1e-6
