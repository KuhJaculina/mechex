"""Delivery branch: volume preparation, the volumetric encoder and the perfusion estimate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.settings import VascularEncoderConfig
from mechex.vascular.encoder import ConvStem, TokenAttention, VolumetricCTEncoder, block_tokens
from mechex.vascular.perfusion import (
    delivery_term,
    is_monotone,
    perfused_fraction,
    perfused_fraction_map,
    phase_fixture,
)
from mechex.vascular.projection import (
    MAXIMUM_DELIVERED_FRACTION,
    MINIMUM_DELIVERED_FRACTION,
    DeliveredFractionHead,
    VascularBranch,
)
from mechex.vascular.volume import (
    fit_shape,
    prepare_pair,
    prepare_phase,
    resample_isotropic,
    to_tensor,
    window_to_contrast_range,
)


@pytest.fixture()
def vascular_config() -> VascularEncoderConfig:
    return VascularEncoderConfig(
        latent_dim=24,
        stem_channels=(8, 16, 24),
        depth=3,
        volume_shape=(16, 16, 12),
    )


def test_resample_isotropic_scales_shape() -> None:
    volume = np.zeros((8, 12, 16), dtype=np.float32)
    resampled = resample_isotropic(volume, (2.0, 1.0, 0.5), target_spacing_mm=1.0)
    assert resampled.shape == (16, 12, 8)


def test_resample_rejects_non_positive_spacing() -> None:
    with pytest.raises(ValueError):
        resample_isotropic(
            np.zeros((4, 4, 4), dtype=np.float32), (1.0, 1.0, 1.0), target_spacing_mm=0.0
        )


def test_contrast_window_is_unit_bounded() -> None:
    volume = np.linspace(-200.0, 1200.0, 500).reshape(10, 10, 5)
    windowed = window_to_contrast_range(volume, 100.0, 700.0)
    assert windowed.min() == pytest.approx(0.0)
    assert windowed.max() == pytest.approx(1.0)


def test_contrast_window_rejects_degenerate_bounds() -> None:
    with pytest.raises(ValueError):
        window_to_contrast_range(np.zeros((2, 2, 2)), 500.0, 500.0)


def test_fit_shape_crops_and_pads_about_the_centre() -> None:
    volume = np.ones((20, 8, 30), dtype=np.float32)
    fitted = fit_shape(volume, (16, 16, 16))
    assert fitted.shape == (16, 16, 16)
    assert fitted[0, 0, 0] == 0.0
    assert fitted[8, 8, 8] == 1.0


def test_prepare_phase_standardises_per_scan(vascular_config: VascularEncoderConfig) -> None:
    volume = np.abs(np.random.default_rng(0).normal(400.0, 60.0, size=(24, 24, 18)))
    prepared = prepare_phase(volume, vascular_config, spacing=(1.0, 1.0, 1.0))
    assert prepared.shape == vascular_config.volume_shape
    assert abs(float(prepared.mean())) < 1e-5


def test_prepare_pair_stacks_two_channels(vascular_config: VascularEncoderConfig) -> None:
    arterial, delayed = phase_fixture(size=(24, 24, 18), occlusion=0.4, seed=5)
    prepared = prepare_pair(arterial, delayed, vascular_config)
    assert prepared.shape == (2, *vascular_config.volume_shape)
    tensor = to_tensor(prepared)
    assert tensor.shape == (1, 2, *vascular_config.volume_shape)


def test_perfused_fraction_is_bounded_and_sensitive_to_occlusion() -> None:
    open_bed = phase_fixture(size=(12, 12, 8), occlusion=0.05, seed=1)
    closed_bed = phase_fixture(size=(12, 12, 8), occlusion=0.85, seed=1)
    open_fraction = perfused_fraction(*open_bed)
    closed_fraction = perfused_fraction(*closed_bed)
    assert 0.0 <= closed_fraction <= 1.0
    assert 0.0 <= open_fraction <= 1.0
    assert open_fraction >= closed_fraction


def test_delivery_term_respects_admissible_interval() -> None:
    phases = phase_fixture(size=(10, 10, 6), occlusion=0.99, seed=2)
    assert MINIMUM_DELIVERED_FRACTION <= delivery_term(*phases) <= MAXIMUM_DELIVERED_FRACTION


def test_perfused_fraction_map_matches_voxelwise_ratio() -> None:
    arterial, delayed = phase_fixture(size=(8, 8, 6), occlusion=0.3, seed=3)
    field = perfused_fraction_map(arterial, delayed)
    numerator = np.maximum(arterial - delayed, 0.0)
    hand = np.clip(numerator / np.maximum(arterial, 1e-6), 0.0, 1.0)
    assert np.allclose(field, hand)


def test_monotonicity_detects_a_decreasing_step() -> None:
    assert is_monotone(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.1, 1.2]))
    assert not is_monotone(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.9, 1.2]))


def test_conv_stem_and_token_attention_shapes(vascular_config: VascularEncoderConfig) -> None:
    stem = ConvStem(2, vascular_config.stem_channels, 3, 0.0)
    features = stem(torch.randn(1, 2, *vascular_config.volume_shape))
    assert features.shape[1] == vascular_config.stem_channels[-1]
    attention = TokenAttention(vascular_config.stem_channels[-1], 4, 0.0, expansion=2)
    tokens = features.flatten(2).transpose(1, 2)
    assert attention(tokens).shape == tokens.shape


def test_encoder_is_frozen_and_reports_token_budget(vascular_config: VascularEncoderConfig) -> None:
    encoder = VolumetricCTEncoder(vascular_config)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    embedding = encoder.embed_volumes(torch.randn(2, 2, *vascular_config.volume_shape))
    assert embedding.shape == (2, vascular_config.latent_dim)
    assert block_tokens(torch.zeros(1, 2, *vascular_config.volume_shape), vascular_config) > 0


def test_delivered_head_is_inside_admissible_range() -> None:
    head = DeliveredFractionHead(latent_dim=24)
    values = head(torch.randn(64, 24))
    assert bool((values > MINIMUM_DELIVERED_FRACTION).all())
    assert bool((values <= MAXIMUM_DELIVERED_FRACTION + 1e-9).all())


def test_vascular_branch_forward(vascular_config: VascularEncoderConfig) -> None:
    branch = VascularBranch(vascular_config)
    fraction, embedding = branch(torch.randn(3, 2, *vascular_config.volume_shape))
    assert fraction.shape == (3,)
    assert embedding.shape == (3, vascular_config.latent_dim)
