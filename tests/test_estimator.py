"""The assembled estimator, its ablation switches and the objective."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.estimator import (
    LOG_FLOOR,
    MechEx,
    apply_ablation,
    count_parameters,
    count_parameters_millions,
    frozen_module_names,
)
from mechex.objective.calibration import (
    adequacy_labels,
    adequacy_loss,
    brier_term,
    calibration_gap,
    evidential_loss,
)
from mechex.objective.composite import CompositeObjective, ObjectiveTerms
from mechex.objective.concordance import (
    concordance_ccc,
    concordance_loss,
    log_concordance_loss,
    pearson_penalty,
    squared_log_error,
)
from mechex.settings import AblationConfig, EstimatorConfig
from tests.conftest import make_batch


def test_forward_emits_every_declared_quantity(config: EstimatorConfig, batch) -> None:
    model = MechEx(config)
    output = model(batch)
    assert output.partition.shape == (batch.batch_size,)
    assert output.delivered_fraction.shape == (batch.batch_size,)
    assert output.interaction.shape == (batch.batch_size,)
    assert output.exposure.shape == (batch.batch_size,)
    assert output.adequacy_logit.shape == (batch.batch_size,)


def test_outputs_are_finite_and_admissible(config: EstimatorConfig, batch) -> None:
    output = MechEx(config)(batch)
    assert torch.isfinite(output.exposure).all()
    assert bool((output.partition > 0).all())
    assert bool((1.0 + output.interaction > 0).all())
    assert bool((output.exposure_variance > 0).all())


def test_conservation_reconstruction_matches_a_hand_product(config: EstimatorConfig, batch) -> None:
    output = MechEx(config)(batch)
    hand = output.partition * output.delivered_fraction * (1.0 + output.interaction)
    assert torch.allclose(hand, output.availability_reconstruction, atol=1e-5)


def test_residual_matches_a_hand_computation(config: EstimatorConfig, batch) -> None:
    output = MechEx(config)(batch)
    hand = (
        output.availability_reference - output.availability_reconstruction
    ).abs() / output.availability_reference.abs()
    assert torch.allclose(hand, output.residual, atol=1e-6)


def test_encoders_are_frozen(config: EstimatorConfig) -> None:
    model = MechEx(config)
    assert all(not parameter.requires_grad for parameter in model.molecular.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.vascular.encoder.parameters())
    assert model.frozen_encoder_parameters() > 0


def test_gradient_reaches_heads_only(config: EstimatorConfig, batch) -> None:
    model = MechEx(config)
    output = model(batch)
    loss = (output.exposure - batch.exposure_reference.clamp_min(LOG_FLOOR)).pow(2).mean()
    loss.backward()
    head_gradients = [parameter.grad for parameter in model.molecular.head.parameters()]
    assert any(
        gradient is not None and float(gradient.abs().sum()) > 0 for gradient in head_gradients
    )
    assert all(
        parameter.grad is None
        for parameter in model.molecular.encoder.parameters()
        if parameter.requires_grad
    )


def test_parameter_update_moves_heads_and_holds_encoders(config: EstimatorConfig, batch) -> None:
    model = MechEx(config)
    before = [parameter.detach().clone() for parameter in model.vascular.encoder.parameters()][:3]
    head_before = [parameter.detach().clone() for parameter in model.vascular.head.parameters()]
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2
    )
    output = model(batch)
    loss = (output.exposure - batch.exposure_reference.clamp_min(LOG_FLOOR)).pow(2).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
    after = [parameter.detach().clone() for parameter in model.vascular.encoder.parameters()][:3]
    head_after = [parameter.detach().clone() for parameter in model.vascular.head.parameters()]
    assert all(torch.equal(first, second) for first, second in zip(before, after, strict=True))
    assert any(
        not torch.equal(first, second)
        for first, second in zip(head_before, head_after, strict=True)
    )


def test_calibration_installs_the_recalibrator(config: EstimatorConfig) -> None:
    model = MechEx(config)
    model.calibrate(make_batch(config, n_records=6))
    assert model.uncertainty.recalibrator.is_fitted()


def test_calibration_requires_a_reference(config: EstimatorConfig) -> None:
    model = MechEx(config)
    batch = make_batch(config, n_records=3)
    batch.exposure_reference = None
    with pytest.raises(ValueError):
        model.calibrate(batch)


def test_molecular_branch_removal_uses_a_constant_partition(config: EstimatorConfig) -> None:
    from dataclasses import replace

    ablated = replace(config, ablation=AblationConfig(molecular_branch=False))
    model = MechEx(ablated)
    values = model(make_batch(ablated, n_records=3)).partition.detach()
    assert float(values.max() - values.min()) < 1e-6


def test_imaging_branch_removal_uses_a_constant_delivery(config: EstimatorConfig) -> None:
    from dataclasses import replace

    ablated = replace(config, ablation=AblationConfig(imaging_branch=False))
    values = MechEx(ablated)(make_batch(ablated, n_records=3)).delivered_fraction.detach()
    assert float(values.max() - values.min()) < 1e-6


def test_decision_layer_removal_zeroes_the_logit(config: EstimatorConfig) -> None:
    from dataclasses import replace

    ablated = replace(config, ablation=AblationConfig(decision_layer=False))
    output = MechEx(ablated)(make_batch(ablated, n_records=3))
    assert float(output.adequacy_logit.abs().sum()) == pytest.approx(0.0)


def test_full_finetuning_unfreezes_the_encoders(config: EstimatorConfig) -> None:
    from dataclasses import replace

    tuned = replace(config, ablation=AblationConfig(adaptation="full_finetune"))
    model = MechEx(tuned)
    assert any(parameter.requires_grad for parameter in model.molecular.encoder.parameters())


def test_frozen_without_adaptation_freezes_the_heads(config: EstimatorConfig) -> None:
    from dataclasses import replace

    frozen = replace(config, ablation=AblationConfig(adaptation="frozen_no_head"))
    model = MechEx(frozen)
    assert all(not parameter.requires_grad for parameter in model.molecular.head.parameters())


def test_random_initialisation_changes_the_encoder_weights(config: EstimatorConfig) -> None:
    from dataclasses import replace

    torch.manual_seed(11)
    pretrained = MechEx(config)
    random_config = replace(config, ablation=AblationConfig(initialisation="random"))
    torch.manual_seed(11)
    randomised = MechEx(random_config)
    difference = (
        next(pretrained.molecular.encoder.parameters()).detach()
        - next(randomised.molecular.encoder.parameters()).detach()
    )
    assert float(difference.abs().sum()) > 0


def test_unknown_adaptation_is_rejected(config: EstimatorConfig) -> None:
    from dataclasses import replace

    model = MechEx(config)
    model.config = replace(config, ablation=AblationConfig(adaptation="not_a_mode"))
    with pytest.raises(ValueError):
        apply_ablation(model)


def test_parameter_counting_helpers(config: EstimatorConfig) -> None:
    model = MechEx(config)
    assert count_parameters(model) > count_parameters(model, trainable_only=True)
    assert count_parameters_millions(model) > 0.0
    assert "molecular" in " ".join(frozen_module_names(model))


def test_composite_objective_returns_all_terms(config: EstimatorConfig, batch) -> None:
    model = MechEx(config)
    objective = CompositeObjective(config)
    terms = objective(model(batch), batch)
    assert isinstance(terms, ObjectiveTerms)
    assert torch.isfinite(terms.total)
    for value in terms.scalar_dict().values():
        assert np.isfinite(value)


def test_objective_requires_a_reference(config: EstimatorConfig) -> None:
    batch = make_batch(config, n_records=2)
    batch.exposure_reference = None
    with pytest.raises(ValueError):
        CompositeObjective(config)(MechEx(config)(batch), batch)


def test_concordance_ccc_bounds() -> None:
    values = torch.linspace(0.0, 1.0, 16)
    assert float(concordance_ccc(values, values)) == pytest.approx(1.0, abs=1e-6)
    assert float(concordance_loss(values, values)) == pytest.approx(0.0, abs=1e-6)


def test_log_concordance_loss_penalises_a_scale_error() -> None:
    reference = torch.linspace(-1.0, 1.0, 8)
    off = reference + 1.0
    assert float(log_concordance_loss(off, reference)) > float(
        log_concordance_loss(reference, reference)
    )


def test_squared_log_error_is_zero_for_a_match() -> None:
    values = torch.randn(6)
    assert float(squared_log_error(values, values)) == pytest.approx(0.0)


def test_pearson_penalty_ignores_an_offset() -> None:
    values = torch.linspace(0.0, 1.0, 20)
    assert float(pearson_penalty(values + 3.0, values)) == pytest.approx(0.0, abs=1e-5)


def test_adequacy_labels_and_loss() -> None:
    exposure = torch.tensor([0.9, 1.4])
    labels = adequacy_labels(exposure, 0.30)
    assert labels.tolist() == [0.0, 1.0]
    assert torch.isfinite(adequacy_loss(torch.tensor([-1.0, 1.0]), labels))


def test_brier_and_calibration_gap_are_finite() -> None:
    probabilities = torch.tensor([0.2, 0.8])
    labels = torch.tensor([0.0, 1.0])
    assert float(brier_term(probabilities, labels)) == pytest.approx(0.04, abs=1e-6)
    assert np.isfinite(float(calibration_gap(probabilities, labels, bins=5)))


def test_evidential_loss_is_finite() -> None:
    from mechex.uncertainty.evidential import EvidentialHead

    head = EvidentialHead(6, 4)
    output = head(torch.randn(4, 6))
    assert torch.isfinite(evidential_loss(torch.randn(4), output))
