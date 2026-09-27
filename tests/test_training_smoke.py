"""Smoke test: two optimiser steps on the smoke overlay, and a short closed loop."""

from __future__ import annotations

import numpy as np
import torch

from mechex.estimator import MechEx, MechExBatch
from mechex.fit.engine import Trainer
from mechex.objective.composite import CompositeObjective
from mechex.seeding import set_seed
from mechex.settings import EstimatorConfig

STEP_SEED = 4242


def _loss(model: MechEx, objective: CompositeObjective, batch: MechExBatch) -> float:
    with torch.no_grad():
        return float(objective(model(batch), batch).total)


def test_two_steps_reduce_the_composite_objective(
    config: EstimatorConfig, batch: MechExBatch
) -> None:
    """Two optimiser steps on the smoke overlay must lower the objective."""
    set_seed(STEP_SEED)
    model = MechEx(config)
    objective = CompositeObjective(config)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimiser.lr,
    )
    before = _loss(model, objective, batch)
    for _ in range(2):
        model.train()
        terms = objective(model(batch), batch)
        optimiser.zero_grad(set_to_none=True)
        terms.total.backward()
        optimiser.step()
    after = _loss(model, objective, batch)
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before


def test_two_steps_leave_the_frozen_encoders_unchanged(
    config: EstimatorConfig, batch: MechExBatch
) -> None:
    set_seed(STEP_SEED)
    model = MechEx(config)
    objective = CompositeObjective(config)
    before = [parameter.detach().clone() for parameter in model.vascular.encoder.parameters()][:3]
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimiser.lr,
    )
    for _ in range(2):
        terms = objective(model(batch), batch)
        optimiser.zero_grad(set_to_none=True)
        terms.total.backward()
        optimiser.step()
    after = [parameter.detach().clone() for parameter in model.vascular.encoder.parameters()][:3]
    assert all(torch.equal(first, second) for first, second in zip(before, after, strict=True))


def test_two_epoch_closed_loop_improves_on_the_smoke_overlay(config, restricted_staged) -> None:
    set_seed(STEP_SEED)
    model = MechEx(config)
    trainer = Trainer(model, config, restricted_staged, device=torch.device("cpu"))
    result = trainer.fit(STEP_SEED)
    totals = [record.train_total for record in result.history]
    assert len(totals) == config.optimiser.epochs
    assert totals[-1] < totals[0] or result.stopped_early
    assert np.isfinite(result.best_validation_ccc)


def test_smoke_overlay_is_smaller_than_the_reported_config(config) -> None:
    """The overlay must be a reduction, never the reported configuration."""
    from pathlib import Path

    from mechex.settings import load_config

    reported = load_config(Path(__file__).resolve().parents[1] / "configs")
    assert config.optimiser.batch_size < reported.optimiser.batch_size
    assert config.vascular.volume_shape != reported.vascular.volume_shape
    assert reported.optimiser.batch_size == 64
    assert reported.optimiser.patience == 12
    assert reported.optimiser.repeats == 5
    assert reported.decision.tau == 0.30
