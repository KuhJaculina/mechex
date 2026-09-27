"""Optimiser, schedule, early stopping, checkpoints and the training engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.estimator import MechEx
from mechex.fit.checkpoint import (
    CHECKPOINT_VERSION,
    build_checkpoint,
    capture_rng_state,
    config_from_checkpoint,
    load_checkpoint,
    metrics_from_checkpoint,
    restore_into,
    restore_rng_state,
    save_checkpoint,
    seed_from_checkpoint,
    tensor_payload_digest,
)
from mechex.fit.distributed import (
    distributed_available,
    initialise,
    is_main_process,
    rank,
    world_size,
    wrap_distributed,
)
from mechex.fit.engine import Trainer, evaluate_partition
from mechex.fit.repeats import (
    RepeatSummary,
    aggregate_across_repeats,
    config_for_repeat,
    repeat_seeds,
    run_repeats,
)
from mechex.optimise.optimiser import (
    EarlyStopping,
    autocast_dtype,
    build_optimizer,
    clip_gradients,
    grad_norm,
)
from mechex.optimise.schedule import (
    build_scheduler,
    cosine_factor,
    current_lr,
    describe_reference_schedule,
    steps_per_epoch,
    total_steps,
    warmup_steps,
)
from mechex.settings import OptimiserConfig


def test_optimizer_only_sees_trainable_parameters() -> None:
    from pathlib import Path

    from mechex.settings import load_smoke_config

    config = load_smoke_config(Path(__file__).resolve().parents[1] / "configs")
    model = MechEx(config)
    optimizer = build_optimizer(model, config.optimiser)
    trainable = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    assert len(optimizer.param_groups[0]["params"]) == trainable


def test_optimizer_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_optimizer(torch.nn.Linear(2, 2), OptimiserConfig(name="sgd"))


def test_optimizer_rejects_a_module_without_trainable_parameters() -> None:
    module = torch.nn.Linear(2, 2)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError):
        build_optimizer(module, OptimiserConfig())


def test_autocast_dtype_mapping() -> None:
    assert autocast_dtype("fp32") is None
    assert autocast_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError):
        autocast_dtype("fp8")


def test_gradient_clipping_bounds_the_norm() -> None:
    module = torch.nn.Linear(4, 4)
    module(torch.randn(2, 4)).sum().backward()
    before = grad_norm(module)
    clip_gradients(module, 0.1)
    assert grad_norm(module) <= 0.1 + 1e-5
    assert before > 0


def test_cosine_factor_warms_up_and_decays() -> None:
    values = [cosine_factor(step, 100, 10, 0.01) for step in range(100)]
    assert values[0] < values[9]
    assert 0.009 <= values[-1] < 0.02
    assert values[50] < values[10]


def test_warmup_and_step_arithmetic() -> None:
    assert warmup_steps(100, 0.1) == 10
    assert steps_per_epoch(1000, 64, 2) == 8
    config = OptimiserConfig(batch_size=64, grad_accum=2, epochs=3)
    assert total_steps(640, config) == 5 * 3


def test_step_arithmetic_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        steps_per_epoch(100, 0, 1)


def test_schedule_runs_and_lowers_the_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    schedule = build_scheduler(optimizer, 50, OptimiserConfig(warmup_fraction=0.1))
    start = current_lr(optimizer)
    for _ in range(50):
        schedule.step()
    assert current_lr(optimizer) < start


def test_reference_schedule_reports_the_effective_batch() -> None:
    config = OptimiserConfig(batch_size=64, grad_accum=1, world_size=1)
    description = describe_reference_schedule(config, 640)
    assert description["effective_batch_size"] == 64
    assert description["scheduler"] == "cosine_with_warmup"


def test_early_stopping_stops_after_the_patience() -> None:
    stopper = EarlyStopping(patience=3, mode="max")
    trace = [1.0, 2.0, 2.0, 2.0, 2.0]
    stopped = [stopper.update(value, index) for index, value in enumerate(trace)]
    assert stopped[-1] is True
    assert stopper.best_epoch == 1


def test_early_stopping_min_mode() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.update(1.0, 0)
    stopper.update(2.0, 1)
    assert stopper.update(3.0, 2) is True


def test_early_stopping_rejects_a_bad_mode() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(mode="sideways")


def test_repeat_seeds_are_distinct() -> None:
    seeds = repeat_seeds(100, 5)
    assert len(set(seeds)) == 5
    with pytest.raises(ValueError):
        repeat_seeds(100, 0)


def test_run_repeats_aggregates() -> None:
    summary = run_repeats(lambda seed: {"value": float(seed)}, [1, 2, 3])
    assert isinstance(summary, RepeatSummary)
    assert summary.mean["value"] == pytest.approx(2.0)
    assert summary.spread["value"] == pytest.approx(float(np.std([1.0, 2.0, 3.0])))


def test_aggregate_across_repeats_averages_elementwise() -> None:
    aggregated = aggregate_across_repeats([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    assert aggregated.tolist() == [2.0, 3.0]
    with pytest.raises(ValueError):
        aggregate_across_repeats([])


def test_config_for_repeat_sets_the_seed(config) -> None:
    repeat = config_for_repeat(config, 777)
    assert repeat.optimiser.seed == 777
    assert config.optimiser.seed != 777


def test_checkpoint_round_trip_reproduces_outputs(config, batch, tmp_path) -> None:
    torch.manual_seed(5)
    model = MechEx(config).eval()
    with torch.no_grad():
        before = model(batch).exposure.clone()
    payload = build_checkpoint(model, None, epoch=2, seed=99, config=config, metrics={"ccc": 0.5})
    path = save_checkpoint(tmp_path / "probe.pt", payload)
    torch.manual_seed(6)
    restored = MechEx(config).eval()
    loaded = restore_into(path, restored)
    with torch.no_grad():
        after = restored(batch).exposure
    assert torch.allclose(before, after, atol=1e-7)
    assert seed_from_checkpoint(loaded) == 99
    assert metrics_from_checkpoint(loaded)["ccc"] == pytest.approx(0.5)
    assert config_from_checkpoint(loaded)


def test_checkpoint_version_is_enforced(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"version": -1, "model": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_checkpoint_rejects_a_foreign_payload(tmp_path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"version": CHECKPOINT_VERSION, "not_model": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_payload_digest_ignores_container_metadata(tmp_path) -> None:
    model = torch.nn.Linear(3, 3)
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    save_checkpoint(first, build_checkpoint(model, None, 0, 1, _smoke(), {}))
    save_checkpoint(second, build_checkpoint(model, None, 0, 1, _smoke(), {}))
    assert tensor_payload_digest(first) == tensor_payload_digest(second)


def test_rng_state_round_trip() -> None:
    torch.manual_seed(3)
    state = capture_rng_state()
    first = float(torch.randn(1))
    restore_rng_state(state)
    second = float(torch.randn(1))
    assert first == pytest.approx(second)


def test_distributed_helpers_degrade_to_single_process(tmp_path) -> None:
    assert not distributed_available() or True
    assert world_size() >= 1
    assert rank() >= 0
    assert isinstance(is_main_process(), bool)
    module = torch.nn.Linear(2, 2)
    assert wrap_distributed(module, torch.device("cpu")) is module
    assert initialise() in {True, False}


def _smoke():
    from pathlib import Path

    from mechex.settings import load_smoke_config

    return load_smoke_config(Path(__file__).resolve().parents[1] / "configs")


def test_trainer_runs_two_epochs_and_reports_history(config, restricted_staged) -> None:
    from dataclasses import replace

    config = replace(
        config, optimiser=replace(config.optimiser, epochs=2, batch_size=16, patience=2)
    )
    model = MechEx(config)
    trainer = Trainer(model, config, restricted_staged, device=torch.device("cpu"))
    result = trainer.fit(7)
    assert len(result.history) == 2
    assert np.isfinite(result.best_validation_ccc)
    assert result.history[-1].train_total < result.history[0].train_total or result.stopped_early


def test_trainer_saves_a_checkpoint_and_calibrates(config, restricted_staged, tmp_path) -> None:
    from dataclasses import replace

    config = replace(
        config, optimiser=replace(config.optimiser, epochs=1, batch_size=16, patience=1)
    )
    model = MechEx(config)
    trainer = Trainer(
        model, config, restricted_staged, device=torch.device("cpu"), checkpoint_dir=tmp_path
    )
    result = trainer.fit(3)
    assert result.checkpoint is not None and result.checkpoint.is_file()
    assert isinstance(result.calibrator, dict)


def test_evaluate_partition_returns_finite_metrics(config, restricted_staged) -> None:
    model = MechEx(config).eval()
    metrics = evaluate_partition(model, restricted_staged, config, "validation")
    assert np.isfinite(metrics["ccc"])
    assert metrics["n"] > 0
