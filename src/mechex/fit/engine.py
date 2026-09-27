"""Training and evaluation engine.

Ref: Sec. 3.8, AdamW with cosine decay, batch size 64 patients, early stopping on the
validation partition with a patience of twelve epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mechex.cohort.cohort import CohortCollator, CohortItem
from mechex.cohort.construction import StagedCohort
from mechex.estimator import MechEx, MechExBatch
from mechex.fit.checkpoint import build_checkpoint, save_checkpoint
from mechex.objective.composite import CompositeObjective
from mechex.optimise.optimiser import EarlyStopping, autocast_dtype, build_optimizer, clip_gradients
from mechex.optimise.schedule import build_scheduler, current_lr, steps_per_epoch
from mechex.quantify.concordance import lin_ccc
from mechex.quantify.roc import auroc
from mechex.seeding import resolve_device, set_seed
from mechex.settings import EstimatorConfig
from mechex.utils.logging import get_logger

_log = get_logger("fit.engine")


@dataclass
class EpochRecord:
    """One epoch of the training history."""

    epoch: int
    train_total: float
    train_terms: dict[str, float]
    validation_ccc: float
    validation_auroc: float
    learning_rate: float
    gradient_norm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "epoch": float(self.epoch),
            "train_total": float(self.train_total),
            "validation_ccc": float(self.validation_ccc),
            "validation_auroc": float(self.validation_auroc),
            "learning_rate": float(self.learning_rate),
            "gradient_norm": float(self.gradient_norm),
            **{f"train_{key}": float(value) for key, value in self.train_terms.items()},
        }


@dataclass
class FitResult:
    """Outcome of one training run."""

    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_ccc: float = float("nan")
    stopped_early: bool = False
    seed: int = 0
    checkpoint: Path | None = None
    calibrator: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_ccc": self.best_validation_ccc,
            "stopped_early": self.stopped_early,
            "seed": self.seed,
            "n_epochs": len(self.history),
            "final_train_total": self.history[-1].train_total if self.history else float("nan"),
            "first_train_total": self.history[0].train_total if self.history else float("nan"),
        }


def _dataloader(
    staged: StagedCohort,
    config: EstimatorConfig,
    partition: str,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader[CohortItem]:
    dataset = staged.dataset(partition, config)
    collator = CohortCollator(dataset.molecule_batch)
    return DataLoader(
        dataset,
        batch_size=config.optimiser.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _predict(model: nn.Module, batch: MechExBatch) -> tuple[torch.Tensor, torch.Tensor]:
    from mechex.estimator import MechExOutput

    output = model(batch)
    assert isinstance(output, MechExOutput)
    return output.exposure, output.exposure_std


@torch.no_grad()
def evaluate_partition(
    model: nn.Module,
    staged: StagedCohort,
    config: EstimatorConfig,
    partition: str,
) -> dict[str, float]:
    """Concordance, discrimination and fold error on one partition.

    The forward pass runs in eval mode with a single call per batch, so the reported
    deviation and the point estimate come from the same pass.
    """
    from mechex.quantify.fold import geometric_fold_error

    model.eval()
    loader = _dataloader(staged, config, partition, shuffle=False, generator=None)
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for batch in loader:
        if batch.exposure_reference is None:
            continue
        prediction, deviation = _predict(model, batch)
        predictions.append(prediction.detach().cpu().numpy())
        references.append(batch.exposure_reference.detach().cpu().numpy())
        stds.append(deviation.detach().cpu().numpy())
    if not predictions:
        return {"ccc": float("nan"), "auroc": float("nan"), "fold_error": float("nan"), "n": 0.0}
    predicted_values: np.ndarray = np.concatenate(predictions)
    reference_values: np.ndarray = np.concatenate(references)
    deviation_values: np.ndarray = np.concatenate(stds)
    labels = (reference_values >= 1.0 + config.decision.tau).astype(np.int64)
    discrimination = auroc(predicted_values, labels) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "ccc": float(lin_ccc(predicted_values, reference_values)),
        "auroc": float(discrimination),
        "fold_error": float(geometric_fold_error(predicted_values, reference_values)),
        "mean_deviation": float(np.mean(deviation_values)),
        "n": float(predicted_values.size),
    }


class Trainer:
    """Trains one MechEx run on the development partition."""

    def __init__(
        self,
        model: MechEx,
        config: EstimatorConfig,
        staged: StagedCohort,
        device: torch.device | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.staged = staged
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device)
        self.objective = CompositeObjective(config)
        self.optimizer = build_optimizer(self.model, config.optimiser)
        self.checkpoint_dir = checkpoint_dir
        self.n_development = int(staged.split.development.size)
        self.n_steps = steps_per_epoch(
            self.n_development, config.optimiser.batch_size, config.optimiser.grad_accum
        ) * max(1, config.optimiser.epochs)
        self.scheduler = build_scheduler(self.optimizer, self.n_steps, config.optimiser)
        self.amp_dtype = autocast_dtype(config.optimiser.precision)
        self.stopping = EarlyStopping(patience=config.optimiser.patience, mode="max")

    def _autocast(self) -> torch.autocast:
        device_type = self.device.type
        if self.amp_dtype is None:
            return torch.autocast(device_type=device_type, enabled=False)
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=True)

    def train_epoch(
        self, epoch: int, generator: torch.Generator | None
    ) -> tuple[float, dict[str, float], float]:
        self.model.train()
        loader = _dataloader(
            self.staged, self.config, "development", shuffle=True, generator=generator
        )
        totals: dict[str, float] = {}
        running = 0.0
        count = 0
        last_norm = 0.0
        accumulation = max(1, self.config.optimiser.grad_accum)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            with self._autocast():
                output = self.model(batch)
                terms = self.objective(output, batch)
            loss = terms.total / accumulation
            loss.backward()
            running += float(terms.total.detach())
            count += 1
            for key, value in terms.scalar_dict().items():
                totals[key] = totals.get(key, 0.0) + value
            if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
                last_norm = clip_gradients(self.model, self.config.optimiser.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        mean_terms = {key: value / max(count, 1) for key, value in totals.items()}
        return running / max(count, 1), mean_terms, last_norm

    def fit(self, seed: int | None = None) -> FitResult:
        """Run the schedule with early stopping on validation concordance."""
        resolved_seed = set_seed(self.config.optimiser.seed if seed is None else seed)
        generator = torch.Generator()
        generator.manual_seed(resolved_seed)
        result = FitResult(seed=resolved_seed)
        for epoch in range(self.config.optimiser.epochs):
            total, terms, norm = self.train_epoch(epoch, generator)
            validation = evaluate_partition(self.model, self.staged, self.config, "validation")
            record = EpochRecord(
                epoch=epoch,
                train_total=total,
                train_terms=terms,
                validation_ccc=validation["ccc"],
                validation_auroc=validation["auroc"],
                learning_rate=current_lr(self.optimizer),
                gradient_norm=norm,
            )
            result.history.append(record)
            _log.info(
                "epoch %d total %.4f validation ccc %.4f",
                epoch,
                total,
                validation["ccc"],
            )
            stop = self.stopping.update(validation["ccc"], epoch)
            if stop:
                result.stopped_early = True
                break
        result.best_epoch = self.stopping.best_epoch
        result.best_validation_ccc = float(self.stopping.best)
        if self.checkpoint_dir is not None:
            result.checkpoint = self.save()
            result.calibrator = self.calibrate()
        return result

    def calibrate(self) -> dict[str, object] | None:
        """Fit the isotonic recalibration on the calibration partition."""
        loader = _dataloader(self.staged, self.config, "calibration", shuffle=False, generator=None)
        model = self.model
        model.eval()
        with torch.no_grad():
            for batch in loader:
                model.calibrate(batch.to(self.device))
                break
        return model.uncertainty.recalibrator.as_dict()

    def save(self, name: str = "mechex.pt") -> Path:
        """Write a checkpoint holding the model, optimiser, seed and calibration."""
        assert self.checkpoint_dir is not None
        metrics = evaluate_partition(self.model, self.staged, self.config, "validation")
        payload = build_checkpoint(
            self.model,
            self.optimizer,
            epoch=self.stopping.best_epoch,
            seed=self.config.optimiser.seed,
            config=self.config,
            metrics=metrics,
            calibrator=self.model.uncertainty.recalibrator.as_dict(),
        )
        return save_checkpoint(self.checkpoint_dir / name, payload)


def fit_once(
    model: MechEx,
    config: EstimatorConfig,
    staged: StagedCohort,
    seed: int | None = None,
    checkpoint_dir: Path | None = None,
) -> FitResult:
    """Convenience wrapper around :class:`Trainer`."""
    trainer = Trainer(model, config, staged, checkpoint_dir=checkpoint_dir)
    return trainer.fit(seed)
