"""Repeated runs with different random initialisations.

Ref: Sec. 3.8, five independent repeats with different random initialisations, with the
reported uncertainty taken from patient-level bootstrap resampling within each site after
averaging across repeats.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mechex.settings import EstimatorConfig


@dataclass
class RepeatSummary:
    """Per-repeat metrics with their across-repeat mean and spread."""

    repeats: list[dict[str, float]] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_repeats": len(self.repeats),
            "seeds": list(self.seeds),
            "mean": dict(self.mean),
            "spread": dict(self.spread),
        }


def repeat_seeds(base_seed: int, n_repeats: int, stride: int = 1013) -> list[int]:
    """Deterministic distinct seeds for the repeats."""
    if n_repeats <= 0:
        raise ValueError("at least one repeat is required")
    return [int(base_seed + index * stride) for index in range(n_repeats)]


def run_repeats(
    fit_callable: Callable[[int], dict[str, float]],
    seeds: list[int],
) -> RepeatSummary:
    """Run a fit once per seed and aggregate the reported metrics.

    The fit callable receives the seed and returns the metrics of that repeat, so the same
    harness serves the real trainer and the smoke path.
    """
    summary = RepeatSummary(seeds=list(seeds))
    for seed in seeds:
        metrics = fit_callable(int(seed))
        summary.repeats.append({key: float(value) for key, value in metrics.items()})
    keys = sorted({key for repeat in summary.repeats for key in repeat})
    for key in keys:
        values = np.array(
            [repeat.get(key, float("nan")) for repeat in summary.repeats], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        summary.mean[key] = float(finite.mean()) if finite.size else float("nan")
        summary.spread[key] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return summary


def aggregate_across_repeats(per_repeat: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean across repeats of an array-valued metric."""
    if not per_repeat:
        raise ValueError("no repeats to aggregate")
    stacked = np.stack([np.asarray(item, dtype=np.float64) for item in per_repeat], axis=0)
    aggregated: np.ndarray = stacked.mean(axis=0)
    return aggregated


def config_for_repeat(config: EstimatorConfig, seed: int) -> EstimatorConfig:
    """Copy of a config with the repeat seed applied."""
    from dataclasses import replace

    optimiser = replace(config.optimiser, seed=int(seed))
    return replace(config, optimiser=optimiser)
