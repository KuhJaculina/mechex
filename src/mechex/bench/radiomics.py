"""Radiomics baseline with gradient boosting.

Ref: Table 1, B7, CTA radiomics with gradient boosting: a hand-designed image description
followed by a boosted tree ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RADIOMIC_FEATURE_NAMES = (
    "mean_enhancement",
    "enhancement_std",
    "delivered_fraction",
    "phase_contrast",
    "high_enhancement_volume_fraction",
    "contrast_gradient_magnitude",
    "radial_heterogeneity",
)


@dataclass(frozen=True)
class RadiomicSummary:
    """Hand-designed description of one imaging episode."""

    values: np.ndarray
    names: tuple[str, ...] = RADIOMIC_FEATURE_NAMES

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values, strict=True)}


def radiomic_features(volume: np.ndarray) -> RadiomicSummary:
    """Describe a prepared two-channel volume.

    The descriptors are means, spreads and contrast terms over the foreground of the scan,
    which is the description a radiomics pipeline would build before boosting.
    """
    array = np.asarray(volume, dtype=np.float64)
    if array.ndim != 4 or array.shape[0] < 2:
        raise ValueError("expected a channel-first two-channel volume")
    arterial = array[0]
    delayed = array[1]
    foreground = np.concatenate([arterial.ravel(), delayed.ravel()])
    active = foreground[foreground > np.quantile(foreground, 0.6)]
    mean_enhancement = float(active.mean()) if active.size else 0.0
    enhancement_std = float(active.std()) if active.size else 0.0
    delivered = float(np.clip((arterial - delayed).mean(), 0.0, 1.0))
    phase_contrast = float(np.abs(arterial - delayed).mean())
    high_fraction = float(np.mean(arterial > np.quantile(arterial, 0.85)))
    gradient = np.gradient(arterial)
    gradient_magnitude = float(np.sqrt(sum(component**2 for component in gradient)).mean())
    centre = np.array(arterial.shape) // 2
    mesh = np.stack(
        np.meshgrid(
            np.arange(arterial.shape[0]) - centre[0],
            np.arange(arterial.shape[1]) - centre[1],
            np.arange(arterial.shape[2]) - centre[2],
            indexing="ij",
        ),
        axis=0,
    )
    radius = np.sqrt((mesh**2).sum(axis=0)) + 1e-6
    radial_heterogeneity = float(np.corrcoef(radius.ravel(), arterial.ravel())[0, 1])
    return RadiomicSummary(
        values=np.array(
            [
                mean_enhancement,
                enhancement_std,
                delivered,
                phase_contrast,
                high_fraction,
                gradient_magnitude,
                radial_heterogeneity,
            ],
            dtype=np.float64,
        )
    )


def radiomic_matrix(volumes: list[np.ndarray]) -> np.ndarray:
    """Stack radiomic features for a list of volumes."""
    if not volumes:
        return np.zeros((0, len(RADIOMIC_FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([radiomic_features(volume).values for volume in volumes])


def gradient_boosting_fit(
    train: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
    n_estimators: int = 120,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> np.ndarray:
    """Boosted-tree regression on the radiomic block.

    Falls back to a ridge fit when the boosting backend is unavailable, so the baseline
    still reports a number instead of failing the whole comparison.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:  # pragma: no cover - the release ships scikit-learn
        from mechex.bench.mechanistic import descriptor_ridge

        return descriptor_ridge(train, target, evaluation)

    model = HistGradientBoostingRegressor(
        max_iter=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0,
    )
    model.fit(np.asarray(train, dtype=np.float64), np.asarray(target, dtype=np.float64))
    prediction = model.predict(np.asarray(evaluation, dtype=np.float64))
    return np.clip(np.asarray(prediction, dtype=np.float64), 1e-4, None)
